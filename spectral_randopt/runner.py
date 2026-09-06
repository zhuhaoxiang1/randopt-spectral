"""Sequential, reproducible random search and independent held-out evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .backend import TransformersBackend
from .config import RunConfig
from .data import load_splits, majority_vote, summarize
from .perturbations import Method, stable_seed
from .targets import WeightManager
from .original import OriginalWeightManager


@dataclass(frozen=True)
class Candidate:
    id: str
    seed: int
    rho: float
    sign: int
    pair_id: str
    sigma: float | None = None


def make_candidates(config: RunConfig) -> list[Candidate]:
    config.validate()
    if config.candidate_sampling == 'randopt':
        # Match randopt.py: unique seeds, then IID draws from the sigma grid.
        rng = np.random.default_rng(config.seed)
        seeds = rng.choice(2**31, size=config.population, replace=False).tolist()
        indices = rng.choice(len(config.original_sigmas), size=config.population).tolist()
        return [Candidate(f'c{i:04d}', seed, config.rhos[j], 1, f'c{i:04d}', config.original_sigmas[j])
                for i, (seed, j) in enumerate(zip(seeds, indices))]
    candidates = []
    signs = (1, -1) if config.antithetic else (1,)
    pairs_per_radius = config.population // (len(config.rhos) * len(signs))
    for ri, rho in enumerate(config.rhos):
        for pi in range(pairs_per_radius):
            pair_id = f'r{ri}-p{pi:04d}'
            seed = stable_seed(config.seed, pair_id)
            for sign in signs:
                candidates.append(Candidate(f'c{len(candidates):04d}', seed, rho, sign, pair_id))
    return candidates


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _compact(predictions: list[dict], save_texts: bool) -> list[dict]:
    return [{k: v for k, v in p.items() if save_texts or k != 'text'} for p in predictions]


def _prediction_record(backend, examples, config) -> dict:
    started = time.perf_counter()
    predictions = backend.generate(examples)
    return {'metrics': summarize(predictions), 'predictions': _compact(predictions, config.save_texts),
            'elapsed_seconds': time.perf_counter() - started}


def _select(records: list[dict], k: int) -> list[dict]:
    return sorted(records, key=lambda r: (-r['selection']['metrics']['accuracy'], r['candidate']['id']))[:k]


def run_experiment(config: RunConfig, backend=None, resume: bool = False, search_only: bool = False) -> Path:
    run_started = time.perf_counter()
    config.validate()
    selection, probe, test, references = load_splits(config)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'{output} is not empty; use --resume or a new output directory')
    backend = backend or TransformersBackend.from_config(config)
    linalg_device = str(backend.device) if config.linalg_device == 'auto' else config.linalg_device
    manager = WeightManager(backend.model, config.projections, config.layers,
                            cache_dir=output / 'svd-cache', linalg_device=linalg_device,
                            diagnostic_heads=config.diagnostic_heads, checkpoint_id=backend.model_id,
                            head_indices=config.head_indices, cache_entries=config.svd_cache_entries)
    methods = [Method(**m) for m in config.methods]
    original = OriginalWeightManager(backend.model, manager) if any(m.kind == 'original' for m in methods) else None

    def apply(method, candidate):
        if method.kind == 'original':
            return original.apply(candidate.seed, candidate.sigma)
        return manager.apply(method, candidate.seed, candidate.rho, candidate.sign)
    root = Path(__file__).resolve().parents[1]
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=False)
    identity = {
        'schema_version': 2, 'config': config.to_dict(), 'model_id': backend.model_id,
        'code_sha256': _code_hash(), 'upstream_commit': git.stdout.strip(),
        'versions': {p: version(p) for p in ('torch', 'transformers', 'numpy')},
        'device': str(backend.device), 'linalg_device': linalg_device,
        'files': {key: _hash_file(getattr(config, key)) for key in ('train_path', 'test_path', 'reference_path')},
        'selection_ids': [e.id for e in selection], 'probe_ids': [e.id for e in probe],
        'test_ids': [e.id for e in test], 'targets': manager.manifest(),
        'candidates': [asdict(c) for c in make_candidates(config)],
        'smoke_only': backend.model_id.startswith('offline-random-'),
        'target_scope': 'individual attention-head matrices; concatenated layer spectra need not be invariant',
        'original_baseline': original.manifest() if original is not None else None,
        'method_scopes': {m.name: ('all_text_parameters' if m.kind == 'original' else 'selected_attention_heads')
                          for m in methods},
    }
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if read_json(manifest_path) != identity:
            raise ValueError('Resume rejected: config, code, model, data, or software identity changed')
    else:
        write_json(manifest_path, identity)
    reference_inputs = backend.reference_inputs(references)
    base_log_probs = [backend.reference_log_probs(ids) for ids in reference_inputs]
    base_path = output / 'base.json'
    if base_path.exists():
        base = read_json(base_path)
    else:
        print('Evaluating base model on selection and held-out probe sets.', flush=True)
        base = {name: _prediction_record(backend, examples, config)
                for name, examples in [('selection', selection), ('probe', probe)]}
        base['reference'] = backend.diagnostics(reference_inputs, base_log_probs)
        write_json(base_path, base)
    candidates = make_candidates(config)
    records_by_method = {m.name: [] for m in methods}
    print(f'{len(methods)} methods, {len(candidates)} candidates each; interleaved search.', flush=True)
    for i, candidate in enumerate(candidates):
        for method in methods:
            path = output / 'candidates' / method.name / f'{candidate.id}.json'
            if path.exists():
                record = read_json(path)
                if record['candidate'] != asdict(candidate) or record['method'] != method.to_dict():
                    raise ValueError(f'Candidate metadata mismatch: {path}')
            else:
                started = time.perf_counter()
                with apply(method, candidate) as perturbation:
                    apply_seconds = time.perf_counter() - started
                    selection_record = _prediction_record(backend, selection, config)
                    probe_record = _prediction_record(backend, probe, config)
                    ref_started = time.perf_counter()
                    reference_record = backend.diagnostics(reference_inputs, base_log_probs)
                    reference_seconds = time.perf_counter() - ref_started
                    record = {
                        'candidate': asdict(candidate), 'method': method.to_dict(),
                        'selection': selection_record, 'probe': probe_record, 'reference': reference_record,
                        'perturbation': perturbation,
                    }
                record['elapsed_seconds'] = time.perf_counter() - started
                record['timing'] = {'apply_and_weight_diagnostics_seconds': apply_seconds,
                                    'reference_seconds': reference_seconds,
                                    'restore_seconds': max(0., record['elapsed_seconds'] - apply_seconds -
                                        selection_record['elapsed_seconds'] - probe_record['elapsed_seconds'] - reference_seconds)}
                write_json(path, record)
            records_by_method[method.name].append(record)
            print(f'  {method.name} {i+1}/{len(candidates)}: selection={record["selection"]["metrics"]["accuracy"]:.4f}, '
                  f'probe={record["probe"]["metrics"]["accuracy"]:.4f}', flush=True)

    for method in methods:
        records = records_by_method[method.name]
        selected = _select(records, max(config.top_k))
        write_json(output / 'selected' / f'{method.name}.json', {
            'criterion': 'selection flexible answer accuracy, then candidate ID',
            'candidate_ids': [r['candidate']['id'] for r in selected]})
    write_json(output / 'search_complete.json', {'status': 'search_complete', 'population': config.population})
    if search_only:
        write_json(output / 'timing.json', {'last_invocation_seconds': time.perf_counter() - run_started,
                                          'note': 'Search only; test predictions have not been generated.'})
        return output
    if 'test' not in base:
        base['test'] = _prediction_record(backend, test, config)
        write_json(base_path, base)
    for method in methods:
        selected = _select(records_by_method[method.name], max(config.top_k))
        member_predictions = []
        for record in selected:
            candidate = Candidate(**record['candidate'])
            path = output / 'test' / method.name / f'{candidate.id}.json'
            if path.exists():
                test_record = read_json(path)
            else:
                started = time.perf_counter()
                with apply(method, candidate):
                    test_record = _prediction_record(backend, test, config)
                test_record['total_elapsed_seconds'] = time.perf_counter() - started
                write_json(path, test_record)
            member_predictions.append(test_record['predictions'])
        ensembles = []
        for k in sorted(set(config.top_k)):
            predictions = majority_vote(member_predictions[:k], test)
            correct = [p['correct'] for p in predictions]
            base_correct = [p['correct'] for p in base['test']['predictions']]
            n = len(test)
            disagreements = [sum(a['correct'] != b['correct'] for a, b in zip(member_predictions[i], member_predictions[j])) / n
                             for i in range(k) for j in range(i + 1, k)]
            ensembles.append({'k': k, 'accuracy': sum(correct) / n,
                              'rescued_fraction': sum(a and not b for a, b in zip(correct, base_correct)) / n,
                              'regressed_fraction': sum(b and not a for a, b in zip(correct, base_correct)) / n,
                              'correctness_disagreement': sum(disagreements) / len(disagreements) if disagreements else None,
                              'selected_ids': [r['candidate']['id'] for r in selected[:k]],
                              'predictions': predictions})
        write_json(output / 'ensembles' / f'{method.name}.json', ensembles)
    manager.restore()
    if original is not None:
        original.restore()
    write_json(output / 'timing.json', {'last_invocation_seconds': time.perf_counter() - run_started,
                                      'note': 'On resume, candidate/test records preserve earlier work timings.'})
    write_json(output / 'complete.json', {'status': 'complete', 'smoke_only': identity['smoke_only']})
    from .analysis import analyze
    analyze(output)
    return output
