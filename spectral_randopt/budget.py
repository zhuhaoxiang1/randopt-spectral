"""Conservative planning from measured timings, without inspecting task scores."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from statistics import median

from .config import load_config
from .runner import _code_hash, read_json, write_json


def plan_budget(calibration_dir, config_path, output_config, hours: float, safety_factor: float = 1.5) -> dict:
    if not math.isfinite(hours) or hours <= 0 or not math.isfinite(safety_factor) or safety_factor < 1:
        raise ValueError('Positive finite hours and safety_factor >= 1 required')
    directory, destination = Path(calibration_dir), Path(output_config).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if not (directory / 'search_complete.json').exists():
        raise ValueError('Complete calibration search is required')
    manifest = read_json(directory / 'manifest.json')
    if manifest['smoke_only']:
        raise ValueError('A random tiny-model smoke cannot estimate GPU runtime')
    if manifest['code_sha256'] != _code_hash():
        raise ValueError('Code changed after calibration; profile the current implementation')
    config = load_config(config_path)
    source = manifest['config']
    comparable = ('model', 'revision', 'device', 'dtype', 'batch_size', 'attention_implementation',
                  'linalg_device', 'rhos', 'original_sigmas', 'candidate_sampling', 'antithetic',
                  'max_new_tokens', 'max_prompt_tokens', 'reference_max_tokens', 'methods',
                  'projections', 'layers', 'head_indices', 'diagnostic_heads', 'svd_cache_entries',
                  'train_path', 'test_path', 'reference_path', 'seed')
    for key in comparable:
        if source[key] != config.to_dict()[key]:
            raise ValueError(f'Calibration and main config differ in {key}; profile the actual settings')
    timing = {}
    cold_seconds = 180.0  # Reserve model loading, file I/O and report generation again.
    for method in config.methods:
        records = [read_json(p) for p in sorted((directory / 'candidates' / method['name']).glob('*.json'))]
        if len(records) != source['population'] or len(records) < 3:
            raise ValueError('At least three complete calibration candidates per method are required')
        rate = max(r[split]['elapsed_seconds'] / r[split]['metrics']['samples']
                   for r in records for split in ('selection', 'probe'))
        overheads = [r['timing']['apply_and_weight_diagnostics_seconds'] + r['timing']['restore_seconds']
                     for r in records]
        overhead = median(overheads)
        reference = max(r['timing']['reference_seconds'] for r in records)
        cold_seconds += sum(max(0., t - overhead) for t in overheads)
        timing[method['name']] = {'seconds_per_example': rate,
                                 'apply_restore_seconds': overhead, 'reference_seconds': reference}
    alternatives = []
    # Fix these alternatives in advance. K=50 is preserved, even in the N=50 fallback.
    for population, test_samples in [(100, 256), (100, 128), (50, 128)]:
        k = 50
        per_method = {name: population * (v['apply_restore_seconds'] + v['reference_seconds'] +
                                          (config.train_samples + config.probe_samples) * v['seconds_per_example']) +
                            k * (v['apply_restore_seconds'] + test_samples * v['seconds_per_example'])
                      for name, v in timing.items()}
        base = (config.train_samples + config.probe_samples + test_samples) * max(
            v['seconds_per_example'] for v in timing.values())
        estimated = safety_factor * (cold_seconds + base + sum(per_method.values()))
        alternatives.append({'population': population, 'test_samples': test_samples, 'k': k,
                             'estimated_hours_with_margin': estimated / 3600,
                             'method_seconds_before_margin': per_method})
    chosen = next((a for a in alternatives if a['estimated_hours_with_margin'] <= hours), None)
    report = {'available_hours': hours, 'safety_factor': safety_factor, 'timing_inputs': timing,
              'alternatives': alternatives, 'chosen': chosen,
              'decision_uses_accuracy': False,
              'limitation': 'An estimate, not a deadline guarantee; use an external wall-clock timeout.'}
    if chosen is None:
        write_json(destination.with_suffix('.budget.json'), report)
        raise ValueError('Even the shared N=50, K=50, test=128 fallback exceeds this budget; do not launch')
    config = replace(config, population=chosen['population'], test_samples=chosen['test_samples'],
                     top_k=[1, 10, 20, 50])
    config.validate()
    write_json(destination, config.to_dict())
    write_json(destination.with_suffix('.budget.json'), report)
    return report
