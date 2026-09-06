from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spectral_randopt.analysis import paired_accuracy_interval
from spectral_randopt.backend import tiny_backend
from spectral_randopt.budget import plan_budget
from spectral_randopt.config import RunConfig, load_config
from spectral_randopt.data import Example
from spectral_randopt.original import OriginalWeightManager
from spectral_randopt.perturbations import Method
from spectral_randopt.runner import _code_hash, make_candidates, run_experiment, write_json
from spectral_randopt.targets import WeightManager
from utils.worker_extn import WorkerExtension


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_original_rule_matches_upstream_and_restores_every_parameter(dtype):
    model = tiny_backend(RunConfig(device='cpu', dtype='float32')).model.to(dtype)
    baseline = {name: p.detach().clone() for name, p in model.named_parameters()}
    reference = deepcopy(model)
    worker = WorkerExtension()
    worker.model_runner = SimpleNamespace(model=reference)
    worker.perturb_self_weights(1234, 0.002)
    head_manager = WeightManager(model, diagnostic_heads=1)
    manager = OriginalWeightManager(model, head_manager)
    with pytest.raises(RuntimeError, match='intentional'):
        with manager.apply(1234, 0.002) as metrics:
            assert metrics['distance_scope'] == 'all_text_parameters'
            assert metrics['relative_distance'] > 0
            for (name, actual), (expected_name, expected) in zip(model.named_parameters(), reference.named_parameters()):
                assert name == expected_name and torch.equal(actual, expected)
                assert not torch.equal(actual, baseline[name]), name
            with pytest.raises(RuntimeError, match='Nested'):
                with manager.apply(1, 0.001):
                    pass
            raise RuntimeError('intentional')
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, baseline[name]), name
    with manager.apply(1234, 0.002):
        assert all(torch.equal(p, q) for p, q in zip(model.parameters(), reference.parameters()))
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_original_candidate_stream_matches_upstream_and_is_shared():
    config = RunConfig(population=100, rhos=[.001, .002, .003], antithetic=False,
                       candidate_sampling='randopt', methods=[Method('original_randopt', 'original').to_dict()])
    candidates = make_candidates(config)
    rng = np.random.default_rng(config.seed)
    assert [c.seed for c in candidates] == rng.choice(2**31, size=100, replace=False).tolist()
    assert [c.sigma for c in candidates] == rng.choice(config.original_sigmas, size=100).tolist()
    assert all(c.sign == 1 and c.rho == c.sigma for c in candidates)
    assert candidates == make_candidates(replace(config, methods=[Method('ours', 'spectral').to_dict()]))
    with pytest.raises(ValueError, match='independent'):
        replace(config, antithetic=True).validate()
    with pytest.raises(ValueError, match='requires candidate_sampling'):
        replace(config, candidate_sampling='balanced', population=102).validate()


@pytest.mark.parametrize('attention', ['eager', 'sdpa'])
def test_batched_greedy_matches_unbatched_for_variable_length_prompts(attention):
    config = RunConfig(device='cpu', dtype='float32', batch_size=1, max_new_tokens=5,
                       attention_implementation=attention)
    backend = tiny_backend(config)
    examples = [Example('a', [{'role': 'user', 'content': 'What is 2 plus 1 ?'}], '3'),
                Example('b', [{'role': 'user', 'content': 'Answer with a number . What is 12 plus 1 ?'}], '13')]
    single = backend.generate(examples)
    config.batch_size = 2
    assert backend.generate(examples) == single
    config.max_prompt_tokens = 2
    with pytest.raises(ValueError, match='refusing silent truncation'):
        backend.generate(examples)


def test_batch_eos_excludes_padding_and_keeps_truncation(monkeypatch):
    backend = tiny_backend(RunConfig(device='cpu', dtype='float32', batch_size=2, max_new_tokens=4))
    def fake_generate(input_ids, **kwargs):
        return torch.cat([input_ids, torch.tensor([[15, 3, 0, 0], [17, 18, 19, 20]])], dim=1)
    monkeypatch.setattr(backend.model, 'generate', fake_generate)
    rows = backend.generate([Example(str(i), [{'role': 'user', 'content': 'What is 2 ?'}], '2') for i in range(2)])
    assert [r['generated_tokens'] for r in rows] == [2, 4]
    assert [r['truncated'] for r in rows] == [False, True]


def test_paired_comparison_aligns_questions_and_sign():
    left = [{'id': str(i), 'correct': x} for i, x in enumerate([True, True, False, False])]
    right = [{'id': str(i), 'correct': x} for i, x in enumerate([False, False, True, False])]
    result = paired_accuracy_interval(left, right, 42)
    assert result['accuracy_delta'] == .25
    assert (result['left_only_correct'], result['right_only_correct']) == (2, 1)
    assert paired_accuracy_interval(left, left, 42)['accuracy_delta'] == 0
    with pytest.raises(ValueError, match='question IDs'):
        paired_accuracy_interval(left, list(reversed(right)), 42)


def test_search_only_keeps_test_sealed_then_resumes(tmp_path):
    for split, count in [('train', 4), ('test', 2)]:
        rows = [{'id': f'{split}-{i}', 'question': f'{split} {i} What is 2 plus 1 ?', 'answer': '3'}
                for i in range(count)]
        (tmp_path / f'{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (tmp_path / 'ref.jsonl').write_text(json.dumps({'text': 'The small model reads text'}) + '\n')
    config = RunConfig(device='cpu', dtype='float32', population=2, rhos=[.003], top_k=[1, 2],
                       methods=[Method('original_randopt', 'original').to_dict(), Method('uv', 'spectral').to_dict()],
                       original_sigmas=[.002], antithetic=False, candidate_sampling='randopt',
                       train_samples=2, probe_samples=2, test_samples=2, max_new_tokens=2,
                       diagnostic_heads=0, batch_size=2, train_path=str(tmp_path / 'train.jsonl'),
                       test_path=str(tmp_path / 'test.jsonl'), reference_path=str(tmp_path / 'ref.jsonl'),
                       output_dir=str(tmp_path / 'run'))
    backend = tiny_backend(config)
    output = run_experiment(config, backend, search_only=True)
    assert (output / 'search_complete.json').exists()
    assert not (output / 'test').exists() and not (output / 'complete.json').exists()
    assert 'test' not in json.loads((output / 'base.json').read_text())
    before = {p: p.read_bytes() for p in (output / 'candidates').glob('*/*.json')}
    parameter = backend.model.model.layers[0].mlp.gate_proj.weight
    original = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(0.1)
    with pytest.raises(ValueError, match='Resume rejected'):
        run_experiment(config, backend, resume=True)
    with torch.no_grad():
        parameter.copy_(original)
    run_experiment(config, backend, resume=True)
    assert (output / 'complete.json').exists()
    assert all(p.read_bytes() == content for p, content in before.items())


def test_budget_uses_timings_only_and_scales_every_arm_together(tmp_path):
    root = Path(__file__).resolve().parents[1]
    main = load_config(root / 'configs/qwen15b-12h.json')
    calibration = replace(main, population=3, train_samples=32, probe_samples=8)
    directory = tmp_path / 'calibration'
    write_json(directory / 'manifest.json', {'config': calibration.to_dict(), 'smoke_only': False,
                                           'code_sha256': _code_hash()})
    write_json(directory / 'search_complete.json', {'status': 'search_complete'})
    for method in main.methods:
        for i in range(3):
            write_json(directory / 'candidates' / method['name'] / f'c{i}.json', {
                'selection': {'elapsed_seconds': 1., 'metrics': {'samples': 32, 'accuracy': 0.}},
                'probe': {'elapsed_seconds': .3, 'metrics': {'samples': 8, 'accuracy': 1.}},
                'timing': {'apply_and_weight_diagnostics_seconds': .2, 'restore_seconds': .01, 'reference_seconds': .1}})
    destination = tmp_path / 'main.json'
    report = plan_budget(directory, root / 'configs/qwen15b-12h.json', destination, hours=9)
    assert report['chosen']['population'] == 100 and report['chosen']['test_samples'] == 256
    assert not report['decision_uses_accuracy']
    assert load_config(destination).methods == main.methods
    assert load_config(destination).top_k == [1, 10, 20, 50]
    for path in (directory / 'candidates').glob('*/*.json'):
        row = json.loads(path.read_text())
        row['selection']['metrics']['accuracy'] = 1.
        row['probe']['metrics']['accuracy'] = 0.
        write_json(path, row)
    repeated = plan_budget(directory, root / 'configs/qwen15b-12h.json', tmp_path / 'main2.json', hours=9)
    assert repeated == report
    alternatives = report['alternatives']
    hours = (alternatives[0]['estimated_hours_with_margin'] + alternatives[1]['estimated_hours_with_margin']) / 2
    fallback = plan_budget(directory, root / 'configs/qwen15b-12h.json', tmp_path / 'smaller.json', hours=hours)
    assert fallback['chosen']['test_samples'] == 128 and fallback['chosen']['population'] == 100
    with pytest.raises(ValueError, match='exceeds this budget'):
        plan_budget(directory, root / 'configs/qwen15b-12h.json', tmp_path / 'impossible.json', hours=.00001)
    assert not (tmp_path / 'impossible.json').exists()
    with pytest.raises(FileExistsError):
        plan_budget(directory, root / 'configs/qwen15b-12h.json', destination, hours=9)
    write_json(directory / 'manifest.json', {'config': calibration.to_dict(), 'smoke_only': True})
    with pytest.raises(ValueError, match='tiny-model'):
        plan_budget(directory, root / 'configs/qwen15b-12h.json', tmp_path / 'bad.json', hours=9)
