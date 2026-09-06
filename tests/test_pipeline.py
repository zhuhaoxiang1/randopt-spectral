from dataclasses import replace
import json
from pathlib import Path

import pytest

from spectral_randopt.backend import tiny_backend
from spectral_randopt.config import RunConfig, load_config
from spectral_randopt.data import Example, load_splits, majority_vote, score_text
from spectral_randopt.runner import _select, make_candidates, run_experiment
from spectral_randopt.smoke import smoke


def test_scoring_and_vote_ties():
    example = Example('a', [{'role': 'user', 'content': 'question'}], '12')
    strict = score_text('work: 3 + 9 = 12. #### 12', example)
    flexible = score_text('The answer is 12', example)
    wrong = score_text('#### 9', example)
    invalid = score_text('I cannot answer.', example)
    assert strict['strict_correct'] and strict['correct']
    assert flexible['correct'] and not flexible['strict_correct']
    assert not wrong['correct'] and not invalid['correct']
    assert majority_vote([[wrong], [strict]], [example])[0]['answer'] == '9'
    assert majority_vote([[strict], [wrong]], [example])[0]['answer'] == '12'
    assert majority_vote([[invalid], [strict]], [example])[0]['correct']
    assert not majority_vote([[invalid], [invalid]], [example])[0]['correct']
    negative = Example('n', [], '-1000')
    assert score_text('#### -1,000', negative)['correct']


def test_candidate_budget_and_test_blind_selection():
    config = RunConfig()
    candidates = make_candidates(config)
    assert len(candidates) == 32
    assert candidates == make_candidates(config)
    assert len({c.id for c in candidates}) == 32
    for i in range(0, len(candidates), 2):
        a, b = candidates[i:i + 2]
        assert (a.seed, a.rho, a.pair_id) == (b.seed, b.rho, b.pair_id)
        assert (a.sign, b.sign) == (1, -1)
    records = [{'candidate': {'id': name}, 'selection': {'metrics': {'accuracy': train}},
                'probe': {'metrics': {'accuracy': test}}}
               for name, train, test in [('c0', 0.8, 0.), ('c1', 0.7, 1.), ('c2', 0.8, 1.)]]
    assert [r['candidate']['id'] for r in _select(records, 2)] == ['c0', 'c2']
    with pytest.raises(ValueError):
        make_candidates(replace(config, population=31))


def test_real_model_configs_are_valid_without_loading_model():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / 'configs').glob('*.json'))
    assert len(paths) >= 7
    for path in paths:
        config = load_config(path)
        assert config.model == 'Qwen/Qwen2.5-1.5B-Instruct'
        assert len(make_candidates(config)) == config.population
        assert Path(config.train_path).is_absolute()


def test_full_offline_smoke_resume_and_data_isolation(tmp_path):
    result = smoke(tmp_path / 'smoke')
    verification = json.loads((tmp_path / 'smoke' / 'smoke-verification.json').read_text())
    assert verification['status'] == 'passed'
    assert verification['methods'] == 15 and verification['candidates'] == 60
    assert verification['all_parameters_restored_exactly']
    manifest = json.loads((result / 'manifest.json').read_text())
    assert manifest['smoke_only']
    assert not set(manifest['selection_ids']) & set(manifest['probe_ids'])
    assert not set(manifest['selection_ids']) & set(manifest['test_ids'])
    assert len(list((result / 'candidates').glob('*/*.json'))) == 60
    assert (result / 'fair_comparison.csv').exists()
    assert (result / 'report.md').exists() and (result / 'summary.csv').exists()
    first = next((result / 'candidates').glob('*/*.json'))
    original = first.read_bytes()
    smoke(tmp_path / 'smoke', resume=True)
    assert first.read_bytes() == original
    config = RunConfig(**manifest['config'])
    with pytest.raises(ValueError, match='Resume rejected'):
        run_experiment(replace(config, margin=0.02), backend=tiny_backend(config), resume=True)
    with pytest.raises(FileExistsError):
        run_experiment(config, backend=tiny_backend(config))
    overlap = replace(config, test_path=config.train_path)
    with pytest.raises(ValueError, match='overlap'):
        load_splits(overlap)
    with pytest.raises(ValueError, match='Insufficient'):
        load_splits(replace(config, train_samples=100))


def test_missing_local_model_fails_without_download(tmp_path):
    from spectral_randopt.backend import TransformersBackend
    config = RunConfig(model=str(tmp_path / 'missing-model'), device='cpu', dtype='float32')
    with pytest.raises((OSError, ValueError)):
        TransformersBackend.from_config(config)
