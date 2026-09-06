import pytest
import torch

from spectral_randopt.backend import tiny_backend
from spectral_randopt.config import RunConfig
from spectral_randopt.perturbations import Method
from spectral_randopt.targets import SVDCache, WeightManager


@pytest.fixture
def backend():
    return tiny_backend(RunConfig(device='cpu', dtype='float32'))


def test_gqa_blocks_and_frozen_parameters(backend, tmp_path):
    model = backend.model
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    manager = WeightManager(model, layers=[0], cache_dir=tmp_path, diagnostic_heads=2)
    assert len(manager.targets) == 6 + 2 + 2 + 6
    assert [t.base.shape for t in manager.targets if '.k_proj.' in t.name] == [(16, 96)] * 2
    assert [t.base.shape for t in manager.targets if '.o_proj.' in t.name] == [(96, 16)] * 6
    with manager.apply(Method('uv', 'spectral'), 91, 0.003) as metrics:
        assert metrics['relative_distance'] == pytest.approx(0.003, rel=0.002)
        for name, p in model.named_parameters():
            selected = 'layers.0.self_attn' in name and name.endswith('.weight')
            assert torch.equal(before[name], p) != selected
    for name, p in model.named_parameters():
        assert torch.equal(before[name], p)


def test_exact_restore_exception_replay_and_logits(backend):
    manager = WeightManager(backend.model, diagnostic_heads=0)
    ids = torch.tensor([[2, 5, 7, 9]])
    def logits():
        with torch.inference_mode():
            return backend.model(input_ids=ids).logits.clone()
    initial = logits()
    with pytest.raises(RuntimeError, match='intentional'):
        with manager.apply(Method('s', 'spectral'), 4, 0.03):
            changed = logits()
            assert not torch.allclose(changed, initial)
            raise RuntimeError('intentional')
    assert torch.equal(logits(), initial)
    with manager.apply(Method('s', 'spectral'), 4, 0.03):
        assert torch.equal(logits(), changed)
        with pytest.raises(RuntimeError, match='Nested'):
            with manager.apply(Method('g', 'gaussian'), 1, 0.001):
                pass
    assert torch.equal(logits(), initial)


def test_restore_when_construction_fails_midway(backend):
    manager = WeightManager(backend.model, projections=['o_proj', 'q_proj'], diagnostic_heads=0)
    before = [t.base.clone() for t in manager.targets]
    with pytest.raises(ValueError, match='no orthogonal complement'):
        with manager.apply(Method('c', 'complement', side='u'), 9, 0.01):
            pass
    for target, old in zip(manager.targets, before):
        assert torch.equal(target.view(), old)


def test_cache_replay_and_bounded_memory(tmp_path):
    cache = SVDCache(tmp_path, max_entries=1)
    a, b = torch.randn(5, 8), torch.randn(7, 4)
    first = cache.get('a', a)
    cache.get('b', b)
    assert len(cache.memory) == 1
    restored = cache.get('a', a)
    assert all(torch.equal(x, y) for x, y in zip(first, restored))


def test_invalid_layer_and_head(backend):
    with pytest.raises(ValueError):
        WeightManager(backend.model, layers=[99])
    with pytest.raises(ValueError):
        WeightManager(backend.model, head_indices=[99])


def test_native_low_precision_restoration_and_diagnostics(backend):
    backend.model.to(torch.bfloat16)
    manager = WeightManager(backend.model, layers=[0], head_indices=[0], diagnostic_heads=1)
    with manager.apply(Method('g', 'gaussian'), 2, 0.003) as metrics:
        assert metrics['changed_fraction'] > 0
        assert 'construction' in metrics['diagnostic_heads'][0]
        assert 'applied' in metrics['diagnostic_heads'][0]
    for target in manager.targets:
        assert torch.equal(target.view(), target.base)
