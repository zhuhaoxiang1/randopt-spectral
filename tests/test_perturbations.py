import pytest
import torch

from spectral_randopt.perturbations import Method, decompose, matrix_metrics, perturb, spectral_bands, stable_seed


@pytest.fixture
def matrix():
    return torch.randn(12, 20, dtype=torch.float64, generator=torch.Generator().manual_seed(8))


@pytest.mark.parametrize('kind,side', [
    ('orthogonal', 'u'), ('orthogonal', 'v'), ('orthogonal', 'both'),
    ('spectral', 'u'), ('spectral', 'v'), ('spectral', 'both'), ('complement', 'v'),
])
@pytest.mark.parametrize('distribution', ['sign', 'uniform', 'gaussian'])
def test_preserves_spectrum_and_distance(matrix, kind, side, distribution):
    base = matrix.clone()
    method = Method('test', kind, side=side, distribution=distribution)
    result, _ = perturb(matrix, method, 4, 0.003)
    assert torch.equal(matrix, base)
    assert torch.allclose(torch.linalg.svdvals(result), torch.linalg.svdvals(base), atol=1e-10, rtol=1e-10)
    assert float(torch.linalg.vector_norm(result - base) / torch.linalg.vector_norm(base)) == pytest.approx(0.003, rel=0.001)
    replay, _ = perturb(matrix, method, 4, 0.003)
    assert torch.equal(result, replay)


def test_gaussian_and_hybrids_share_source(matrix):
    gaussian, _ = perturb(matrix, Method('g', 'gaussian'), 5, 0.01)
    direction, _ = perturb(matrix, Method('d', 'direction'), 5, 0.01)
    values, _ = perturb(matrix, Method('s', 'values'), 5, 0.01)
    u0, s0, vh0 = decompose(matrix)
    u1, s1, vh1 = decompose(gaussian)
    assert torch.allclose(direction, (u1 * s0) @ vh1)
    assert torch.allclose(values, (u0 * s1) @ vh0)
    assert torch.allclose(torch.linalg.svdvals(direction), s0, atol=1e-10)
    minus, _ = perturb(matrix, Method('g', 'gaussian'), 5, 0.01, sign=-1)
    assert torch.allclose(gaussian + minus, 2 * matrix)


@pytest.mark.parametrize('kind', ['gaussian', 'direction', 'values', 'spectral', 'orthogonal', 'complement'])
def test_zero_is_exact(matrix, kind):
    result, _ = perturb(matrix, Method(kind, kind), 9, 0)
    assert torch.equal(result, matrix)


def test_rank_deficiency_and_repeated_singular_values():
    w = torch.diag(torch.tensor([3., 3., 3., 1., 0., 0.], dtype=torch.float64))
    for kind in ['spectral', 'orthogonal', 'direction']:
        result, _ = perturb(w, Method(kind, kind, side='u'), 7, 0.01)
        assert torch.allclose(torch.linalg.svdvals(result), torch.linalg.svdvals(w), atol=1e-10)
    assert spectral_bands(torch.ones(8)) == [list(range(8)), [], []]
    with pytest.raises(ValueError, match='No rotation pairs'):
        perturb(w, Method('empty', 'spectral', band='middle'), 1, 0.01)


def test_gauge_rotation_has_no_functional_change():
    w = torch.eye(6, dtype=torch.float64) * 2
    q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
    reconstructed = q @ w @ q.T
    metrics = matrix_metrics(w, reconstructed)
    assert metrics['relative_distance'] < 1e-12
    assert metrics['bands']['top']['reconstruction_change'] < 1e-12


def test_subspace_outside_moves_rectangular_span(matrix):
    result, _ = perturb(matrix, Method('c', 'complement', side='v'), 14, 0.02)
    v0, v1 = decompose(matrix)[2].T, decompose(result)[2].T
    assert torch.linalg.vector_norm(v0 @ v0.T - v1 @ v1.T) > 0.001
    with pytest.raises(ValueError, match='no orthogonal complement'):
        perturb(matrix, Method('c', 'complement', side='u'), 14, 0.02)


@pytest.mark.parametrize('band,pairing', [('top','within'), ('middle','within'), ('tail','within'), ('all','cross')])
def test_bands(matrix, band, pairing):
    result, _ = perturb(matrix, Method('s', 'spectral', band=band, pairing=pairing), 11, 0.003)
    assert matrix_metrics(matrix, result)['relative_spectral_drift'] < 1e-12


def test_invalid_inputs_and_impossible_radius(matrix):
    for rho in [-1, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            perturb(matrix, Method('g', 'gaussian'), 1, rho)
    with pytest.raises(ValueError, match='unreachable'):
        perturb(matrix, Method('s', 'spectral'), 1, 5)
    with pytest.raises(ValueError):
        perturb(torch.zeros_like(matrix), Method('g', 'gaussian'), 1, 0.1)
    assert stable_seed(4, 'layer0.q.head0') != stable_seed(4, 'layer1.q.head0')


def test_float32(matrix):
    w = matrix.float()
    for kind in ['spectral', 'complement', 'direction']:
        result, _ = perturb(w, Method(kind, kind), 5, 0.003)
        assert matrix_metrics(w, result)['relative_spectral_drift'] < 1e-5
