"""Small matrix perturbations. All distances refer to the actual weight matrix."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Callable

import torch
from torch import Tensor


KINDS = {"gaussian", "direction", "values", "orthogonal", "spectral", "complement", "original"}


@dataclass(frozen=True)
class Method:
    name: str
    kind: str
    side: str = "both"
    band: str = "all"
    pairing: str = "within"
    distribution: str = "sign"
    pairs: int = 4
    max_angle: float = 0.2

    def __post_init__(self) -> None:
        if self.kind not in KINDS or not re.fullmatch(r'[a-zA-Z0-9_-]+', self.name):
            raise ValueError(f"Invalid method: {self.name!r}, {self.kind!r}")
        if self.side not in {"u", "v", "both"}:
            raise ValueError("side must be u, v, or both")
        if self.band not in {"all", "top", "middle", "tail"}:
            raise ValueError("band must be all, top, middle, or tail")
        if self.pairing not in {"within", "cross"}:
            raise ValueError("pairing must be within or cross")
        if self.distribution not in {"sign", "uniform", "gaussian"}:
            raise ValueError("Unknown angle distribution")
        if self.pairs < 1 or not 0 < self.max_angle <= 0.5:
            raise ValueError("pairs >= 1 and 0 < max_angle <= 0.5 required")
        if self.pairing == "cross" and self.band != "all":
            raise ValueError("Cross-band rotation requires band=all")

    def to_dict(self) -> dict:
        return asdict(self)


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def decompose(w: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if w.ndim != 2 or not torch.isfinite(w).all():
        raise ValueError("SVD requires a finite 2-D matrix")
    return torch.linalg.svd(w, full_matrices=False)


def spectral_bands(s: Tensor, gap_tolerance: float = 1e-3) -> list[list[int]]:
    """Index quartiles, moving boundaries past near-degenerate groups."""
    r = len(s)
    cuts = []
    for fraction in (0.25, 0.75):
        cut = max(1, min(r, int(round(r * fraction))))
        while cut < r:
            scale = max(float(s[cut - 1]), float(s[cut]), 1e-12)
            if abs(float(s[cut - 1] - s[cut])) > gap_tolerance * scale:
                break
            cut += 1
        cuts.append(cut)
    return [list(range(0, cuts[0])), list(range(cuts[0], cuts[1])), list(range(cuts[1], r))]


def _pairs(s: Tensor, method: Method, rng: torch.Generator) -> list[tuple[int, int]]:
    bands = spectral_bands(s)
    if method.pairing == "cross":
        choices = [(i, j) for a in range(3) for b in range(a + 1, 3)
                   for i in bands[a] for j in bands[b]]
    else:
        ids = list(range(len(s))) if method.band == "all" else bands[
            {"top": 0, "middle": 1, "tail": 2}[method.band]]
        choices = [(i, j) for k, i in enumerate(ids) for j in ids[k + 1:]]
    if not choices:
        raise ValueError(f"No rotation pairs for band={method.band}, pairing={method.pairing}; "
                         "the band may be empty after merging near-degenerate values")
    order = torch.randperm(len(choices), generator=rng)[:method.pairs].tolist()
    return [choices[i] for i in order]


def _angles(count: int, distribution: str, rng: torch.Generator) -> list[float]:
    if distribution == "sign":
        x = torch.randint(0, 2, (count,), generator=rng).float() * 2 - 1
    elif distribution == "uniform":
        x = torch.rand(count, generator=rng) * 2 - 1
    else:
        x = torch.randn(count, generator=rng)
    # Sampling controls relative angles; calibration sets the total distance.
    return (x / x.abs().max().clamp_min(1e-12)).tolist()


def _rotate_columns(x: Tensor, i: int, j: int, angle: float) -> None:
    a, b = x[:, i].clone(), x[:, j].clone()
    c, s = math.cos(angle), math.sin(angle)
    x[:, i] = c * a + s * b
    x[:, j] = -s * a + c * b


def _calibrate(base: Tensor, build: Callable[[float], Tensor], rho: float,
               max_angle: float) -> tuple[Tensor, float]:
    norm = float(torch.linalg.vector_norm(base))
    if norm == 0:
        raise ValueError("Relative distance is undefined for an all-zero matrix")
    target = rho * norm
    lo, hi = 0.0, min(0.005, max_angle)
    result = build(hi)
    while float(torch.linalg.vector_norm(result - base)) < target and hi < max_angle:
        hi = min(hi * 2, max_angle)
        result = build(hi)
    if float(torch.linalg.vector_norm(result - base)) < target * 0.999:
        raise ValueError(f"rho={rho} is unreachable within max_angle={max_angle}; "
                         "increase pairs or lower rho")
    for _ in range(32):
        angle = (lo + hi) / 2
        result = build(angle)
        distance = float(torch.linalg.vector_norm(result - base))
        if abs(distance - target) <= target * 1e-4:
            return result, angle
        if distance < target:
            lo = angle
        else:
            hi = angle
    angle = (lo + hi) / 2
    result = build(angle)
    if abs(float(torch.linalg.vector_norm(result - base)) / target - 1) > 0.01:
        raise ValueError("Distance calibration failed, possibly below numerical precision")
    return result, angle


def perturb(base: Tensor, method: Method, seed: int, rho: float, sign: int = 1,
            svd: tuple[Tensor, Tensor, Tensor] | None = None) -> tuple[Tensor, dict]:
    """Return a new matrix, never modify base. Hybrids share the same Gaussian source."""
    if method.kind == 'original':
        raise ValueError('Original RandOpt requires OriginalWeightManager, not a matrix-relative perturbation')
    if base.ndim != 2 or base.dtype not in {torch.float32, torch.float64}:
        raise ValueError("Perturbations require a 2-D FP32 or FP64 matrix")
    if not math.isfinite(rho) or rho < 0 or sign not in {-1, 1}:
        raise ValueError("Require finite rho >= 0 and sign in {-1, 1}")
    if rho == 0:
        return base.clone(), {"angle": 0.0, "source_rho": 0.0}
    if float(torch.linalg.vector_norm(base)) == 0:
        raise ValueError("Cannot perturb an all-zero matrix with a relative radius")
    if method.kind in {"gaussian", "direction", "values"}:
        noise = torch.randn(base.shape, dtype=base.dtype, device=base.device,
                            generator=generator(seed, base.device))
        noise *= sign * rho * torch.linalg.vector_norm(base) / torch.linalg.vector_norm(noise)
        candidate = base + noise
        if method.kind != "gaussian":
            u0, s0, vh0 = svd if svd is not None else decompose(base)
            u1, s1, vh1 = decompose(candidate)
            candidate = (u1 * s0) @ vh1 if method.kind == "direction" else (u0 * s1) @ vh0
        return candidate, {"angle": None, "source_rho": rho}

    rng = generator(seed)
    if method.kind == "orthogonal":
        plans = []
        for dimension in (base.shape[0], base.shape[1]):
            # Coordinate pairs are sampled without constructing an O(d^2) list.
            pairs = []
            if dimension < 2:
                raise ValueError("Coordinate rotation requires dimensions >= 2")
            for _ in range(method.pairs):
                i = int(torch.randint(dimension, (), generator=rng))
                j = int(torch.randint(dimension - 1, (), generator=rng))
                pairs.append((i, j + (j >= i)))
            plans.append((pairs, _angles(len(pairs), method.distribution, rng)))

        def build(angle: float) -> Tensor:
            w = base.clone()
            for side, (pairs, coefficients) in enumerate(plans):
                if (side == 0 and method.side == "v") or (side == 1 and method.side == "u"):
                    continue
                for (i, j), c in zip(pairs, coefficients):
                    _rotate_columns(w.T if side == 0 else w, i, j, sign * angle * c)
            return w
    else:
        u, s, vh = svd if svd is not None else decompose(base)
        if method.kind == "spectral":
            plans = [(_pairs(s, method, rng), None) for _ in range(2)]
            plans = [(pairs, _angles(len(pairs), method.distribution, rng)) for pairs, _ in plans]

            def build(angle: float) -> Tensor:
                left, right = u.clone(), vh.T.clone()
                for side, (pairs, coefficients) in enumerate(plans):
                    if (side == 0 and method.side == "v") or (side == 1 and method.side == "u"):
                        continue
                    for (i, j), c in zip(pairs, coefficients):
                        _rotate_columns(left if side == 0 else right, i, j, sign * angle * c)
                return (left * s) @ right.T
        else:
            basis_options = [("u", u), ("v", vh.T)]
            plans = []
            for side, basis in basis_options:
                if method.side not in {side, "both"} or basis.shape[0] <= basis.shape[1]:
                    continue
                bands = spectral_bands(s)
                ids = list(range(len(s))) if method.band == "all" else bands[
                    {"top": 0, "middle": 1, "tail": 2}[method.band]]
                if not ids:
                    raise ValueError("Empty band for complement rotation")
                selected = [ids[i] for i in torch.randperm(len(ids), generator=rng)[:method.pairs]]
                vectors = torch.randn((len(selected), basis.shape[0]), generator=rng,
                                      dtype=base.dtype).to(base.device)
                plans.append((side, selected, vectors, _angles(len(selected), method.distribution, rng)))
            if not plans:
                raise ValueError("Requested side has no orthogonal complement")

            def build(angle: float) -> Tensor:
                left, right = u.clone(), vh.T.clone()
                for side, selected, vectors, coefficients in plans:
                    basis = left if side == "u" else right
                    for i, raw, c in zip(selected, vectors, coefficients):
                        z = raw.clone()
                        # Reproject against the current basis after every rotation.
                        for _ in range(2):
                            z = z - basis @ (basis.T @ z)
                        norm = torch.linalg.vector_norm(z)
                        if float(norm) < 1e-8:
                            raise ValueError("Degenerate sampled complement direction")
                        theta = sign * angle * c
                        basis[:, i] = math.cos(theta) * basis[:, i] + math.sin(theta) * z / norm
                return (left * s) @ right.T
    result, angle = _calibrate(base, build, rho, method.max_angle)
    return result, {"angle": angle, "source_rho": rho}


def matrix_metrics(base: Tensor, result: Tensor, svd=None) -> dict:
    u0, s0, vh0 = svd if svd is not None else decompose(base)
    u1, s1, vh1 = decompose(result)
    norm = float(torch.linalg.vector_norm(base))
    bands = spectral_bands(s0)
    output = {
        "relative_distance": float(torch.linalg.vector_norm(result - base)) / max(norm, 1e-30),
        "relative_spectral_drift": float(torch.linalg.vector_norm(s1 - s0)) /
                                   max(float(torch.linalg.vector_norm(s0)), 1e-30),
        "changed_fraction": float((base != result).float().mean()),
        "bands": {},
    }
    for name, ids in zip(("top", "middle", "tail"), bands):
        if not ids:
            continue
        # The residual avoids cancellation in k - ||U0^T U1||_F^2 near identical spans.
        def distance(a: Tensor, b: Tensor) -> float:
            if a.shape[0] == a.shape[1]:
                return 0.0
            residual = b - a @ (a.T @ b)
            return min(1.0, float(torch.linalg.vector_norm(residual)) / math.sqrt(len(ids)))
        block0 = (u0[:, ids] * s0[ids]) @ vh0[ids, :]
        block1 = (u1[:, ids] * s1[ids]) @ vh1[ids, :]
        output["bands"][name] = {
            "rank": len(ids), "u_subspace_distance": distance(u0[:, ids], u1[:, ids]),
            "v_subspace_distance": distance(vh0[ids, :].T, vh1[ids, :].T),
            "reconstruction_change": float(torch.linalg.vector_norm(block1 - block0)) /
                                     max(float(torch.linalg.vector_norm(block0)), 1e-30),
        }
    return output
