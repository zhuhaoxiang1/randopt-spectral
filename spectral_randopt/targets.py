"""GQA-aware head views, bounded SVD caching, and exact weight restoration."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch import Tensor, nn

from .perturbations import Method, decompose, matrix_metrics, perturb, stable_seed


@dataclass
class Target:
    name: str
    parameter: nn.Parameter
    axis: int
    start: int
    stop: int
    base: Tensor
    fingerprint: str

    def view(self) -> Tensor:
        return self.parameter[self.start:self.stop, :] if self.axis == 0 else self.parameter[:, self.start:self.stop]


class SVDCache:
    def __init__(self, directory: Path | None = None, max_entries: int = 16):
        self.directory, self.max_entries = directory, max_entries
        self.memory: OrderedDict[str, tuple[Tensor, Tensor, Tensor]] = OrderedDict()
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, base: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if key in self.memory:
            value = self.memory.pop(key)
        else:
            path = self.directory / f'{key}.npz' if self.directory else None
            if path is not None and path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    value = tuple(torch.from_numpy(saved[name].copy()) for name in ('u', 's', 'vh'))
                if value[0].shape != (base.shape[0], min(base.shape)) or value[2].shape != (min(base.shape), base.shape[1]):
                    raise ValueError(f'Invalid SVD cache shape: {path}')
            else:
                value = tuple(t.detach().cpu() for t in decompose(base))
                if path is not None:
                    temporary = path.with_suffix('.tmp')
                    with temporary.open('wb') as stream:
                        np.savez(stream, u=value[0].numpy(), s=value[1].numpy(), vh=value[2].numpy())
                    temporary.replace(path)
        self.memory[key] = value
        while len(self.memory) > self.max_entries:
            self.memory.popitem(last=False)
        return tuple(t.to(device=base.device, dtype=base.dtype) for t in value)


class WeightManager:
    def __init__(self, model: nn.Module, projections=('q_proj', 'k_proj', 'v_proj', 'o_proj'),
                 layers: list[int] | None = None, cache_dir: Path | None = None,
                 linalg_device: str = 'cpu', diagnostic_heads: int = 8,
                 checkpoint_id: str = 'unknown', head_indices: list[int] | None = None,
                 cache_entries: int = 16):
        if not set(projections) <= {'q_proj', 'k_proj', 'v_proj', 'o_proj'} or not projections:
            raise ValueError('Only attention q_proj/k_proj/v_proj/o_proj are supported')
        config = model.config
        nheads, nkv = config.num_attention_heads, config.num_key_value_heads
        if nheads % nkv:
            raise ValueError('Query head count must be divisible by KV head count')
        self.targets: list[Target] = []
        self.cache = SVDCache(cache_dir, max_entries=cache_entries)
        self.linalg_device = torch.device(linalg_device)
        self.active = False
        self.diagnostic_heads = diagnostic_heads
        seen_layers = set()
        for module_name, attention in model.named_modules():
            if not module_name.endswith('self_attn'):
                continue
            match = re.search(r'layers\.(\d+)\.self_attn$', module_name)
            if match is None:
                raise ValueError(f'Unsupported attention module name: {module_name}')
            layer = int(match[1])
            if layers is not None and layer not in layers:
                continue
            seen_layers.add(layer)
            dim = attention.q_proj.weight.shape[0] // nheads
            for projection in projections:
                parameter = getattr(attention, projection).weight
                axis = 1 if projection == 'o_proj' else 0
                count = nkv if projection in {'k_proj', 'v_proj'} else nheads
                if parameter.shape[axis] != count * dim:
                    raise ValueError(f'Unexpected GQA shape: {module_name}.{projection}')
                for head in range(count):
                    if head_indices is not None and head not in head_indices:
                        continue
                    start, stop = head * dim, (head + 1) * dim
                    view = parameter[start:stop, :] if axis == 0 else parameter[:, start:stop]
                    base = view.detach().cpu().contiguous().clone()
                    name = f'{module_name}.{projection}.weight:head{head}'
                    digest = hashlib.sha256()
                    digest.update(f'svd-v1:{checkpoint_id}:{name}:{tuple(base.shape)}:{base.dtype}'.encode())
                    digest.update(base.view(torch.uint8).numpy().tobytes())
                    self.targets.append(Target(name, parameter, axis, start, stop, base, digest.hexdigest()))
        if not self.targets or (layers is not None and set(layers) != seen_layers):
            raise ValueError('Requested layers/heads did not match the model')
        if head_indices is not None and any(i < 0 or i >= nheads for i in head_indices):
            raise ValueError('Invalid head index')

    @torch.no_grad()
    def restore(self) -> None:
        for target in self.targets:
            target.view().copy_(target.base.to(target.parameter.device))

    def manifest(self) -> list[dict]:
        return [{'name': t.name, 'shape': list(t.base.shape), 'fingerprint': t.fingerprint}
                for t in self.targets]

    @torch.no_grad()
    def current_diagnostics(self) -> list[dict]:
        indices = np.linspace(0, len(self.targets) - 1,
                              min(self.diagnostic_heads, len(self.targets)), dtype=int)
        rows = []
        for index in indices:
            target = self.targets[index]
            base = target.base.to(device=self.linalg_device, dtype=torch.float32)
            applied = target.view().detach().to(device=self.linalg_device, dtype=torch.float32)
            svd = self.cache.get(target.fingerprint, base)
            rows.append({'target': target.name, 'applied': matrix_metrics(base, applied, svd)})
        return rows

    @contextmanager
    def apply(self, method: Method, seed: int, rho: float, sign: int = 1):
        if self.active:
            raise RuntimeError('Nested perturbations are not supported')
        self.active = True
        try:
            self.restore()
            indices = set(np.linspace(0, len(self.targets) - 1,
                                      min(self.diagnostic_heads, len(self.targets)), dtype=int).tolist())
            total_base, total_delta, total_changed, total_elements = 0., 0., 0, 0
            distances, diagnostics = [], []
            with torch.no_grad():
                for index, target in enumerate(self.targets):
                    dtype = torch.float64 if target.base.dtype == torch.float64 else torch.float32
                    base = target.base.to(device=self.linalg_device, dtype=dtype)
                    need_svd = method.kind not in {'gaussian', 'orthogonal'} or index in indices
                    svd = self.cache.get(target.fingerprint, base) if need_svd else None
                    changed, detail = perturb(base, method, stable_seed(seed, target.name), rho, sign, svd)
                    native = changed.to(dtype=target.parameter.dtype, device=target.parameter.device)
                    target.view().copy_(native)
                    applied = native.to(device=base.device, dtype=base.dtype)
                    norm2 = float(base.square().sum())
                    delta2 = float((applied - base).square().sum())
                    total_base += norm2
                    total_delta += delta2
                    total_changed += int((applied != base).sum())
                    total_elements += base.numel()
                    distances.append(math.sqrt(delta2 / max(norm2, 1e-30)))
                    if index in indices:
                        diagnostics.append({'target': target.name, **detail,
                                            'construction': matrix_metrics(base, changed, svd),
                                            'applied': matrix_metrics(base, applied, svd)})
            if rho > 0 and total_changed == 0:
                raise ValueError('Perturbation vanished at the model dtype; increase rho or use FP32')
            yield {
                'distance_scope': 'selected_attention_heads',
                'relative_distance': math.sqrt(total_delta / max(total_base, 1e-30)),
                'min_head_distance': min(distances), 'max_head_distance': max(distances),
                'changed_fraction': total_changed / total_elements,
                'target_count': len(self.targets), 'diagnostic_heads': diagnostics,
            }
        finally:
            self.restore()
            self.active = False
