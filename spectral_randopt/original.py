"""Original RandOpt perturbation rule on a shared Transformers parameter layout.

Mirrors utils/worker_extn.py: native-dtype Gaussian noise with the SAME seed
restarted for every parameter. Exact snapshot restoration prevents BF16 drift.
vLLM fuses some parameters, so this is not a bitwise vLLM reproduction.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math

import torch


class OriginalWeightManager:
    def __init__(self, model, head_manager):
        self.head_manager = head_manager
        self.active = False
        # Keep a single native-precision snapshot on the model device; no N copies.
        self.parameters = [(name, p, p.detach().clone()) for name, p in model.named_parameters()
                           if not name.startswith(('visual.', 'model.visual.'))]
        if not self.parameters:
            raise ValueError('No text parameters to perturb')
        self.elements = sum(p.numel() for _, p, _ in self.parameters)
        digest = hashlib.sha256()
        for name, _, base in self.parameters:
            digest.update(f'{name}:{tuple(base.shape)}:{base.dtype}'.encode())
            raw = base.contiguous().view(torch.uint8).reshape(-1)
            for start in range(0, raw.numel(), 8 * 2**20):
                digest.update(raw[start:start + 8 * 2**20].cpu().numpy().tobytes())
        self.weights_sha256 = digest.hexdigest()

    @torch.no_grad()
    def restore(self):
        for _, parameter, base in self.parameters:
            parameter.copy_(base)

    def manifest(self):
        return {'scope': 'all_text_parameters_including_embeddings_biases_and_norms',
                'weights_sha256': self.weights_sha256,
                'parameter_tensors': len(self.parameters), 'parameter_elements': self.elements,
                'noise_rule': 'native dtype; generator reset to the same candidate seed per parameter',
                'backend_adaptation': 'Transformers parameter layout, not vLLM fused layout',
                'restoration': 'exact native snapshot for every independent candidate'}

    @contextmanager
    def apply(self, seed: int, sigma: float):
        if self.active or self.head_manager.active:
            raise RuntimeError('Nested perturbations are not supported')
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError('Original RandOpt sigma must be finite and positive')
        self.active = True
        try:
            self.restore()
            device = self.parameters[0][1].device
            base2, delta2 = torch.zeros((), device=device), torch.zeros((), device=device)
            changed = torch.zeros((), dtype=torch.int64, device=device)
            with torch.no_grad():
                for _, parameter, base in self.parameters:
                    gen = torch.Generator(device=parameter.device).manual_seed(int(seed))
                    noise = torch.randn(parameter.shape, dtype=parameter.dtype,
                                        device=parameter.device, generator=gen)
                    parameter.add_(float(sigma) * noise)
                    del noise
                    # Chunk diagnostics to avoid full FP32 copies of the embedding tensor.
                    flat_base, flat_parameter = base.reshape(-1), parameter.reshape(-1)
                    for start in range(0, parameter.numel(), 2**20):
                        b = flat_base[start:start + 2**20].float()
                        p = flat_parameter[start:start + 2**20].float()
                        base2 += b.square().sum()
                        delta2 += (p - b).square().sum()
                        changed += (p != b).sum()
                count = int(changed)
                if count == 0:
                    raise ValueError('Original perturbation vanished at the model dtype')
                yield {'sigma': sigma, 'distance_scope': 'all_text_parameters',
                       'relative_distance': math.sqrt(float(delta2) / max(float(base2), 1e-30)),
                       'changed_fraction': count / self.elements,
                       'target_count': len(self.parameters),
                       'diagnostic_heads': self.head_manager.current_diagnostics()}
        finally:
            self.restore()
            self.active = False
