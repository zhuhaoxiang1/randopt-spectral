"""JSON experiment configurations; loading a config never accesses the network."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from .perturbations import Method


def core_methods() -> list[dict]:
    return [Method(name, kind, side=side).to_dict() for name, kind, side in [
        ('gaussian', 'gaussian', 'both'), ('direction', 'direction', 'both'),
        ('values', 'values', 'both'), ('orthogonal', 'orthogonal', 'both'),
        ('spectral_u', 'spectral', 'u'), ('spectral_v', 'spectral', 'v'),
        ('spectral_uv', 'spectral', 'both')]]


@dataclass
class RunConfig:
    model: str = 'Qwen/Qwen2.5-1.5B-Instruct'
    revision: str = 'main'
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    linalg_device: str = 'auto'
    seed: int = 42
    population: int = 32
    rhos: list[float] = field(default_factory=lambda: [0.001, 0.003])
    antithetic: bool = True
    candidate_sampling: str = 'balanced'
    original_sigmas: list[float] = field(default_factory=lambda: [0.001, 0.002, 0.003])
    top_k: list[int] = field(default_factory=lambda: [1, 5])
    train_samples: int = 64
    probe_samples: int = 32
    test_samples: int | None = 128
    max_new_tokens: int = 512
    batch_size: int = 1
    attention_implementation: str = 'eager'
    max_prompt_tokens: int = 1024
    reference_max_tokens: int = 128
    train_path: str = '../data/spectral/gsm8k_train.jsonl'
    test_path: str = '../data/spectral/gsm8k_test.jsonl'
    reference_path: str = '../data/spectral/reference.jsonl'
    output_dir: str = '../runs/qwen15b-pilot'
    projections: list[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
    layers: list[int] | None = None
    head_indices: list[int] | None = None
    diagnostic_heads: int = 8
    svd_cache_entries: int = 16
    margin: float = 0.01
    save_texts: bool = False
    methods: list[dict] = field(default_factory=core_methods)

    def validate(self) -> None:
        methods = [Method(**m) for m in self.methods]
        if not methods or len({m.name for m in methods}) != len(methods):
            raise ValueError('Method names must be nonempty and unique')
        if self.dtype not in {'float32', 'bfloat16', 'float16'}:
            raise ValueError('Invalid dtype')
        if not self.rhos or any(not math.isfinite(r) or r <= 0 for r in self.rhos):
            raise ValueError('rhos must be finite and positive')
        if not self.original_sigmas or any(not math.isfinite(s) or s <= 0 for s in self.original_sigmas):
            raise ValueError('original_sigmas must be finite and positive')
        if self.batch_size < 1 or self.attention_implementation not in {'eager', 'sdpa'}:
            raise ValueError('Invalid batch_size or attention_implementation')
        if self.candidate_sampling not in {'balanced', 'randopt'}:
            raise ValueError('Unknown candidate_sampling')
        if self.candidate_sampling == 'randopt':
            if self.antithetic or len(self.rhos) != len(self.original_sigmas):
                raise ValueError('randopt sampling requires independent candidates and equal grid sizes')
        if any(m.kind == 'original' for m in methods) and self.candidate_sampling != 'randopt':
            raise ValueError('Original RandOpt baseline requires candidate_sampling=randopt')
        divisor = len(self.rhos) * (2 if self.antithetic else 1)
        if self.population < 1 or self.population > 2**31:
            raise ValueError('Invalid population')
        if self.candidate_sampling == 'balanced' and (self.population < divisor or self.population % divisor):
            raise ValueError('population must divide evenly into radii and antithetic pairs')
        if not self.top_k or min(self.top_k) < 1 or max(self.top_k) > self.population:
            raise ValueError('top_k must be between 1 and population')
        if min(self.train_samples, self.probe_samples) < 1 or (self.test_samples is not None and self.test_samples < 1):
            raise ValueError('Sample counts must be positive')
        if min(self.max_new_tokens, self.max_prompt_tokens, self.reference_max_tokens) < 2:
            raise ValueError('Token limits must be at least 2')
        if self.diagnostic_heads < 0 or self.svd_cache_entries < 1 or not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError('Invalid diagnostic_heads or margin')

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> RunConfig:
    path = Path(path).resolve()
    with path.open() as stream:
        raw = json.load(stream)
    config = RunConfig(**raw)
    for name in ('train_path', 'test_path', 'reference_path', 'output_dir'):
        value = Path(getattr(config, name)).expanduser()
        setattr(config, name, str(value if value.is_absolute() else (path.parent / value).resolve()))
    config.validate()
    return config
