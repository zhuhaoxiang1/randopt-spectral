"""Offline end-to-end verification, with socket connections explicitly forbidden."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import torch

from .backend import tiny_backend
from .config import RunConfig, core_methods
from .perturbations import Method
from .runner import run_experiment, write_json


def smoke(output_dir: str | Path, resume: bool = False) -> Path:
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(2)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f'{output} is not empty; choose another smoke output or --resume')
    fixtures = output / 'fixtures'
    fixtures.mkdir(parents=True, exist_ok=True)
    for split, size in [('train', 6), ('test', 2)]:
        rows = [{'id': f'{split}-{i}', 'messages': [{'role': 'user',
                 'content': f'{split} {i} What is {i} plus 1 ? Answer with a number .'}],
                 'answer': str(i + 1)} for i in range(size)]
        (fixtures / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (fixtures / 'reference.jsonl').write_text(json.dumps({'text': 'The small model reads text and writes an answer .'}) + '\n')
    methods = [Method('original_randopt', 'original').to_dict()] + core_methods() + [Method('complement', 'complement').to_dict()]
    methods += [Method(f'band_{band}', 'spectral', band=band).to_dict() for band in ('top', 'middle', 'tail')]
    methods += [Method('band_cross', 'spectral', pairing='cross').to_dict(),
                Method('angle_uniform', 'spectral', distribution='uniform').to_dict(),
                Method('angle_gaussian', 'spectral', distribution='gaussian').to_dict()]
    config = RunConfig(model='offline-random-tiny-qwen2', device='cpu', dtype='float32',
                       seed=42, population=4, rhos=[0.003], top_k=[1, 2], train_samples=2,
                       original_sigmas=[0.001], candidate_sampling='randopt', antithetic=False, batch_size=2,
                       probe_samples=2, test_samples=2, max_new_tokens=4, max_prompt_tokens=64,
                       reference_max_tokens=32, diagnostic_heads=2, methods=methods, save_texts=True,
                       train_path=str(fixtures / 'train.jsonl'), test_path=str(fixtures / 'test.jsonl'),
                       reference_path=str(fixtures / 'reference.jsonl'), output_dir=str(output / 'run'))
    with patch('socket.socket.connect', side_effect=RuntimeError('Network is forbidden during smoke')), \
         patch('socket.create_connection', side_effect=RuntimeError('Network is forbidden during smoke')):
        backend = tiny_backend(config)
        before = {name: p.detach().clone() for name, p in backend.model.named_parameters()}
        result = run_experiment(config, backend=backend, resume=resume)
        for name, parameter in backend.model.named_parameters():
            if not torch.equal(before[name], parameter):
                raise AssertionError(f'Weights were not restored exactly: {name}')
    write_json(output / 'smoke-verification.json', {
        'status': 'passed', 'network_connections': 'forbidden', 'pretrained_weights_downloaded': False,
        'parameters': sum(p.numel() for p in backend.model.parameters()),
        'methods': len(methods), 'candidates': len(methods) * config.population,
        'all_parameters_restored_exactly': True, 'report': str(result / 'report.md'),
        'interpretation': 'Pipeline validation only; no pretrained 1.5B performance experiment was run.',
    })
    return result
