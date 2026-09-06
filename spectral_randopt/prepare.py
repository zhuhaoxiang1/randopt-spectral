"""Explicit network-enabled preparation; never imported by the offline smoke path."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from .runner import write_json


def prepare(output_dir: str, model: str, revision: str, data_only: bool, model_only: bool) -> None:
    if not data_only:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=model, revision=revision,
                                 allow_patterns=['*.json', '*.safetensors', '*.model', 'merges.txt', 'vocab.txt'])
        print(f'Model cached at: {path}', flush=True)
    if model_only:
        return
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for split in ('train', 'test'):
        target = output / f'gsm8k_{split}.jsonl'
        if target.exists():
            raise FileExistsError(f'{target} already exists; choose a new preparation directory')
        url = f'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl'
        print(f'Downloading official GSM8K {split} JSONL.', flush=True)
        with urlopen(url, timeout=60) as response:
            raw = response.read()
        rows = []
        for index, line in enumerate(raw.decode('utf-8').splitlines()):
            item = json.loads(line)
            answer = item['answer'].rsplit('####', 1)[-1].strip().replace(',', '')
            rows.append({'id': f'gsm8k-{split}-{index}', 'messages': [{'role': 'user',
                         'content': item['question'] + '\nPlease reason step by step, and put your final answer after ####.'}],
                         'answer': answer})
        temporary = target.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        temporary.replace(target)
        provenance[split] = {'source': url, 'source_sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rows)}
    reference = [
        'A library keeps books in an organized collection. Readers can borrow a book and return it later.',
        'Water flows downhill through streams and rivers. Some of it eventually reaches the ocean.',
        'A person writes a letter to describe an event. Another person reads it and replies with a question.',
        'Plants use light and water as they grow. Their leaves and roots have different roles.',
    ]
    (output / 'reference.jsonl').write_text(''.join(json.dumps({'text': text}) + '\n' for text in reference))
    write_json(output / 'provenance.json', {'gsm8k': provenance,
               'reference': 'Four fixed diagnostic snippets; not a general-capability benchmark.'})
