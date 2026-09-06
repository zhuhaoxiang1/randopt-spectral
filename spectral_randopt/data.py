"""Local-only input parsing and GSM8K scoring using upstream answer extraction."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import random

from utils.reward_score.gsm8k import extract_solution


def numeric_answer(value: str | None) -> str:
    if value is None:
        return ''
    try:
        number = Decimal(value.strip().replace(',', ''))
    except InvalidOperation:
        return ''
    if not number.is_finite():
        return ''
    return str(number.normalize()) if number != 0 else '0'


@dataclass(frozen=True)
class Example:
    id: str
    messages: list[dict[str, str]]
    answer: str

    def content_key(self) -> str:
        return hashlib.sha256(json.dumps(self.messages, sort_keys=True).encode()).hexdigest()


def read_examples(path: str) -> list[Example]:
    examples = []
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get('messages')
            if messages is None and isinstance(row.get('question'), str):
                messages = [{'role': 'user', 'content': row['question']}]
            if not isinstance(messages, list) or not messages or any(
                not isinstance(m, dict) or m.get('role') not in {'system', 'user', 'assistant'}
                or not isinstance(m.get('content'), str) for m in messages
            ):
                raise ValueError(f'{path}:{line_number}: invalid messages')
            answer = row.get('ground_truth', row.get('answer'))
            if answer is None:
                raise ValueError(f'{path}:{line_number}: missing answer')
            if '####' in str(answer):
                answer = extract_solution(str(answer), method='strict')
            if answer is None or not numeric_answer(str(answer)):
                raise ValueError(f'{path}:{line_number}: invalid answer')
            key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
            examples.append(Example(str(row.get('id', key)), messages, numeric_answer(str(answer))))
    if not examples or len({e.id for e in examples}) != len(examples):
        raise ValueError(f'{path}: empty data or duplicate IDs')
    if len({e.content_key() for e in examples}) != len(examples):
        raise ValueError(f'{path}: duplicate prompts')
    return examples


def load_splits(config) -> tuple[list[Example], list[Example], list[Example], list[str]]:
    train, test = read_examples(config.train_path), read_examples(config.test_path)
    if {e.id for e in train} & {e.id for e in test} or {e.content_key() for e in train} & {e.content_key() for e in test}:
        raise ValueError('Selection and test sets overlap; supply disjoint data')
    rng = random.Random(config.seed)
    rng.shuffle(train)
    rng.shuffle(test)
    if len(train) < config.train_samples + config.probe_samples or (config.test_samples and len(test) < config.test_samples):
        raise ValueError('Insufficient data; reduce sample counts explicitly')
    references = []
    with Path(config.reference_path).open() as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)['text']
                if not isinstance(value, str) or not value.strip():
                    raise ValueError('Reference text must be nonempty')
                references.append(value)
    if not references:
        raise ValueError('Reference dataset is empty')
    selection = train[:config.train_samples]
    probe = train[config.train_samples:config.train_samples + config.probe_samples]
    return selection, probe, test[:config.test_samples], references


def score_text(text: str, example: Example) -> dict:
    strict = extract_solution(text, method='strict')
    answer = strict if strict is not None else extract_solution(text, method='flexible')
    strict = numeric_answer(strict)
    answer = numeric_answer(answer)
    expected = numeric_answer(example.answer)
    return {'id': example.id, 'text': text, 'answer': answer, 'strict_answer': strict,
            'correct': bool(answer) and answer == expected, 'strict_correct': bool(strict) and strict == expected}


def summarize(predictions: list[dict]) -> dict:
    n = len(predictions)
    if n == 0:
        raise ValueError('Cannot summarize empty predictions')
    return {'accuracy': sum(p['correct'] for p in predictions) / n,
            'strict_accuracy': sum(p['strict_correct'] for p in predictions) / n,
            'truncation_rate': sum(p.get('truncated', False) for p in predictions) / n,
            'generated_tokens': sum(p.get('generated_tokens', 0) for p in predictions), 'samples': n}


def majority_vote(member_predictions: list[list[dict]], examples: list[Example]) -> list[dict]:
    if not member_predictions or any(len(p) != len(examples) for p in member_predictions):
        raise ValueError('Invalid ensemble dimensions')
    voted = []
    for i, example in enumerate(examples):
        if any(member[i]['id'] != example.id for member in member_predictions):
            raise ValueError('Ensemble prediction IDs are misaligned')
        answers = [member[i]['answer'] for member in member_predictions if member[i]['answer']]
        counts = Counter(answers)
        # Equal votes go to the highest selection-ranked member, never test accuracy.
        winner = next((a for a in answers if counts[a] == max(counts.values())), '')
        voted.append({'id': example.id, 'answer': winner, 'correct': bool(winner) and winner == numeric_answer(example.answer),
                      'strict_correct': False})
    return voted
