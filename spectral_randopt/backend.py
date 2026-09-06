"""Transformers inference with no implicit model or tokenizer downloads."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from .data import Example, score_text


class TransformersBackend:
    def __init__(self, model, tokenizer, config, model_id: str):
        self.model, self.tokenizer, self.config = model.eval(), tokenizer, config
        self.model_id = model_id
        self.device = next(model.parameters()).device
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError('A pad token or EOS token is required for batching')
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_config(cls, config):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if config.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Use the offline smoke command locally.')
        if config.device == 'cpu' and config.dtype != 'float32':
            raise ValueError('CPU experiments require dtype=float32')
        torch.manual_seed(config.seed)
        # Download is a separate, explicit prepare command, including on GPU hosts.
        tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision,
                                                  local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError('The selected model must provide a chat template')
        model = AutoModelForCausalLM.from_pretrained(
            config.model, revision=config.revision, local_files_only=True, trust_remote_code=False,
            dtype=getattr(torch, config.dtype), attn_implementation=config.attention_implementation,
        ).to(config.device)
        resolved = getattr(model.config, '_commit_hash', None) or str(Path(config.model).resolve())
        return cls(model, tokenizer, config, f'{config.model}@{resolved}')

    @torch.inference_mode()
    def generate(self, examples: list[Example]) -> list[dict]:
        results = []
        for start in range(0, len(examples), self.config.batch_size):
            batch = examples[start:start + self.config.batch_size]
            prompts = [self.tokenizer.apply_chat_template(e.messages, tokenize=False, add_generation_prompt=True)
                       for e in batch]
            encoded = self.tokenizer(prompts, return_tensors='pt', add_special_tokens=False, padding=True)
            length = encoded['input_ids'].shape[1]
            if length > self.config.max_prompt_tokens:
                raise ValueError(f'Batch starting at {batch[0].id} has a {length}-token prompt, '
                                 'above max_prompt_tokens; refusing silent truncation')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            output = self.model.generate(
                **encoded, do_sample=False, max_new_tokens=self.config.max_new_tokens,
                pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
                use_cache=True,
            )
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            for example, row in zip(batch, output):
                tokens = row[length:].tolist()
                stop = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), len(tokens))
                tokens = tokens[:stop]  # Exclude padding generated after an earlier EOS in this batch.
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                ended = bool(tokens and tokens[-1] in eos_ids)
                results.append({**score_text(text, example), 'generated_tokens': len(tokens),
                                'truncated': len(tokens) >= self.config.max_new_tokens and not ended})
        return results

    def reference_inputs(self, texts: list[str]) -> list[torch.Tensor]:
        inputs = []
        for text in texts:
            ids = self.tokenizer(text, return_tensors='pt', truncation=True,
                                 max_length=self.config.reference_max_tokens)['input_ids']
            if ids.shape[1] < 2:
                raise ValueError('Reference text must contain at least two tokens')
            inputs.append(ids)
        return inputs

    @torch.inference_mode()
    def reference_log_probs(self, ids: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_ids=ids.to(self.device), use_cache=False).logits[0, :-1].float()
        return logits.log_softmax(-1).cpu()

    def diagnostics(self, inputs: list[torch.Tensor], base_log_probs: list[torch.Tensor]) -> dict:
        kl, nll, base_nll, tokens = 0., 0., 0., 0
        for ids, base in zip(inputs, base_log_probs):
            current = self.reference_log_probs(ids)
            labels = ids[0, 1:, None]
            kl += float((base.exp() * (base - current)).sum())
            nll -= float(current.gather(1, labels).sum())
            base_nll -= float(base.gather(1, labels).sum())
            tokens += len(labels)
        return {'reference_kl': max(0., kl / tokens), 'reference_nll': nll / tokens,
                'reference_nll_delta': (nll - base_nll) / tokens, 'reference_tokens': tokens}


def tiny_backend(config) -> TransformersBackend:
    """A real Qwen2 forward/generation path, initialized entirely in memory."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    words = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'user', 'assistant', 'system',
             'What', 'is', 'plus', 'Answer', 'with', 'a', 'number', '####', '.', '?',
             'The', 'small', 'model', 'reads', 'text', 'and', 'writes', 'an', 'answer',
             'Water', 'flows', 'through', 'river', 'Books', 'contain', 'many', 'words',
             'train', 'test'] + [str(i) for i in range(80)]
    vocab = {word: index for index, word in enumerate(words)}
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]',
                                       pad_token='[PAD]', bos_token='[BOS]', eos_token='[EOS]')
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    model_config = Qwen2Config(vocab_size=len(vocab), hidden_size=96, intermediate_size=192,
                              num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=2,
                              max_position_embeddings=256, bos_token_id=2, eos_token_id=3,
                              pad_token_id=0, tie_word_embeddings=True)
    model_config._attn_implementation = config.attention_implementation
    torch.manual_seed(config.seed)
    model = Qwen2ForCausalLM(model_config).eval()
    return TransformersBackend(model, tokenizer, config, f'offline-random-tiny-qwen2:seed{config.seed}')
