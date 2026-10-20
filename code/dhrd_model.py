"""Paper-based DHRD-style adaptation to the matched sparse-rationale corpus.

Two-layer MLP width/activation are implementation choices (not specified by
the paper): hidden_size -> GELU -> two class logits. No new tokenizer tokens.
All valid tokens receive causal LM supervision, including label-only rows.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2ForCausalLM
from peft import LoraConfig, TaskType

LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def encode_example(prompt, reason, label, tokenizer, max_length=512):
    if label not in (0, 1):
        raise ValueError('Class label must be 0 or 1')
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    suffix = ('<REASON>' + reason if reason else '') + '<ANS>' + str(int(label))
    suffix_ids = tokenizer(suffix, add_special_tokens=False)['input_ids']
    if tokenizer.eos_token_id is not None:
        suffix_ids.append(tokenizer.eos_token_id)
    ids = prompt_ids + suffix_ids
    if not prompt_ids or len(ids) > max_length:
        raise ValueError(f'DHRD sequence length {len(ids)} exceeds {max_length}; no silent truncation')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': ids.copy(),
            'class_labels': float(label), 'cls_token_index': len(prompt_ids) - 1,
            'rationale_truncated': False}


def encode_input(prompt, label, tokenizer, max_length=512):
    """Classification validation/test inputs contain neither reason nor answer."""
    ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    if not ids or len(ids) > max_length:
        raise ValueError('Invalid inference prompt length')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': [-100] * len(ids),
            'class_labels': float(label), 'cls_token_index': len(ids) - 1,
            'rationale_truncated': False}


class DHRDStyleQwen(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.classification_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Linear(config.hidden_size, 2))

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                class_labels=None, cls_token_index=None, compute_lm=False,
                inputs_embeds=None, **kwargs):
        for key in ('return_dict', 'output_hidden_states', 'use_cache',
                    'num_logits_to_keep', 'logits_to_keep'):
            kwargs.pop(key, None)
        states = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            inputs_embeds=inputs_embeds, use_cache=False,
                            return_dict=True, output_hidden_states=False, **kwargs).last_hidden_state
        if cls_token_index is None:
            raise ValueError('Explicit input boundary is required to prevent answer leakage')
        index = cls_token_index.to(device=states.device, dtype=torch.long)
        pooled = states[torch.arange(states.shape[0], device=states.device), index]
        raw = self.classification_head(pooled)
        cls_loss = states.new_zeros((), dtype=torch.float32)
        if class_labels is not None:
            cls_loss = F.cross_entropy(raw.float(), class_labels.long())
        lm_loss = states.new_zeros((), dtype=torch.float32)
        if compute_lm and labels is not None:
            targets = labels[:, 1:]
            active = targets.ne(-100)
            if active.any():
                lm_logits = self.lm_head(states[:, :-1][active])
                lm_loss = F.cross_entropy(lm_logits.float(), targets[active])
        # sigmoid(z1-z0) == softmax([z0,z1])[1], preserving common evaluator.
        return {'raw_class_logits': raw, 'class_logits': raw[:, 1] - raw[:, 0],
                'loss_cls': cls_loss, 'loss_lm': lm_loss, 'loss': cls_loss + lm_loss}


def initialize_classifier(model, seed):
    head_prefix = 'classification_head.'
    unexpected = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
                  if tensor.is_meta and not name.startswith(head_prefix)]
    if unexpected:
        raise RuntimeError('Pretrained tensors remain on meta: ' + str(unexpected[:5]))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    initial = []
    for module in model.classification_head:
        if isinstance(module, nn.Linear):
            dtype = module.weight.dtype
            weight = torch.randn(module.weight.shape, generator=generator) * model.config.initializer_range
            bias = torch.zeros(module.bias.shape)
            module.weight = nn.Parameter(weight.to(dtype))
            module.bias = nn.Parameter(bias.to(dtype))
            initial.extend([weight.flatten(), bias.flatten()])
    return torch.cat(initial)


def lora_config():
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                      lora_dropout=.05, target_modules=LORA_TARGETS,
                      modules_to_save=['classification_head'], bias='none')
