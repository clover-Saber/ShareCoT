from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd

from protocol import FEATURE_COLS, TARGET, build_prompt, rationale_is_clean

LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
MAX_LENGTH = 512
MAX_NEW_TOKENS = 256
OUTPUT_INSTRUCTION = (
    '\n请仅输出严格 JSON：先给出简短的分析理由 "reason"，再给出最终标签 "pred"。'
    '最终标签必须是整数 0 或 1。格式：{"reason":"简短理由","pred":0}。'
    '理由应简洁并保证最后的标签完整，整个回答不超过 256 个 token。'
)
LABEL_ONLY_INSTRUCTION = (
    '\n请仅输出严格 JSON：给出最终标签 \"pred\"。'
    '最终标签必须是整数 0 或 1。格式：{\"pred\":0}。'
    '请保证标签完整，整个回答不超过 256 个 token。'
)
SCORE_CONVENTION = 'discrete_generated_label_0_or_1; invalid_prediction_minus_1'


def structured_prompt(row, tokenizer, with_reason=True):
    class PromptTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            messages = [dict(message) for message in messages]
            messages[-1]['content'] += OUTPUT_INSTRUCTION if with_reason else LABEL_ONLY_INSTRUCTION
            return tokenizer.apply_chat_template(messages, **kwargs)
    return build_prompt({key: row[key] for key in FEATURE_COLS}, PromptTokenizer())


def encode_example(prompt, reason, label, tokenizer, max_length=MAX_LENGTH,
                   max_new_tokens=MAX_NEW_TOKENS):
    if type(label) not in (int, np.int64, np.int32) or int(label) not in (0, 1):
        raise ValueError('Training label must be binary integer')
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError('Accepted reason must be nonempty text')
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    eos = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
    budget = min(max_length - len(prompt_ids), max_new_tokens)
    if not prompt_ids or budget < 1:
        raise ValueError('No room in token budget for final label')

    def encode_target(text):
        content = {'reason': text, 'pred': int(label)} if reason is not None else {'pred': int(label)}
        serialized = json.dumps(content, ensure_ascii=False, separators=(',', ':'))
        return tokenizer(serialized, add_special_tokens=False)['input_ids'] + eos

    text = reason.strip() if reason is not None else None
    target = encode_target(text)
    truncated = len(target) > budget
    if truncated and reason is not None:
        if len(encode_target('')) > budget:
            raise ValueError('No room in token budget for JSON reason and final label')
        while len(target) > budget:
            remove = max(1, math.ceil(len(text) * (len(target) - budget) / len(target)))
            text = text[:-remove]
            target = encode_target(text)
    if len(target) > budget:
        raise ValueError('No room in token budget for complete final label')
    ids = prompt_ids + target
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt_ids) + target, 'rationale_truncated': truncated}


def encode_frame(frame, tokenizer, reasons=None):
    reasons = reasons or {}
    return [encode_example(structured_prompt(record, tokenizer,
                                               with_reason=str(record['sample_id']) in reasons),
                           reasons.get(str(record['sample_id'])), int(record[TARGET]), tokenizer)
            for record in frame.to_dict('records')]


class RightPadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        width = max(len(row['input_ids']) for row in rows)
        return {key: torch.tensor([row[key] + [pad] * (width - len(row[key])) for row in rows],
                                  dtype=torch.long)
                for key, pad in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]}


def lora_config():
    from peft import LoraConfig, TaskType
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                      lora_dropout=.05, target_modules=LORA_TARGETS, bias='none')


def parse_prediction(raw):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate output key')
            result[key] = value
        return result
    try:
        obj = json.loads(raw.strip(), object_pairs_hook=unique_pairs)
    except (ValueError, TypeError, AttributeError):
        return None
    if not isinstance(obj, dict) or set(obj) not in ({'pred'}, {'reason', 'pred'}):
        return None
    if type(obj['pred']) is not int or obj['pred'] not in (0, 1):
        return None
    if 'reason' in obj and not isinstance(obj['reason'], str):
        return None
    return obj['pred']


def prediction_records(ids, labels, outputs):
    if len(ids) != len(labels) or len(labels) != len(outputs):
        raise ValueError('Prediction/label lengths differ')
    records = []
    for sid, label, raw in zip(ids, labels, outputs):
        label = int(label)
        parsed = parse_prediction(raw)
        predicted = parsed if parsed is not None else -1
        records.append({'sample_id': str(sid), 'y_true': label, 'parsed_pred': parsed,
                        'valid_output': parsed is not None, 'y_pred': predicted,
                        'prob': float(parsed) if parsed is not None else None, 'raw_generation': raw,
                        'score_convention': SCORE_CONVENTION})
    return records


def generation_metrics(records):
    from baselines_suite import compute_metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef
    if not records:
        raise ValueError('No generation records to evaluate')
    valid = [r for r in records if r['valid_output']]
    metrics = (compute_metrics([r['y_true'] for r in valid], [r['prob'] for r in valid])
               if valid else {'auc_roc': None, 'auc_pr': None, 'brier_score': None, 'log_loss': None})
    labels = [r['y_true'] for r in records]
    predictions = [r['y_pred'] for r in records]
    report = classification_report(labels, predictions, labels=[0, 1], output_dict=True, zero_division=0)
    invalid_by_class = {str(c): sum(not r['valid_output'] and r['y_true'] == c for r in records) for c in (0, 1)}
    matrix = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()
    metrics.update(n=len(records), threshold=.5, accuracy=float(accuracy_score(labels, predictions)),
                   balanced_accuracy=float(np.mean([report[str(c)]['recall'] for c in sorted(set(labels))])),
                   macro_precision=float(report['macro avg']['precision']),
                   macro_recall=float(report['macro avg']['recall']), macro_f1=float(report['macro avg']['f1-score']),
                   weighted_f1=float(report['weighted avg']['f1-score']), f1=float(report['1']['f1-score']),
                   mcc=float(matthews_corrcoef(labels, predictions)),
                   per_class={str(c): {'precision': float(report[str(c)]['precision']),
                                       'recall': float(report[str(c)]['recall']),
                                       'f1': float(report[str(c)]['f1-score']),
                                       'support': int(report[str(c)]['support'])} for c in (0, 1)},
                   confusion_matrix=matrix,
                   confusion_matrix_with_invalid=[matrix[c] + [invalid_by_class[str(c)]] for c in (0, 1)],
                   confusion_matrix_with_invalid_columns=[0, 1, -1], invalid_by_true_class=invalid_by_class,
                   classification_report=report, probability_metrics_population='valid_generations_only',
                   probability_metrics_n=len(valid))
    metrics.update(invalid_count=sum(not r['valid_output'] for r in records),
                   invalid_fraction=sum(not r['valid_output'] for r in records) / len(records),
                   score_convention=SCORE_CONVENTION, probability_metrics_comparable=False,
                   probability_metrics_note='ROC/PR AUC, Brier and log loss use only valid discrete generated labels; '
                                            'they are not probability-calibration comparisons. Classification '
                                            'metrics include invalid -1 as incorrect, creating a false negative '
                                            'for the true class without inventing a binary prediction.')
    objects = [json.loads(r['raw_generation']) for r in valid]
    nonempty_reason_count = sum(isinstance(obj.get('reason'), str) and bool(obj['reason'].strip())
                                for obj in objects)
    metrics.update(parsed_valid_output_count=len(valid), nonempty_reason_count=nonempty_reason_count,
                   nonempty_reason_rate=nonempty_reason_count / len(records),
                   pred_only_count=sum(set(obj) == {'pred'} for obj in objects),
                   reason_audit_population='all_generations; nonempty_reason_requires_valid_parsed_output')
    return metrics


def read_split(path):
    frame = pd.read_csv(path, dtype={'sample_id': str})
    if set(frame.columns) != {'sample_id', TARGET, *FEATURE_COLS}:
        raise ValueError(f'{path} must contain exactly the registered features, sample_id and target')
    if frame.empty or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError(f'{path} has zero rows or missing/duplicate sample IDs')
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError(f'{path} contains invalid binary labels')
    return frame


def load_rationales(path, train_ids, train_labels):
    accepted, seen = {}, set()
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sid = str(record['sample_id'])
            if sid not in train_ids:
                raise ValueError(f'Rationale ID {sid} is outside training')
            if sid in seen:
                raise ValueError(f'Duplicate rationale ID {sid}')
            seen.add(sid)
            if 'label' in record and (type(record['label']) is not int or
                                     record['label'] != int(train_labels[sid])):
                raise ValueError(f'Rationale {sid} label differs from training label')
            if record.get('accepted') is True:
                if type(record.get('label')) is not int or record['label'] not in (0, 1):
                    raise ValueError(f'Accepted rationale {sid} requires integer binary label')
                reason = record.get('reason')
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f'Accepted rationale {sid} has no reason text')
                if not rationale_is_clean(reason):
                    raise ValueError(f'Accepted rationale {sid} contains forbidden content')
                accepted[sid] = reason.strip()
    return accepted


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def write_predictions(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.tmp')
    pd.DataFrame(records).to_csv(temp, index=False)
    os.replace(temp, path)


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_trainable(model, state):
    import torch
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(params) != set(state):
        raise ValueError('Checkpoint trainable parameters differ from model')
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != params[name].shape:
                raise ValueError(f'Checkpoint tensor shape differs: {name}')
            params[name].copy_(value)


def verify_serialized_adapter(model, directory, selected_state):
    import torch
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    weights = load_peft_weights(str(directory), device='cpu', local_files_only=True)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.fill_(float('nan'))
    result = set_peft_model_state_dict(model, weights, adapter_name='default')
    if result.unexpected_keys:
        raise RuntimeError('Serialized adapter has unexpected tensors: ' + ', '.join(result.unexpected_keys))
    reloaded = trainable_state(model)
    if set(reloaded) != set(selected_state) or any(
            not torch.equal(reloaded[name], selected_state[name]) for name in selected_state):
        raise RuntimeError('Serialized selected adapter does not reproduce every selected trainable tensor')
    return {'verified': True, 'adapter': str(directory), 'tensor_count': len(selected_state)}


def save_resume(path, model, optimizer, scheduler, state):
    import torch
    payload = dict(state, trainable=trainable_state(model), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate())
    temp = Path(str(path) + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def restore_resume(path, model, optimizer, scheduler):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    restore_trainable(model, checkpoint['trainable'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    torch.set_rng_state(checkpoint['torch_rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    np.random.set_state(checkpoint['numpy_rng'])
    random.setstate(checkpoint['python_rng'])
    keys = ('epoch', 'offset', 'step', 'best_score', 'best_epoch', 'best_step', 'best_state', 'history')
    return {key: checkpoint[key] for key in keys}


def generate_predictions(model, frame, tokenizer, device):
    import torch
    model.eval()
    outputs = []
    with torch.inference_mode():
        for record in frame.to_dict('records'):
            prompt = structured_prompt(record, tokenizer)
            batch = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
            batch = {key: value.to(device) for key, value in batch.items()}
            if batch['input_ids'].shape[1] > MAX_LENGTH:
                raise ValueError('Generation prompt exceeds fixed input budget')
            generated = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS,
                                       do_sample=False, num_beams=1, use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
            outputs.append(tokenizer.decode(generated[0, batch['input_ids'].shape[1]:],
                                            skip_special_tokens=True))
    return prediction_records(frame.sample_id.tolist(), frame[TARGET].tolist(), outputs)


def finish_selected_run(root, model, tokenizer, device, state, seed, smoke=False,
                        development_ids=None, serialization_verified=False):
    if state['best_epoch'] is None or state['best_step'] is None or state['best_score'] < 0:
        raise RuntimeError('No validation-selected checkpoint exists')
    root = Path(root)
    output = root / 'results' / 'single_cot'
    metadata = {'method': 'single_cot', 'seed': seed, 'best_epoch': state['best_epoch'],
                'best_step': state['best_step'], 'best_validation_macro_f1': state['best_score'],
                'step': state['step'], 'trained_steps': state['step'], 'smoke_run': smoke,
                'checkpoint': str(output / 'model'), 'threshold': .5}
    if smoke:
        metrics = {**metadata, 'validation_history': state['history']}
        atomic_json(output / 'smoke_metrics.json', metrics)
        atomic_json(output / 'logs' / 'progress.json', {'status': 'smoke_complete', **metadata})
        return metrics
    if not serialization_verified:
        raise RuntimeError('Selected serialized adapter reload must be verified before test access')
    test = read_split(root / 'data' / 'test.csv')
    if development_ids and set(test.sample_id) & set(development_ids):
        raise ValueError('Test sample IDs overlap development')
    records = generate_predictions(model, test, tokenizer, device)
    metrics = {**generation_metrics(records), **metadata,
               'test_n': len(test)}
    write_predictions(output / 'predictions.csv', records)
    atomic_json(output / 'metrics.json', metrics)
    atomic_json(output / 'logs' / 'progress.json', {'status': 'complete', **metadata, 'metrics': metrics})
    return metrics


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--gpu', type=int, default=0, help='Logical index after CUDA_VISIBLE_DEVICES')
    parser.add_argument('--checkpoint-steps', type=int, default=100)
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.checkpoint_steps < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('epochs, checkpoint-steps and max-steps must be positive')
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, set_seed
    from peft import get_peft_model
    from train_llm import base_manifest
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the matched 7B training protocol')
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    torch.cuda.set_device(args.gpu)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 GPU support is required')
    device = torch.device('cuda', args.gpu)
    root = args.root.resolve()
    output = root / 'results' / 'single_cot'
    data = root / 'data'
    train, val = read_split(data / 'train.csv'), read_split(data / 'val.csv')
    if set(train.sample_id) & set(val.sample_id):
        raise ValueError('Training and validation sample IDs overlap')
    reasons = load_rationales(data / 'rationales.jsonl', set(train.sample_id),
                              dict(zip(train.sample_id, train[TARGET])))
    if not reasons:
        raise ValueError('Single-CoT requires accepted clean training rationales')
    manifest = base_manifest(args.model_dir)
    config = {'method': 'single_cot', 'model_dir': str(args.model_dir.resolve()),
              'seed': args.seed, 'epochs': args.epochs,
              'max_steps': args.max_steps, 'checkpoint_steps': args.checkpoint_steps,
              'features': list(FEATURE_COLS), 'target': TARGET, 'loss': 'causal_lm_cross_entropy',
              'classification_head': False, 'train_batch_size': 1, 'gradient_accumulation': 8,
              'max_length': MAX_LENGTH, 'max_new_tokens': MAX_NEW_TOKENS,
              'output_instruction': OUTPUT_INSTRUCTION, 'generation': 'greedy',
              'label_only_instruction': LABEL_ONLY_INSTRUCTION,
              'training_instruction_routing': 'accepted_reason_mask; reason_examples_use_original_output_instruction; otherwise_label_only',
              'generation_instruction_routing': 'original_output_instruction_for_all_validation_and_test',
              'lr': 1e-4, 'optimizer': 'AdamW', 'weight_decay': 0., 'scheduler': 'cosine',
              'warmup_ratio': .1, 'precision': 'bfloat16', 'gradient_checkpointing': True,
              'clip_grad_norm': 1., 'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': .05,
              'lora_targets': LORA_TARGETS, 'threshold': .5, 'selection_metric': 'validation_macro_f1',
              'score_convention': SCORE_CONVENTION, 'torch_version': torch.__version__,
              'transformers_version': transformers.__version__, 'peft_version': peft.__version__}
    config_path = output / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Existing run config differs; use a new root')
    atomic_json(config_path, config)
    atomic_json(output / 'base_model_manifest.json', manifest)
    progress = output / 'logs' / 'progress.json'
    smoke = args.max_steps is not None
    expected_status = 'smoke_complete' if smoke else 'complete'
    if progress.exists() and json.loads(progress.read_text(encoding='utf-8')).get('status') == expected_status:
        required = [output / 'model' / 'adapter_config.json',
                    output / ('smoke_metrics.json' if smoke else 'metrics.json')]
        if not smoke:
            required.append(output / 'predictions.csv')
        if not all(path.exists() for path in required):
            raise RuntimeError('Completed run is missing artifacts')
        print(f'Already complete: {output}', flush=True)
        return 0
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = encode_frame(train, tokenizer, reasons)
    atomic_json(output / 'encoding_report.json', {'train_rows': len(train_rows), 'validation_rows': len(val),
                'accepted_rationales': len(reasons), 'used_rationales': len(reasons),
                'reason_instruction_rows': len(reasons), 'label_only_instruction_rows': len(train_rows) - len(reasons),
                'rationales_truncated': sum(row['rationale_truncated'] for row in train_rows),
                'max_train_tokens': max(len(row['input_ids']) for row in train_rows)})
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError('Base tensors remain on meta; refusing to reinitialize')
    set_seed(args.seed)
    model = get_peft_model(model, lora_config()).to(device)
    atomic_json(output / 'initialization.json', {'seed': args.seed, 'fresh_base': True, 'classification_head': False})
    atomic_json(output / 'base_config.json', model.config.to_dict())
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.)
    total_steps = math.ceil(len(train_rows) / 8) * args.epochs
    if smoke:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps * .1), total_steps)
    collator = RightPadCollator(tokenizer.pad_token_id)
    resume_path = output / 'resume.pt'
    state = {'epoch': 0, 'offset': 0, 'step': 0, 'best_score': -1., 'best_epoch': None,
             'best_step': None, 'best_state': None, 'history': []}
    if resume_path.exists():
        state = restore_resume(resume_path, model, optimizer, scheduler)
        print(f'Resumed epoch={state["epoch"]}, offset={state["offset"]}, step={state["step"]}', flush=True)
    started = time.time()
    try:
        for epoch in range(state['epoch'], args.epochs):
            order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
            offset = state['offset'] if epoch == state['epoch'] else 0
            model.train()
            while offset < len(order) and state['step'] < total_steps:
                group = order[offset:offset + 8]
                optimizer.zero_grad(set_to_none=True)
                loss_total = 0.
                for index in group:
                    batch = {key: value.to(device) for key, value in collator([train_rows[index]]).items()}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        objective = model(**batch, use_cache=False).loss
                        loss = objective / len(group)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, sample={index}')
                    loss.backward()
                    loss_total += objective.detach().float().item() / len(group)
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                offset += len(group)
                state.update(epoch=epoch, offset=offset, step=state['step'] + 1)
                report = {'status': 'training', 'method': 'single_cot', 'epoch': epoch + 1,
                          'sample_offset': offset, 'step': state['step'], 'total_steps': total_steps,
                          'loss': loss_total, 'learning_rate': scheduler.get_last_lr()[0],
                          'elapsed_seconds': time.time() - started,
                          'best_validation_macro_f1': state['best_score']}
                atomic_json(progress, report)
                if state['step'] == 1 or state['step'] % 10 == 0:
                    print(json.dumps(report), flush=True)
                if state['step'] % args.checkpoint_steps == 0:
                    save_resume(resume_path, model, optimizer, scheduler, state)
            if offset > 0:
                records = generate_predictions(model, val, tokenizer, device)
                metrics = generation_metrics(records)
                entry = {'epoch': epoch + 1, 'step': state['step'], 'partial_epoch': offset < len(order), **metrics}
                state['history'].append(entry)
                write_predictions(output / 'validation' / f'epoch_{epoch + 1}_step_{state["step"]}.csv', records)
                if metrics['macro_f1'] > state['best_score']:
                    state.update(best_score=metrics['macro_f1'], best_epoch=epoch + 1,
                                 best_step=state['step'], best_state=trainable_state(model))
                    write_predictions(output / 'validation_predictions.csv', records)
                atomic_json(output / 'validation_history.json', state['history'])
                print('Validation: ' + json.dumps(entry), flush=True)
            state.update(epoch=epoch + 1, offset=0)
            save_resume(resume_path, model, optimizer, scheduler, state)
            if state['step'] >= total_steps:
                break
        if state['best_state'] is None:
            raise RuntimeError('No validation-selected checkpoint exists')
        restore_trainable(model, state['best_state'])
        temp_model = output / 'model.tmp'
        temp_model.mkdir(exist_ok=True)
        model.save_pretrained(temp_model, safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(temp_model)
        model.config.save_pretrained(temp_model)
        model.generation_config.save_pretrained(temp_model)
        model_dir = output / 'model'
        if model_dir.exists():
            shutil.rmtree(model_dir)
        os.replace(temp_model, model_dir)
        verification = verify_serialized_adapter(model, model_dir, state['best_state'])
        atomic_json(output / 'serialized_reload_verification.json', verification)
        metrics = finish_selected_run(root, model, tokenizer, device, state, args.seed, smoke=smoke,
                                      development_ids=set(train.sample_id) | set(val.sample_id),
                                      serialization_verified=verification['verified'])
        resume_path.unlink(missing_ok=True)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        atomic_json(progress, {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                              'method': 'single_cot', 'step': state['step'], 'error': repr(error),
                              'resume_checkpoint': str(resume_path) if resume_path.exists() else None})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
