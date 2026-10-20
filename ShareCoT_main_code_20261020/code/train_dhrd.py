"""Matched DHRD-style baseline, fixed 1:1 objective. No remote model downloads or test selection.

CUDA_VISIBLE_DEVICES is set by the caller; --gpu is a logical CUDA device index.
Checkpoints contain trainable tensors and optimizer state only, never the base 7B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen2ForCausalLM, get_cosine_schedule_with_warmup, set_seed
from peft import LoraConfig, TaskType, get_peft_model

from protocol import FEATURE_COLS, TARGET, build_prompt, configure_semantic_dictionary
from loss_weighting import LossWeighting
from dhrd_model import DHRDStyleQwen as DualHeadQwen, encode_example, encode_input, lora_config, initialize_classifier

LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']




class RightPadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        width = max(len(row['input_ids']) for row in rows)
        batch = {}
        for key, pad in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]:
            batch[key] = torch.tensor([r[key] + [pad] * (width - len(r[key])) for r in rows], dtype=torch.long)
        batch['class_labels'] = torch.tensor([r['class_labels'] for r in rows], dtype=torch.float32)
        batch['cls_token_index'] = torch.tensor([r['cls_token_index'] for r in rows], dtype=torch.long)
        return batch








def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def base_manifest(model_dir):
    """Hash actual local weights and model/tokenizer metadata on every invocation."""
    model_dir = Path(model_dir)
    suffixes = {'.safetensors', '.bin', '.json', '.txt', '.model', '.tiktoken', '.jinja'}
    files = []
    for path in sorted(model_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() in suffixes and '.cache' not in path.relative_to(model_dir).parts:
            files.append({'path': path.relative_to(model_dir).as_posix(),
                          'size_bytes': path.stat().st_size, 'sha256': file_hash(path)})
    if not files or not any(Path(entry['path']).suffix in {'.safetensors', '.bin'} for entry in files):
        raise ValueError('Base model manifest contains no local weight files')
    canonical = json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return {'sha256': hashlib.sha256(canonical).hexdigest(), 'files': files}


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def validate_config(output, config):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'run_config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != config:
        raise ValueError(f'Existing run config differs: {path}. Use a new root for a different experiment.')
    atomic_json(path, config)


def load_rationales(path, train_ids, train_labels=None):
    accepted = {}
    seen = set()
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sid = str(record['sample_id'])
            if sid not in train_ids:
                raise ValueError(f'Rationale ID {sid} is outside the training split')
            if sid in seen:
                raise ValueError(f'Duplicate rationale ID {sid}')
            seen.add(sid)
            if record.get('accepted') is True:
                reason = record.get('reason')
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(f'Accepted rationale {sid} has no reason text')
                if train_labels is not None and 'label' in record:
                    if record['label'] not in (0, 1) or record['label'] != train_labels[sid]:
                        raise ValueError(f'Accepted rationale {sid} label differs from training label')
                accepted[sid] = reason.strip()
    return accepted


def read_split(path):
    frame = pd.read_csv(path, dtype={'sample_id': str})
    expected = {'sample_id', TARGET, *FEATURE_COLS}
    if set(frame.columns) != expected:
        raise ValueError(f'{path} must have exactly sample_id, target and the nine registered features')
    if frame.empty or frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError(f'{path} has empty/duplicate sample IDs or zero rows')
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError(f'{path} has invalid binary labels')
    return frame




def encode_frame(frame, tokenizer, reasons=None):
    rows = []
    for record in frame.to_dict('records'):
        prompt = build_prompt({k: record[k] for k in FEATURE_COLS}, tokenizer)
        if reasons is None:
            row = encode_input(prompt, record[TARGET], tokenizer)
        else:
            row = encode_example(prompt, reasons.get(str(record['sample_id'])), record[TARGET], tokenizer)
        rows.append(row)
    return rows


def epoch_order(length, seed, epoch):
    return torch.randperm(length, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def classification_metrics(labels, probabilities):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    return {'accuracy': float(accuracy_score(labels, predictions)),
            'macro_f1': float(f1_score(labels, predictions, average='macro', labels=[0, 1], zero_division=0)),
            'macro_precision': float(precision_score(labels, predictions, average='macro', labels=[0, 1], zero_division=0)),
            'macro_recall': float(recall_score(labels, predictions, average='macro', labels=[0, 1], zero_division=0)),
            'positive_f1': float(f1_score(labels, predictions, zero_division=0)),
            'positive_recall': float(recall_score(labels, predictions, zero_division=0)),
            'roc_auc': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None}


@torch.no_grad()
def evaluate(model, rows, collator, device, batch_size=8):
    model.eval()
    probabilities = []
    for batch in DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collator):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(**batch, compute_lm=False)['class_logits']
        probabilities.extend(logits.float().sigmoid().cpu().tolist())
    return probabilities


def trainable_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def restore_trainable(model, state):
    parameters = dict(model.named_parameters())
    if set(state) != {name for name, parameter in parameters.items() if parameter.requires_grad}:
        raise ValueError('Checkpoint trainable parameter names differ from model')
    with torch.no_grad():
        for name, tensor in state.items():
            parameters[name].copy_(tensor)


def save_resume(path, model, optimizer, scheduler, state):
    payload = dict(state)
    payload.update(trainable=trainable_state(model), optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                   python_rng=random.getstate())
    temp = Path(str(path) + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def completed_run(output, data, smoke=False):
    output = Path(output)
    progress = output / 'logs' / 'progress.json'
    if not progress.exists():
        return False
    status = json.loads(progress.read_text(encoding='utf-8')).get('status')
    expected_status = 'smoke_complete' if smoke else 'complete'
    if status != expected_status:
        return False
    artifacts = [output / 'model' / 'adapter_config.json']
    artifacts += [output / 'smoke_metrics.json'] if smoke else [output / 'metrics.json', output / 'predictions.csv']
    for path in artifacts:
        if not path.exists():
            raise RuntimeError(f'Complete run missing artifact {path}')
    if not smoke:
        prior = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
        if prior['test_sha256'] != file_hash(Path(data) / 'test.csv'):
            raise ValueError('Completed run test data config differs')
    return True


def finish_selected_run(root, method, model, tokenizer, collator, device,
                        eval_batch_size, state, seed, smoke=False,
                        development_ids=None):
    """Smoke terminates at validation; only a full run may open the test split."""
    root = Path(root)
    output = root / 'results' / method
    metadata = {'method': method, 'seed': seed, 'threshold': 0.5,
                'best_epoch': state['best_epoch'], 'best_step': state['best_step'],
                'best_validation_macro_f1': state['best_score'], 'trained_steps': state['step'], 'step': state['step'],
                'checkpoint': str(output / 'model'), 'smoke_run': smoke}
    if hasattr(model, 'loss_weighting'):
        metadata['selected_loss_weights'] = model.loss_weighting.snapshot()
    if smoke:
        metrics = {**metadata, 'validation_history': state['history']}
        atomic_json(output / 'smoke_metrics.json', metrics)
        atomic_json(output / 'logs' / 'progress.json', {'status': 'smoke_complete', **metadata})
        return metrics
    # First test-set access occurs only after full-run checkpoint selection.
    test = read_split(root / 'data' / 'test.csv')
    if development_ids and set(development_ids) & set(test.sample_id):
        raise ValueError('Test sample IDs overlap training or validation')
    test_rows = encode_frame(test, tokenizer)
    probabilities = evaluate(model, test_rows, collator, device, eval_batch_size)
    metrics = {**classification_metrics(test[TARGET], probabilities), **metadata,
               'test_sha256': file_hash(root / 'data' / 'test.csv'), 'test_n': len(test)}
    predictions = pd.DataFrame({'sample_id': test.sample_id, 'y_true': test[TARGET],
                               'y_pred': (np.asarray(probabilities) >= 0.5).astype(int), 'prob': probabilities})
    predictions.to_csv(output / 'predictions.csv.tmp', index=False)
    os.replace(output / 'predictions.csv.tmp', output / 'predictions.csv')
    atomic_json(output / 'metrics.json', metrics)
    atomic_json(output / 'logs' / 'progress.json', {'status': 'complete', **metadata, 'metrics': metrics})
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--method', required=True, choices=['dhrd_style'])
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--gpu', type=int, default=0, help='Logical index after CUDA_VISIBLE_DEVICES')
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--checkpoint-steps', type=int, default=100)
    parser.add_argument('--weighting', choices=['fixed_1'], default='fixed_1')
    args = parser.parse_args(argv)
    if args.method != 'dhrd_style' and args.weighting != 'fixed_3':
        parser.error('New weighting modes apply only to ShareCoT')
    if args.epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error('epochs/max-steps must be positive')
    if args.checkpoint_steps < 1 or args.eval_batch_size < 1:
        parser.error('checkpoint-steps/eval-batch-size must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this 7B experiment; run CPU unit tests separately')
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    torch.cuda.set_device(args.gpu)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This matched protocol requires a GPU supporting bfloat16')
    device = torch.device('cuda', args.gpu)
    root = args.root.resolve()
    output = root / 'results' / args.method
    data = root / 'data'
    serialization_metadata = configure_semantic_dictionary(data)
    progress = output / 'logs' / 'progress.json'
    train = read_split(data / 'train.csv')
    val = read_split(data / 'val.csv')
    if set(train.sample_id) & set(val.sample_id):
        raise ValueError('Training and validation sample IDs overlap')
    rationale_path = data / 'rationales.jsonl'
    # Bind this run to the newly generated no-attribution rationale artifact.
    reasons = load_rationales(rationale_path, set(train.sample_id), dict(zip(train.sample_id, train[TARGET])))
    if args.method == 'dhrd_style' and not reasons:
        raise ValueError('DHRD requires at least one accepted training rationale')
    import transformers, peft
    print('Hashing base weights and tokenizer for immutable provenance...', flush=True)
    manifest = base_manifest(args.model_dir)
    config = {'method': args.method, 'model_dir': str(args.model_dir.resolve()),
              'base_manifest_sha256': manifest['sha256'],
              'seed': args.seed, 'epochs': args.epochs, 'max_steps': args.max_steps,
              'train_sha256': file_hash(data / 'train.csv'), 'val_sha256': file_hash(data / 'val.csv'),
              'rationales_sha256': file_hash(rationale_path),
              'code_sha256': file_hash(Path(__file__)),
              'protocol_sha256': file_hash(Path(__file__).with_name('protocol.py')),
              **serialization_metadata,
              'features': list(FEATURE_COLS), 'target': TARGET,
              'train_batch_size': 1, 'gradient_accumulation': 8, 'max_length': 512,
              'lr': 1e-4, 'scheduler': 'cosine', 'warmup_ratio': 0.1,
              'loss_lm_weight': (None if args.weighting == 'adaptive' else
                                float(args.weighting.split('_')[1])) if args.method == 'dhrd_style' else 0.0,
              'loss_cls_weight': None if args.weighting == 'adaptive' else 1.0,
              'class_weight': None, 'weighting_mode': args.weighting,
              'weighting_code_sha256': file_hash(Path(__file__).with_name('loss_weighting.py')),
              'adaptive_initial_log_variances': [0., 0.] if args.weighting == 'adaptive' else None,
              'adaptive_objective': '0.5*(exp(-s_cls)*BCE+s_cls)+0.5*mask*(exp(-s_lm)*CE+s_lm)' if args.weighting == 'adaptive' else None,
              'adaptive_lr': 1e-4 if args.weighting == 'adaptive' else None,
              'lm_normalization': 'full_sequence_token_mean_per_microbatch_then_accumulation_mean',
              'lora_r': 8, 'lora_alpha': 16, 'lora_dropout': 0.05, 'lora_targets': LORA_TARGETS,
              'threshold': 0.5, 'selection_metric': 'validation_macro_f1',
              'eval_batch_size': args.eval_batch_size, 'checkpoint_steps': args.checkpoint_steps,
              'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
              'peft_version': peft.__version__}
    config.update(
        implementation='paper_based_DHRD_style_sparse_rationale_adaptation',
        dhrd_model_sha256=file_hash(Path(__file__).with_name('dhrd_model.py')),
        classifier='Linear(hidden,hidden,bias=True), GELU, Linear(hidden,2,bias=True)',
        classifier_objective='two_class_cross_entropy',
        reasoning_objective='causal_LM_all_nonpadding_tokens_including_prompt_and_answer',
        no_reason_policy='prompt+<ANS>+gold_label+EOS; no invented rationale',
        rationale_policy='same946 training IDs; new Gemini2.5Flash label-conditioned reasons without SHAP; format and gold-answer validation only',
        delimiters='ordinary tokenizer strings <REASON> and <ANS>; no vocabulary expansion',
        architecture_reference='2509.21487v1 Sections3.1-3.2',
        matched_overrides='main backbone, LoRA8/16/.05, lr1e-4, batch1*accum8, 5epochs, cosine10percentwarmup',
        hidden_width_activation='unspecified by reference; hidden_size and GELU chosen explicitly',
        historical_holdout_reuse=True)
    validate_config(output, config)
    atomic_json(output / 'base_model_manifest.json', manifest)
    if completed_run(output, data, smoke=args.max_steps is not None):
        print(f'Already complete with identical config: {output}', flush=True)
        return 0
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = encode_frame(train, tokenizer, reasons if args.method == 'dhrd_style' else {})
    val_rows = encode_frame(val, tokenizer)
    truncations = sum(row['rationale_truncated'] for row in train_rows)
    atomic_json(output / 'encoding_report.json', {'train_rows': len(train_rows), 'validation_rows': len(val_rows),
                'accepted_rationales': len(reasons), 'used_rationales': len(reasons) if args.method == 'dhrd_style' else 0,
                'rationales_truncated': truncations,
                'lm_supervised_rows': len(train_rows),
                'max_sequence_tokens': max(len(row['input_ids']) for row in train_rows),
                'max_prompt_tokens': max(row['cls_token_index']+1 for row in train_rows+val_rows)})
    print(f'{args.method}: {len(train_rows)} train, {len(val_rows)} val; rationale truncations={truncations}', flush=True)
    model = DualHeadQwen.from_pretrained(args.model_dir, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa', low_cpu_mem_usage=True)
    # Explicit seeded CPU materialization gives both methods an identical head,
    # including when low-memory loading left that checkpoint-absent head on meta.
    initial_head = initialize_classifier(model, args.seed)
    atomic_json(output / 'initialization.json', {'classification_head_sha256':
                hashlib.sha256(initial_head.numpy().tobytes()).hexdigest()})
    set_seed(args.seed)
    model = get_peft_model(model, lora_config()).to(device)
    if args.method == 'dhrd_style':
        # Zero scalars consume no RNG and remain FP32 despite BF16 backbone.
        model.add_module('loss_weighting', LossWeighting(args.weighting).to(device))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
    per_epoch = math.ceil(len(train_rows) / 8)
    total_steps = per_epoch * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(total_steps * 0.1), total_steps)
    collator = RightPadCollator(tokenizer.pad_token_id)
    resume_path = output / 'resume.pt'
    state = {'epoch': 0, 'offset': 0, 'step': 0, 'best_score': -1.0,
             'best_epoch': None, 'best_step': None, 'best_state': None, 'history': []}
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        restore_trainable(model, checkpoint['trainable'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        for key in state:
            state[key] = checkpoint[key]
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        del checkpoint
        print(f'Resumed epoch={state["epoch"]}, sample offset={state["offset"]}, step={state["step"]}', flush=True)
    started = time.time()
    try:
        for epoch in range(state['epoch'], args.epochs):
            order = epoch_order(len(train_rows), args.seed, epoch)
            offset = state['offset'] if epoch == state['epoch'] else 0
            model.train()
            while offset < len(order) and state['step'] < total_steps:
                group = order[offset:offset+8]
                optimizer.zero_grad(set_to_none=True)
                loss_total = lm_total = cls_total = 0.0
                lm_supervised_rows = 0
                for index in group:
                    batch = {key: value.to(device) for key, value in collator([train_rows[index]]).items()}
                    # Base weights are BF16; PEFT may keep trainable weights FP32.
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        losses = model(**batch, compute_lm=args.method == 'dhrd_style')
                        has_rationale = bool(batch['labels'][:, 1:].ne(-100).any().item())
                        objective = (model.loss_weighting(losses['loss_cls'], losses['loss_lm'], has_rationale)
                                     if args.method == 'dhrd_style' else losses['loss_cls'])
                        loss = objective / len(group)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, sample={index}')
                    loss.backward()
                    loss_total += objective.detach().float().item()/len(group)
                    lm_supervised_rows += int(has_rationale)
                    lm_total += losses['loss_lm'].detach().float().item()/len(group)
                    cls_total += losses['loss_cls'].detach().float().item()/len(group)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                offset += len(group)
                state.update(epoch=epoch, offset=offset, step=state['step']+1)
                report = {'status': 'training', 'method': args.method, 'epoch': epoch+1,
                          'sample_offset': offset, 'step': state['step'], 'total_steps': total_steps,
                          'loss': loss_total, 'loss_lm': lm_total, 'loss_cls': cls_total,
                          'learning_rate': scheduler.get_last_lr()[0], 'elapsed_seconds': time.time()-started,
                          'best_validation_macro_f1': state['best_score']}
                if args.method == 'dhrd_style':
                    report.update(loss_weights=model.loss_weighting.snapshot(), lm_supervised_rows=lm_supervised_rows,
                                  loss_weights_timing='after_optimizer_step', loss_timing='before_optimizer_step')
                    with (output / 'weight_history.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write(json.dumps(report, allow_nan=False) + '\n')
                atomic_json(progress, report)
                if state['step'] % 10 == 0 or state['step'] == 1:
                    print(json.dumps(report), flush=True)
                if state['step'] % args.checkpoint_steps == 0:
                    save_resume(resume_path, model, optimizer, scheduler, state)
            # Validation alone selects checkpoints; a max-steps run validates its
            # final partial epoch and is explicitly marked as a smoke run.
            if offset > 0:
                probabilities = evaluate(model, val_rows, collator, device, args.eval_batch_size)
                metrics = classification_metrics(val[TARGET], probabilities)
                record = {'epoch': epoch+1, 'step': state['step'], 'partial_epoch': offset < len(order), **metrics}
                if args.method == 'dhrd_style':
                    record['loss_weights'] = model.loss_weighting.snapshot()
                state['history'].append(record)
                if metrics['macro_f1'] > state['best_score']:
                    state.update(best_score=metrics['macro_f1'], best_epoch=epoch+1,
                                 best_step=state['step'], best_state=trainable_state(model))
                atomic_json(output / 'validation_history.json', state['history'])
                print('Validation: ' + json.dumps(record), flush=True)
            state.update(epoch=epoch+1, offset=0)
            save_resume(resume_path, model, optimizer, scheduler, state)
            if state['step'] >= total_steps:
                break
        if state['best_state'] is None:
            raise RuntimeError('No validation-selected checkpoint exists')
        if args.method == 'dhrd_style':
            atomic_json(output / 'final_training_loss_weights.json', model.loss_weighting.snapshot())
        restore_trainable(model, state['best_state'])
        temp_model = output / 'model.tmp'
        temp_model.mkdir(exist_ok=True)
        model.save_pretrained(temp_model, safe_serialization=True, save_embedding_layers=False)
        tokenizer.save_pretrained(temp_model)
        if args.method == 'dhrd_style':
            atomic_json(temp_model / 'loss_weights.json', model.loss_weighting.snapshot())
        model_dir = output / 'model'
        if model_dir.exists():
            shutil.rmtree(model_dir)
        os.replace(temp_model, model_dir)
        metrics = finish_selected_run(root, args.method, model, tokenizer, collator,
                    device, args.eval_batch_size, state, args.seed, smoke=args.max_steps is not None,
                    development_ids=set(train.sample_id) | set(val.sample_id))
        resume_path.unlink(missing_ok=True)
        print(json.dumps(metrics, indent=2), flush=True)
    except BaseException as error:
        atomic_json(progress, {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                              'method': args.method, 'step': state['step'], 'error': repr(error),
                              'resume_checkpoint': str(resume_path) if resume_path.exists() else None})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
