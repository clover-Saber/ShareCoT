"""Genuine serialized-English TextCNN, Transformer and BERT; seed 44.

Five classical methods delegate to baselines_suite unchanged. Neural inputs use
the same fixed local BERT tokenizer, with no fitted numerical preprocessing.
No model selection or length selection reads test rows. Completed runs can be
resumed only when provenance and every persisted artifact match exactly.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines_suite as classical
from protocol import FEATURE_COLS, TARGET, sha256, write_json, configure_semantic_dictionary, semantic_label

METHODS = classical.METHODS
TEXT_METHODS = ('textcnn', 'transformer', 'bert')
THRESHOLD = .5
SERIALIZATION = 'english-nine-semantic-values-units-v2'
# Dataset: Chicago taxi fares/charges in USD, distances km/miles, dewpoint Celsius.
FIELD_SPEC = (
    ('fare', 'Fare', 'USD'), ('trip_miles', 'Trip distance', 'miles'),
    ('trip_seconds', 'Trip duration', 'seconds'),
    ('additional_charges', 'Additional charges', 'USD'),
    ('pickup_to_center_distance_km', 'Pickup distance to city center', 'km'),
    ('dropoff_to_center_distance_km', 'Dropoff distance to city center', 'km'),
    ('trip_start_timestamp', 'Trip start timestamp', ''),
    ('dropoff_poi_category', 'Dropoff POI category', ''),
    ('weather_dewpoint_2m', 'Weather dewpoint at 2 meters', 'degrees Celsius'),
)


def rows_to_texts(frame):
    """Allowlisted values with manuscript categorical decoding and explicit units.

    CSVs are read as strings so lexical precision and missing cells survive.
    Timestamp strings are kept in their original ISO-compatible date/time form.
    For perturbation DataFrames, numerical values use Python's full str precision.
    """
    missing = set(FEATURE_COLS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing allowed features: {sorted(missing)}')
    texts = []
    for row in frame[FEATURE_COLS].itertuples(index=False, name=None):
        lines = []
        for (field, label, unit), value in zip(FIELD_SPEC, row):
            absent = pd.isna(value) or (isinstance(value, str) and value == '')
            rendered = ('missing' if absent else semantic_label(field, value)
                        if field in ('trip_start_timestamp', 'dropoff_poi_category') else str(value))
            lines.append(f'{label}: {rendered}' + (f' {unit}' if unit and not absent else ''))
        texts.append('\n'.join(lines))
    return texts


def load_raw_splits(root, names=('train', 'val'), development=None):
    # Reuse validated binary labels / unique IDs / non-overlap checks, then retain
    # original CSV feature strings instead of numerical conversion for inputs.
    validated = classical.load_splits(root, names, development)
    result = {}
    for split in names:
        raw = pd.read_csv(Path(root) / 'data' / f'{split}.csv', dtype=str, keep_default_na=False)
        if raw.sample_id.tolist() != validated[split].sample_id.tolist():
            raise ValueError('Raw/validated sample IDs differ')
        raw[TARGET] = validated[split][TARGET]
        result[split] = raw
    return result


def choose_max_length(lengths, limit, requested=None):
    """Development-only length cap rounded up to 32; refuse lossy truncation."""
    if set(lengths) != {'train', 'val'} or any(not values for values in lengths.values()):
        raise ValueError('Length selection requires only nonempty train and val')
    maximum = max(max(values) for values in lengths.values())
    cap = min(int(limit), int(math.ceil(maximum / 32) * 32)) if requested is None else int(requested)
    if cap < maximum or cap > limit or cap < 4:
        raise ValueError(f'Length cap {cap} cannot retain all nine features (development max={maximum}, limit={limit})')
    return cap, {'selection_splits': ['train', 'val'], 'max_length': cap,
                 'rule': 'round development maximum up to 32, bounded by backbone limit; reject any truncation',
                 'requested': requested, 'backbone_limit': int(limit),
                 'splits': {s: length_summary(v, cap) for s, v in lengths.items()}}


def length_summary(lengths, cap):
    values = np.asarray(lengths, dtype=int)
    return {'n': len(lengths), 'min': int(values.min()), 'max': int(values.max()),
            'mean': float(values.mean()), 'p50': float(np.percentile(values, 50)),
            'p95': float(np.percentile(values, 95)), 'p99': float(np.percentile(values, 99)),
            'would_truncate': int((values > cap).sum()), 'actual_truncated': 0}


def token_lengths(tokenizer, texts):
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=True,
                                         truncation=False, padding=False)['input_ids']]


def make_text_network(config, backbone=None):
    import torch
    from torch import nn
    import torch.nn.functional as F

    class TextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'],
                                          padding_idx=config['pad_token_id'])
            self.kernels = config['kernels']
            self.convs = nn.ModuleList([nn.Conv1d(config['embedding_dim'], config['cnn_filters'], k)
                                       for k in self.kernels])
            self.dropout = nn.Dropout(config['dropout'])
            self.head = nn.Linear(config['cnn_filters'] * len(self.kernels), 1)

        def forward(self, input_ids, attention_mask, **unused):
            mask = attention_mask.bool()
            x = (self.embedding(input_ids) * mask.unsqueeze(-1)).transpose(1, 2)
            mask = mask.to(x.dtype)
            if x.shape[-1] < max(self.kernels):
                amount = max(self.kernels) - x.shape[-1]
                x, mask = F.pad(x, (0, amount)), F.pad(mask, (0, amount))
            pooled = []
            for k, conv in zip(self.kernels, self.convs):
                counts = F.conv1d(mask.unsqueeze(1), torch.ones(1, 1, k, device=x.device))
                required = mask.sum(1).clamp(max=k).view(-1, 1, 1)
                valid = (counts >= required) & (counts > 0)
                values = torch.relu(conv(x)).masked_fill(~valid, float('-inf')).amax(-1)
                pooled.append(torch.where(torch.isfinite(values), values, torch.zeros_like(values)))
            return self.head(self.dropout(torch.cat(pooled, 1))).squeeze(-1)

    class TextTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            dim = config['embedding_dim']
            self.embedding = nn.Embedding(config['vocab_size'], dim, padding_idx=config['pad_token_id'])
            positions = torch.arange(config['max_length']).float().unsqueeze(1)
            scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.) / dim))
            pe = torch.zeros(config['max_length'], dim)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(positions * scale), torch.cos(positions * scale)
            self.register_buffer('position_encoding', pe.unsqueeze(0))
            layer = nn.TransformerEncoderLayer(dim, config['heads'], config['feedforward_dim'],
                                               config['dropout'], activation='gelu', batch_first=True,
                                               norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, config['layers'], enable_nested_tensor=False)
            self.norm = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(config['dropout'])
            self.head = nn.Linear(dim, 1)

        def forward(self, input_ids, attention_mask, **unused):
            if input_ids.shape[1] > config['max_length']:
                raise ValueError('Input exceeds saved positional encoding capacity')
            mask = attention_mask.bool()
            if not mask[:, 0].all():
                raise ValueError('First token must be the valid BERT CLS token')
            x = self.embedding(input_ids) * math.sqrt(config['embedding_dim'])
            x = self.dropout(x + self.position_encoding[:, :x.shape[1]])
            x = self.encoder(x, src_key_padding_mask=~mask)
            return self.head(self.dropout(self.norm(x[:, 0]))).squeeze(-1)

    if config['method'] == 'textcnn':
        return TextCNN()
    if config['method'] == 'transformer':
        return TextTransformer()
    if config['method'] == 'bert':
        if backbone is None:
            raise ValueError('BERT needs a backbone')
        return classical.make_bert_network(backbone)
    raise ValueError(f'Not a text method: {config["method"]}')


def save_checkpoint(path, model, config, epoch, seed):
    import torch
    path = Path(path)
    temp = path.with_suffix('.pt.tmp')
    torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'config': config, 'epoch': epoch, 'seed': seed,
                'serialization': SERIALIZATION}, temp)
    temp.replace(path)


def load_network(out, device='cpu'):
    """Reconstruct from saved configuration; BERT does not reread source weights."""
    import torch
    out = Path(out)
    checkpoint = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    if checkpoint.get('serialization') != SERIALIZATION:
        raise ValueError('Incompatible text serialization checkpoint')
    config = checkpoint['config']
    backbone = None
    if config['method'] == 'bert':
        from transformers import AutoConfig, AutoModel
        backbone = AutoModel.from_config(AutoConfig.from_pretrained(out / 'bert_config', local_files_only=True))
    net = make_text_network(config, backbone)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    return net.to(device).eval(), checkpoint


class TextPredictor:
    """Callable raw DataFrame -> P(y=1); also exposes model/tokenizer for timing."""
    def __init__(self, model, tokenizer, config, device='cpu', batch_size=8):
        self.model, self.net, self.tokenizer = model, model, tokenizer
        self.config, self.device, self.batch_size = config, device, batch_size

    def encode(self, texts):
        lengths = token_lengths(self.tokenizer, texts)
        if lengths and max(lengths) > self.config['max_length']:
            raise ValueError(f'Input length {max(lengths)} exceeds saved cap {self.config["max_length"]}; truncation is forbidden')
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        return {k: v.to(self.device) for k, v in encoded.items()}

    def predict_texts(self, texts):
        import torch
        self.model.eval()
        result = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                logits = self.model(**self.encode(texts[start:start + self.batch_size]))
                p = torch.softmax(logits, 1)[:, 1] if self.config['method'] == 'bert' else torch.sigmoid(logits)
                result.append(p.cpu().numpy())
        return np.concatenate(result).astype(float) if result else np.array([], dtype=float)

    def __call__(self, frame):
        return self.predict_texts(rows_to_texts(frame))


def load_model(out, device='cpu', batch_size=None):
    """Load neural or classical model for perturbation/efficiency evaluation.

    Neural callers must supply raw timestamp/value features, without prepare_features.
    Source BERT weights need not be available: config + state_dict + saved tokenizer
    suffice. Classical callables perform prepare_features internally.
    """
    out = Path(out)
    if (out / 'model.pt').exists():
        from transformers import AutoTokenizer
        net, checkpoint = load_network(out, device)
        tokenizer = AutoTokenizer.from_pretrained(out / 'tokenizer', local_files_only=True)
        cfg = checkpoint['config']
        return TextPredictor(net, tokenizer, cfg, device,
                             batch_size if batch_size is not None else cfg['hyperparameters']['batch_size'])
    import joblib
    prep, model = joblib.load(out / 'preprocessor.joblib'), joblib.load(out / 'model.joblib')
    return lambda frame: model.predict_proba(prep.transform(classical.prepare_features(frame)))[:, 1]


def neural_defaults(method):
    if method == 'bert':
        return {'epochs': 8, 'batch_size': 8, 'learning_rate': 2e-5, 'weight_decay': .01,
                'patience': 3, 'warmup_ratio': .06, 'max_grad_norm': 1.}
    return {'epochs': 60, 'batch_size': 64, 'learning_rate': .001, 'weight_decay': .0001,
            'patience': 12, 'max_grad_norm': 1.}


def fit_text(method, train, val, tokenizer, cap, out, seed, device, bert_model, options):
    import torch
    from torch import nn
    classical.seed_everything(seed, neural=True)
    hp = neural_defaults(method)
    hp.update({k: v for k, v in options.items() if k != 'max_length'})
    config = {'method': method, 'vocab_size': len(tokenizer), 'pad_token_id': tokenizer.pad_token_id,
              'max_length': cap, 'embedding_dim': 128, 'cnn_filters': 100,
              'kernels': [3, 4, 5], 'heads': 4, 'layers': 2, 'feedforward_dim': 256,
              'dropout': .2 if method == 'textcnn' else .1, 'hyperparameters': hp,
              'embedding_initialization': 'pretrained BERT' if method == 'bert' else 'from scratch',
              'serialization': SERIALIZATION}
    backbone = None
    if method == 'bert':
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(str(bert_model), local_files_only=True)
        backbone.config.save_pretrained(out / 'bert_config')
    net = make_text_network(config, backbone).to(device)
    predictor = TextPredictor(net, tokenizer, config, device, hp['batch_size'])
    tokenizer.save_pretrained(out / 'tokenizer')
    write_json(out / 'model_config.json', config)
    texts = {s: rows_to_texts(df) for s, df in [('train', train), ('val', val)]}
    y = train[TARGET].to_numpy(dtype=int)
    yval = val[TARGET].to_numpy(dtype=int)
    ratio = float((y == 0).sum() / (y == 1).sum())
    criterion = (nn.CrossEntropyLoss(weight=torch.tensor([1., ratio], device=device)) if method == 'bert'
                 else nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio], device=device)))
    opt = torch.optim.AdamW(net.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
    scheduler = None
    if method == 'bert':
        steps = math.ceil(len(y) / hp['batch_size']) * hp['epochs']
        warmup = max(1, int(steps * hp['warmup_ratio']))
        def lr_factor(step):
            return step / warmup if step < warmup else max(0., .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    rng = np.random.RandomState(seed)
    history, best, bad, best_epoch = [], -1., 0, None
    for epoch in range(1, hp['epochs'] + 1):
        net.train()
        losses = []
        order = rng.permutation(len(y))
        for start in range(0, len(y), hp['batch_size']):
            idx = order[start:start + hp['batch_size']]
            opt.zero_grad(set_to_none=True)
            logits = net(**predictor.encode([texts['train'][int(i)] for i in idx]))
            labels = torch.as_tensor(y[idx], device=device)
            loss = criterion(logits, labels.long() if method == 'bert' else labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp['max_grad_norm'])
            opt.step()
            if scheduler is not None:
                scheduler.step()
            losses.append((float(loss.detach()), len(idx)))
        metrics = classical.compute_metrics(yval, predictor.predict_texts(texts['val']))
        history.append({'epoch': epoch, 'train_loss': sum(v * n for v, n in losses) / len(y), 'val': metrics})
        if metrics['macro_f1'] > best:
            best, best_epoch, bad = metrics['macro_f1'], epoch, 0
            save_checkpoint(out / 'model.pt', net, config, epoch, seed)
        else:
            bad += 1
        write_json(out / 'validation_search/history.json', {'criterion': 'validation_macro_f1',
                   'threshold': THRESHOLD, 'tie_break': 'earliest_epoch', 'trials': history,
                   'best_epoch': best_epoch})
        print(f'{method} epoch={epoch} validation MacroF1={metrics["macro_f1"]:.6f}', flush=True)
        if bad >= hp['patience']:
            break
    return load_model(out, device), {'model_class': type(net).__name__, 'best_epoch': best_epoch,
                                    'config': config, 'hyperparameters': hp,
                                    'positive_class_training_weight': ratio,
                                    'input_format': SERIALIZATION}


def verify_resume(out, identity):
    # The existing verifier recomputes stored metrics and verifies ALL artifacts.
    # Interrupted training restarts deterministically; never mixes a partial epoch.
    out = Path(out)
    if not (out / 'run_config.json').is_file() and out.exists() and any(
            path.is_file() for path in out.rglob('*')):
        raise ValueError('Existing experiment artifacts have missing run_config.json; refusing overwrite')
    return classical.verify_resume(out, identity)


def _persist_inputs(out, split, frame, texts):
    path = out / 'inputs' / f'{split}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.jsonl.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        for sample_id, text in zip(frame.sample_id, texts):
            handle.write(json.dumps({'sample_id': sample_id, 'serialization': SERIALIZATION,
                                     'text': text}, ensure_ascii=False) + '\n')
    temp.replace(path)


def run(root, method, seed=44, device='cpu', bert_model='models/bert-base-uncased',
        neural_options=None):
    if method not in METHODS:
        raise ValueError(f'Unsupported method: {method}')
    options = neural_options or {}
    if method not in TEXT_METHODS:
        if options:
            raise ValueError('Neural options are invalid for classical methods')
        return classical.run(root, method, seed, device, bert_model)
    if set(options) - {'epochs', 'batch_size', 'max_length'} or any(v < 1 for v in options.values()):
        raise ValueError('Only positive epochs, batch_size and max_length overrides are supported')
    import torch
    from transformers import AutoConfig, AutoTokenizer
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'Requested {device} but CUDA unavailable')
    root, bert_model = Path(root), Path(bert_model)
    serialization_metadata = configure_semantic_dictionary(root / 'data')
    splits = load_raw_splits(root)
    tokenizer = AutoTokenizer.from_pretrained(str(bert_model), local_files_only=True)
    limit = min(int(AutoConfig.from_pretrained(str(bert_model), local_files_only=True).max_position_embeddings),
                int(tokenizer.model_max_length))
    texts = {s: rows_to_texts(df) for s, df in splits.items()}
    lengths = {s: token_lengths(tokenizer, value) for s, value in texts.items()}
    cap, audit = choose_max_length(lengths, limit, options.get('max_length'))
    source_files = sorted(p for p in bert_model.iterdir() if p.is_file() and
                          p.suffix in ('.json', '.txt', '.bin', '.safetensors', '.model'))
    identity = {'schema_version': 2, 'runner': 'baselines_text_suite', 'method': method, 'seed': seed,
                'device': device, 'threshold': THRESHOLD, 'selection': 'validation_macro_f1',
                'features': FEATURE_COLS, 'label': TARGET, 'serialization': SERIALIZATION,
                'field_spec': [list(f) for f in FIELD_SPEC], 'preprocessing_fit_split': None,
                'fixed_vocabulary_source': str(bert_model), 'bert_model': str(bert_model),
                'source_sha256': {p.name: sha256(p) for p in source_files},
                'code_sha256': sha256(Path(__file__)),
                'classical_runner_sha256': sha256(Path(classical.__file__)),
                'protocol_sha256': sha256(Path(__file__).with_name('protocol.py')),
                **serialization_metadata,
                'split_sha256': {s: sha256(root / 'data' / f'{s}.csv') for s in ('train', 'val', 'test')},
                'split_rows': {s: len(df) for s, df in splits.items()},
                'split_sample_ids_sha256': {s: hashlib.sha256(json.dumps(df.sample_id.tolist(), ensure_ascii=False).encode()).hexdigest()
                                            for s, df in splits.items()},
                'input_text_sha256': {s: hashlib.sha256(json.dumps(v, ensure_ascii=False).encode()).hexdigest() for s, v in texts.items()},
                'length_selection': copy.deepcopy(audit), 'neural_options': options,
                'hyperparameters': neural_defaults(method) | {k: v for k, v in options.items() if k != 'max_length'},
                'versions': {n: importlib.metadata.version(n) for n in
                             ('numpy', 'pandas', 'scikit-learn', 'torch', 'transformers')}}
    out = root / 'results' / method
    resumed = verify_resume(out, identity)
    if resumed:
        print(f'{method}: verified completed text artifacts; skipping retraining', flush=True)
        return resumed
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'run_config.json', identity)
    write_json(out / 'token_length_audit.json', audit)
    write_json(out / 'preprocessor.json', {'serialization': SERIALIZATION, 'features': FEATURE_COLS,
               'field_spec': [list(f) for f in FIELD_SPEC], 'fit_split': None,
               'max_length': cap, 'truncation': 'forbidden', 'vocabulary': 'fixed local bert-base-uncased'})
    for split in ('train', 'val'):
        _persist_inputs(out, split, splits[split], texts[split])
    start = time.monotonic()
    predict, info = fit_text(method, splits['train'], splits['val'], tokenizer, cap, out,
                             seed, device, bert_model, options)
    # A selected checkpoint now exists; only here do held-out rows enter memory.
    splits.update(load_raw_splits(root, names=('test',), development=splits))
    test_texts = rows_to_texts(splits['test'])
    test_lengths = token_lengths(tokenizer, test_texts)
    audit['splits']['test'] = length_summary(test_lengths, cap)
    write_json(out / 'token_length_audit.json', audit)
    _persist_inputs(out, 'test', splits['test'], test_texts)
    result = {'status': 'complete', 'method': method, 'seed': seed, 'threshold': THRESHOLD,
              'provenance': identity, 'model': info, 'test_rows': len(splits['test']),
              'test_sample_ids_sha256': hashlib.sha256(json.dumps(splits['test'].sample_id.tolist(), ensure_ascii=False).encode()).hexdigest(),
              'test_input_text_sha256': hashlib.sha256(json.dumps(test_texts, ensure_ascii=False).encode()).hexdigest()}
    for split, filename in [('val', 'validation_predictions.csv'), ('test', 'predictions.csv')]:
        frame = splits[split]
        probabilities = predict(frame)
        result[split] = classical.compute_metrics(frame[TARGET], probabilities)
        pd.DataFrame({'sample_id': frame.sample_id, 'y_true': frame[TARGET],
                      'y_pred': (probabilities >= THRESHOLD).astype(int), 'prob_1': probabilities}).to_csv(
                          out / filename, index=False, float_format='%.17g')
    result['elapsed_seconds'] = time.monotonic() - start
    result['artifact_sha256'] = {str(p.relative_to(out)).replace('\\', '/'): sha256(p)
                                for p in sorted(out.rglob('*')) if p.is_file() and
                                p.name not in ('metrics.json', 'run_config.json')}
    write_json(out / 'metrics.json', result)
    print(f'{method} complete: test MacroF1={result["test"]["macro_f1"]:.6f}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--bert-model', default='models/bert-base-uncased')
    for name in ('epochs', 'batch-size', 'max-length'):
        parser.add_argument('--' + name, type=int, help='Explicit override recorded in provenance')
    args = parser.parse_args()
    options = {k: getattr(args, k) for k in ('epochs', 'batch_size', 'max_length') if getattr(args, k) is not None}
    run(args.root, args.method, args.seed, args.device, args.bert_model, options)


if __name__ == '__main__':
    main()
