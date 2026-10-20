"""Sequential single-method no-tip baseline runner, seed 44 by default.

TextCNN/Transformer retain the reference tabular feature-token architectures.
BERT retains the reference column/value sentence serialization and pooled head.
Two misleading reference identities are corrected: LogitBoost uses actual IRLS
regression-tree boosting; Gradient Boosting uses GradientBoostingClassifier.
All preprocessing fits train only, all selection uses validation Macro-F1 at .5.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, classification_report, confusion_matrix,
                             f1_score, log_loss, matthews_corrcoef, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from protocol import FEATURE_COLS, TARGET, sha256, write_json

METHODS = ('logistic_regression', 'random_forest', 'xgboost', 'logit_boost',
           'gradient_boosting', 'textcnn', 'transformer', 'bert')
NUMERIC_COLS = [c for c in FEATURE_COLS if c != 'dropoff_poi_category']
CAT_COLS = ['dropoff_poi_category']
THRESHOLD = .5


def prepare_features(df):
    """Explicit allowlist prevents extra CSV fields becoming model inputs."""
    missing = set(FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing allowed features: {sorted(missing)}')
    x = df.loc[:, FEATURE_COLS].copy()
    ts = pd.to_datetime(x['trip_start_timestamp'], errors='coerce', utc=True)
    x['trip_start_timestamp'] = ts.astype('int64') / 1e9
    x.loc[ts.isna(), 'trip_start_timestamp'] = np.nan
    for col in NUMERIC_COLS:
        x[col] = pd.to_numeric(x[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    x['dropoff_poi_category'] = x['dropoff_poi_category'].map(
        lambda v: str(v) if pd.notna(v) else np.nan)
    return x


def load_splits(root, names=('train', 'val'), development=None):
    """Development splits by default; test parsing is explicitly deferred."""
    splits, seen = {}, set()
    for df in (development or {}).values():
        seen.update(df.sample_id)
    for name in names:
        df = pd.read_csv(Path(root) / 'data' / f'{name}.csv', dtype={'sample_id': str})
        prepare_features(df)
        if 'sample_id' not in df or df.sample_id.isna().any() or df.sample_id.duplicated().any():
            raise ValueError(f'{name} must have unique non-null sample_id')
        ids = set(df.sample_id)
        if seen.intersection(ids):
            raise ValueError('Split sample_id overlap is forbidden')
        seen.update(ids)
        if TARGET not in df:
            raise ValueError(f'Missing label column {TARGET}')
        labels = pd.to_numeric(df[TARGET], errors='coerce')
        if labels.isna().any() or not set(labels.unique()).issubset({0, 1}):
            raise ValueError(f'{name} labels must be binary 0/1')
        df[TARGET] = labels.astype(int)
        if not len(df) or set(df[TARGET]) != {0, 1}:
            raise ValueError(f'{name} must contain both classes for manuscript evaluation')
        splits[name] = df
    return splits


def make_preprocessor(method):
    num = [('imputer', SimpleImputer(strategy='median'))]
    if method == 'logistic_regression':
        num.append(('scaler', StandardScaler(with_mean=False)))
    try:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
    return ColumnTransformer([
        ('num', Pipeline(num), NUMERIC_COLS),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', onehot)]), CAT_COLS)], remainder='drop')


class LogitBoostClassifier(ClassifierMixin, BaseEstimator):
    """Binary additive logistic regression via bounded Newton pseudo-responses.

    Friedman/Hastie/Tibshirani convention: p = sigmoid(2 F), regression weights
    p(1-p), working response (y-p)/[p(1-p)], F += learning_rate * tree/2.
    Clipping the response and positive weight floor stabilize nearly pure nodes.
    """
    def __init__(self, n_estimators=200, learning_rate=.08, max_depth=3,
                 min_samples_leaf=20, random_state=44, response_bound=4.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.response_bound = response_bound

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('LogitBoost requires both binary classes')
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = x.shape[1]
        self.algorithm_ = 'binary_logitboost_irls'
        self.estimators_ = []
        score = np.zeros(len(y))
        rng = np.random.RandomState(self.random_state)
        for _ in range(self.n_estimators):
            p = np.clip(expit(2 * score), 1e-8, 1 - 1e-8)
            weights = np.maximum(p * (1 - p), 1e-8)
            response = np.clip((y - p) / weights, -self.response_bound, self.response_bound)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         random_state=int(rng.randint(0, 2**31 - 1)))
            tree.fit(x, response, sample_weight=weights)
            score += .5 * self.learning_rate * tree.predict(x)
            self.estimators_.append(tree)
        return self

    def predict_proba(self, x):
        score = np.zeros(len(x))
        for tree in self.estimators_:
            score += .5 * self.learning_rate * tree.predict(x)
        p = np.clip(expit(2 * score), 1e-12, 1 - 1e-12)
        return np.column_stack([1 - p, p])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= THRESHOLD).astype(int)


def make_estimator(method, params, seed, y):
    if method == 'logistic_regression':
        base = dict(C=1., max_iter=2000, solver='lbfgs', class_weight='balanced')
        cls = LogisticRegression
    elif method == 'random_forest':
        base = dict(n_estimators=500, max_depth=None, class_weight='balanced', n_jobs=-1)
        cls = RandomForestClassifier
    elif method == 'xgboost':
        from xgboost import XGBClassifier
        base = dict(n_estimators=800, learning_rate=.05, max_depth=6, subsample=.9,
                    colsample_bytree=.9, reg_lambda=1., objective='binary:logistic',
                    eval_metric='logloss', tree_method='hist', n_jobs=4,
                    scale_pos_weight=float(np.sum(y == 0) / max(np.sum(y == 1), 1)))
        cls = XGBClassifier
    elif method == 'logit_boost':
        base = dict(n_estimators=200, learning_rate=.08, max_depth=3, min_samples_leaf=20)
        cls = LogitBoostClassifier
    elif method == 'gradient_boosting':
        base = dict(n_estimators=300, learning_rate=.08, max_depth=6,
                    min_samples_leaf=20, loss='log_loss', n_iter_no_change=None)
        cls = GradientBoostingClassifier
    else:
        raise ValueError(f'Not a classical method: {method}')
    base.update(params)
    return cls(random_state=seed, **base)


def search_grid(method):
    # Prespecified modest grids include the reference settings. Tie: first trial.
    if method == 'logistic_regression':
        return [{'C': c} for c in (1., .1, 10.)]
    if method == 'random_forest':
        return [{'n_estimators': 500, 'max_depth': d} for d in (None, 16, 32)]
    if method == 'xgboost':
        return [{'n_estimators': n, 'max_depth': d} for n, d in ((800, 6), (400, 4), (800, 4))]
    if method == 'logit_boost':
        return [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr}
                for n, d, lr in ((200, 3, .08), (400, 3, .05), (200, 2, .08))]
    if method == 'gradient_boosting':
        return [{'n_estimators': n, 'max_depth': d, 'learning_rate': lr}
                for n, d, lr in ((300, 6, .08), (200, 3, .08), (300, 3, .05))]
    return []


def compute_metrics(y, p):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability predictions')
    pred = (p >= THRESHOLD).astype(int)
    report = classification_report(y, pred, labels=[0, 1], output_dict=True, zero_division=0)
    both = len(np.unique(y)) == 2
    return {'n': int(len(y)), 'threshold': THRESHOLD,
            'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_precision': float(report['macro avg']['precision']),
            'macro_recall': float(report['macro avg']['recall']),
            'macro_f1': float(report['macro avg']['f1-score']),
            'weighted_f1': float(report['weighted avg']['f1-score']),
            'f1': float(report['1']['f1-score']),
            'auc_roc': float(roc_auc_score(y, p)) if both else None,
            'auc_pr': float(average_precision_score(y, p)) if both else None,
            'mcc': float(matthews_corrcoef(y, pred)),
            'brier_score': float(brier_score_loss(y, p)),
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'per_class': {k: {'precision': float(report[k]['precision']),
                              'recall': float(report[k]['recall']),
                              'f1': float(report[k]['f1-score']),
                              'support': int(report[k]['support'])} for k in ('0', '1')},
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            'classification_report': report}


def fit_tabular_prep(x):
    nums = x[NUMERIC_COLS]
    medians = nums.median().fillna(0.)
    scaler = StandardScaler().fit(nums.fillna(medians).to_numpy(dtype=float))
    maps = [{v: i for i, v in enumerate(sorted(x[c].fillna('__nan__').unique()))} for c in CAT_COLS]
    return {'numeric_cols': NUMERIC_COLS, 'categorical_cols': CAT_COLS,
            'medians': medians.to_dict(), 'scaler': scaler,
            'category_maps': maps, 'vocab_sizes': [len(m) + 1 for m in maps],
            'fit_split': 'train'}


def transform_tabular(x, prep):
    num = prep['scaler'].transform(x[NUMERIC_COLS].fillna(prep['medians']).to_numpy(dtype=float))
    cat = np.column_stack([[m.get(v, len(m)) for v in x[c].fillna('__nan__')]
                           for c, m in zip(CAT_COLS, prep['category_maps'])])
    return num.astype(np.float32), cat.astype(np.int64)


def make_tabular_network(method, n_num, vocab_sizes):
    # Lazy torch import keeps CPU sklearn methods independent of torch.
    import torch
    from torch import nn

    class TabTextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_projs = nn.ModuleList([nn.Linear(1, 64) for _ in range(n_num)])
            self.cat_embs = nn.ModuleList([nn.Embedding(n, 64) for n in vocab_sizes])
            self.convs = nn.ModuleList([nn.Conv1d(64, 100, k) for k in (2, 3, 4)])
            self.dropout = nn.Dropout(.2)
            self.fc = nn.Linear(300, 1)

        def forward(self, xn, xc):
            parts = [p(xn[:, i:i+1]).unsqueeze(1) for i, p in enumerate(self.num_projs)]
            parts += [e(xc[:, j]).unsqueeze(1) for j, e in enumerate(self.cat_embs)]
            seq = torch.cat(parts, dim=1).transpose(1, 2)
            h = torch.cat([torch.relu(c(seq)).max(dim=2).values for c in self.convs], dim=1)
            return self.fc(self.dropout(h)).squeeze(-1)

    class TabTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_projs = nn.ModuleList([nn.Linear(1, 64) for _ in range(n_num)])
            self.cat_embs = nn.ModuleList([nn.Embedding(n, 64) for n in vocab_sizes])
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 64))
            self.col_pos = nn.Parameter(torch.zeros(1, n_num + len(vocab_sizes) + 1, 64))
            layer = nn.TransformerEncoderLayer(64, 4, 128, .1, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(64)
            self.head = nn.Linear(64, 1)
            nn.init.trunc_normal_(self.cls_token, std=.02)
            nn.init.trunc_normal_(self.col_pos, std=.02)

        def forward(self, xn, xc):
            parts = [p(xn[:, i:i+1]).unsqueeze(1) for i, p in enumerate(self.num_projs)]
            parts += [e(xc[:, j]).unsqueeze(1) for j, e in enumerate(self.cat_embs)]
            seq = torch.cat([self.cls_token.expand(len(xn), -1, -1), *parts], dim=1)
            return self.head(self.norm(self.encoder(seq + self.col_pos)[:, 0])).squeeze(-1)

    return TabTextCNN() if method == 'textcnn' else TabTransformer()


def rows_to_texts(x):
    def cell(c, v):
        if pd.isna(v):
            return f'{c} is unknown'
        if isinstance(v, (float, np.floating)):
            return f'{c} is {float(v):.6g}'
        return f'{c} is {str(v).strip()}'
    return ['. '.join(cell(c, v) for c, v in zip(FEATURE_COLS, row)) + '.'
            for row in x[FEATURE_COLS].itertuples(index=False, name=None)]


def make_bert_network(backbone):
    from torch import nn
    class BertTabularClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = backbone
            self.dropout = nn.Dropout(.1)
            self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

        def forward(self, **kwargs):
            output = self.bert(**kwargs)
            if output.pooler_output is None:
                raise ValueError('BERT pooler_output is required for the specified pooled-head architecture')
            return self.classifier(self.dropout(output.pooler_output))
    return BertTabularClassifier()


def seed_everything(seed, neural=False):
    random.seed(seed)
    np.random.seed(seed)
    if neural:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


def fit_classical(method, xtrain, ytrain, xval, yval, seed, out):
    prep = make_preprocessor(method)
    xt = prep.fit_transform(xtrain)
    xv = prep.transform(xval)
    best, best_f1, trials = None, -1., []
    for idx, params in enumerate(search_grid(method)):
        model = make_estimator(method, params, seed, ytrain)
        model.fit(xt, ytrain)
        m = compute_metrics(yval, model.predict_proba(xv)[:, 1])
        trials.append({'trial': idx, 'params': params, 'val': m})
        if m['macro_f1'] > best_f1:
            best, best_f1, best_idx = model, m['macro_f1'], idx
        write_json(out / 'validation_search/history.json',
                   {'criterion': 'validation_macro_f1', 'threshold': THRESHOLD,
                    'tie_break': 'first_trial', 'trials': trials, 'best_trial': best_idx})
        print(f'{method} trial={idx} validation MacroF1={m["macro_f1"]:.6f}', flush=True)
    joblib.dump(prep, out / 'preprocessor.joblib')
    joblib.dump(best, out / 'model.joblib')
    return lambda x: best.predict_proba(prep.transform(x))[:, 1], {
        'model_class': type(best).__name__, 'selected_params': trials[best_idx]['params'],
        'best_trial': best_idx, 'input_format': 'numeric tabular fields + one-hot POI category'}


def neural_defaults(method):
    if method == 'bert':
        return {'epochs': 8, 'batch_size': 8, 'learning_rate': 2e-5,
                'weight_decay': .01, 'patience': 3, 'max_length': 256,
                'warmup_ratio': .06, 'max_grad_norm': 1.}
    return {'epochs': 60, 'batch_size': 256, 'learning_rate': .001,
            'weight_decay': .0001, 'patience': 12, 'max_grad_norm': 1.}


def fit_neural(method, xtrain, ytrain, xval, yval, seed, device, bert_model, out, options):
    import torch
    from torch import nn
    seed_everything(seed, neural=True)
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'Requested {device} but CUDA unavailable')
    hp = neural_defaults(method)
    hp.update(options)
    batch_size = hp['batch_size']
    bert = method == 'bert'
    if bert:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(bert_model), local_files_only=True)
        backbone = AutoModel.from_pretrained(str(bert_model), local_files_only=True)
        net = make_bert_network(backbone).to(dev)
        train_data, val_data = rows_to_texts(xtrain), rows_to_texts(xval)
        tokenizer.save_pretrained(out / 'tokenizer')
        net.bert.config.save_pretrained(out / 'bert_config')
        write_json(out / 'preprocessor.json', {'fit_split': None, 'features': FEATURE_COLS,
                   'kind': 'fixed column/value sentence serialization; epoch timestamp; no fitted statistics',
                   'max_length': hp['max_length'], 'local_pretrained_tokenizer': str(bert_model)})
    else:
        prep = fit_tabular_prep(xtrain)
        train_data, val_data = transform_tabular(xtrain, prep), transform_tabular(xval, prep)
        net = make_tabular_network(method, len(NUMERIC_COLS), prep['vocab_sizes']).to(dev)
        joblib.dump(prep, out / 'preprocessor.joblib')

    def forward(data, idx):
        if bert:
            texts = [data[int(i)] for i in idx]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=hp['max_length'], return_tensors='pt')
            return net(**{k: v.to(dev) for k, v in enc.items()})
        return net(torch.from_numpy(data[0][idx]).to(dev), torch.from_numpy(data[1][idx]).to(dev))

    def probabilities(data, n):
        net.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                logits = forward(data, np.arange(start, min(start + batch_size, n)))
                p = torch.softmax(logits, dim=1)[:, 1] if bert else torch.sigmoid(logits)
                outputs.append(p.cpu().numpy())
        return np.concatenate(outputs).astype(float)

    ratio = float(np.sum(ytrain == 0) / max(np.sum(ytrain == 1), 1))
    if bert:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., ratio], device=dev))
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio], device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=hp['learning_rate'], weight_decay=hp['weight_decay'])
    scheduler = None
    if bert:
        total_steps = int(np.ceil(len(ytrain) / batch_size)) * hp['epochs']
        warmup = max(1, int(total_steps * hp['warmup_ratio']))
        def lr_factor(step):
            if step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return max(0., .5 * (1. + np.cos(np.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    rng = np.random.RandomState(seed)
    history, best_f1, bad = [], -1., 0
    for epoch in range(1, hp['epochs'] + 1):
        net.train()
        perm = rng.permutation(len(ytrain))
        losses = []
        for start in range(0, len(ytrain), batch_size):
            idx = perm[start:start + batch_size]
            labels = torch.from_numpy(ytrain[idx]).to(dev)
            opt.zero_grad(set_to_none=True)
            logits = forward(train_data, idx)
            loss = criterion(logits, labels.long() if bert else labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp['max_grad_norm'])
            opt.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(float(loss.detach().item()))
        m = compute_metrics(yval, probabilities(val_data, len(yval)))
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'val': m})
        if m['macro_f1'] > best_f1:
            best_f1, best_epoch, bad = m['macro_f1'], epoch, 0
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in net.state_dict().items()},
                        'method': method, 'seed': seed, 'epoch': epoch, 'hyperparameters': hp,
                        'features': FEATURE_COLS, 'vocab_sizes': None if bert else prep['vocab_sizes']},
                       out / 'model.pt')
        else:
            bad += 1
        write_json(out / 'validation_search/history.json', {
            'criterion': 'validation_macro_f1', 'threshold': THRESHOLD,
            'tie_break': 'earliest_epoch', 'trials': history, 'best_epoch': best_epoch})
        print(f'{method} epoch={epoch} validation MacroF1={m["macro_f1"]:.6f}', flush=True)
        if bad >= hp['patience']:
            break
    checkpoint = torch.load(out / 'model.pt', map_location=dev, weights_only=True)
    net.load_state_dict(checkpoint['state_dict'])
    def predict(x):
        data = rows_to_texts(x) if bert else transform_tabular(x, prep)
        return probabilities(data, len(x))
    return predict, {'model_class': type(net).__name__, 'best_epoch': best_epoch,
                     'hyperparameters': hp, 'positive_class_training_weight': ratio,
                     'input_format': ('serialized English column-name/value sentences, local bert-base-uncased'
                                      if bert else 'tabular feature tokens: numeric Linear(1,64), categorical Embedding; numeric tokens precede categorical tokens'),
                     'architecture': ('AutoModel pooled output + Dropout(.1) + Linear(hidden,2)' if bert else
                         '64-dim; Conv1d(100 filters, kernels 2/3/4), ReLU, global maxpool, Dropout(.2), Linear(300,1)'
                         if method == 'textcnn' else
                         '64-dim; learned CLS/column positions; 2 TransformerEncoder layers, 4 heads, FF128, GELU, dropout .1, norm_first; LayerNorm + Linear(64,1)')}


def provenance(root, method, seed, device, bert_model, options):
    splits = {s: sha256(Path(root) / 'data' / f'{s}.csv') for s in ('train', 'val', 'test')}
    versions = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib',
                 *(['torch'] if method in ('textcnn', 'transformer', 'bert') else []),
                 *(['transformers'] if method == 'bert' else []),
                 *(['xgboost'] if method == 'xgboost' else [])):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not-installed'
    bert_hashes = None
    if method == 'bert':
        path = Path(bert_model)
        if not path.is_dir():
            raise FileNotFoundError(f'Local BERT model directory missing: {path}')
        files = sorted(p for p in path.iterdir() if p.is_file() and
                       p.suffix in ('.json', '.txt', '.bin', '.safetensors', '.model'))
        bert_hashes = {p.name: sha256(p) for p in files}
    return {'schema_version': 1, 'method': method, 'seed': int(seed), 'threshold': THRESHOLD,
            'features': FEATURE_COLS, 'label': TARGET, 'selection': 'validation_macro_f1',
            'preprocessing_fit_split': 'train', 'device': device,
            'code_sha256': sha256(Path(__file__)),
            'protocol_sha256': sha256(Path(__file__).with_name('protocol.py')),
            'split_sha256': splits, 'versions': versions,
            'search_grid': search_grid(method), 'neural_options': options,
            'bert_model': str(bert_model) if method == 'bert' else None,
            'bert_model_sha256': bert_hashes,
            'reference_discrepancy': {
                'logit_boost': 'Original 3.4 used GradientBoostingClassifier(log_loss), not LogitBoost; this runner corrects to bounded-response binary LogitBoost IRLS regression trees.',
                'gradient_boosting': 'Original 3.5 used HistGradientBoostingClassifier; this runner uses genuine GradientBoostingClassifier to match manuscript method name.'}.get(method)}


def verify_resume(out, expected):
    config_path = out / 'run_config.json'
    if not config_path.exists():
        return None
    actual = json.loads(config_path.read_text(encoding='utf-8'))
    if actual != expected:
        raise ValueError('Experiment provenance differs; use a new directory rather than mixing results')
    metrics_path = out / 'metrics.json'
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    if metrics.get('provenance') != expected:
        raise ValueError('Completed metrics provenance differs')
    for filename, digest in metrics['artifact_sha256'].items():
        if not (out / filename).is_file() or sha256(out / filename) != digest:
            raise ValueError(f'Completed artifact provenance differs: {filename}')
    for split, filename in [('test', 'predictions.csv'), ('val', 'validation_predictions.csv')]:
        pred = pd.read_csv(out / filename, dtype={'sample_id': str})
        # CSV decimal roundtrip may differ in the last bit, so compare numerically.
        recomputed = compute_metrics(pred.y_true, pred.prob_1)
        def same(actual, recorded):
            if isinstance(actual, dict):
                return isinstance(recorded, dict) and set(actual) == set(recorded) and all(
                    same(actual[k], recorded[k]) for k in actual)
            if isinstance(actual, list):
                return isinstance(recorded, list) and len(actual) == len(recorded) and all(
                    same(a, b) for a, b in zip(actual, recorded))
            if isinstance(actual, (int, float)):
                return isinstance(recorded, (int, float)) and bool(np.isclose(actual, recorded, atol=1e-12, rtol=1e-12))
            return actual == recorded
        if not same(recomputed, metrics[split]):
            raise ValueError(f'Completed metrics provenance differs: {split}')
        if not np.array_equal(pred.y_pred.to_numpy(), (pred.prob_1.to_numpy() >= THRESHOLD).astype(int)):
            raise ValueError(f'Completed prediction provenance differs: {split}')
    return {'status': 'resumed_complete', 'metrics': metrics}


def run(root, method, seed=44, device='cpu', bert_model='models/bert-base-uncased',
        neural_options=None):
    if method not in METHODS:
        raise ValueError(f'Unsupported method: {method}')
    root = Path(root)
    options = neural_options or {}
    splits = load_splits(root)
    out = root / 'results' / method
    identity = provenance(root, method, seed, device, bert_model, options)
    for split, df in splits.items():
        identity.setdefault('split_sample_ids_sha256', {})[split] = hashlib.sha256(
            json.dumps(df.sample_id.tolist(), ensure_ascii=False).encode('utf-8')).hexdigest()
        identity.setdefault('split_rows', {})[split] = len(df)
    resumed = verify_resume(out, identity)
    if resumed:
        print(f'{method}: verified completed artifacts; skipping retraining', flush=True)
        return resumed
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'run_config.json', identity)
    seed_everything(seed)
    start = time.monotonic()
    x = {s: prepare_features(df) for s, df in splits.items()}
    y = {s: df[TARGET].to_numpy(dtype=int) for s, df in splits.items()}
    if method in ('textcnn', 'transformer', 'bert'):
        predict, modelinfo = fit_neural(method, x['train'], y['train'], x['val'], y['val'],
                                       seed, device, bert_model, out, options)
    else:
        predict, modelinfo = fit_classical(method, x['train'], y['train'], x['val'], y['val'], seed, out)
    # Selected model/checkpoint has been saved. Only now parse held-out test.
    splits.update(load_splits(root, names=('test',), development=splits))
    x['test'] = prepare_features(splits['test'])
    y['test'] = splits['test'][TARGET].to_numpy(dtype=int)
    metrics = {'status': 'complete', 'method': method, 'seed': seed, 'threshold': THRESHOLD,
               'provenance': identity, 'model': modelinfo,
               'test_sample_ids_sha256': hashlib.sha256(json.dumps(
                   splits['test'].sample_id.tolist(), ensure_ascii=False).encode('utf-8')).hexdigest(),
               'test_rows': len(splits['test'])}
    for split, filename in [('val', 'validation_predictions.csv'), ('test', 'predictions.csv')]:
        probabilities = predict(x[split])
        metrics[split] = compute_metrics(y[split], probabilities)
        pd.DataFrame({'sample_id': splits[split].sample_id, 'y_true': y[split],
                      'y_pred': (probabilities >= THRESHOLD).astype(int),
                      'prob_1': probabilities}).to_csv(out / filename, index=False, float_format='%.17g')
    metrics['elapsed_seconds'] = time.monotonic() - start
    metrics['artifact_sha256'] = {str(p.relative_to(out)).replace('\\', '/'): sha256(p)
                                  for p in sorted(out.rglob('*')) if p.is_file() and
                                  p.name not in ('metrics.json', 'run_config.json')}
    write_json(out / 'metrics.json', metrics)
    print(f'{method} complete: test MacroF1={metrics["test"]["macro_f1"]:.6f}', flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--seed', type=int, default=44)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cpu')
    parser.add_argument('--bert-model', default='models/bert-base-uncased')
    parser.add_argument('--epochs', type=int, help='Explicit neural override, recorded in provenance')
    parser.add_argument('--batch-size', type=int, help='Explicit neural override, recorded in provenance')
    args = parser.parse_args()
    options = {k: v for k, v in {'epochs': args.epochs, 'batch_size': args.batch_size}.items() if v is not None}
    if any(v < 1 for v in options.values()):
        parser.error('epochs and batch size must be positive')
    run(args.root, args.method, args.seed, args.device, args.bert_model, options)


# Preserve an importable joblib class identity even when invoked as a script.
if __name__ == '__main__':
    sys.modules.setdefault('baselines_suite', sys.modules[__name__])
    LogitBoostClassifier.__module__ = 'baselines_suite'
    main()
