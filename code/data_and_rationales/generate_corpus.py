from base64 import urlsafe_b64encode
import argparse
import concurrent.futures
import copy
import datetime
import json
import math
import re
import threading
import time
from pathlib import Path
from protocol import FEATURE_COLS, TARGET, rationale_is_clean, write_json

JUDGES = ['deepseek-v3.2', 'qwen3.5-plus']
TEACHER = 'gpt-5.4'
TEACHER_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'trip_rationale', 'strict': True,
    'schema': {'type': 'object', 'properties': {'reason': {'type': 'string'}, 'pred': {'type': 'integer', 'enum': [0, 1]}},
               'required': ['reason', 'pred'], 'additionalProperties': False}}}
JUDGE_FORMAT = {'type': 'json_schema', 'json_schema': {'name': 'rationale_verdict', 'strict': True,
    'schema': {'type': 'object', 'properties': {'pass': {'type': 'boolean'}, 'score': {'type': 'number', 'minimum': 0, 'maximum': 10}, 'feedback': {'type': 'string'}},
               'required': ['pass', 'score', 'feedback'], 'additionalProperties': False}}}


def valid_teacher(value, label):
    return (isinstance(value, dict) and type(value.get('pred')) is int and value['pred'] == label
            and rationale_is_clean(value.get('reason'))
            and not re.search(r'[\u4e00-\u9fff]|\b(label|ground truth)\b', value['reason'], re.I))


def valid_verdict(value):
    return (isinstance(value, dict) and type(value.get('pass')) is bool
            and type(value.get('score')) in (int, float) and math.isfinite(value['score']) and 0 <= value['score'] <= 10
            and isinstance(value.get('feedback'), str))


def accepted(stages):
    return all(stages.get(k, {}).get('status') == 'valid'
               and valid_verdict(stages[k].get('verdict'))
               and stages[k]['verdict']['pass'] and stages[k]['verdict']['score'] >= 8
               for k in ['judge1', 'judge2'])


def curate(item, previous, teacher_system, judge_system, call, checkpoint=lambda value: None):
    result = copy.deepcopy(item)
    result['stages'] = copy.deepcopy(previous.get('stages', {}))
    stages = result['stages']
    result.update(accepted=False, curation_complete=False, human_reviewed=False)
    teacher = stages.get('teacher', {})
    if teacher.get('status') != 'valid' or not valid_teacher(teacher.get('output'), item['label']):
        stages['teacher'] = call(TEACHER, teacher_system, item['teacher_user'], item['sample_id'], 'teacher')
        teacher = stages['teacher']
        checkpoint(result)
    if teacher.get('status') != 'valid' or not valid_teacher(teacher.get('output'), item['label']):
        result['review_outcome'] = 'teacher_technical_failure'
        checkpoint(result)
        return result
    result.update(reason=teacher['output']['reason'], pred=teacher['output']['pred'])
    user = item['teacher_user'] + '\nRationale to review: ' + result['reason']
    for i, model in enumerate(JUDGES, 1):
        role = f'judge{i}'
        stage = stages.get(role, {})
        if stage.get('status') != 'valid' or not valid_verdict(stage.get('verdict')):
            stages[role] = call(model, judge_system, user, item['sample_id'], role)
            checkpoint(result)
    result['curation_complete'] = all(stages[k].get('status') == 'valid' and valid_verdict(stages[k].get('verdict')) for k in ['judge1', 'judge2'])
    result['accepted'] = accepted(stages)
    result['review_outcome'] = ('accepted' if result['accepted'] else 'content_rejected') if result['curation_complete'] else 'judge_technical_failure'
    result['judges'] = [stages[k] for k in ['judge1', 'judge2']]
    checkpoint(result)
    return result


def prepare(root, source):
    import numpy as np
    import pandas as pd
    from xgboost import XGBClassifier, DMatrix
    frame = pd.read_csv(root / 'data/train.csv', dtype={'sample_id': str})
    original = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    ids = [r['sample_id'] for r in original]
    if len(ids) != 1000 or len(set(ids)) != 1000 or not set(ids) <= set(frame.sample_id):
        raise ValueError('Expected 1000 unique original training-only candidate IDs')
    heldout = set()
    for split in ['val', 'test']:
        heldout.update(pd.read_csv(root / f'data/{split}.csv', usecols=['sample_id']).sample_id)
    if set(ids) & heldout or set(frame.sample_id) & heldout:
        raise ValueError('Candidate/training versus held-out overlap')
    metadata = json.loads((root / 'data/shap_metadata.json').read_text())
    if metadata.get('fit_split') != 'train_only':
        raise ValueError('Attribution model must be fitted on the training split')
    selected = frame.set_index('sample_id').loc[ids]
    x = selected[FEATURE_COLS].copy()
    x['dropoff_poi_category'] = x.dropoff_poi_category.astype(str).map(metadata['category_mapping']).fillna(-1)
    x['trip_start_timestamp'] = pd.to_datetime(x.trip_start_timestamp).astype('int64') / 1e9
    x = x.astype(float).fillna(pd.Series(metadata['medians']))
    model = XGBClassifier()
    model.load_model(root / 'data/shap_model.json')
    contributions = model.get_booster().predict(DMatrix(x), pred_contribs=True)
    probabilities = model.predict_proba(x)[:, 1]
    items = []
    for j, (sid, row) in enumerate(selected.iterrows()):
        hints = [{'feature': FEATURE_COLS[i], 'value': str(row[FEATURE_COLS[i]]), 'shap_log_odds': round(float(contributions[j, i]), 6)}
                 for i in np.argsort(-np.abs(contributions[j, :-1]))]
        features = {k: str(row[k]) for k in FEATURE_COLS}
        probability, base = float(probabilities[j]), float(contributions[j, -1])
        user = ('Trip features (raw values; nan means missing): ' + json.dumps(features)
                + '\nUnits: fare and additional charges in USD; trip distance in miles; duration in seconds; center distances in km; dew point in degrees Celsius.'
                + '\nObserved training label (for JSON pred only): ' + str(int(row[TARGET]))
                + '\nAttribution model: XGBoost\nModel authorization probability: ' + str(round(probability, 6))
                + '\nSHAP baseline log-odds: ' + str(base)
                + '\nTop four absolute SHAP contributions: ' + json.dumps(hints[:4]))
        items.append({'sample_id': sid, 'label': int(row[TARGET]), 'features': features, 'shap': hints,
                      'teacher_shap': hints[:4], 'xgb_probability': probability, 'shap_base_log_odds': base, 'teacher_user': user})
    return items


def run(root, source, workers):
    from api_suite import client_for_suite
    root, source = Path(root), Path(source)
    out = root / 'curation/main_english_v1'
    cache = out / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    teacher_system = (root / 'prompts/teacher_system_en.txt').read_text(encoding='utf-8')
    judge_system = (root / 'prompts/judge_system_en.txt').read_text(encoding='utf-8')
    items = prepare(root, source)
    config = {'version': 'main_english_v1', 'teacher': TEACHER, 'judges': JUDGES,
              'teacher_system': teacher_system, 'judge_system': judge_system,
              'teacher_response_format': TEACHER_FORMAT, 'judge_response_format': JUDGE_FORMAT,
              'candidate_ids': [r['sample_id'] for r in items],
              'temperature': 0, 'max_completion_tokens': 512, 'max_attempts_per_invocation': 3,
              'acceptance': 'Both judges must have a valid boolean pass=true and numeric score>=8/10.',
              'human_reviewed': 0, 'rationale_language': 'English', 'shap_policy': 'Recomputed from the unchanged training-only model; top four absolute contributions supplied.'}
    cfg_path = out / 'run_config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise ValueError('Curation configuration changed; use another version directory')
    write_json(cfg_path, config)
    client, key = client_for_suite()
    lock, stop = threading.Lock(), threading.Event()
    ledger = out / 'api_calls.jsonl'
    started = time.time()
    def call(model, system, user, sid, role):
        attempts = []
        for attempt in range(3):
            if stop.is_set():
                return {'status': 'error', 'model': model, 'error': 'Stopped after quota failure', 'attempts': attempts}
            entry = {'sample_id': sid, 'stage': role, 'requested_model': model, 'attempt': attempt + 1,
                     'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
            t0 = time.time()
            try:
                extra = {'extra_body': {'enable_thinking': False}} if model == 'qwen3.5-plus' else {}
                response = client.chat.completions.create(model=model, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                    temperature=0, max_completion_tokens=512, response_format=TEACHER_FORMAT if role == 'teacher' else JUDGE_FORMAT, **extra)
                raw = response.choices[0].message.content or ''
                entry.update(raw=raw, actual_model=response.model, request_id=response.id,
                    usage=response.usage.model_dump() if response.usage else None, finish_reason=response.choices[0].finish_reason)
                value = json.loads(raw)
                label = next(r['label'] for r in items if r['sample_id'] == sid) if role == 'teacher' else None
                legal = valid_teacher(value, label) if role == 'teacher' else valid_verdict(value)
                if not legal:
                    raise ValueError('Malformed output or teacher language/label/no-tip guard violation')
                entry['success'] = True
            except Exception as exc:
                entry.update(success=False, error=str(exc).replace(key, '[redacted]')[:600])
                if any(token in entry['error'].lower() for token in ['insufficient_user_quota', 'insufficient_quota', 'insufficient balance']):
                    stop.set()
            entry['seconds'] = time.time() - t0
            with lock:
                with ledger.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
            attempts.append(entry)
            if entry['success']:
                return {'status': 'valid', 'model': model, 'actual_model': response.model,
                        'output' if role == 'teacher' else 'verdict': value, 'attempts': attempts}
            if attempt < 2 and not stop.is_set():
                time.sleep(2 ** (attempt + 1))
        return {'status': 'error', 'model': model, 'error': attempts[-1]['error'], 'attempts': attempts}
    def process(item):
        if stop.is_set():
            return None
        path = cache / (urlsafe_b64encode(item['sample_id'].encode()).decode() + '.json')
        previous = json.loads(path.read_text()) if path.exists() else {}
        def checkpoint(record):
            write_json(path, record)
        return curate(item, previous, teacher_system, judge_system, call, checkpoint)
    results = []
    def status(stage):
        records = [json.loads(path.read_text()) for path in cache.glob('*.json')]
        value = {'stage': stage, 'selected': len(items), 'teacher_valid': sum(r.get('stages', {}).get('teacher', {}).get('status') == 'valid' for r in records),
                 'judge1_valid': sum(r.get('stages', {}).get('judge1', {}).get('status') == 'valid' for r in records),
                 'judge2_valid': sum(r.get('stages', {}).get('judge2', {}).get('status') == 'valid' for r in records),
                 'reviewed': sum(r.get('curation_complete', False) for r in records), 'accepted': sum(r.get('accepted', False) for r in records),
                 'content_rejected': sum(r.get('review_outcome') == 'content_rejected' for r in records),
                 'technical_failures': sum(r.get('review_outcome', '').endswith('technical_failure') for r in records),
                 'workers': workers, 'elapsed_seconds': time.time() - started,
                 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'human_reviewed': 0,
                 'new_student_training_started': False, 'bulk_corpus_generated': stage == 'complete'}
        write_json(out / 'status.json', value)
        write_json(root / 'status.json', value)
    status('running')
    first = process(items[0])
    if not first or not first.get('curation_complete'):
        status('finished_with_failures')
        raise RuntimeError('Teacher/two-judge canary failed; no bulk dispatch')
    results.append(first)
    print('Canary complete', first['sample_id'], first['review_outcome'], flush=True)
    status('running')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, item) for item in items[1:]]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            if len(results) % 10 == 0:
                status('running')
                print('Processed', len(results), 'accepted', sum(r.get('accepted', False) for r in results), flush=True)
    index = {r['sample_id']: r for r in results}
    ordered = [index[item['sample_id']] for item in items if item['sample_id'] in index]
    complete = len(ordered) == len(items) and all(r.get('curation_complete') for r in ordered)
    full = out / 'rationales.jsonl'
    full.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ordered), encoding='utf-8')
    retained = out / 'accepted.jsonl'
    retained.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in ordered if r['accepted']), encoding='utf-8')
    summary = {'complete': complete, 'selected': len(items), 'processed': len(ordered),
               'accepted': sum(r['accepted'] for r in ordered), 'content_rejected': sum(r.get('review_outcome') == 'content_rejected' for r in ordered),
               'technical_failures': sum(not r['curation_complete'] for r in ordered), 'not_processed': len(items) - len(ordered),
               'human_reviewed': 0, 'teacher': TEACHER, 'judges': JUDGES,
               'elapsed_seconds': time.time() - started}
    write_json(out / 'summary.json', summary)
    if complete:
        (root / 'data/rationales.jsonl').write_bytes(full.read_bytes())
        (root / 'data/rationale_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    status('complete' if complete else 'finished_with_failures')
    print(json.dumps(summary), flush=True)
    return 0 if complete else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-candidates', required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    raise SystemExit(run(args.root, args.source_candidates, args.workers))
