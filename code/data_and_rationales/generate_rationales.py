import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
import threading
import time
from protocol import FEATURE_COLS, TARGET, feature_text, rationale_is_clean, write_json

TEACHER = 'gpt-5.4'
JUDGES = ['deepseek-v3.2', 'qwen3.5-plus']
PROMPT_VERSION = 'fresh_no_tip_v3_explicit_label_semantics'
POLICY_VERSION = 'v1_content_rejection_transport_stage_resume'


def api_options(model):
    return {'extra_body': {'enable_thinking': False}} if model == 'qwen3.5-plus' else {}


def parse_json(text):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start:
        raise ValueError('JSON object absent')
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError('Expected JSON object')
    return obj


def valid_teacher(obj, label):
    return isinstance(obj, dict) and type(obj.get('pred')) is int and obj['pred'] == int(label) and rationale_is_clean(obj.get('reason'))


def judge_passes(obj):
    return (obj.get('pass') is True and isinstance(obj.get('score'), (int, float))
            and 8 <= obj['score'] <= 10)


def valid_judge(obj):
    return (isinstance(obj, dict) and type(obj.get('pass')) is bool
            and type(obj.get('score')) in (int, float) and 0 <= obj['score'] <= 10
            and isinstance(obj.get('feedback'), str))


def judges_accepted(judges):
    return (len(judges) == len(JUDGES) == 2
            and [item.get('model') for item in judges] == JUDGES
            and all(item.get('status') == 'valid' and valid_judge(item.get('verdict'))
                    and judge_passes(item['verdict']) for item in judges))


def sufficient_coverage(accepted, selected):
    return selected > 0 and accepted * 2 >= selected


def request_stage(client, model, system, user, max_tokens, sample_id, role,
                  log_attempt, api_key, transport_errors, sleep=time.sleep):
    """Retry bounded failures; returned malformed content never becomes approval."""
    outcome = {'model': model, 'role': role, 'attempts': []}
    for attempt in range(3):
        started = time.time()
        entry = {'sample_id': sample_id, 'role': role, 'model': model,
                 'attempt': attempt + 1}
        try:
            response = client.chat.completions.create(model=model, messages=[
                {'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                temperature=0., max_completion_tokens=max_tokens, **api_options(model))
        except transport_errors as exc:
            safe = str(exc).replace(api_key, '[redacted]')[:400]
            entry.update(status='error', success=False, error=safe,
                         seconds=time.time() - started, request_id=getattr(exc, 'request_id', None))
            outcome.update(status='error', error=safe)
        else:
            text = response.choices[0].message.content or '' if response.choices else ''
            entry.update(raw=text, request_id=response.id, seconds=time.time() - started,
                         usage=response.usage.model_dump() if response.usage else None,
                         finish_reason=response.choices[0].finish_reason if response.choices else None)
            outcome.pop('error', None)
            outcome.update(raw=text)
            try:
                parsed = parse_json(text)
                if role.startswith('judge') and not valid_judge(parsed):
                    raise ValueError('Judge schema requires score 0..10, pass boolean, feedback string')
            except ValueError as exc:
                entry.update(status='rejected', success=True, validation_error=str(exc))
                outcome.update(status='rejected', rejection='malformed_content')
            else:
                entry.update(status='valid', success=True)
                outcome.update(status='valid', verdict=parsed)
                outcome.pop('rejection', None)
        outcome['attempts'].append(entry)
        log_attempt(entry)
        if outcome['status'] == 'valid':
            return outcome
        if attempt < 2:
            sleep(2 ** (attempt + 1))
    return outcome


def curate_stages(result, teacher_system, judge_system, user, call, checkpoint):
    """Resume only unfinished/transport-failed stages, retaining prior attempts."""
    result.update(accepted=False, curation_complete=False)
    result.pop('error', None)
    result.pop('rejection', None)
    stages = result.setdefault('stages', {})

    def stage(model, system, prompt, limit, role):
        previous = stages.get(role)
        if previous and previous.get('model') != model:
            raise ValueError('Cached stage model differs from registered model')
        if not previous or previous.get('status') == 'error':
            value = call(model, system, prompt, limit, result['sample_id'], role)
            value['attempts'] = (previous or {}).get('attempts', []) + value['attempts']
            stages[role] = value
        value = stages[role]
        if role == 'teacher':
            result['teacher_raw'] = value.get('raw', '')
        result['judges'] = [stages[f'judge{i+1}'] for i in range(len(JUDGES)) if f'judge{i+1}' in stages]
        checkpoint(result)
        return value

    def stop_for(value):
        if value['status'] == 'error':
            result['error'] = value['error']
        else:
            result.update(rejection=f"{value['role']}:{value.get('rejection', 'judge_rejected')}",
                          curation_complete=True)
        checkpoint(result)
        return result

    teacher = stage(TEACHER, teacher_system, user, 512, 'teacher')
    if teacher['status'] != 'valid':
        return stop_for(teacher)
    if not valid_teacher(teacher['verdict'], result['label']):
        teacher.update(status='rejected', rejection='invalid_label_or_forbidden_content')
        return stop_for(teacher)
    result['reason'] = teacher['verdict']['reason'].strip()
    for index, judge in enumerate(JUDGES):
        verdict = stage(judge, judge_system, user + '\n待审核解释：' + result['reason'], 320, f'judge{index+1}')
        if verdict['status'] != 'valid':
            return stop_for(verdict)
        verdict['passed'] = valid_judge(verdict['verdict']) and judge_passes(verdict['verdict'])
    result.update(accepted=judges_accepted(result['judges']), curation_complete=True)
    if not result['accepted']:
        result['rejection'] = 'judge_rejected'
    checkpoint(result)
    return result


def credentials(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        text = line.strip()
        if text and not text.startswith('#') and '=' in text:
            k, v = text.split('=', 1); values[k.strip()] = v.strip().strip('\"').strip("'")
    key = values.get('AIHUBMIX_API_KEY') or os.environ.get('AIHUBMIX_API_KEY') or values.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('API credential missing; no secret written to experiment outputs')
    return key


def build_shap(root, count):
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    train = pd.read_csv(root / 'data/train.csv')
    x = train[FEATURE_COLS].copy()
    categories = sorted(x['dropoff_poi_category'].dropna().astype(str).unique())
    mapping = {name: i for i, name in enumerate(categories)}
    x['dropoff_poi_category'] = x['dropoff_poi_category'].astype(str).map(mapping).fillna(-1)
    x['trip_start_timestamp'] = pd.to_datetime(x['trip_start_timestamp']).astype('int64') / 1e9
    x = x.astype(float)
    medians = x.median(); x = x.fillna(medians)
    if x.isna().any().any():
        raise ValueError('Cannot fit SHAP with entirely missing feature')
    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.05,
                             subsample=.9, colsample_bytree=.9, random_state=42,
                             tree_method='hist', n_jobs=8)
    model.fit(x, train[TARGET])
    model.save_model(root / 'data/shap_model.json')
    sampled = train.sample(n=count, random_state=42).sort_index()
    contrib = model.get_booster().predict(xgb.DMatrix(x.loc[sampled.index]), pred_contribs=True)
    records = []
    for position, (_, row) in enumerate(sampled.iterrows()):
        values = contrib[position, :-1]
        hints = [{'feature': FEATURE_COLS[i], 'value': str(row[FEATURE_COLS[i]]),
                  'shap_log_odds': round(float(values[i]), 6)} for i in np.argsort(-np.abs(values))]
        records.append({'sample_id': row.sample_id, 'label': int(row[TARGET]),
                        'features': {k: str(row[k]) for k in FEATURE_COLS}, 'shap': hints,
                        'shap_base_log_odds': float(contrib[position, -1]),
                        'xgb_probability': float(model.predict_proba(x.loc[[row.name]])[0, 1])})
    write_json(root / 'data/shap_metadata.json', {'fit_split': 'train_only', 'seed': 42,
               'features': FEATURE_COLS, 'category_mapping': mapping, 'medians': medians.to_dict(),
               'selected_sample_ids': sampled.sample_id.tolist(), 'count': count})
    return records


def run(root, env_file, count=1000, workers=6, retry_errors=False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / '.runtime_deps'))
    from openai import OpenAI, APIError
    root = Path(root)
    api_key = credentials(env_file)
    client = OpenAI(api_key=api_key, base_url='https://aihubmix.com/v1', timeout=90., max_retries=0)
    cache = root / 'data/rationale_cache'; cache.mkdir(parents=True, exist_ok=True)
    records = build_shap(root, count)
    train_file = root / 'data/train.csv'
    train_info = train_file.stat()
    cache_context = {'prompt': PROMPT_VERSION, 'policy': POLICY_VERSION,
                     'teacher': TEACHER, 'judges': JUDGES, 'count': count,
                     'train_file': str(train_file.resolve()),
                     'train_size': train_info.st_size,
                     'train_modified_ns': train_info.st_mtime_ns}
    api_lock = threading.Lock()
    usage_path = root / 'logs/api_calls.jsonl'; usage_path.parent.mkdir(parents=True, exist_ok=True)

    def log_attempt(entry):
        with api_lock:
            with usage_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def call(model, system, user, max_tokens, sample_id, role):
        return request_stage(client, model, system, user, max_tokens, sample_id, role,
                             log_attempt, api_key, (APIError,))

    teacher_system = (
        '你为行程级拼车授权分类生成简短的事后归因解释。只能依据输入字段、给定标签和XGBoost SHAP。'
        '标签0表示未授权，标签1表示授权。SHAP正值提高授权评分，负值降低授权评分，属于模型归因而不是因果效应。'
        '用“正向归因/负向归因/模型评分”等表述，禁止用“导致/主要因为”做因果解释，禁止推断个人心理属性。'
        '选取2至4个主要贡献特征即可，无须罗列所有字段；不要自由翻译POI类别，若提及则保留输入原文。'
        '不得提及小费、打赏、价格敏感或体验敏感；数据不包含这些信息。'
        '若XGBoost预测概率所支持的类别与给定标签不同，要承认模型与观察标签的分歧，不得篡改归因方向。'
        '中文理由60至110字。仅输出JSON对象，字段reason和pred，其中pred必须等于给定标签。')
    judge_system = (
        '审核行程分类的事后归因解释。仅验证所给字段、标签和SHAP的一致性，不验证心理真实性。'
        '标签0=未授权，标签1=授权；正SHAP提高授权评分，负SHAP降低授权评分。负SHAP与标签0并不矛盾。'
        '每个样本允许同时出现正负归因，不要求每个特征方向均与标签一致，也不要求列举全部特征。'
        '检查是否使用不存在的信息、捏造事实、反转SHAP方向、做因果或个人心理断言。'
        '解释不能包含小费、打赏、价格敏感或体验敏感。SHAP与标签可以不一致，但解释必须如实承认。'
        '评分规则：事实和方向正确且无额外断言为8至10分；有明确错误为0至7分。无明确错误则不要捏造错误。'
        'score必须是0至10的数值，不能使用负分；pass为布尔值；feedback不超过40个汉字。8分及以上且无违规才通过。'
        '例如合法通过格式：{"pass":true,"score":9,"feedback":"归因方向与输入一致"}；'
        '合法拒绝格式：{"pass":false,"score":3,"feedback":"明确指出哪项事实错误"}。')

    def process(record):
        name = record['sample_id'].replace(':', '_') + '.json'
        path = cache / name
        old = None
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old.get('cache_context') != cache_context:
                raise ValueError('Rationale cache context mismatch; use a new experiment directory')
            if old.get('curation_complete') or (old.get('error') and not retry_errors):
                return old
        result = old or {'sample_id': record['sample_id'], 'label': record['label'], 'cache_context': cache_context,
                  'teacher_model': TEACHER, 'judge_models': JUDGES, 'accepted': False,
                  'human_reviewed': False, 'reason': '', 'shap': record['shap'],
                  'shap_base_log_odds': record['shap_base_log_odds'], 'xgb_probability': record['xgb_probability']}
        user = (feature_text(record['features']) + '\n给定观察标签：' + str(record['label'])
                + '\nXGBoost授权预测概率：' + str(round(record['xgb_probability'], 6))
                + '\nSHAP基准log-odds：' + str(round(record['shap_base_log_odds'], 6))
                + '\nSHAP归因：' + json.dumps(record['shap'], ensure_ascii=False))
        return curate_stages(result, teacher_system, judge_system, user, call,
                             lambda value: write_json(path, value))

    first = process(records[0])
    if first.get('error'):
        raise RuntimeError('API canary failed: ' + first['error'])
    results = [first]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process, record) for record in records[1:]]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            if len(results) % 10 == 0:
                status = {'processed': len(results), 'total': count,
                          'accepted': sum(x['accepted'] for x in results),
                          'rejected': sum(bool(x.get('rejection')) for x in results),
                          'errors': sum(bool(x.get('error')) for x in results)}
                write_json(root / 'data/rationale_progress.json', status); print(status, flush=True)
    results.sort(key=lambda x: x['sample_id'])
    path = root / 'data/rationales.jsonl'
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as f:
        for record in results: f.write(json.dumps(record, ensure_ascii=False) + '\n')
    temp.replace(path)
    summary = {'selected': count, 'accepted': sum(x['accepted'] for x in results),
               'rejected': sum(bool(x.get('rejection')) for x in results),
               'malformed_rejections': sum('malformed_content' in x.get('rejection', '') for x in results),
               'errors': sum(bool(x.get('error')) for x in results), 'cache_context': cache_context,
               'prompt_version': PROMPT_VERSION, 'policy_version': POLICY_VERSION,
               'teacher': TEACHER, 'judges': JUDGES, 'judge_api_options': {m: api_options(m) for m in JUDGES}, 'human_reviewed': 0,
               'tokens_and_calls': str(usage_path), 'monetary_cost': 'not estimated; use provider billing'}
    write_json(root / 'data/rationale_summary.json', summary)
    if summary['errors']:
        raise RuntimeError('Rationale API errors remain; retry --retry-errors before student training')
    if not sufficient_coverage(summary['accepted'], count):
        raise RuntimeError('Less than half of rationale candidates passed; inspect quality before training')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--env-file', required=True); parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=6); parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args(); run(args.root, args.env_file, args.count, args.workers, args.retry_errors)
