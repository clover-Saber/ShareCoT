import argparse
import collections
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from protocol import TARGET, FEATURE_COLS, build_prompt, write_json

NOISE_COLS = ['fare', 'trip_miles', 'trip_seconds', 'additional_charges']
LEVELS = [0.05, 0.10, 0.20]
REPEATS = 5


def macro_f1_rows(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    scores = []
    for label in [0, 1]:
        tp = ((y == label) & (pred == label)).sum(axis=-1)
        fp = ((y != label) & (pred == label)).sum(axis=-1)
        fn = ((y == label) & (pred != label)).sum(axis=-1)
        denominator = 2 * tp + fp + fn
        scores.append(np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0))
    return (scores[0] + scores[1]) / 2


def align_to_reference(reference, other):
    if reference.sample_id.duplicated().any() or other.sample_id.duplicated().any() or set(reference.sample_id) != set(other.sample_id):
        raise ValueError('Paired comparison requires identical unique sample IDs')
    aligned = other.set_index('sample_id').loc[reference.sample_id]
    if not np.array_equal(reference.y_true.to_numpy(), aligned.y_true.to_numpy()):
        raise ValueError('Paired labels differ')
    return aligned.y_pred.to_numpy(dtype=np.int8)


def perturb(frame, sigma, seed):
    if sigma < 0:
        raise ValueError('Noise level must be nonnegative')
    out = frame.copy()
    if sigma == 0:
        return out
    values = frame[NOISE_COLS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Noise protocol expects finite, nonnegative source quantities')
    error = np.random.RandomState(seed).normal(size=values.shape)
    out[NOISE_COLS] = np.maximum(0, values * (1 + sigma * error))
    return out


def statistics(suite):
    suite = Path(suite)
    out = suite / 'experiments/14_statistics'
    out.mkdir(parents=True, exist_ok=True)
    root = suite / 'experiments/02_main/results'
    reference = pd.read_csv(root / 'adaptive/predictions.csv', dtype={'sample_id': str})
    paths = {p.parent.name: p for p in sorted(root.glob('*/predictions.csv')) if not p.parent.name.startswith('api_') and p.parent.name!='sharecot'}
    shuffle = suite / 'experiments/12_shuffle/results/sharecot/predictions.csv'
    if not shuffle.exists():
        raise ValueError('Shuffled-rationale experiment must complete before statistics')
    paths['shuffled_reason'] = shuffle
    predictions = {name: align_to_reference(reference, pd.read_csv(path, dtype={'sample_id': str})) for name, path in paths.items()}
    y = reference.y_true.to_numpy(dtype=np.int8)
    n, resamples = len(y), 5000
    rng = np.random.RandomState(44)
    bootstrap = {name: [] for name in predictions}
    for start in range(0, resamples, 100):
        ids = rng.randint(0, n, (min(100, resamples - start), n))
        for name, pred in predictions.items():
            bootstrap[name].extend(macro_f1_rows(y[ids], pred[ids]).tolist())
    rows = []
    a = predictions['adaptive']
    base = float(macro_f1_rows(y, a))
    for name, b in predictions.items():
        score = float(macro_f1_rows(y, b))
        difference = base - score
        ci = np.percentile(bootstrap[name], [2.5, 97.5]).tolist()
        delta_ci = np.percentile(np.asarray(bootstrap['adaptive']) - bootstrap[name], [2.5, 97.5]).tolist()
        pvalue = None
        if name != 'adaptive':
            prng = np.random.RandomState(44)
            exceed = 0
            for start in range(0, resamples, 100):
                swap = prng.rand(min(100, resamples - start), n) < .5
                pa, pb = np.where(swap, b, a), np.where(swap, a, b)
                difference_null = macro_f1_rows(y, pa) - macro_f1_rows(y, pb)
                exceed += int((np.abs(difference_null) >= abs(difference) - 1e-12).sum())
            pvalue = (exceed + 1) / (resamples + 1)
        rows.append({'method': name, 'accuracy': float((b == y).mean()), 'macro_f1': score,
                     'macro_f1_95_ci': ci, 'adaptive_minus_method': difference,
                     'paired_difference_95_ci': delta_ci, 'paired_permutation_two_sided_p': pvalue})
    np.savez_compressed(out / 'bootstrap_macro_f1.npz', **{name: np.asarray(v) for name, v in bootstrap.items()})
    pd.DataFrame(rows).to_csv(out / 'comparisons.csv', index=False)
    write_json(out / 'summary.json', {
        'results': rows, 'n': n, 'resamples': resamples, 'resampling_seed': 44, 'training_seed': 44,
        'interpretation': 'Paired fixed-test uncertainty conditional on selected seed44 checkpoints; does not estimate training-seed uncertainty. Multiple baseline comparisons are exploratory and unadjusted.',
        'complete': True,
    })
    print('Statistics complete:', len(rows), 'methods', flush=True)


def noise_frames(main):
    path=Path(main)/'data/test.csv'
    numeric=pd.read_csv(path,dtype={'sample_id':str})
    raw=pd.read_csv(path,dtype=str,keep_default_na=False)
    raw[TARGET]=numeric[TARGET].to_numpy()
    if raw.sample_id.tolist()!=numeric.sample_id.tolist():
        raise ValueError('Raw and numeric input memberships differ')
    return numeric,raw


def cached_probabilities(cached,frame):
    reference=pd.DataFrame({'sample_id':frame.sample_id,'y_true':frame[TARGET]})
    align_to_reference(reference,cached)
    aligned=cached.set_index('sample_id').loc[frame.sample_id]
    prob=aligned.prob.to_numpy(float)
    if not np.array_equal(aligned.y_pred.to_numpy(),(prob>=.5).astype(int)):
        raise ValueError('Cached probability/classes differ')
    return prob


def verify_noise_identity(out,config):
    out=Path(out);cfg=out/'run_config.json'
    if not cfg.exists() and any(out.iterdir()):
        raise ValueError('Orphan noise artifacts without run configuration')
    if cfg.exists() and json.loads(cfg.read_text())!=config:
        raise ValueError('Noise protocol changed; use a new output folder')
    write_json(cfg,config)


def noise(suite):
    import torch
    from evaluate_suite import load_qwen, classify, neural_baseline_predictor
    suite = Path(suite)
    main = suite / 'experiments/02_main'
    out = suite / 'experiments/13_estimation_noise'
    out.mkdir(parents=True, exist_ok=True)
    numeric_frame,raw_frame = noise_frames(main)
    frame=numeric_frame
    methods = ['adaptive', 'direct_ft', 'transformer']
    config = {'methods': methods, 'columns': NOISE_COLS, 'levels': LEVELS, 'repeats': REPEATS,
              'noise_seed_base': 4400, 'training_seed': 44, 'threshold': .5,
              'input_serialization': 'Each model retains its training-time parser on untouched features; equal numerical errors on perturbed quantities. Text baseline keeps original CSV lexical precision.',
              'distribution': 'Independent multiplicative Gaussian errors, clipped at zero; identical perturbed inputs across methods; zero source quantities remain zero.',
              'interpretation': 'Assumed error sensitivity scenarios, not calibrated booking estimates or an external deployment validation.'}
    verify_noise_identity(out,config)
    scenarios = [(0., 0)] + [(level, repeat) for level in LEVELS for repeat in range(REPEATS)]
    rows = []
    for method in methods:
        frame=raw_frame if method=='transformer' else numeric_frame
        result = main / 'results' / method
        if method == 'transformer':
            model, predict = neural_baseline_predictor(result, method)
            def inference(x):
                return np.concatenate([predict(x.iloc[start:start + 64]) for start in range(0, len(x), 64)])
        else:
            model, tok, dev = load_qwen(result)
            def inference(x, model=model, tok=tok, dev=dev):
                return classify(model, tok, dev, x)
        saved = pd.read_csv(result / 'predictions.csv', dtype={'sample_id': str}).set_index('sample_id')
        probability_col = 'prob_1' if method == 'transformer' else 'prob'
        if not np.allclose(inference(frame.iloc[:32]), saved.loc[frame.sample_id.iloc[:32], probability_col], atol=1e-5):
            raise ValueError('Selected model reload mismatch: ' + method)
        for level, repeat in scenarios:
            altered = perturb(frame, level, 4400 + repeat)
            name = f'level_{level:.2f}_repeat_{repeat}'
            inputs = out / 'inputs' / (name + '.csv')
            inputs.parent.mkdir(exist_ok=True)
            perturb(raw_frame, level, 4400 + repeat).to_csv(inputs, index=False)
            path = out / 'predictions' / (method + '_' + name + '.csv')
            path.parent.mkdir(exist_ok=True)
            if path.exists():
                cached = pd.read_csv(path, dtype={'sample_id': str})
                prob=cached_probabilities(cached,frame)
            else:
                prob = inference(altered)
                pd.DataFrame({'sample_id': frame.sample_id, 'y_true': frame[TARGET],
                              'y_pred': (prob >= .5).astype(int), 'prob': prob}).to_csv(path, index=False,float_format='%.17g')
            if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
                raise ValueError('Invalid perturbation probabilities')
            pred = (prob >= .5).astype(int)
            if level == 0 and (not np.allclose(prob, saved.loc[frame.sample_id, probability_col], atol=1e-5)
                               or not np.array_equal(pred, saved.loc[frame.sample_id, 'y_pred'])):
                raise ValueError('Clean noise condition differs from original predictions')
            rows.append({'method': method, 'relative_error_sd': level, 'repeat': repeat,
                         'accuracy': float((pred == frame[TARGET].to_numpy()).mean()),
                         'macro_f1': float(macro_f1_rows(frame[TARGET].to_numpy(), pred))})
            write_json(out / 'progress.json', {'processed_conditions': len(rows), 'total_conditions': len(methods) * len(scenarios), 'method': method, 'level': level})
            print(method, name, rows[-1]['macro_f1'], flush=True)
        del model, inference
        if method == 'transformer':
            del predict
        else:
            del tok
        gc.collect()
        torch.cuda.empty_cache()
    table = pd.DataFrame(rows)
    table.to_csv(out / 'conditions.csv', index=False)
    grouped = table.groupby(['method', 'relative_error_sd'])[['accuracy', 'macro_f1']].agg(['mean', 'std'])
    grouped.to_csv(out / 'sensitivity.csv')
    write_json(out / 'summary.json', {'status':'complete', 'config': config, 'conditions': rows, 'complete': True,
                                    'repeat_variation': 'Five assumed input-error realizations; these are not training seeds.'})


def stage_statistics(records):
    stages = {}
    attempts = 0
    prompt_tokens = completion_tokens = 0
    request_seconds = 0.
    seen = set()
    for record in records:
        for role, stage in record.get('stages', {}).items():
            item = stages.setdefault(role, {'observed': 0, 'valid_verdicts': 0, 'passing_verdicts': 0, 'status_counts': collections.Counter()})
            item['observed'] += 1
            item['status_counts'][stage.get('status', 'unknown')] += 1
            valid = stage.get('status') == 'valid'
            item['valid_verdicts'] += int(valid)
            verdict = stage.get('verdict', {})
            item['passing_verdicts'] += int(valid and verdict.get('pass') is True and verdict.get('score', -1) >= 8)
            for attempt in stage.get('attempts', []):
                attempts += 1
                request_seconds += attempt.get('seconds', 0) or 0
                rid = attempt.get('request_id')
                if rid and rid in seen:
                    continue
                if rid:
                    seen.add(rid)
                usage = attempt.get('usage') or {}
                prompt_tokens += usage.get('prompt_tokens', 0) or 0
                completion_tokens += usage.get('completion_tokens', 0) or 0
    accepted = sum(bool(r.get('accepted')) for r in records)
    return {'selected': len(records), 'accepted': accepted, 'rejected': len(records) - accepted,
            'acceptance_rate': accepted / len(records) if records else None,
            'human_reviewed': sum(bool(r.get('human_reviewed')) for r in records),
            'human_rejection_rate': None, 'human_review_time': None, 'monetary_cost': None,
            'stages': stages, 'recorded_attempts': attempts, 'unique_response_ids': len(seen),
            'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
            'sum_recorded_request_seconds': request_seconds,
            'time_definition': 'Summed request durations, not parallel pipeline wallclock.',
            'cost_scope': 'Attempts attached to final corpus records; preliminary canaries or other model calls may be absent. Monetary cost requires provider billing; human rejection/time requires actual annotations.'}


def audit(suite):
    from transformers import AutoTokenizer
    suite = Path(suite)
    main = suite / 'experiments/02_main'
    out = suite / 'experiments/15_curation_runtime_audit'
    out.mkdir(parents=True, exist_ok=True)
    corpus = main / 'data/rationales.jsonl'
    records = [json.loads(line) for line in corpus.read_text(encoding='utf-8').splitlines()]
    curation = stage_statistics(records)
    curation['existing_summary'] = json.loads((main / 'data/rationale_summary.json').read_text())
    write_json(out / 'curation.json', curation)
    from evaluate_suite import selected_adapter
    tokenizer = AutoTokenizer.from_pretrained(selected_adapter(main / 'results/adaptive'), local_files_only=True)
    frame = pd.read_csv(main / 'data/test.csv')
    lengths = [len(tokenizer(build_prompt(dict(row), tokenizer), add_special_tokens=False)['input_ids']) for _, row in frame.iterrows()]
    pd.DataFrame({'sample_id': frame.sample_id, 'input_tokens': lengths}).to_csv(out / 'input_lengths.csv', index=False)
    timings = []
    for path in sorted((suite / 'experiments/03_efficiency').glob('*.json')):
        if path.name == 'summary.json':
            continue
        info = json.loads(path.read_text())
        values = info.get('seconds', [])
        if not values:
            continue
        timings.append({'source': str(path),
                        'method': info.get('method', path.stem), 'n_observations': len(values),
                        'mean_seconds': float(np.mean(values)), 'p95_seconds': float(np.percentile(values, 95)),
                        'sequential_batch1_requests_per_second': float(1 / np.mean(values)),
                        'peak_allocated_bytes': info.get('peak_allocated_bytes'),
                        'definition': 'Observed sequential batch-one request rate; not batched saturation throughput.'})
    write_json(out / 'summary.json', {'curation': curation, 'timings': timings,
                                      'input_token_quantiles': dict(zip(['min', 'p25', 'median', 'p75', 'p95', 'max'], np.percentile(lengths, [0, 25, 50, 75, 95, 100]).tolist())),
                                      'test_n': len(frame), 'weight_storage': 'Existing local BF16 checkpoints, no new quantization.',
                                      'complete': True})
    print('Curation and runtime audit complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True)
    parser.add_argument('--mode', required=True, choices=['noise', 'statistics', 'audit'])
    args = parser.parse_args()
    globals()[args.mode](args.suite)
