import collections
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SCORES = ('fact_score', 'grounding_score', 'coherence_score')
FEATURES = ['fare', 'trip_miles', 'trip_seconds', 'additional_charges',
            'pickup_to_center_distance_km', 'dropoff_to_center_distance_km',
            'trip_start_timestamp', 'dropoff_poi_category', 'weather_dewpoint_2m']


def read_lines(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def write_json(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def same_value(feature, a, b):
    if feature in ('trip_start_timestamp', 'dropoff_poi_category'):
        return str(a) == str(b)
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or a == b


def agreement(a, b, ordinal=False):
    assert len(a) == len(b) and len(a)
    categories = list(range(4)) if ordinal else sorted(set(a) | set(b))
    idx = {x: i for i, x in enumerate(categories)}
    conf = np.zeros((len(idx), len(idx)), dtype=float)
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    p = conf / conf.sum()
    dist = np.arange(len(idx))[:, None] - np.arange(len(idx))[None, :]
    weights = dist.astype(float)**2 / 9 if ordinal else (dist != 0).astype(float)
    observed = float((p * weights).sum())
    expected = float((p.sum(1)[:, None] * p.sum(0)[None, :] * weights).sum())
    return {'n': len(a), 'raw_agreement': float(np.trace(p)),
            'kappa': None if expected == 0 else float(1-observed/expected),
            'weighting': 'quadratic' if ordinal else 'unweighted',
            'categories': categories, 'counts': conf.astype(int).tolist()}


def alignment(conclusions, heads):
    values = {'accept': 1, 'reject': 0}
    assessable = sum(c in values for c in conclusions)
    matched = sum(c in values and values[c] == h for c, h in zip(conclusions, heads))
    return {'n': len(heads), 'assessable': assessable, 'matches': matched,
            'rate_all': matched/len(heads), 'rate_assessable': matched/assessable if assessable else None}


def cluster_mean_interval(values, repetitions=5000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n_samples': 0, 'mean': None, 'ci95': None}
    rng = np.random.default_rng(20261015)
    sampled = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(1)
    return {'n_samples': len(values), 'mean': float(values.mean()),
            'ci95': np.quantile(sampled, [.025, .975]).tolist(), 'bootstrap_repetitions': repetitions}


def validate(rows, packet):
    assert len(rows) == len(packet)
    lookup = {x['item_id']: x for x in rows}
    assert len(lookup) == len(rows) and set(lookup) == set(packet)
    for row in rows:
        assert all(type(row[k]) is int and 0 <= row[k] <= 3 for k in SCORES), row['item_id']
        assert row['conclusion'] in ('accept', 'reject', 'unclear', 'contradictory')
        assert row['human_reviewed'] is False and row['reviewer_kind'] == 'AI_preaudit'
        for key in ('cited_features', 'supporting_features', 'opposing_features', 'dominant_features'):
            assert set(row[key]) <= set(FEATURES), (row['item_id'], key)
        assert set(row['supporting_features']) | set(row['opposing_features']) <= set(row['cited_features'])
    return lookup


def teacher_direction_counts(rating, shap):
    good = bad = ambiguous = 0
    supporting, opposing = set(rating['supporting_features']), set(rating['opposing_features'])
    for feature in supporting | opposing:
        if feature in supporting & opposing:
            ambiguous += 1
        elif abs(shap[feature]) < 1e-8:
            ambiguous += 1
        elif (shap[feature] > 0) == (feature in supporting):
            good += 1
        else:
            bad += 1
    return {'consistent': good, 'inconsistent': bad, 'ambiguous': ambiguous}


def main():
    packet = {x['item_id']: x for x in read_lines(ROOT/'blind_packet.jsonl')}
    mapping = json.loads((ROOT/'private_mapping.json').read_text())
    raters = {name: validate(read_lines(ROOT/f'rater_{name}.jsonl'), packet) for name in ('a', 'b')}
    summary = {'status': 'complete_model_preaudit', 'human_annotations': 0,
               'rater_type': 'two independently prompted same-model AI subagents; not human experts',
               'n_unique_texts': len(packet), 'n_student_cases': len(mapping['cases']),
               'n_intervention_samples': len({x['sample_id'] for x in mapping['pairs']}),
               'n_raw_intervention_pairs': len(mapping['pairs']),
               'student_ratings': {}, 'interventions': {}, 'inter_rater': {}}
    all_details = []
    cases = mapping['cases']
    for name, ratings in raters.items():
        selected = [ratings[x['item_id']] for x in cases]
        report = {k: {'distribution': dict(collections.Counter(x[k] for x in selected)),
                       'mean': float(np.mean([x[k] for x in selected])),
                       'score_at_least2': sum(x[k] >= 2 for x in selected),
                       'score3': sum(x[k] == 3 for x in selected)} for k in SCORES}
        report['head_alignment'] = alignment([x['conclusion'] for x in selected], [x['head_pred'] for x in cases])
        report['label_alignment'] = alignment([x['conclusion'] for x in selected], [x['label'] for x in cases])
        report['unverified_attribution_claims'] = sum(x['attribution_claim_unverified'] for x in selected)
        report['explicit_conditional_claims'] = sum(len(x['conditional_claims']) for x in selected)
        summary['student_ratings'][name] = report
    for field in (*SCORES, 'conclusion'):
        summary['inter_rater'][field] = agreement(
            [raters['a'][x['item_id']][field] for x in cases],
            [raters['b'][x['item_id']][field] for x in cases], ordinal=field in SCORES)
    disagreement = []
    for item_id, item in packet.items():
        a, b = raters['a'][item_id], raters['b'][item_id]
        fields = [k for k in (*SCORES, 'conclusion', 'cited_features', 'supporting_features', 'opposing_features', 'dominant_features')
                  if (set(a[k]) != set(b[k]) if isinstance(a[k], list) else a[k] != b[k])]
        if fields:
            disagreement.append({'item_id': item_id, 'fields': fields, 'features': item['features'],
                                 'reason': item['reason'], 'rater_a': a, 'rater_b': b})
    write_json(ROOT/'disagreements.json', disagreement)
    summary['disagreement_items_all294'] = len(disagreement)
    pairs = mapping['pairs']
    no_change = [x for x in pairs if same_value(x['feature'], packet[x['original_id']]['features'][x['feature']], x['replacement'])]
    eligible = [x for x in pairs if x not in no_change]
    summary['n_no_value_change_pairs_excluded'] = len(no_change)
    summary['n_eligible_pairs'] = len(eligible)
    summary['no_change_probability_drift_max'] = max((abs(x['modified_probability']-x['original_probability']) for x in no_change), default=0)
    summary['no_change_class_flips'] = sum((x['original_probability'] >= .5) != (x['modified_probability'] >= .5) for x in no_change)
    summary['head_flips_eligible'] = sum((x['original_probability'] >= .5) != (x['modified_probability'] >= .5) for x in eligible)
    for name, ratings in raters.items():
        detail = []
        for x in eligible:
            a, b = ratings[x['original_id']], ratings[x['modified_id']]
            ca, cb = a['conclusion'], b['conclusion']
            ha, hb = int(x['original_probability'] >= .5), int(x['modified_probability'] >= .5)
            dec = {'accept': 1, 'reject': 0}
            assess = ca in dec and cb in dec
            detail.append({**x, 'rater': name, 'cited': x['feature'] in a['cited_features'],
                'dominant': x['feature'] in a['dominant_features'],
                'abs_probability_change': abs(x['modified_probability']-x['original_probability']),
                'original_conclusion': ca, 'modified_conclusion': cb,
                'head_flip': ha != hb, 'text_flip': dec[ca] != dec[cb] if assess else None,
                'both_conclusions_match_head': assess and dec[ca] == ha and dec[cb] == hb,
                'transition_match': (dec[ca] != dec[cb]) == (ha != hb) if assess else None,
                'modified_teacher_direction': teacher_direction_counts(b, x['modified_teacher_shap'])})
        samples = []
        for sid in sorted({x['sample_id'] for x in detail}):
            group = [x for x in detail if x['sample_id'] == sid]
            cited = [x['abs_probability_change'] for x in group if x['cited']]
            uncited = [x['abs_probability_change'] for x in group if not x['cited']]
            if cited and uncited:
                samples.append({'sample_id': sid, 'cited_mean': float(np.mean(cited)),
                                'uncited_mean': float(np.mean(uncited)),
                                'difference': float(np.mean(cited)-np.mean(uncited))})
        original = {}
        for x in pairs:
            if x['sample_id'] in original:
                assert original[x['sample_id']]['original_id'] == x['original_id']
            original[x['sample_id']] = x
        assessable = [x for x in detail if x['transition_match'] is not None]
        flips = [x for x in detail if x['head_flip']]
        counts = collections.Counter()
        for x in detail:
            counts.update(x['modified_teacher_direction'])
        result = {
            'original_alignment20': alignment([ratings[x['original_id']]['conclusion'] for x in original.values()],
                                             [int(x['original_probability'] >= .5) for x in original.values()]),
            'modified_alignment': alignment([ratings[x['modified_id']]['conclusion'] for x in eligible],
                                            [int(x['modified_probability'] >= .5) for x in eligible]),
            'transition_assessable': len(assessable),
            'transition_matches': sum(x['transition_match'] for x in assessable),
            'both_conclusions_match_head_pairs': sum(x['both_conclusions_match_head'] for x in detail),
            'head_flip_pairs': len(flips), 'head_flip_text_flip_pairs': sum(x['text_flip'] is True for x in flips),
            'head_flip_both_aligned': sum(x['both_conclusions_match_head'] for x in flips),
            'cited_probability_change': cluster_mean_interval([x['cited_mean'] for x in samples]),
            'uncited_probability_change': cluster_mean_interval([x['uncited_mean'] for x in samples]),
            'paired_cited_minus_uncited': cluster_mean_interval([x['difference'] for x in samples]),
            'per_sample': samples,
            'modified_teacher_direction_statements': dict(counts),
            'teacher_alignment_scope': 'textual sign claims versus separate XGBoost; not student attribution'}
        summary['interventions'][name] = result
        all_details.extend(detail)
    write_json(ROOT/'intervention_details.json', all_details)
    write_json(ROOT/'summary.json', summary)
    human = ROOT/'human_review';human.mkdir(exist_ok=True)
    columns = ['item_id', *FEATURES, 'reason', 'reviewer_id', 'fact_score', 'grounding_score',
               'coherence_score', 'conclusion', 'cited_features', 'supporting_features',
               'opposing_features', 'dominant_features', 'conditional_claims', 'notes', 'review_seconds']
    for reviewer in (1, 2):
        p = human/f'independent_reviewer_{reviewer}.csv'
        if p.exists():
            continue
        with p.open('w', encoding='utf-8-sig', newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
            for item in packet.values():
                w.writerow({'item_id':item['item_id'],**item['features'],'reason':item['reason']})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
