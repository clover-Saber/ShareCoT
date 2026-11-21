import argparse
import json
import os
from pathlib import Path

from fresh_queue import atomic, validate_jobs

VERSION = 'paper_continuation_seed44_20261009_v1'
MISSING_METHODS = ('logistic_regression', 'random_forest', 'logit_boost',
                   'gradient_boosting', 'textcnn', 'bert', 'single_cot')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def paper_phases():
    rows = [
        ('5.1 Setup', 'preserved', 'Formal swapped holdouts; train8000/val1000/test1000; nine no-tip inputs; final946 corpus.'),
        ('5.2 Fig 4', 'ready', 'Export the frozen training-only teacher SHAP; teacher seed42 is separate from student seed44. No all-candidate feature selection claim.'),
        ('5.3.1 Table 4 + Fig 5', 'ready', 'Seven missing main methods sequentially; preserve ShareCoT, DirectFT, XGBoost and text Transformer.'),
        ('5.3.2 Table 5', 'pending', 'Matched inference timing and parameter/trainable-parameter accounting; existing artifacts must be audited before reporting.'),
        ('5.4.1 Table 6', 'pending', '50K/100K splits require overlap audit of teacher SHAP fitting rows versus holdouts, chronology check and split-specific corpus. Do not blindly reuse final946. Add matched DirectFT comparator.'),
        ('5.4.2 Table 7', 'pending', 'Chronological time-shift splits, training-only teacher/corpus and matched DirectFT comparator.'),
        ('5.4.3 Table 8', 'pending', 'Disjoint regional protocol, train-only teacher/corpus and matched DirectFT comparator.'),
        ('5.4.4 Fig 6', 'pending', 'Local 1.5B/3B backbones and train_size.py available; prepare new variant roots only after earlier paper gates. Reuse primary7B; verify tokenizer/BPE and provenance.'),
        ('5.4.5 Table 9', 'pending', 'CoT-only NoSHAP/raw-feature/LogitSHAP/XGB-final construction needs separately verified corpora and explicit accepted membership; XGB-final reuses final946 only in matched formal split.'),
        ('5.4.6 Fig 7', 'pending', 'Architecture comparison reuses adopted main44 and completed SingleCoT; no duplicate training.'),
        ('5.4.7 Table 10', 'pending', 'Fixed1/3/5 via train_llm.py --weighting fixed_1/fixed_3/fixed_5 in new roots; reuse adaptive44. Same initialization/protocol and corpus; blocked by earlier gates.'),
        ('5.4.8 Cases', 'pending', 'Choose cases from persisted predictions without altering training or selecting checkpoints on test.'),
        ('Reviewer student rationales / intervention / expert agreement', 'pending', 'Reuse completed shuffle44 and64-condition noise; audit student rationale and intervention requirements. Real human expert annotation count is0; do not label automated agreement as expert agreement.'),
        ('Paired statistics', 'pending', 'Pair by sample_id on identical formal test membership; retain43/44/50 evidence only. Separate conditional split statistics from historical holdout reuse limitation.'),
        ('API prediction baselines (11)', 'pending', 'Deferred by user; no API tasks or credentials in this continuation.'),
    ]
    return [{'paper_location': location, 'status': status, 'requirement': reason,
             'depends_on_previous_paper_phase': i > 0} for i, (location, status, reason) in enumerate(rows)]


def make_manifest(suite, model_dir, bert_model):
    suite = Path(suite).resolve()
    code = Path(__file__).resolve().parents[1]
    models = code / 'models_and_baselines'
    main = suite / 'experiments/02_main'
    data = main / 'data'
    formal = read(suite / 'formal_dataset.json')
    if (formal['training_seed'], formal['primary_weighting'], formal['accepted_rationales']) != (44, 'adaptive', 946):
        raise ValueError('Formal primary protocol differs')
    primary = {}
    for method in ('sharecot', 'direct_ft'):
        entry = formal['formal_main_models'][method]
        if entry['new_training_required'] is not False:
            raise ValueError('Adopted model unexpectedly requests training')
        result = Path(entry['source_result'])
        checkpoint = sorted(p for p in (result / 'model').rglob('*') if p.is_file())
        if not checkpoint:
            raise ValueError('Adopted checkpoint is missing: ' + str(result))
        primary[method] = {'source_result': str(result)}
    jobs = []
    feature = suite / 'experiments/01_features'
    feature_manifest = feature / 'manifest.json'
    jobs.append({'id': 'features_frozen_teacher44', 'kind': 'evaluation', 'resource': 'cpu',
                 'script': str(code / 'data_and_rationales/export_features.py'),
                 'args': ['--suite', str(suite)], 'depends': [],
                 'code_dependencies': [str(models / 'protocol.py')],
                 'inputs': [str(data / name) for name in ('train.csv', 'shap_model.json', 'shap_metadata.json')],
                 'minimum_free_bytes': 2 * 1024**3,
                 'artifacts': [str(feature / name) for name in ('global_importance.csv', 'train_local_shap.csv', 'manifest.json')],
                 'expected_json': {str(feature_manifest): {'fit_split': 'train_only', 'training_rows': 8000, 'student_seed': 44, 'export_complete': True}},
                 'terminal_contract': {'path': str(feature_manifest), 'metrics_path': str(feature_manifest),
                                       'expected': {'status': 'complete'}, 'required_fields': ['features']}})
    for index, method in enumerate(MISSING_METHODS):
        result = main / 'results' / method
        if result.exists() and any(result.iterdir()):
            raise ValueError('Existing result requires reconciliation, refusing new queue overwrite: ' + str(result))
        config, metrics = result / 'run_config.json', result / 'metrics.json'
        single, text = method == 'single_cot', method in ('textcnn', 'bert')
        script = models / ('train_singlecot.py' if single else 'baselines_text_suite.py' if text else 'baselines_suite.py')
        deps = [str(models / 'protocol.py'), str(models / 'baselines_suite.py')]
        inputs = [str(data / (s + '.csv')) for s in ('train', 'val', 'test')]
        expected = {'method': method, 'seed': 44, 'threshold': .5}
        terminal = {'path': str(metrics), 'metrics_path': str(metrics),
                    'expected': {'status': 'complete', 'method': method, 'seed': 44},
                    'required_fields': ['provenance', 'test']}
        artifacts = [str(config), str(metrics), str(result / 'predictions.csv'), str(result / 'validation_predictions.csv')]
        args = ['--root', str(main), '--seed', '44']
        job = {'id': 'main44_' + method, 'kind': 'training', 'resource': 'gpu0' if single or text else 'cpu',
               'script': str(script), 'depends': [jobs[-1]['id']], 'priority': index + 1,
               'minimum_free_bytes': 3 * 1024**3 if single or text else 2 * 1024**3}
        if single:
            args += ['--model-dir', str(model_dir), '--epochs', '5', '--checkpoint-steps', '100']
            inputs += [str(data / 'rationales.jsonl')]
            deps += [str(models / 'train_llm.py'), str(models / 'loss_weighting.py')]
            expected.update(epochs=5, max_steps=None, selection_metric='validation_macro_f1', classification_head=False)
            artifacts += [str(result / name) for name in ('logs/progress.json', 'validation_history.json',
                          'model/adapter_model.safetensors', 'model/adapter_config.json', 'serialized_reload_verification.json')]
            terminal = {'path': str(result / 'logs/progress.json'), 'metrics_path': str(metrics), 'metrics_field': 'metrics',
                        'expected': {'status': 'complete', 'method': method, 'seed': 44, 'trained_steps': 5000},
                        'required_fields': ['metrics', 'checkpoint']}
            job.update(expected_steps=5000, required_metrics={'path': str(metrics), 'required_fields': ['macro_f1', 'accuracy', 'trained_steps']})
        else:
            args += ['--method', method, '--device', 'cuda:0' if text else 'cpu']
            artifacts += [str(result / ('model.pt' if text else 'model.joblib'))]
            if text:
                args += ['--max-length', '256', '--bert-model', str(bert_model)]
                expected.update(runner='baselines_text_suite', serialization='english-nine-raw-values-units-v1', neural_options={'max_length': 256})
            job['required_metrics'] = {'path': str(metrics), 'required_fields': ['test.macro_f1', 'val.macro_f1']}
        job.update(args=args, inputs=inputs, code_dependencies=deps, run_config=str(config), artifacts=artifacts,
                   expected_json={str(config): expected, str(metrics): {'method': method, 'seed': 44, **({'test_n': 1000, 'smoke_run': False} if single else {'status': 'complete', 'test_rows': 1000})}},
                   terminal_contract=terminal)
        jobs.append(job)
    manifest = {'version': VERSION, 'suite': str(suite), 'implementation_complete': False,
                'primary_seed': 44, 'primary_weighting': 'adaptive', 'currentValidSeeds': [43, 44, 50],
                'api_prediction_baselines_excluded': True, 'human_expert_annotations': 0,
                'initial_primary_checkpoints': primary,
                'completed_preserved': ['sharecot44', 'direct_ft44', 'xgboost44', 'transformer44', 'shuffle44', 'noise64', 'seeds43/44/50'],
                'paper_phases': paper_phases(), 'jobs': jobs}
    validate_jobs(jobs)
    return manifest


def runtime_directory(suite):
    return (Path(suite) / os.environ.get('SHARECOT_RUNTIME_DIR', 'continuation')).resolve()


def build(suite, model_dir=None, bert_model=None):
    suite = Path(suite).resolve()
    root = runtime_directory(suite)
    queue = root / 'execution_queue.json'
    if queue.exists():
        manifest = read(queue)
        if manifest['version'] != VERSION:
            raise ValueError('Existing continuation version differs')
        print('Existing continuation retained; use its controller to resume:', root)
        return root
    if model_dir is None or bert_model is None:
        raise ValueError('--model-dir and --bert-model are required to create a new queue')
    manifest = make_manifest(suite, Path(model_dir).resolve(), Path(bert_model).resolve())
    atomic(queue, manifest)
    atomic(root / 'execution_status.json', {'stage': 'prepared_not_started', 'runs': {},
           'primary_seed': 44, 'primary_weighting': 'adaptive', 'currentValidSeeds': [43, 44, 50]})
    active = {'version': VERSION, 'runtime_root': str(root),
           'queue': str(queue), 'status': str(root / 'execution_status.json'),
           'stage': 'prepared_not_started', 'currentValidSeeds': [43, 44, 50]}
    if controller := os.environ.get('SHARECOT_CONTROLLER_SCRIPT'):
        active['controller'] = str((suite / controller).resolve())
    atomic(suite / ('active_' + root.name + '.json'), active)
    print('Prepared eight sequential tasks; later paper sections pending:', root)
    return root


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--bert-model', type=Path)
    args = parser.parse_args()
    build(args.suite, args.model_dir, args.bert_model)
