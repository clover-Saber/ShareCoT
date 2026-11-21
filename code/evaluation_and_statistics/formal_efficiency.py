import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

from fresh_queue import atomic, stamp

METHODS = ('adaptive', 'direct_ft', 'single_cot', 'logistic_regression',
    'random_forest', 'xgboost', 'logit_boost', 'gradient_boosting',
    'textcnn', 'transformer', 'bert', 'adaptive_reason_generation')
TEXT = ('textcnn', 'transformer', 'bert')
QWEN = ('adaptive', 'direct_ft', 'single_cot', 'adaptive_reason_generation')
SAMPLE_COUNT, REPEATS, WARMUP = 30, 3, 5


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def percentile(values, quantile):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def summarize(seconds):
    if not seconds or any(not math.isfinite(s) or s <= 0 for s in seconds):
        raise ValueError('Timing must contain finite positive observed seconds')
    return {'n_requests': len(seconds), 'mean_seconds': statistics.mean(seconds),
        'median_seconds': statistics.median(seconds), 'p95_seconds': percentile(seconds, .95),
        'sequential_batch1_requests_per_second': len(seconds) / sum(seconds),
        'rate_definition': 'observed sequential batch1 requests / summed request latency; not saturation throughput'}


def sample_ids(ids):
    ids = list(ids)
    if len(ids) < SAMPLE_COUNT or len(set(ids)) != len(ids) or any(not str(i) for i in ids):
        raise ValueError('Sampling requires at least30 unique nonempty IDs')
    return random.Random(44).sample(ids, SAMPLE_COUNT)


def write_raw(path, rows):
    path = Path(path)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['method', 'mode', 'sample_id', 'repeat', 'seconds',
            'generated_output_tokens', 'generated_non_eos_tokens', 'ends_with_eos'])
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_loaded_gpu_uuid(expected, loaded):
    if loaded is None or str(expected).removeprefix('GPU-').lower() != str(loaded).removeprefix('GPU-').lower():
        raise ValueError('Actual loaded GPU UUID differs from pinned physicalGPU0 UUID')
    return True


def generated_token_receipt(sequence, prompt_tokens, eos_token_id):
    if prompt_tokens < 0 or len(sequence) < prompt_tokens:
        raise ValueError('Generated sequence shorter than its native prompt')
    output = sequence[prompt_tokens:]
    eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
    return {'generated_output_tokens':len(output),
        'generated_non_eos_tokens':sum(token not in eos_ids for token in output),
        'ends_with_eos':bool(output and output[-1] in eos_ids)}


def checkpoint_files(result, method):
    from evaluate_suite import selected_adapter
    files = [result / 'run_config.json', result / 'metrics.json', result / 'predictions.csv']
    if method in QWEN:
        files += [p for p in selected_adapter(result).rglob('*') if p.is_file()]
        for name in ('selected_checkpoint.json', 'serialized_reload_verification.json', 'reuse.json'):
            if (result / name).is_file():
                files.append(result / name)
    elif method in TEXT:
        files += [result / 'model.pt']
        files += [p for name in ('tokenizer', 'bert_config') for p in (result / name).rglob('*') if p.is_file()]
    else:
        files += [result / 'model.joblib', result / 'preprocessor.joblib']
    return sorted(set(files))


def environment(torch, gpu):
    versions = {}
    for name in ('numpy', 'pandas', 'scikit-learn', 'torch', 'transformers', 'peft', 'xgboost', 'joblib', 'safetensors'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpus = []
    if Path('/proc/cpuinfo').is_file():
        cpus = sorted(set(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
            if line.startswith('model name')))
    info = {'hostname': platform.node(), 'platform': platform.platform(), 'python': sys.version,
        'cpu_models': cpus, 'logical_cpu_count': os.cpu_count(), 'versions': versions,
        'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads(),
        'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'TOKENIZERS_PARALLELISM')}, 'cuda_runtime': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
    if gpu:
        properties = torch.cuda.get_device_properties(0)
        info['gpu'] = {'logical_device': 'cuda:0', 'name': properties.name,
            'loaded_device_uuid':str(getattr(properties, 'uuid', None)),
            'total_memory_bytes': properties.total_memory, 'compute_capability': [properties.major, properties.minor]}
        driver = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, check=True)
        info['nvidia_smi_gpu_identity'] = driver.stdout.strip()
    return info


def parameter_accounting(model, result, method):
    info = {'num_parameters': None, 'loaded_requires_grad_parameters': None,
        'parameter_dtype_elements': None, 'serialized_selected_trainable_tensor_elements': None,
        'quantization': 'none', 'parameter_count_note': 'Classical parameter counting not comparable to neural weights'}
    if model is not None:
        dtypes = {}
        for p in model.parameters():
            dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
        info.update(num_parameters=sum(p.numel() for p in model.parameters()),
            loaded_requires_grad_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            parameter_dtype_elements=dtypes,
            parameter_count_note='Counts observed loaded parameters; requires_grad at inference is not a reconstructed historical training count')
        if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
            raise ValueError('Matched protocol forbids quantized checkpoints')
    if method in QWEN:
        from evaluate_suite import selected_adapter
        from safetensors import safe_open
        adapter = selected_adapter(result) / 'adapter_model.safetensors'
        with safe_open(adapter, framework='pt', device='cpu') as handle:
            info['serialized_selected_trainable_tensor_elements'] = sum(math.prod(handle.get_slice(k).get_shape()) for k in handle.keys())
        info['selected_trainable_count_definition'] = 'Elements actually present in selected adapter safetensors, including serialized modules; optimizer-only loss variables may be absent'
        base = read(result / 'run_config.json')
        directory = Path(base.get('model_dir', base.get('base_model_dir')))
        weights = sorted(directory.glob('*.safetensors'))
        info['base_model_weight_storage_bytes'] = sum(p.stat().st_size for p in weights)
        info['base_model_weight_storage_files'] = [str(p) for p in weights]
    files = checkpoint_files(result, method)
    info['selected_checkpoint_storage_bytes'] = sum(p.stat().st_size for p in files
        if p.suffix in ('.safetensors', '.pt', '.joblib'))
    info['storage_definition'] = 'Selected serialized model/preprocessor files only; excludes epoch archives, caches and tokenizer files; shared base separately reported'
    return info


def assert_probabilities(actual, saved, frame, field):
    import numpy as np
    if saved.sample_id.duplicated().any() or frame.sample_id.duplicated().any() or set(saved.sample_id) != set(frame.sample_id):
        raise ValueError('Reload verification requires identical unique formal test IDs')
    aligned = saved.set_index('sample_id').loc[frame.sample_id]
    if len(actual) != len(frame) or not np.allclose(actual, aligned[field].to_numpy(), atol=1e-5, rtol=0):
        raise ValueError('Reloaded selected probabilities differ from original full-test batch inference')
    from protocol import TARGET
    if not np.array_equal(frame[TARGET].astype(int).to_numpy(), aligned.y_true.astype(int).to_numpy()):
        raise ValueError('Saved labels differ from formal test membership')
    return float(np.max(np.abs(actual - aligned[field].to_numpy())))


def text_probability_receipt(actual, reference, saved, frame, batch, atol, scope, device):
    import numpy as np
    actual, reference = np.asarray(actual, dtype=float), np.asarray(reference, dtype=float)
    if (actual.shape != (len(frame),) or reference.shape != actual.shape
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()
            or ((actual < 0) | (actual > 1) | (reference < 0) | (reference > 1)).any()):
        raise ValueError('Text reload comparison requires finite probability vectors in [0,1]')
    classes = saved.set_index('sample_id').loc[frame.sample_id, 'y_pred'].to_numpy()
    if (not np.isin(classes, [0, 1]).all() or not np.array_equal(actual >= .5, classes)
            or not np.array_equal(reference >= .5, classes)):
        raise ValueError('Text reload comparison changed saved threshold class labels')
    error = np.abs(actual - reference)
    if not np.allclose(actual, reference, atol=atol, rtol=0):
        raise ValueError('Text probability comparison exceeded its bounded tolerance: ' + scope)
    return {'verified': True, 'rows': len(frame), 'original_eval_batch_size': batch,
        'scope': scope, 'inference_device': device, 'prob_atol': atol, 'prob_rtol': 0,
        'threshold': .5, 'label_mismatches': 0, 'max_absolute_error': float(error.max()),
        'median_absolute_error': float(np.median(error)), 'p99_absolute_error': float(np.percentile(error, 99)),
        'count_above_1e_minus_5': int((error > 1e-5).sum())}


def verify_text_reload(result, method, config, saved, frame):
    from protocol import FEATURE_COLS
    from evaluate_suite import neural_baseline_predictor
    from baselines_text_suite import load_model
    batch = config['hyperparameters']['batch_size']
    if config.get('device') == 'cpu':
        cpu_predictor = load_model(result, device='cpu', batch_size=batch)
        native_probabilities = cpu_predictor(frame[FEATURE_COLS])
        assert_probabilities(native_probabilities, saved, frame, 'prob_1')
        saved_probabilities = saved.set_index('sample_id').loc[frame.sample_id, 'prob_1'].to_numpy()
        native_receipt = text_probability_receipt(native_probabilities, saved_probabilities, saved, frame,
            batch, 1e-5, 'Native CPU reload versus originally saved CPU probabilities', 'cpu')
        del cpu_predictor
    model, predictor = neural_baseline_predictor(result, method, device='cuda:0')
    predictor.batch_size = batch
    probabilities = predictor(frame[FEATURE_COLS])
    if config.get('device') == 'cpu':
        verification = {'rows': len(frame), 'original_eval_batch_size': batch,
            'native_probability_verification': native_receipt,
            'cross_device_portability': text_probability_receipt(probabilities, native_probabilities, saved, frame,
                batch, 1e-4, 'GPU portability versus strictly verified CPU reload; unchanged threshold decisions', 'cuda:0')}
    else:
        maximum_error = assert_probabilities(probabilities, saved, frame, 'prob_1')
        verification = {'rows': len(frame), 'original_eval_batch_size': batch,
            'prob_atol': 1e-5, 'prob_rtol': 0, 'max_absolute_error': maximum_error}
    verification.update(csv_parser={'dtype': 'str', 'keep_default_na': False}, serialization=config['serialization'])
    return model, predictor, verification


def measure_method(args):
    import numpy as np
    import pandas as pd
    import resource
    import torch
    from protocol import FEATURE_COLS, build_prompt
    from evaluate_suite import load_qwen, classify, generate_reason, neural_baseline_predictor
    from baselines_text_suite import load_model, rows_to_texts, token_lengths
    from run_formal_efficiency_seed44 import verify_snapshot, require_gpu_zero_idle
    suite, method = Path(args.suite).resolve(), args.method
    ready, _, _ = verify_snapshot(suite)
    if not ready:
        raise ValueError('Main phase is not verified idle complete')
    out, main = suite / 'experiments/03_efficiency', suite / 'experiments/02_main'
    result = main / 'results' / ('sharecot' if method in ('adaptive', 'adaptive_reason_generation') else method)
    if method in ('adaptive', 'adaptive_reason_generation', 'direct_ft'):
        formal = read(suite / 'formal_dataset.json')
        result = Path(formal['formal_main_models']['sharecot' if method != 'direct_ft' else 'direct_ft']['source_result'])
    config = read(result / 'run_config.json')
    if config['seed'] != 44:
        raise ValueError('Only seed44 selected models permitted')
    native_csv = dict(dtype=str, keep_default_na=False) if method in TEXT else dict(dtype={'sample_id': str})
    frame = pd.read_csv(main / 'data/test.csv', **native_csv)
    if len(frame) != 1000:
        raise ValueError('Formal test must have1000 rows')
    selected = sample_ids(frame.sample_id.tolist())
    inputs = frame[['sample_id', *FEATURE_COLS]].set_index('sample_id', drop=False)
    sample = inputs.loc[selected]
    saved = pd.read_csv(result / 'predictions.csv', dtype={'sample_id': str}, keep_default_na=False)
    gpu = method in QWEN or method in TEXT
    gpu_uuid = require_gpu_zero_idle()
    if gpu:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid or not torch.cuda.is_available():
            raise ValueError('All neural methods require UUID-pinned physicalGPU0 visibility')
        verify_loaded_gpu_uuid(gpu_uuid, getattr(torch.cuda.get_device_properties(0), 'uuid', None))
    model = None
    token_audit = None
    generation = method in ('single_cot', 'adaptive_reason_generation')
    mode = 'single_cot_autoregressive_json' if method == 'single_cot' else ('sharecot_reason_generation' if generation else 'classification_forward')
    if method in QWEN:
        model, tokenizer, device = load_qwen(result, device='cuda:0', classification=method != 'single_cot')
        from train_singlecot import structured_prompt
        prompts = [structured_prompt(row, tokenizer) if method == 'single_cot' else build_prompt(dict(row), tokenizer)
            for _, row in frame[FEATURE_COLS].iterrows()]
        lengths = [len(tokenizer(p, add_special_tokens=False)['input_ids']) for p in prompts]
        cap = None if generation else 512
        token_audit = {'native_input_lengths': lengths, 'add_special_tokens': False,
            'input_limit': cap, 'truncation': not generation,
            'generation_max_new_tokens': 256 if generation else 0,
            'do_sample': False if generation else None}
        if method == 'single_cot' and max(lengths) > 512:
            raise ValueError('Single-CoT native prompt exceeds registered512 limit')
        if method != 'single_cot':
            batch = config.get('eval_batch_size', 8)
            maximum_error = assert_probabilities(classify(model, tokenizer, device, frame[FEATURE_COLS], batch=batch), saved, frame, 'prob')
            verification = {'rows': len(frame), 'original_eval_batch_size': batch, 'prob_atol': 1e-5,
                'prob_rtol': 0, 'max_absolute_error': maximum_error}
        else:
            receipt = read(result / 'serialized_reload_verification.json')
            if receipt.get('verified') is not True:
                raise ValueError('Single-CoT serialized selected tensor receipt invalid')
            if saved.sample_id.duplicated().any() or set(saved.sample_id) != set(frame.sample_id):
                raise ValueError('Single-CoT saved test IDs mismatch')
            saved_index = saved.set_index('sample_id')
            from protocol import TARGET
            if not np.array_equal(frame[TARGET].astype(int).to_numpy(),
                    saved_index.loc[frame.sample_id, 'y_true'].astype(int).to_numpy()):
                raise ValueError('Single-CoT saved labels differ from formal test')
            checked = selected[:3]
            for sid in checked:
                raw = generate_reason(model, tokenizer, device, inputs.loc[sid], single_cot=True)
                if raw != saved_index.loc[sid, 'raw_generation']:
                    raise ValueError('Deterministic selected Single-CoT reload output differs: ' + sid)
            verification = {'serialized_tensor_receipt': receipt, 'deterministic_generation_checked_ids': checked,
                'generation_batch_size': 1, 'raw_output_exact_match': True,
                'full1000_generation_repeated': False}
        def predict(one):
            if generation:
                return generate_reason(model, tokenizer, device, one.iloc[0], single_cot=method == 'single_cot')
            return classify(model, tokenizer, device, one, batch=1)
    elif method in TEXT:
        model, predictor, verification = verify_text_reload(result, method, config, saved, frame)
        lengths = token_lengths(predictor.tokenizer, rows_to_texts(frame[FEATURE_COLS]))
        token_audit = {'native_input_lengths': lengths, 'add_special_tokens': True,
            'input_limit': predictor.config['max_length'], 'truncation': False, 'generation_max_new_tokens': 0}
        if max(lengths) > predictor.config['max_length']:
            raise ValueError('Native text input exceeds saved cap')
        predictor.batch_size = 1
        predict = predictor
    else:
        predict = load_model(result, device='cpu')
        maximum_error = assert_probabilities(predict(frame[FEATURE_COLS]), saved, frame, 'prob_1')
        verification = {'rows': len(frame), 'original_eval_batch_size': 'full_test_cpu_array',
            'prob_atol': 1e-5, 'prob_rtol': 0, 'max_absolute_error': maximum_error}
    if token_audit is not None:
        token_audit.update(n=len(lengths), min=min(lengths), max=max(lengths),
            mean=statistics.mean(lengths), p50=percentile(lengths,.5), p95=percentile(lengths,.95), p99=percentile(lengths,.99),
            would_exceed_input_limit=sum(n > token_audit['input_limit'] for n in lengths) if token_audit['input_limit'] else 0)
    generation_capture = {}
    if generation:
        native_generate = model.generate
        def capture_generate(*positional, **keyword):
            ids = native_generate(*positional, **keyword)
            generation_capture.update(ids=ids, prompt_tokens=keyword['input_ids'].shape[1])
            return ids
        model.generate = capture_generate
    sync = (lambda: torch.cuda.synchronize(0)) if gpu else (lambda: None)
    if gpu:
        torch.cuda.empty_cache()
    with torch.inference_mode():
        for sid in selected[:WARMUP]:
            predict(sample.loc[[sid], FEATURE_COLS])
        sync()
        if gpu:
            torch.cuda.reset_peak_memory_stats(0)
        raw = []
        for repeat in range(REPEATS):
            for sid in selected:
                sync()
                start = time.perf_counter()
                predict(sample.loc[[sid], FEATURE_COLS])
                sync()
                record = dict(method=method, mode=mode, sample_id=sid, repeat=repeat,
                    seconds=time.perf_counter() - start)
                if generation:
                    record.update(generated_token_receipt(generation_capture['ids'][0].tolist(),
                        generation_capture['prompt_tokens'], tokenizer.eos_token_id))
                raw.append(record)
    metadata = {'method': method, 'mode': mode, 'seed':44, 'device': 'cuda:0' if gpu else 'cpu',
        'batch_size':1, 'sample_ids': selected, 'sampling': 'random.Random(44), without replacement; IDs only, no label/performance selection',
        'physical_gpu0_uuid':gpu_uuid,
        'repetitions':REPEATS, 'warmup_queries':WARMUP, 'checkpoint':str(result),
        'verification':verification, 'input_token_audit':token_audit, 'environment':environment(torch, gpu),
        'parameters':parameter_accounting(model, result, method),
        'model_load_excluded':True, 'inference_mode':True, 'timing_clock':'time.perf_counter',
        'includes':'single-row selection/DataFrame construction, allowed-feature preprocessing/prompt formatting, tokenization, H2D, forward and probability conversion or generation and decoding; sync before/after each GPU request',
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(0) if gpu else None,
        'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(0) if gpu else None,
        'cuda_peak_definition':'Reset after warmup; includes loaded model baseline and observed batch1 requests; larger verification-batch cache released before warmup',
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'rss_definition':'Linux process high-water resident memory, includes Python/libraries/model loading and verification; not GPU memory or standalone model weight memory',
        **summarize([r['seconds'] for r in raw])}
    if generation:
        counts = [r['generated_output_tokens'] for r in raw]
        metadata['generated_output_token_audit'] = {'counts':counts, 'n':len(counts),
            'min':min(counts), 'max':max(counts), 'mean':statistics.mean(counts),
            'p50':percentile(counts,.5), 'p95':percentile(counts,.95),
            'ended_with_eos_requests':sum(r['ends_with_eos'] for r in raw),
            'definition':'Actual model.generate returned token IDs after native input prefix; includes emitted EOS; no decoded-text retokenization; count extracted after measured timer stops',
            'max_new_tokens_limit':256}
    write_raw(out / (method + '_raw.csv'), raw)
    atomic(out / (method + '.json'), metadata)


def run(args):
    suite = Path(args.suite).resolve()
    out = suite / 'experiments/03_efficiency'
    out.mkdir(parents=True, exist_ok=True)
    from run_formal_efficiency_seed44 import require_gpu_zero_idle
    gpu_uuid = require_gpu_zero_idle()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('Benchmark parent lacks pinned physicalGPU0 UUID visibility')
    atomic(out / 'run_config.json', {'seed':44, 'methods':list(METHODS), 'sample_count':SAMPLE_COUNT,
        'physical_gpu0_uuid':gpu_uuid, 'cuda_visible_devices':gpu_uuid,
        'repeats':REPEATS, 'warmup':WARMUP, 'batch_size':1,
        'historical_holdout_reuse':True, 'training_or_selection_performed':False,
        'test_labels_role':'saved prediction/membership verification only; never model input', 'started_at':stamp()})
    results, combined = [], []
    for method in METHODS:
        if require_gpu_zero_idle() != gpu_uuid:
            raise ValueError('PhysicalGPU0 UUID changed during benchmark')
        argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--suite', str(suite),
            '--method', method]
        progress = {'status':'launching', 'method':method, 'worker_argv':argv,
            'completed_methods':[r['method'] for r in results], 'updated_at':stamp()}
        atomic(out / 'progress.json', progress)
        worker = subprocess.Popen(argv, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid, TOKENIZERS_PARALLELISM='false'))
        from fresh_queue import ticks
        progress.update(status='running', worker_pid=worker.pid, worker_process_start_ticks=ticks(worker.pid), updated_at=stamp())
        atomic(out / 'progress.json', progress)
        code = worker.wait()
        if code != 0:
            progress.update(status='failed', returncode=code, updated_at=stamp())
            atomic(out / 'progress.json', progress)
            raise ValueError('Efficiency worker failed: ' + method + ', exit=' + str(code))
        entry = read(out / (method + '.json'))
        results.append(entry)
        with (out / (method + '_raw.csv')).open(encoding='utf-8', newline='') as handle:
            combined.extend(dict(r, repeat=int(r['repeat']), seconds=float(r['seconds'])) for r in csv.DictReader(handle))
    write_raw(out / 'raw_timings.csv', combined)
    fields = ['method', 'mode', 'device', 'n_requests', 'mean_seconds', 'median_seconds', 'p95_seconds',
        'sequential_batch1_requests_per_second', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes', 'process_peak_rss_bytes']
    with (out / 'summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in results)
    atomic(out / 'summary.json', {'status':'complete', 'results':results, 'seed':44,
        'classification_forward_methods':[m for m in METHODS if m not in ('single_cot','adaptive_reason_generation')],
        'generation_rows':['single_cot','adaptive_reason_generation'],
        'comparison_note':'Classification forward and autoregressive generation are distinct task modes; sequential batch1 rate is not saturation throughput',
        'costs_or_human_time':'Not measured; no invented API/training/human cost',
        'completed_at':stamp()})
    atomic(out / 'progress.json', {'status':'complete','completed_methods':list(METHODS),'updated_at':stamp()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--method', choices=METHODS)
    args = parser.parse_args()
    if args.method:
        measure_method(args)
    else:
        run(args)
