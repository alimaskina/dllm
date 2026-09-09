#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from bitsieve_fastdllm.eval.benchmarks import max_new_tokens_for  # noqa: E402

METHODS = (
    'official_dense_bf16',
    'dense_kivi4_k4v4_r0',
    'dense_kivi2_k2v2_r0',
    'mage_bf16_all_a_k512',
    'herald_middle_bf16_a',
    'proposed_a_k4v4_k512',
    'proposed_a_k2v2_k512',
    'proposed_a_k4v4_p5',
    'proposed_a_k2v2_p5',
)
LABELS = {
    'official_dense_bf16': 'Dense BF16',
    'dense_kivi4_k4v4_r0': 'KIVI-style K4/V4',
    'dense_kivi2_k2v2_r0': 'KIVI-style K2/V2',
    'mage_bf16_all_a_k512': 'MAGE BF16, k512',
    'herald_middle_bf16_a': 'HERALD-middle proxy, k512',
    'herald_middle1_bf16_proxy_semantic_a': 'HERALD-middle proxy, k512',
    'proposed_a_k4v4_k512': 'Proposed A K4/V4, k512',
    'proposed_a_k2v2_k512': 'Proposed A K2/V2, k512',
    'proposed_a_k4v4_p5': 'Proposed A K4/V4, 5% budget',
    'proposed_a_k2v2_p5': 'Proposed A K2/V2, 5% budget',
}
BENCHMARKS = (
    'gsm8k', 'math500', 'hotpotqa', 'narrativeqa', 'qasper', 'qmsum',
    'lcc', 'repobench-p', 'triviaqa', '2wikimqa', 'musique', 'niah',
)
STOP = threading.Event()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_rows(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f'{path}:{number}: expected a JSON object')
        rows.append(value)
    return rows


def csv_items(text):
    values = [s.strip() for s in text.split(',') if s.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError('Use a nonempty, comma-separated list without duplicates.')
    return values


def positive_int(text):
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError('Expected a positive integer.')
    return value


def positive_ints(text):
    return [positive_int(x) for x in csv_items(text)]


def parser():
    p = argparse.ArgumentParser(description='Run and summarize the retained BitSieve experiments.')
    sub = p.add_subparsers(dest='action', required=True)
    for action in ('quality', 'performance', 'memory'):
        q = sub.add_parser(action)
        q.add_argument('--configs', type=csv_items, default=list(METHODS))
        q.add_argument('--config-root', type=Path, default=ROOT / 'configs')
        q.add_argument('--gpus', type=csv_items)
        q.add_argument('--output-root', type=Path, default=ROOT / 'results' / 'release_runs')
        q.add_argument('--model', default='Efficient-Large-Model/Fast_dLLM_v2_7B')
        q.add_argument('--revision', default='0661abf5f9f0ee338970d091052a26c8efa51974')
        q.add_argument('--dtype', choices=('bf16', 'fp16'), default='bf16')
        q.add_argument('--attn-implementation')
        q.add_argument('--max-new-tokens', type=positive_int)
        q.add_argument('--max-cache-tokens', type=positive_int, default=32768)
        q.add_argument('--retry-failed', action='store_true')
        q.add_argument('--dry-run', action='store_true')
        if action == 'quality':
            q.add_argument('--benchmarks', type=csv_items,
                           default=['gsm8k', 'hotpotqa', 'narrativeqa', 'qasper', 'qmsum', 'lcc', 'repobench-p'])
            q.add_argument('--limit', type=positive_int)
            q.add_argument('--plain-prompt', action='store_true')
        else:
            q.add_argument('--contexts', type=positive_ints, default=[512, 2048, 8192, 16384, 28672])
            q.add_argument('--batch-sizes', type=positive_ints, default=[1, 4, 8, 16, 32] if action == 'memory' else [1, 4])
            q.add_argument('--capacity', choices=('fixed', 'workload'),
                           default='workload' if action == 'memory' else 'fixed')
            q.add_argument('--fixed-steps', type=positive_int, default=20)
            q.add_argument('--warmup', type=int, default=2)
            q.add_argument('--repeats', type=positive_int, default=3)
    q = sub.add_parser('summarize')
    q.add_argument('--input', type=Path, action='append', required=True)
    q.add_argument('--output', type=Path, required=True)
    return p


def source_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / 'src').rglob('*.py')):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def make_jobs(args):
    import yaml

    root = args.output_root.resolve()
    unsupported = set(args.configs) - set(METHODS)
    if unsupported:
        raise ValueError(f'Unknown configurations: {sorted(unsupported)}')
    if args.action == 'quality' and set(args.benchmarks) - set(BENCHMARKS):
        raise ValueError(f'Unknown benchmarks: {sorted(set(args.benchmarks) - set(BENCHMARKS))}')
    if args.action != 'quality' and args.warmup < 0:
        raise ValueError('--warmup must be nonnegative')
    src_hash = source_hash()
    # Each child receives a remapped CUDA_VISIBLE_DEVICES, so its selected
    # physical GPU is always local device cuda:0. An unindexed "cuda" is
    # rejected by torch.cuda.set_device on current PyTorch releases.
    common = ['--model', args.model, '--revision', args.revision, '--device', 'cuda:0', '--dtype', args.dtype]
    if args.attn_implementation:
        common += ['--attn-implementation', args.attn_implementation]
    jobs = []
    for name in args.configs:
        cfg_path = (args.config_root / f'{name}.yaml').resolve()
        cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        block = cfg.get('generation', {}).get('block_size', 32)
        if args.action == 'quality':
            if cfg.get('generation', {}).get('schedule', 'threshold') != 'threshold':
                raise ValueError(f'{cfg_path}: quality requires a threshold schedule')
            points = [(bench, None, None) for bench in args.benchmarks]
        else:
            points = [(None, context, batch) for context in args.contexts for batch in args.batch_sizes]
        for benchmark, context, batch in points:
            new_tokens = args.max_new_tokens or max_new_tokens_for(benchmark)
            capacity = args.max_cache_tokens
            if context is not None and args.capacity == 'workload':
                capacity = math.ceil((context + new_tokens) / block) * block
            if capacity <= new_tokens or (context is not None and context + new_tokens > capacity):
                raise ValueError(f'Insufficient cache capacity for {name}: {capacity}')
            if benchmark:
                rel = Path('quality/core') / benchmark / f'{name}.jsonl'
                cmd = [sys.executable, '-m', 'bitsieve_fastdllm.eval.quality', *common,
                       '--config', str(cfg_path), '--benchmark', benchmark, '--split', 'test',
                       '--max-new-tokens', str(new_tokens), '--max-cache-tokens', str(capacity),
                       '--output', str(root / rel)]
                if args.limit is not None:
                    cmd += ['--limit', str(args.limit)]
                if args.plain_prompt:
                    cmd += ['--plain-prompt']
            else:
                rel = Path(args.action) / name / f'ctx{context}_b{batch}.jsonl'
                cmd = [sys.executable, '-m', 'bitsieve_fastdllm.eval.performance', *common,
                       '--config', str(cfg_path), '--contexts', str(context), '--batch-sizes', str(batch),
                       '--max-new-tokens', str(new_tokens), '--max-cache-tokens', str(capacity),
                       '--schedule', 'fixed', '--fixed-steps', str(args.fixed_steps),
                       '--warmup', str(args.warmup), '--repeats', str(args.repeats),
                       '--output', str(root / rel)]
            settings = dict(kind=args.action, config_file=name, config=cfg, model=args.model,
                            revision=args.revision, dtype=args.dtype, benchmark=benchmark,
                            context_tokens=context, batch_size=batch, max_new_tokens=new_tokens,
                            max_cache_tokens=capacity, limit=getattr(args, 'limit', None),
                            plain_prompt=getattr(args, 'plain_prompt', False),
                            attn_implementation=args.attn_implementation,
                            fixed_steps=getattr(args, 'fixed_steps', None),
                            warmup=getattr(args, 'warmup', None), repeats=getattr(args, 'repeats', None),
                            source_sha256=src_hash,
                            kernel_environment={k: v for k, v in sorted(os.environ.items())
                                                if k.startswith('BITSIEVE_') or k in ('PYTORCH_CUDA_ALLOC_CONF', 'LONG_BENCH_REVISION', 'LONG_BENCH_ARCHIVE_PATH')})
            jobs.append(dict(output=root / rel, log=root / 'logs' / rel.with_suffix('.log'), command=cmd,
                             settings=settings, fingerprint=digest(json.dumps(settings, sort_keys=True).encode())))
    return jobs


def process_memory(pid):
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = []
    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            values.append(int(parts[1]))
    return sum(values) if values else None


def validate_output(job):
    path, s = job['output'], job['settings']
    rows = read_rows(path)
    if not rows:
        raise ValueError('The command produced no measurement rows.')
    for row in rows:
        if row['config']['name'] != s['config']['name']:
            raise ValueError('Unexpected configuration in the result file.')
        if row['config']['max_cache_tokens'] != s['max_cache_tokens'] or row['config']['generation']['max_new_tokens'] != s['max_new_tokens']:
            raise ValueError('Result token budgets do not match the requested point.')
        if row.get('model_revision') != s['revision'] or row.get('dtype') != s['dtype']:
            raise ValueError('Unexpected model revision or dtype in the result file.')
    if s['kind'] == 'quality':
        ids = [str(row['id']) for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate quality example IDs.')
        summary = read_json(path.with_suffix('.summary.json'))
        if summary['num_examples'] != len(rows) or any(r['benchmark'] != s['benchmark'] for r in rows):
            raise ValueError('Quality summary does not match the result file.')
    else:
        if len(rows) != s['repeats'] or {r['repeat'] for r in rows} != set(range(s['repeats'])):
            raise ValueError('Incomplete or duplicated performance repeats.')
        if any(r['batch_size'] != s['batch_size'] or r['context_tokens_per_request'] != s['context_tokens'] for r in rows):
            raise ValueError('Performance geometry does not match the requested point.')
    return len(rows)


def run_job(job, gpu, args):
    out = job['output']
    status_path = Path(str(out) + '.status.json')
    if status_path.exists():
        state = read_json(status_path)
        if state.get('fingerprint') != job['fingerprint']:
            raise ValueError(f'{out}: settings or source changed; use a new output root.')
        if state['status'] == 'PASS' and out.exists():
            if state.get('output_sha256') == digest(out.read_bytes()):
                print(f'SKIP {out}', flush=True)
                return 'PASS'
            raise ValueError(f'{out}: a previously completed file has changed.')
        if state['status'] in ('OOM', 'ERROR') and not args.retry_failed:
            print(f"SKIP {state['status']} {out}", flush=True)
            return state['status']
    elif out.exists() and out.stat().st_size:
        raise ValueError(f'{out}: untracked output exists; use a new output root.')
    if STOP.is_set():
        return 'INTERRUPTED'
    out.parent.mkdir(parents=True, exist_ok=True)
    job['log'].parent.mkdir(parents=True, exist_ok=True)
    state = dict(status='RUNNING', fingerprint=job['fingerprint'], settings=job['settings'],
                 gpu=gpu, started_utc=utc(), command=job['command'], log=str(job['log']))
    save_json(status_path, state)
    print(f'RUN GPU={gpu} {out}', flush=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = gpu
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('BITSIEVE_CUDA_STRICT', '1')
    peak, samples, problem = None, 0, None
    with job['log'].open('a', encoding='utf-8') as log:
        log.write(f"\n[{utc()}] CUDA_VISIBLE_DEVICES={gpu} {shlex.join(job['command'])}\n")
        log.flush()
        start = log.tell()
        try:
            process = subprocess.Popen(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except OSError as exc:
            state.update(status='ERROR', ended_utc=utc(), validation_error=str(exc), exit_code=None)
            save_json(status_path, state)
            return 'ERROR'
        terminated = False
        termination_time = None
        while process.poll() is None:
            if STOP.is_set() and not terminated:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                terminated, termination_time = True, time.monotonic()
            if terminated and time.monotonic() - termination_time > 10:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if args.action == 'memory':
                value = process_memory(process.pid)
                if value is not None:
                    peak = value if peak is None else max(peak, value)
                    samples += 1
            time.sleep(0.2)
        rc = process.wait()
    text = job['log'].read_bytes()[start:].decode('utf-8', errors='replace')
    if STOP.is_set():
        status = 'INTERRUPTED'
    elif rc != 0:
        status = 'OOM' if re.search(r'CUDA out of memory|OutOfMemoryError', text, re.I) else 'ERROR'
    else:
        try:
            state['rows'] = validate_output(job)
            status = 'PASS'
        except Exception as exc:
            status, problem = 'ERROR', str(exc)
    state.update(status=status, exit_code=rc, ended_utc=utc(), validation_error=problem,
                 process_peak_mib_sampled=peak, memory_samples=samples,
                 process_peak_scope='process lifetime including load, warmup, prefill and decode' if args.action == 'memory' else None)
    if out.exists():
        state['output_sha256'] = digest(out.read_bytes())
    save_json(status_path, state)
    print(f'{status} GPU={gpu} {out}', flush=True)
    return status


def run(args):
    jobs = make_jobs(args)
    gpu_text = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpus = args.gpus or (csv_items(gpu_text) if gpu_text else ['0'])
    if len(set(gpus)) != len(gpus) or any(not re.fullmatch(r'[A-Za-z0-9_./:-]+', g) for g in gpus):
        raise ValueError('Invalid or duplicated CUDA device selectors.')
    if args.dry_run:
        for i, job in enumerate(jobs):
            print(f'CUDA_VISIBLE_DEVICES={gpus[i % len(gpus)]} {shlex.join(job["command"])}')
        return 0
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.experiment.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another launcher is using {root}.') from None
        packages = {}
        for pkg in ('torch', 'triton', 'transformers', 'datasets', 'numpy'):
            try:
                packages[pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                packages[pkg] = None
        save_json(root / 'environment.json', dict(timestamp_utc=utc(), python=sys.version,
                                               executable=sys.executable, packages=packages,
                                               source_sha256=source_hash(), cuda_selectors=gpus))
        pending = queue.Queue()
        for job in jobs:
            pending.put(job)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: STOP.set())

        def worker(gpu):
            outcomes = []
            while not STOP.is_set():
                try:
                    job = pending.get_nowait()
                except queue.Empty:
                    break
                try:
                    outcomes.append(run_job(job, gpu, args))
                except Exception as exc:
                    print(f'ERROR {job["output"]}: {exc}', file=sys.stderr, flush=True)
                    outcomes.append('ERROR')
                finally:
                    pending.task_done()
            return outcomes

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            outcomes = [item for items in pool.map(worker, gpus) for item in items]
        print(json.dumps({value: outcomes.count(value) for value in sorted(set(outcomes))}, indent=2))
        return 130 if STOP.is_set() else (1 if any(x != 'PASS' for x in outcomes) else 0)


def finite(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def median(rows, key, divisor=1):
    values = [v for r in rows if (v := finite(r.get(key))) is not None]
    return statistics.median(values) / divisor if values else None


def write_table(prefix, rows, columns):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with prefix.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    def fmt(value):
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.4f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    text = ['| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    text.extend('| ' + ' | '.join(fmt(row.get(c)) for c in columns) + ' |' for row in rows)
    prefix.with_suffix('.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


def summarize(args):
    files, failures = set(), []
    for path in args.input:
        if not path.exists():
            raise FileNotFoundError(path)
        files.update([path.resolve()] if path.is_file() else (p.resolve() for p in path.rglob('*.jsonl')))
        if path.is_dir():
            for suffix in ('.status.json', '.failed', '.error'):
                for marker in path.rglob('*.jsonl' + suffix):
                    files.add(Path(str(marker)[:-len(suffix)]).resolve())
    quality, performance, seen, errors = [], [], set(), []
    for path in sorted(files):
        display = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        state_path = Path(str(path) + '.status.json')
        state = read_json(state_path) if state_path.exists() else {}
        bad = state.get('status') in ('ERROR', 'OOM', 'INTERRUPTED', 'RUNNING')
        bad = bad or Path(str(path) + '.failed').exists() or Path(str(path) + '.error').exists()
        if bad:
            failures.append(dict(file=display, status=state.get('status', 'FAILED'),
                                 gpu=state.get('gpu'), exit_code=state.get('exit_code')))
            continue
        try:
            rows = read_rows(path)
        except Exception as exc:
            errors.append(dict(file=display, error=str(exc)))
            continue
        if not rows:
            continue
        sha = digest(path.read_bytes())
        if sha in seen:
            continue
        seen.add(sha)
        groups = {}
        for row in rows:
            cfg = row.get('config', {})
            if not isinstance(cfg, dict):
                continue
            key = (row.get('benchmark'), row.get('config_name') or cfg.get('name'),
                   row.get('context_tokens_per_request'), row.get('batch_size'),
                   json.dumps(cfg, sort_keys=True), row.get('model_id'),
                   row.get('model_revision'), row.get('dtype'), row.get('device_name'))
            groups.setdefault(key, []).append(row)
        for key, group in groups.items():
            benchmark, name, context, batch, _, model, revision, dtype, device = key
            if name is None:
                continue
            cfg = group[0]['config']
            base = dict(source_file=display, config=name, method=LABELS.get(name, name),
                        model=model, revision=revision, dtype=dtype, device=device,
                        max_cache_tokens=cfg.get('max_cache_tokens'),
                        max_new_tokens=cfg.get('generation', {}).get('max_new_tokens'))
            if benchmark:
                ids = [str(r['id']) for r in group]
                if len(set(ids)) != len(ids):
                    errors.append(dict(file=display, error='Duplicate quality example IDs.'))
                    continue
                s_path = path.with_suffix('.summary.json')
                saved = read_json(s_path) if s_path.exists() else {}
                complete = saved.get('num_examples') == len(group) and saved.get('benchmark') == benchmark
                scores = [float(r['score']) for r in group]
                rt = [r.get('runtime', {}) for r in group]
                tokens = sum(float(r.get('generated_tokens') or 0) for r in rt)
                decode = sum(float(r.get('decode_ms') or 0) for r in rt)
                quality.append(dict(**base, benchmark=benchmark, examples=len(group),
                                    status='COMPLETE' if complete else 'INCOMPLETE',
                                    mean_score=statistics.mean(scores), score_x100=100 * statistics.mean(scores),
                                    aggregate_decode_tokens_per_second=1000 * tokens / decode if decode else None,
                                    mean_decode_ms=decode / len(group)))
            elif context is not None:
                rt = group
                peaks = []
                for sidecar in (Path(str(path) + '.nvidia_smi_peak_mib'),):
                    if sidecar.exists():
                        peaks.append(finite(sidecar.read_text().strip()))
                if state.get('process_peak_mib_sampled') is not None:
                    peaks.append(finite(state['process_peak_mib_sampled']))
                peaks = [p for p in peaks if p is not None and p > 0]
                cache = [r.get('packed_cache', {}).get('total', r.get('native_cache_bytes')) for r in group]
                cache = [finite(x) for x in cache if finite(x) is not None]
                generation = cfg.get('generation', {})
                performance.append(dict(**base, context_tokens=context, batch_size=batch, repeats=len(group),
                                        status='PASS', schedule=generation.get('schedule'),
                                        fixed_steps=generation.get('fixed_steps_per_block'),
                                        median_prefill_ms=median(rt, 'prefill_ms'),
                                        median_prefill_pack_ms=median(rt, 'prefill_pack_ms'),
                                        median_decode_ms=median(rt, 'decode_ms'),
                                        median_tpob_ms=median(rt, 'mean_tpob_ms'),
                                        median_tokens_per_second=median(rt, 'tokens_per_second'),
                                        decode_peak_allocated_gib=median(rt, 'peak_cuda_allocated_bytes', 2**30),
                                        decode_peak_reserved_gib=median(rt, 'peak_cuda_reserved_bytes', 2**30),
                                        process_peak_gib_sampled=max(peaks) / 1024 if peaks else None,
                                        logical_kv_gib=statistics.median(cache) / 2**30 if cache else None,
                                        cache_compression_ratio=median(rt, 'cache_compression_ratio')))
    order = {name: i for i, name in enumerate(METHODS)}
    order['herald_middle1_bf16_proxy_semantic_a'] = order['herald_middle_bf16_a']
    quality.sort(key=lambda r: (r['benchmark'], order.get(r['config'], 99), r['source_file']))
    performance.sort(key=lambda r: (r['context_tokens'], r['batch_size'], order.get(r['config'], 99), r['source_file']))
    out = args.output
    write_table(out / 'quality', quality, ['benchmark', 'config', 'examples', 'status', 'score_x100',
                                         'mean_score', 'mean_decode_ms', 'aggregate_decode_tokens_per_second',
                                         'max_new_tokens', 'max_cache_tokens', 'source_file'])
    write_table(out / 'performance_memory', performance,
                ['context_tokens', 'batch_size', 'config', 'status', 'repeats', 'median_prefill_ms',
                 'median_prefill_pack_ms', 'median_decode_ms', 'median_tpob_ms',
                 'median_tokens_per_second', 'decode_peak_allocated_gib', 'decode_peak_reserved_gib',
                 'process_peak_gib_sampled', 'logical_kv_gib', 'cache_compression_ratio', 'max_cache_tokens',
                 'max_new_tokens', 'schedule', 'fixed_steps', 'device', 'source_file'])
    write_table(out / 'failures', failures, ['file', 'status', 'gpu', 'exit_code'])
    if errors:
        save_json(out / 'parse_errors.json', errors)
    else:
        (out / 'parse_errors.json').unlink(missing_ok=True)
    print(f'quality_groups={len(quality)} performance_groups={len(performance)} failures={len(failures)} parse_errors={len(errors)}')
    print(out.resolve())
    return 1 if errors else 0


def main():
    args = parser().parse_args()
    try:
        return summarize(args) if args.action == 'summarize' else run(args)
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
