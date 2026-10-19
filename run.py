"""论文主入口：三个数据集、两种描述子；中文创新标记见 h4_registration/core_*.py。"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_THREAD_LIMIT']:
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime

from h4_registration import runtime

ROOT = Path(__file__).resolve().parent
PRIMARY = 'diverse_scale_rank'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def manifest(domain):
    cases = read_csv(ROOT / 'manifests' / (domain + '.csv'))
    for row in cases:
        for key in row:
            if key.endswith('_path'):
                row[key] = runtime.manifest_path(row[key])
    return cases


def selected_cases(args):
    domains = ['indoor', 'kitti'] if args.scope == 'all' else [args.scope]
    result = []
    for domain in domains:
        cases = manifest(domain)
        datasets = ['3DMatch', '3DLoMatch'] if domain == 'indoor' else ['KITTI']
        for dataset in datasets:
            for descriptor in ['fpfh', 'fcgf']:
                group = [c for c in cases if c['dataset'] == dataset and c['descriptor'] == descriptor]
                # 固定取清单前 N 例，既不按成功率，也不按耗时挑选。
                if args.per_group:
                    group = group[:args.per_group]
                result.append((domain, dataset, descriptor, group))
    return result


def check_environment(groups):
    import numpy
    import scipy
    import open3d
    for name in ['mac_producer', 'indoor_tools', 'kitti_tools', 'indoor_control', 'kitti_control']:
        if not runtime.binary(name).is_file():
            raise FileNotFoundError(f'Missing {name}; first run python scripts/build.py')
        check = subprocess.run(runtime.command_for(['ldd', runtime.native_path(runtime.binary(name))]),
                               capture_output=True, text=True, errors='replace', timeout=30)
        if check.returncode or 'not found' in check.stdout:
            raise RuntimeError(f'Native dependency check failed: {name}\n{check.stdout}\n{check.stderr}')
    missing = {c[k + '_path'] for _, _, _, group in groups for c in group
               for k in ['source', 'target', 'corr', 'gt'] if not Path(c[k + '_path']).is_file()}
    if missing:
        raise FileNotFoundError(f'{len(missing)} input files missing; configure data roots. Example: {sorted(missing)[0]}')
    print('Environment OK:', numpy.__version__, scipy.__version__, open3d.__version__, flush=True)


def execute(arguments, log, timeout=1800):
    with Path(log).open('w', encoding='utf-8') as f:
        process = subprocess.run(runtime.command_for(arguments), stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f'Native process exited with {process.returncode}; log: {log}')


def prepare_case(case, core, block, memo):
    import numpy as np
    cid = case['case_id']
    for name in ['source', 'target', 'corr', 'gt']:
        # 只验证输入文件哈希；位姿选择不读取真实变换的数值。
        if sha(case[name + '_path']) != case[name + '_sha256']:
            raise ValueError(f'Input hash mismatch: {name}, case {cid}')
    dummy = block / 'dummy_identity.txt'
    np.savetxt(dummy, np.eye(4), fmt='%.17g')
    labels = block / 'generated' / (cid + '_dummy_labels.txt')
    count = len(np.atleast_2d(np.loadtxt(case['corr_path'])))
    labels.write_text('0\n' * count, encoding='ascii')
    provenance = block / 'generated' / (cid + '_provenance.bin')
    dataset = {'3DMatch': '3dmatch', '3DLoMatch': '3dlomatch', 'KITTI': 'KITTI'}[case['dataset']]
    # ================= 继承 MAC：图构建、极大团、节点引导、SVD =================
    execute([runtime.native_path(runtime.binary('mac_producer')), dataset,
             *[runtime.native_path(case[k + '_path']) for k in ['source', 'target', 'corr']],
             runtime.native_path(labels), runtime.native_path(dummy),
             runtime.native_path(block / 'generated' / cid), case['descriptor'],
             runtime.native_path(provenance)], block / 'generated' / (cid + '.log'), 620)
    poses = core.decode_cache(provenance)
    if not len(poses):
        raise RuntimeError('MAC generated no pose hypotheses')
    cache = block / 'generated' / (cid + '.bin')
    with cache.open('wb') as f:
        f.write(b'MACPOSE1'); f.write(struct.pack('<Q', len(poses))); f.write(poses.astype('<f8').tobytes())
    row = dict(case, cache_path=str(cache), cache_sha256=sha(cache))
    for name in ['source', 'target']:
        digest = case[name + '_sha256']
        if digest not in memo:
            with runtime.cloud_input(case[name + '_path']) as input_path:
                values = core.preprocess(input_path)
            path = block / 'clouds' / (digest + '.npz')
            np.savez(path, points=values[0], normals=values[1], weights=values[2], normal_valid=values[3])
            memo[digest] = path
        row[name + '_npz'] = str(memo[digest])
        row[name + '_npz_sha256'] = sha(memo[digest])
    return row


def failed_rows(case, reason, variants):
    common = {k: case[k] for k in ['case_id', 'dataset', 'descriptor', 'scene', 'pair_id']}
    return [dict(common, variant=v, status='execution_failed', reason=reason, success=False,
                 RE_deg='', TE_cm='', candidate_id='', dense_count=0, diverse_count=0) for v in variants]


def evaluate_block(domain, cases, block, memo):
    marker = block / 'completion.json'
    if marker.is_file():
        old = json.loads(marker.read_text(encoding='utf-8'))
        if sha(block / 'results_raw.csv') != old['results_sha256']:
            raise ValueError('Completed block was modified: ' + str(block))
        return read_csv(block / 'results_raw.csv')
    if block.exists():
        raise RuntimeError('Incomplete block retained for investigation; use a new run: ' + str(block))
    block.mkdir(parents=True)
    for name in ['runtime', 'initial', 'generated', 'clouds', 'control']:
        (block / name).mkdir()
    core = importlib.import_module('h4_registration.core_' + domain)
    good, failures = [], []
    for case in cases:
        print('Preparing', case['dataset'], case['descriptor'], case['case_id'], flush=True)
        try:
            good.append(prepare_case(case, core, block, memo))
        except Exception as error:
            failures.extend(failed_rows(case, repr(error), core.VARIANTS))
            print('Failure retained:', case['case_id'], repr(error), flush=True)
    if good:
        tsv = block / 'control' / 'inputs.tsv'
        tsv.write_text(''.join(f"{c['case_id']}\t{runtime.native_path(c['cache_path'])}\t{runtime.native_path(c['corr_path'])}\n" for c in good), encoding='utf-8')
        execute([runtime.native_path(runtime.binary(domain + '_control')), runtime.native_path(tsv),
                 runtime.native_path(block / 'control' / 'engine')], block / 'control' / 'run.log')
        shutil.copy2(runtime.binary(domain + '_tools'), block / 'runtime' / 'tools')
        shutil.copy2(ROOT / 'h4_registration' / ('core_' + domain + '.py'), block / 'runtime' / 'geometry_verify.py')
        write_csv(block / 'case_manifest.csv', good)
        save_json(block / 'protocol.json', {'runtime_hashes': {p.name: sha(p) for p in (block / 'runtime').iterdir()}})
        (block / 'protocol.sha256').write_text(sha(block / 'protocol.json'), encoding='ascii')
        core.OUT, core.PRE = block, block / 'control'
        # ★ 本文新增后端入口：DS → 加权精化 → BG/NC/DB → MS，真值只用于末端评价。
        core.run()
        rows = read_csv(block / 'results_raw.csv')
    else:
        rows = []
    rows += failures
    if len(rows) != len(cases) * len(core.VARIANTS):
        raise RuntimeError('Incomplete result record count')
    write_csv(block / 'results_raw.csv', rows)
    save_json(marker, {'complete': True, 'cases': len(cases), 'results_sha256': sha(block / 'results_raw.csv')})
    return rows


def summarize(rows):
    import numpy as np
    result = []
    for dataset in ['3DMatch', '3DLoMatch', 'KITTI']:
        for desc in ['fpfh', 'fcgf']:
            group = [r for r in rows if r['dataset'] == dataset and r['descriptor'] == desc]
            if not group:
                continue
            wins = [r for r in group if str(r['success']).lower() == 'true']
            result.append(dict(dataset=dataset, descriptor=desc, cases=len(group), successes=len(wins),
                               RR_percent=100 * len(wins) / len(group),
                               RE_success_deg=float(np.mean([float(r['RE_deg']) for r in wins])) if wins else '',
                               TE_success_cm=float(np.mean([float(r['TE_cm']) for r in wins])) if wins else ''))
    return result


def main():
    parser = argparse.ArgumentParser(description='MAC hypotheses + H4 backend benchmark runner')
    parser.add_argument('--config')
    parser.add_argument('--scope', choices=['all', 'indoor', 'kitti'], default='all')
    parser.add_argument('--per-group', type=int, default=0, help='First N pairs per dataset/descriptor; 0 = full benchmark')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.per_group < 0:
        parser.error('--per-group must be non-negative')
    settings = runtime.configure(args.config)
    groups = selected_cases(args)
    for _, dataset, desc, cases in groups:
        print(f'{dataset:10s} {desc:4s}: {len(cases)} pairs', flush=True)
    if args.plan:
        return
    check_environment(groups)
    if args.check:
        return
    config = {'scope': args.scope, 'per_group': args.per_group, 'settings': settings,
              'manifest_hashes': {domain: sha(ROOT / 'manifests' / (domain + '.csv')) for domain in {g[0] for g in groups}},
              'source_hashes': {str(p.relative_to(ROOT)): sha(p) for folder in ['h4_registration', 'cpp'] for p in (ROOT / folder).iterdir() if p.is_file()}}
    if args.resume:
        out = args.resume.resolve()
        if Path(settings['output_dir']).resolve() not in out.parents:
            parser.error('--resume must point inside output_dir')
        if json.loads((out / 'run_config.json').read_text(encoding='utf-8')) != config:
            raise ValueError('Resume configuration/source differs from original run')
    else:
        out = Path(settings['output_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out.mkdir(parents=True, exist_ok=False)
        save_json(out / 'run_config.json', config)
    primary, all_rows, memo = [], [], {}
    start = time.perf_counter()
    for domain, dataset, desc, cases in groups:
        for offset in range(0, len(cases), int(settings['batch_size'])):
            batch = cases[offset:offset + int(settings['batch_size'])]
            block = out / f'{domain}_{desc}_{dataset}' / f'{offset // int(settings["batch_size"]):05d}'
            rows = evaluate_block(domain, batch, block, memo)
            all_rows.extend(rows)
            primary.extend(r for r in rows if r['variant'] == PRIMARY)
            write_csv(out / 'results_raw.csv', primary)
            write_csv(out / 'all_controls.csv', all_rows)
            write_csv(out / 'summary.csv', summarize(primary))
            print('Evaluated:', len(primary), 'Output:', out, flush=True)
    errors = sum(r['status'] != 'completed' for r in primary)
    save_json(out / 'completion.json', {'complete': True, 'cases': len(primary), 'noncompleted_cases': errors,
                                      'seconds': time.perf_counter() - start})
    print('Finished:', out, flush=True)
    for row in summarize(primary):
        print(row)
    if any(r['status'] == 'execution_failed' for r in primary):
        raise SystemExit('Execution failures retained in output; inspect reasons before reporting results')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit('Interrupted; completed blocks and logs are retained.')
