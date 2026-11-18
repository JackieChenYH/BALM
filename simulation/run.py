"""Run one or more manuscript cases: python -m simulation.run --help."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
from simulation.designs import (EXPERIMENTS, INITIALIZATION, SAMPLER, all_jobs,
                                expanded_start, generate, generation_seeds, observation_masks,
                                select_jobs, tbm_start_seeds)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def digest_arrays(arrays):
    h = hashlib.sha256()
    for key, array in sorted(arrays.items()):
        value = np.ascontiguousarray(array)
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


@contextmanager
def file_lock(path):
    """Serialize writers; the OS releases the lock if a worker exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b'0')
                stream.flush()
            while True:
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.25)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


def prepare_dataset(job, output):
    with file_lock(Path(output) / 'locks' / ('data_' + job['dataset'] + '.lock')):
        return _prepare_dataset(job, output)


def _prepare_dataset(job, output):
    folder = Path(output) / 'data' / job['dataset']
    expected = {key: job[key] for key in ('dataset', 'generator_rep', 'config', 'family', 'setting')}
    expected['data_seeds'] = generation_seeds(job['family'], job['setting'], job['generator_rep'])
    if folder.exists():
        metadata = read_json(folder / 'metadata.json')
        if metadata['design'] != expected:
            raise ValueError(f'Data configuration or seeds differ from the existing directory: {folder}. '
                             'Choose a new --output directory.')
        arrays = {name: load_npz(folder / (name + '.npz')) for name in ('data', 'truth', 'masks')}
        for name, values in arrays.items():
            if digest_arrays(values) != metadata['array_hashes'][name]:
                raise ValueError(f'Changed saved arrays: {folder / name}')
        return arrays['data'], arrays['truth'], arrays['masks'], metadata
    data, truth, masks, audit = generate(job['config'], job['generator_rep'], job['setting'], job['family'])
    arrays = dict(data=data, truth=truth, masks=masks)
    metadata = dict(design=expected, audit=audit,
                    array_hashes={name: digest_arrays(value) for name, value in arrays.items()})
    folder.parent.mkdir(parents=True, exist_ok=True)
    stage = folder.with_name(folder.name + '.pending-' + uuid.uuid4().hex)
    stage.mkdir()
    for name, values in arrays.items():
        np.savez_compressed(stage / (name + '.npz'), **values)
    write_json(stage / 'metadata.json', metadata)
    stage.rename(folder)
    return data, truth, masks, metadata


def case_masks(job, masks):
    return observation_masks(masks, job['rep'], reduced=job['mask'] == 'less_observed',
                             selection_true_m=job['config']['M'] if 2 in job['experiments'] else None)


def fit_signature(job, metadata, masks, sampler, initialization):
    return dict(dataset=metadata['design'], data_hashes=metadata['array_hashes'],
                train_hash=digest_arrays({'train': masks['train']}), method=job['method'],
                n=job['config']['n'], M=job['M'], K=job['K'], alpha=job['alpha'],
                seeds=job['seeds'], sampler=sampler, initialization=initialization,
                start_scheme=job.get('start_scheme'), depends_on=job.get('depends_on'))


def fit_case(job, output, *, sampler=None, initialization=None, lookup=None):
    with file_lock(Path(output) / 'locks' / ('fit_' + job['fit_key'] + '.lock')):
        return _fit_case(job, output, sampler=sampler, initialization=initialization, lookup=lookup)


def _fit_case(job, output, *, sampler=None, initialization=None, lookup=None):
    from model.balm import FitResult, fit_balm
    sampler = dict(SAMPLER if sampler is None else sampler)
    initialization = dict(INITIALIZATION if initialization is None else initialization)
    data, truth, masks, metadata = prepare_dataset(job, output)
    masks = case_masks(job, masks)
    folder = Path(output) / 'fits' / job['fit_key']
    signature = fit_signature(job, metadata, masks, sampler, initialization)
    if (folder / 'complete.json').exists():
        complete = read_json(folder / 'complete.json')
        if complete['signature'] != signature:
            raise ValueError(f'Existing fit has different settings: {folder}. Choose a new --output directory.')
        return folder
    if folder.exists():
        raise ValueError(f'Incomplete fit directory retained: {folder}. Choose a new output directory or inspect it.')
    initial = None
    if job.get('depends_on'):
        if lookup is None:
            lookup = {x['id']: x for x in all_jobs(include_sensitivity=True)}
        source = lookup[job['depends_on']]
        parent = fit_case(source, output, sampler=sampler, initialization=initialization, lookup=lookup)
        previous = FitResult.load(parent)
        if len(previous.q) < 200:
            raise ValueError('The reported initialization transfer requires retained draw 200')
        initial = {key: value[199].copy() for key, value in previous.samples.items()}
        if job['start_scheme'] == 'hard_draw199':
            initial['W'] = .98 * initial['W'] + .02 / job['M']
        elif job['start_scheme'] == 'fixed_zero_draw199':
            initial.pop('a1', None)
        else:
            initial = expanded_start(initial, job['M'], job['start_scheme'])
    z = np.where(masks['train'], data['z'], 0)
    y = np.where(masks['train'] & (data['z'] == 1), data['y'], 0.)
    folder.parent.mkdir(parents=True, exist_ok=True)
    stage = folder.with_name(folder.name + '.pending-' + uuid.uuid4().hex)
    stage.mkdir()
    started = time.perf_counter()
    print(f'Fitting {job["id"]}', flush=True)
    if job['method'] in ('alma', 'tbm'):
        if job['method'] == 'alma':
            from model.alma import fit_alma
            result = fit_alma(z, masks['train'], job['config']['n'], job['M'], job['K'], job['seeds']['fit'])
            if not result['optimization']['converged']:
                result = fit_alma(z, masks['train'], job['config']['n'], job['M'], job['K'],
                                  job['seeds']['fit'], state=result, max_iter=None)
        else:
            from model.tbm import fit_tbm
            result = fit_tbm(z, masks['train'], job['config']['n'], job['M'], job['K'],
                             job['seeds']['fit'], start_seeds=tbm_start_seeds(job))
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        np.savez_compressed(stage / 'estimates.npz', **arrays)
        write_json(stage / 'optimization.json', result['optimization'])
    else:
        result = fit_balm(z, y, masks['train'], job['config']['n'], job['M'], job['K'],
                          method=job['method'], alpha=job['alpha'],
                          seed=job['seeds']['fit'], init_seed=job['seeds']['init'],
                          gig_seed=job['seeds']['gig'], init_values=initial,
                          initialization=initialization, **sampler)
        result.save(stage)
    write_json(stage / 'complete.json', dict(signature=signature,
               wall_seconds=time.perf_counter() - started))
    stage.rename(folder)
    print(f'Completed {job["id"]}', flush=True)
    return folder


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('list', 'prepare', 'fit', 'summarize', 'plot'))
    p.add_argument('--experiment', type=int, nargs='+', choices=range(1, 9), default=list(range(1, 9)))
    p.add_argument('--rep', type=int, nargs='+', default=list(range(50)), help='Zero-based replications, 0..49')
    p.add_argument('--method', nargs='+', choices=('coupled', 'null', 'hard', 'weighted', 'independent', 'student3', 'alma', 'tbm'))
    p.add_argument('--case', help='Run exactly this case ID from the list command')
    p.add_argument('--task', type=int, help='Zero-based index in the selected list; useful for Slurm arrays')
    p.add_argument('--sensitivity', action='store_true', help='Use only the 17 targeted initialization cases')
    p.add_argument('--output', type=Path, default=Path('outputs/simulation'))
    p.add_argument('--warmup', type=int, default=SAMPLER['warmup'])
    p.add_argument('--draws', type=int, default=SAMPLER['draws'])
    p.add_argument('--allow-partial', action='store_true', help='Summarize completed cases without treating them as the full experiment')
    return p


def main():
    args = parser().parse_args()
    rows = select_jobs(args.experiment, args.rep, args.method, sensitivity=args.sensitivity)
    if args.case:
        rows = [row for row in rows if row['id'] == args.case]
        if not rows:
            raise SystemExit('Case not found in the requested experiment/replication/method selection')
    if args.task is not None:
        if not 0 <= args.task < len(rows):
            raise SystemExit(f'Task outside 0..{len(rows)-1}')
        rows = [rows[args.task]]
    if args.command == 'list':
        writer = csv.writer(sys.stdout, lineterminator='\n')
        writer.writerow(('task', 'case', 'experiments', 'method', 'M', 'K', 'depends_on'))
        for index, row in enumerate(rows):
            writer.writerow((index, row['id'], ','.join(map(str, row['experiments'])),
                             row['method'], row['M'], row['K'], row.get('depends_on', '')))
        return
    if args.command == 'prepare':
        seen = set()
        for row in rows:
            if row['dataset'] not in seen:
                prepare_dataset(row, args.output)
                seen.add(row['dataset'])
        print(f'Prepared {len(seen)} datasets')
    elif args.command == 'fit':
        if args.draws < 1 or args.warmup < 1:
            raise SystemExit('Warmup and draws must be positive')
        lookup = {row['id']: row for row in all_jobs(include_sensitivity=True)}
        sampler = dict(SAMPLER, warmup=args.warmup, draws=args.draws)
        for row in rows:
            fit_case(row, args.output, sampler=sampler, lookup=lookup)
    elif args.command == 'summarize':
        from simulation.analysis import summarize
        summarize(rows, args.output, allow_partial=args.allow_partial)
    elif args.command == 'plot':
        from simulation.analysis import plot_results
        plot_results(args.output)


if __name__ == '__main__':
    main()
