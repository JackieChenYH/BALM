"""Explicit entry points for HCP fits, comparisons and fixed-template transfer."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('JAX_ENABLE_X64', 'true')

from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, gammaln, logsumexp
from sklearn.metrics import roc_auc_score

from .data import (COHORTS, file_hash, load_cohort, observations, prepare, read_npz,
                   target_rows, write_json)


TRANSFER_SEED_BASES = {'M22_25': 63000, 'F31_35': 64000}


def auc(z, probability):
    return float(roc_auc_score(z, probability)) if len(np.unique(z)) == 2 else None


def aligned_means(q, w):
    """Use the exact HCP median-total-norm anchor and signed-distance assignment."""
    anchor_index = np.argsort(np.sum(q * q, axis=(1, 2)), kind='stable')[len(q) // 2]
    order = np.argsort(-np.sum(q[anchor_index] ** 2, axis=1), kind='stable')
    anchor = q[anchor_index, order]
    q_sum = np.zeros_like(anchor)
    w_sum = np.zeros_like(w[0])
    for q_draw, w_draw in zip(q, w):
        cost = ((anchor[:, None] - q_draw[None, :]) ** 2).mean(-1)
        permutation = linear_sum_assignment(cost)[1]
        q_sum += q_draw[permutation]
        w_sum += w_draw[:, permutation]
    return q_sum / len(q), w_sum / len(w)


def predictive(q, w, z, y, a0, a1, sigma2):
    """Pointwise Student-t5 hurdle predictions, averaging density before logging."""
    q, w = np.asarray(q), np.atleast_2d(w)
    mu = w @ q if q.ndim == 2 else np.einsum('sm,sme->se', w, q)
    a0, a1, sigma2 = [np.asarray(x).reshape(-1, 1) for x in (a0, a1, sigma2)]
    logits = a0 + a1 * mu
    probability = expit(logits)
    binary = z[None, :] * logits - np.logaddexp(0, logits)
    strength = (gammaln(3.) - gammaln(2.5) - .5 * np.log(5 * np.pi * sigma2)
                - 3 * np.log1p((y[None, :] - mu) ** 2 / (5 * sigma2)))
    density = binary + z[None, :] * strength
    return dict(probability=probability.mean(0), log_score=logsumexp(density, axis=0) - np.log(len(mu)),
                positive_location=np.sum(probability * mu, axis=0) / probability.sum(0))


def validation_predictions(q, w, scalars, data):
    masks = data['validation']
    output = {key: [] for key in ('z', 'probability', 'log_score', 'positive_location')}
    for subject, mask in enumerate(masks):
        z, y = data['z'][subject, mask], data['y'][subject, mask]
        predicted = predictive(q[:, :, mask], w[:, subject], z, y, **scalars)
        output['z'].append(z)
        for key, value in predicted.items():
            output[key].append(value)
    return {key: np.stack(value) for key, value in output.items()}


def fit(args):
    from model.balm import fit_balm
    cohort = load_cohort(args.prepared, args.cohort)
    index = COHORTS.index(args.cohort) * 3 + (args.templates - 2) + 1
    seeds = dict(init_seed=41000 + 10 * index + 1, seed=41000 + 10 * index + 2,
                 gig_seed=41000 + 10 * index + 3)
    destination = Path(args.out) / f'{args.cohort}_M{args.templates}'
    destination.mkdir(parents=True, exist_ok=False)
    config = dict(cohort=args.cohort, M=args.templates, K=4, alpha=1., method='student5',
                  warmup=args.warmup, draws=args.draws, target_accept=.85, max_tree_depth=9,
                  initialization={'profile': 'hcp'}, prepared_sha256=file_hash(Path(args.prepared) / 'networks.npz'), **seeds)
    write_json(destination / 'config.json', config)
    result = fit_balm(cohort['z'], cohort['y'], cohort['train'], n=68,
                      M=args.templates, K=4, method='student5', alpha=1., warmup=args.warmup,
                      draws=args.draws, initialization={'profile': 'hcp'}, target_accept=.85,
                      max_tree_depth=9, **seeds)
    result.save(destination / 'fit')
    q, w = np.asarray(result.q), np.asarray(result.samples['W'])
    q_mean, w_mean = aligned_means(q, w)
    scalars = {key: np.asarray(result.samples[key]) for key in ('a0', 'a1', 'sigma2')}
    source_probability = ((cohort['z'] * cohort['train']).sum(0) + .5) / (cohort['train'].sum(0) + 1)
    np.savez_compressed(destination / 'source.npz', Q_mean=q_mean, W_mean=w_mean,
                        source_average_W=w_mean.mean(0), source_edge_probability=source_probability,
                        subject_ids=cohort['subject_ids'], roi_labels=cohort['roi_labels'],
                        training_scale=cohort['training_scale'], cohort=args.cohort,
                        **{key: value.mean() for key, value in scalars.items()})
    predictions = validation_predictions(q, w, scalars, cohort)
    np.savez_compressed(destination / 'validation.npz', subject_ids=cohort['subject_ids'], **predictions)
    summary = dict(cohort=args.cohort, M=args.templates, subjects=len(w_mean), retained=len(q),
                   complete=len(q) == args.draws, validation_auc=auc(predictions['z'].ravel(), predictions['probability'].ravel()),
                   validation_log_score=float(predictions['log_score'].mean()),
                   training_scale=cohort['training_scale'], test_evaluated=False)
    write_json(destination / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def compare(args):
    cohort = load_cohort(args.prepared, args.cohort)
    position = COHORTS.index(args.cohort) * 2 + (args.method == 'tbm')
    seed = 52000 + 1009 * position
    destination = Path(args.out) / f'{args.cohort}_{args.method}'
    state = None
    if args.resume:
        if args.method != 'alma':
            raise ValueError('Only ALMA iterations can continue from a saved endpoint')
        old = read_npz(destination / 'estimates.npz')
        old_status = json.loads((destination / 'optimization.json').read_text())
        state = {'loading': old['loading'], 'Q_binary': old['Q_binary'],
                 'input_digest': old_status.pop('input_digest'), 'optimization': old_status}
    else:
        destination.mkdir(parents=True, exist_ok=False)
    if args.method == 'alma':
        from model.alma import fit_alma
        result = fit_alma(cohort['z'], cohort['train'], n=68, M=4, K=4, seed=seed,
                          max_iter=args.iterations, tol=1e-5, state=state)
    else:
        from model.tbm import fit_tbm, tbm_start_seeds
        result = fit_tbm(cohort['z'], cohort['train'], n=68, M=4, K=4, seed=seed,
                         start_seeds=tbm_start_seeds(seed, policy='hcp'), rscript=args.rscript)
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(destination / 'estimates.npz', **arrays)
    optimization = dict(result['optimization'])
    if args.method == 'alma':
        optimization['input_digest'] = result['input_digest']
    write_json(destination / 'optimization.json', optimization)
    summary = dict(cohort=args.cohort, method=args.method, M=4, seed=seed,
                   validation_auc=auc(cohort['z'][cohort['validation']], result['topology_score'][cohort['validation']]),
                   optimization=result['optimization'], test_evaluated=False)
    write_json(destination / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def select_dimensions(args):
    rows = []
    for cohort in COHORTS:
        candidates = {}
        identity = None
        for m in (2, 3, 4):
            folder = Path(args.fits) / f'{cohort}_M{m}'
            if not (folder / 'summary.json').exists():
                continue
            summary = json.loads((folder / 'summary.json').read_text())
            prediction = read_npz(folder / 'validation.npz')
            if not summary['complete']:
                continue
            if identity is None:
                identity = prediction['subject_ids']
            if not np.array_equal(identity, prediction['subject_ids']):
                raise ValueError('Candidates must have identical subject order')
            candidates[m] = prediction['log_score'].mean(1)
        if not candidates:
            continue
        best = max(candidates, key=lambda m: candidates[m].mean())
        eligible = []
        for m, values in candidates.items():
            difference = candidates[best] - values
            se = difference.std(ddof=1) / np.sqrt(len(difference))
            if difference.mean() <= se:
                eligible.append(m)
            rows.append(dict(cohort=cohort, M=m, mean_validation_log_score=float(values.mean()),
                             difference_from_best=float(difference.mean()), paired_subject_se=float(se),
                             selected=None, complete_candidate_grid=len(candidates) == 3))
        for row in rows:
            if row['cohort'] == cohort:
                row['selected'] = min(eligible) if len(candidates) == 3 else None
    pd.DataFrame(rows).to_csv(args.out, index=False)


def membership_model(q, z, y, a0, a1, sigma2):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    w = numpyro.sample('W', dist.Dirichlet(jnp.full((4,), .25)))
    mu = w @ q
    numpyro.sample('Z', dist.Bernoulli(logits=a0 + a1 * mu).to_event(1), obs=z)
    strength = dist.StudentT(df=5., loc=mu, scale=jnp.sqrt(sigma2)).log_prob(y)
    numpyro.factor('positive_strength', jnp.sum(jnp.where(z > 0, strength, 0.)))


def transfer_worker(payload):
    prepared, source_path, destination, indices, warmup, draws = payload
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS, init_to_value
    jax.config.update('jax_enable_x64', True)
    data = read_npz(Path(prepared) / 'networks.npz')
    source = read_npz(source_path)
    source_name = str(source['cohort'])
    rows = target_rows(data, source_name)
    seed_base = TRANSFER_SEED_BASES[source_name]
    fixed = {key: float(source[key]) for key in ('a0', 'a1', 'sigma2')}
    kernel = NUTS(membership_model, init_strategy=init_to_value(values={'W': jnp.asarray(source['source_average_W'])}),
                  dense_mass=True, target_accept_prob=.9, max_tree_depth=10)
    sampler = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=1, progress_bar=False, jit_model_args=True)
    for index in indices:
        output = Path(destination) / 'subjects' / f'{index:04d}.npz'
        row = rows[index]
        seed = int(np.random.SeedSequence(seed_base, spawn_key=(index,)).generate_state(1)[0])
        if output.exists():
            previous = read_npz(output)
            if (str(previous['subject_id']) != str(data['subject_ids'][row])
                    or int(previous['seed']) != seed or previous['W'].shape != (draws, 4)
                    or not np.isfinite(previous['W']).all()):
                raise ValueError(f'Existing transfer output has a different identity or incomplete draws: {output}')
            continue
        z, y = observations(data['count'][row], float(source['training_scale']))
        train, test = data['train'][row], data['test'][row]
        sampler.post_warmup_state = None
        sampler.run(jax.random.PRNGKey(seed), jnp.asarray(source['Q_mean'][:, train]),
                    jnp.asarray(z[train]), jnp.asarray(y[train]),
                    *[jnp.asarray(fixed[key]) for key in ('a0', 'a1', 'sigma2')],
                    extra_fields=('num_steps', 'accept_prob', 'potential_energy'))
        w = np.asarray(sampler.get_samples()['W'])
        predicted = predictive(source['Q_mean'][:, test], w, z[test], y[test], **fixed)
        average = predictive(source['Q_mean'][:, test], source['source_average_W'], z[test], y[test], **fixed)
        diagnostics = {key: np.asarray(value) for key, value in sampler.get_extra_fields().items()}
        temporary = output.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, W=w, z=z[test], y=y[test], edge_indices=np.flatnonzero(test),
                            subject_id=data['subject_ids'][row], cohort=data['cohort'][row], seed=seed,
                            empirical_probability=source['source_edge_probability'][test],
                            fixed_probability=average['probability'], fixed_log_score=average['log_score'],
                            **predicted, **diagnostics)
        temporary.replace(output)
        print(f'{source_name}: participant {index + 1}/{len(rows)} complete', flush=True)


def transfer(args):
    data = read_npz(Path(args.prepared) / 'networks.npz')
    source = read_npz(args.source)
    if source['Q_mean'].shape != (4, 2278) or not np.array_equal(source['roi_labels'], data['roi_labels']):
        raise ValueError('Transfer requires an M=4 source in the identical DK68 ROI order')
    rows = target_rows(data, str(source['cohort']))
    if set(source['subject_ids'].astype(str)) & set(data['subject_ids'][rows].astype(str)):
        raise ValueError('Source and target participants overlap')
    destination = Path(args.out)
    config = dict(source=str(source['cohort']), source_sha256=file_hash(args.source),
                  prepared_sha256=file_hash(Path(args.prepared) / 'networks.npz'),
                  subjects=len(rows), warmup=args.warmup, draws=args.draws,
                  seed_base=TRANSFER_SEED_BASES[str(source['cohort'])],
                  target_accept=.9, max_tree_depth=10, alpha=1., df=5.)
    if destination.exists():
        if not args.resume or json.loads((destination / 'config.json').read_text()) != config:
            raise ValueError('Existing output requires --resume and identical inputs/settings')
    else:
        (destination / 'subjects').mkdir(parents=True)
        write_json(destination / 'config.json', config)
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    jobs = [(str(args.prepared), str(args.source), str(destination), list(range(slot, len(rows), args.workers)),
             args.warmup, args.draws) for slot in range(args.workers)]
    if args.workers == 1:
        transfer_worker(jobs[0])
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(transfer_worker, jobs))
    from .analysis import summarize_transfer
    summarize_transfer(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Create private counts, metadata and frozen edge masks')
    p.add_argument('--networks', required=True)
    p.add_argument('--people', required=True)
    p.add_argument('--masks')
    p.add_argument('--out', required=True)
    p = commands.add_parser('fit', help='Fit one cohort and template dimension')
    p.add_argument('--prepared', required=True)
    p.add_argument('--cohort', choices=COHORTS, required=True)
    p.add_argument('--templates', choices=(2, 3, 4), type=int, required=True)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--draws', type=int, default=1500)
    p.add_argument('--out', required=True)
    p = commands.add_parser('compare', help='Run ALMA or TBM on identical binary training entries')
    p.add_argument('--prepared', required=True)
    p.add_argument('--cohort', choices=COHORTS, required=True)
    p.add_argument('--method', choices=('alma', 'tbm'), required=True)
    p.add_argument('--iterations', type=int, default=5000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--rscript', default='Rscript')
    p.add_argument('--out', required=True)
    p = commands.add_parser('select', help='Summarize candidate validation scores and paired one-SE choices')
    p.add_argument('--fits', required=True)
    p.add_argument('--out', required=True)
    p = commands.add_parser('transfer', help='Infer W for new participants with source parameters fixed')
    p.add_argument('--prepared', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--draws', type=int, default=1500)
    p.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.networks, args.people, args.out, masks_path=args.masks)
    else:
        {'fit': fit, 'compare': compare, 'select': select_dimensions, 'transfer': transfer}[args.command](args)


if __name__ == '__main__':
    main()
