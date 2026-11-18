"""The eight manuscript experiment designs, with fixed reproducible seeds.

Experiments 1--3 share random streams. Experiments 4--8 use setting-specific
streams. The released seed values differ from those of the saved study results.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.metrics import roc_auc_score

EXPERIMENTS = {
    1: 'Template and membership recovery with method comparison',
    2: 'Selecting the number of templates',
    3: 'Coupling and edge density',
    4: 'Block-structured networks',
    5: 'Degree of membership overlap',
    6: 'Network size and template rank',
    7: 'Signal and noise',
    8: 'Model components and distributional assumptions',
}
BASE = dict(n=68, L=100, M=3, K=3, alpha=1., template_rms=.6,
            sigma=.3, a1=2., density=.5, holdout_fraction=.15, kind='coupled')
INITIALIZATION = dict(screen_iterations=80, refine_candidates=12, maxiter=1000,
                      ftol=1e-10, gtol=1e-5, maxls=30, maxfun=50000, families=5)
SAMPLER = dict(warmup=300, draws=1000, target_accept=.8, max_tree_depth=9)
FRESH = {
    'block': dict(kind='block'),
    'normal_fresh': {}, 'alpha01': dict(alpha=.1), 'alpha3': dict(alpha=3.),
    'n30': dict(n=30), 'L200': dict(L=200), 'K2': dict(K=2),
    'weak_signal': dict(template_rms=.15), 'high_noise': dict(sigma=1.),
    't3': {}, 'separate': dict(kind='independent'),
}
FRESH_EXPERIMENTS = {
    'block': [4], 'normal_fresh': [5, 6, 7, 8], 'alpha01': [5], 'alpha3': [5],
    'n30': [6], 'L200': [6], 'K2': [6], 'weak_signal': [7], 'high_noise': [7],
    't3': [8], 'separate': [8],
}


def seed_for(*parts):
    raw = json.dumps(parts, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:16], 16) % (2**32 - 1)


def data_seed(family, setting, rep, stream):
    if family == 'legacy':
        return seed_for(31415, rep, stream)
    if family == 'fresh':
        return seed_for(27182, setting, rep, stream)
    if family == 'whole_network':
        return seed_for(16180, rep, stream)
    raise ValueError(f'Unknown data family: {family}')


def generation_seeds(family, setting, rep):
    return {key: data_seed(family, setting, rep, key)
            for key in ('template', 'membership', 'presence', 'noise', 'mask', 'binary', 'fold')}


def generate(cfg, rep, setting="normal", family="legacy"):
    n,L,M,K=[cfg[k] for k in ('n','L','M','K')];ij=np.triu_indices(n,1);P=len(ij[0])
    # Shared underlying streams across cells where dimensions allow pairing.
    seeds = generation_seeds(family, setting, rep)
    rng=np.random.default_rng(seeds['template']);raw=rng.normal(size=(M,n,K))
    if cfg['kind']=='close':raw=raw[:1]+cfg['direction_perturbation']*raw
    nodes=np.arange(n)%K
    if cfg['kind']=='block':
        rng.shuffle(nodes);basis=np.eye(K)[nodes]/np.sqrt(np.bincount(nodes))[None,:]
        # Block-constant rank-K signed means; probability templates also rank <=K.
        raw=np.array([basis@np.linalg.qr(rng.normal(size=(K,K)))[0] for _ in range(M)])
    U=np.linalg.qr(raw)[0]
    lam=np.array([(-1.)**np.arange(K)*np.linspace(2.,1.,K) for _ in range(M)])
    full=np.einsum('mik,mk,mjk->mij',U,lam,U);q=full[:,ij[0],ij[1]]
    scales=cfg['template_rms']/np.sqrt(np.mean(q*q,axis=1));q*=scales[:,None];full*=scales[:,None,None]
    w_rng=np.random.default_rng(seeds['membership'])
    W=w_rng.dirichlet(np.repeat(cfg['alpha']/M,M),L)
    if cfg['kind']=='block':W=np.eye(M)[w_rng.permutation(np.arange(L)%M)]
    mu=W@q
    qb=cfg['a1']*q
    if cfg['kind']=='independent':
        u=np.linalg.qr(np.random.default_rng(seeds['binary']).normal(size=(M,n,K)))[0]
        b=np.einsum('mik,mk,mjk->mij',u,lam,u)[:,ij[0],ij[1]]
        qb=b*(cfg['a1']*cfg['template_rms']/np.sqrt(np.mean(b*b,axis=1)))[:,None]
    logits=W@qb
    a0=brentq(lambda a:expit(a+logits).mean()-cfg['density'],-100,100)
    pi=expit(a0+logits)
    if cfg['kind']=='continuous':pi=np.ones_like(mu)
    z=(np.random.default_rng(seeds['presence']).random(mu.shape)<pi).astype(np.int32)
    latent=mu+cfg['sigma']*np.random.default_rng(seeds['noise']).normal(size=mu.shape)
    y=np.where(z==1,latent,0.)
    rng=np.random.default_rng(seeds['mask']);test=np.zeros((L,P),bool)
    for row in test:row[rng.choice(P,max(1,int(round(P*cfg['holdout_fraction']))),replace=False)]=True
    fold=np.empty(L,int);fold[np.random.default_rng(seeds['fold']).permutation(L)]=np.arange(L)%2
    obs=test&(z==1);qdist=np.sqrt(np.mean((q[:,None]-q[None,:])**2,axis=-1))
    audit=dict(seeds=seeds,observed_density=float(z.mean()),expected_density=float(pi.mean()),template_rms=np.sqrt(np.mean(q*q,axis=1)).tolist(),
        template_pair_distance=qdist.tolist(),mu_rms=float(np.sqrt(np.mean(mu**2))),sigma=cfg['sigma'],
        oracle_auc=float(roc_auc_score(z[test],pi[test])) if np.unique(z[test]).size==2 else None,
        oracle_conditional_mse=float(np.mean((y[obs]-mu[obs])**2)),
        kind=cfg['kind'],weight_definition='Z is generated separately; present Y is a signed Gaussian score, not thresholded or re-transformed.',
        a1_truth_applicable=cfg['kind'] not in ('independent','continuous'),selection='Every prespecified seed retained; no oracle threshold or rejection sampling')
    truth=dict(q=q,Q=full,W=W,mu=mu,pi=pi,q_binary=qb,a0=np.array(a0),a1=np.array(cfg['a1']),nodes=nodes,sigma2=np.array(cfg['sigma']**2))
    if setting == 't3':
        scale = cfg['sigma'] * np.sqrt(1 / 3)
        noise = np.random.default_rng(seeds['noise']).standard_t(3, size=y.shape) * scale
        y = np.where(z == 1, mu + noise, 0.)
        truth['noise_scale'] = np.asarray(scale)
        truth['noise_variance'] = np.asarray(cfg['sigma'] ** 2)
        audit.update(noise_df=3, noise_scale=float(scale),
                     oracle_conditional_mse=float(np.mean((y[obs] - mu[obs]) ** 2)),
                     weight_definition='Separate binary Z and signed present Student-t Y; no truncation')
    return dict(z=z,y=y),truth,dict(train=~test,test=test,fold=fold),audit

def observation_masks(masks, rep, *, reduced=False, selection_true_m=None):
    """Create nested S1 masks and the paired S3 validation split."""
    result = {key: value.copy() for key, value in masks.items()}
    if reduced:
        rng = np.random.default_rng(seed_for(14142, 'ext_mixed', rep, 'training_subset'))
        train = np.zeros_like(masks['train'])
        for row, available in enumerate(masks['train']):
            indices = np.flatnonzero(available)
            train[row, rng.choice(indices, max(1, int(np.floor(len(indices) * .5))), replace=False)] = True
        result['train'] = train
    if selection_true_m is not None:
        rng = np.random.default_rng(17320 + 100 * selection_true_m + rep)
        validation = np.zeros_like(masks['test'])
        for source, target in zip(masks['test'], validation):
            edges = np.flatnonzero(source)
            target[rng.permutation(edges)[:len(edges) // 2]] = True
        result['validation'] = validation
        result['test'] &= ~validation
    result['unused'] = ~(result['train'] | result['test'] | result.get('validation', False))
    return result


def _record(case, dataset, rep, config, method, experiments, *, family='legacy',
            setting='normal', mask='reference', phase='', fit_m=None, seeds=None):
    return dict(id=case, fit_key=case, dataset=dataset, rep=rep, generator_rep=rep,
                config=config, method=method, experiments=experiments, family=family,
                setting=setting, mask=mask, phase=phase, M=fit_m or config['M'],
                K=config['K'], alpha=config['alpha'], seeds=seeds)


def all_jobs(reps=range(50), *, include_sensitivity=False):
    reps = tuple(reps)
    if len(set(reps)) != len(reps) or any(r not in range(50) for r in reps):
        raise ValueError('Replications must be distinct integers in 0..49')
    rows = []
    for rep, mask, method in itertools.product(reps, ('reference', 'less_observed'), ('coupled', 'alma', 'tbm')):
        paired = f'ext_mixed__rep{rep:03d}__{method}'
        seeds = dict(init=seed_for('ab_colab_pilot_v1', paired, 'init'),
                     fit=seed_for('ab_colab_pilot_v1', paired, 'fit'),
                     gig=seed_for('S1_NORMAL_MIXED_50x2_GIG_v1', paired, 'gig'))
        case = f'normal_rep{rep:03d}_{mask}_{method}'
        rows.append(_record(case, f'normal_rep{rep:03d}', rep, dict(BASE), method, [1],
                            mask=mask, seeds=seeds))
    for rep, true_m, phase, fit_m in itertools.product(reps, (3, 2, 4), ('primary', 'restart'), (2, 3, 4, 5)):
        case = f'r{rep:03d}_t{true_m}_m{fit_m}_{phase}'
        seeds = {key: seed_for('S3_candidates_v1', case, old)
                 for key, old in [('init', 'initializer_seed'), ('fit', 'fit_seed'), ('gig', 'gig_seed')]}
        row = _record(case, f'normal_rep{rep:03d}' if true_m == 3 else f'selection_t{true_m}_rep{rep:03d}',
                      rep, dict(BASE, M=true_m), 'coupled', [2], fit_m=fit_m, phase=phase, seeds=seeds)
        row['generator_rep'] = rep if true_m == 3 else 10000 * true_m + rep
        if true_m == fit_m == 3 and phase == 'primary':
            parent = next(x for x in rows if x['id'] == f'normal_rep{rep:03d}_reference_coupled')
            row['fit_key'] = parent['fit_key']
            row['seeds'] = parent['seeds'].copy()
        rows.append(row)
    for cell, (density, coupling) in enumerate(itertools.product((.15, .30, .50), (0., .5, 1.))):
        for rep, method in itertools.product(reps, ('coupled', 'null')):
            dataset = f'cell{cell:02d}_rep{rep:03d}'
            seeds = {key: seed_for('S4_COUPLING_90_v1', dataset, method, key) for key in ('init', 'fit', 'gig')}
            rows.append(_record(dataset + '__' + method, dataset, rep,
                                dict(BASE, a1=coupling, density=density), method, [3],
                                setting='coupling', seeds=seeds))
    for setting, changes in FRESH.items():
        methods = ['coupled', 'alma', 'tbm']
        if setting == 'block':
            methods.insert(0, 'hard')
        elif setting == 'normal_fresh':
            methods += ['hard', 'weighted', 'independent']
        elif setting == 't3':
            methods = ['coupled', 'student3']
        elif setting == 'separate':
            methods = ['coupled', 'independent']
        for rep, method in itertools.product(reps, methods):
            original_method = 'tbm_r' if method == 'tbm' else method
            seeds = {key: seed_for(27182, setting, rep, key, original_method, 0)
                     for key in ('init', 'fit', 'gig')}
            case = f'{setting}_rep{rep:03d}_{method}'
            row = _record(case, f'{setting}_rep{rep:03d}', rep, dict(BASE, **changes), method,
                          FRESH_EXPERIMENTS[setting], family='fresh', setting=setting, seeds=seeds)
            if setting == 'block':
                row['alpha'] = .1
            if setting == 'normal_fresh' and method in ('hard', 'weighted', 'independent'):
                row['experiments'] = [8]
            if setting == 'block' and method == 'coupled':
                row['depends_on'] = f'block_rep{rep:03d}_hard'
                row['start_scheme'] = 'hard_draw199'
            rows.append(row)
    if include_sensitivity:
        lookup = {row['id']: row for row in rows}
        sparse = (5, 9, 16, 19, 25, 26, 29, 41, 43, 45, 47)
        for index, rep in enumerate(sparse):
            if rep not in reps:
                continue
            original = lookup[f'cell00_rep{rep:03d}__null']
            row = dict(original, id=f'null_rep{rep:03d}_crossstart',
                       fit_key=f'null_rep{rep:03d}_crossstart', sensitivity=True,
                       depends_on=f'cell00_rep{rep:03d}__coupled', start_scheme='fixed_zero_draw199',
                       seeds=dict(init=22361 + index, fit=24494 + index, gig=26457 + index))
            rows.append(row)
        for index, (rep, m, scheme) in enumerate([(28, 5, 'small_extras'), (28, 5, 'equal_split'),
                                                 (45, 3, 'small_extras'), (45, 3, 'equal_split'),
                                                 (45, 4, 'small_extras'), (45, 4, 'equal_split')], 11):
            if rep not in reps:
                continue
            phase = 'primary' if scheme == 'small_extras' else 'restart'
            original = lookup[f'r{rep:03d}_t2_m{m}_{phase}']
            case = f'r{rep:03d}_t2_m{m}_{scheme}'
            rows.append(dict(original, id=case, fit_key=case, phase=scheme, sensitivity=True,
                             depends_on=f'r{rep:03d}_t2_m2_{phase}', start_scheme=scheme,
                             seeds=dict(init=22361 + index, fit=24494 + index, gig=26457 + index)))
    return rows


def select_jobs(experiments=range(1, 9), reps=range(50), methods=None, *, sensitivity=False):
    wanted = set(experiments)
    if not wanted or not wanted <= set(EXPERIMENTS):
        raise ValueError('Experiments must be selected from 1..8')
    return [row for row in all_jobs(reps, include_sensitivity=sensitivity)
            if wanted.intersection(row['experiments']) and bool(row.get('sensitivity')) == sensitivity
            and (methods is None or row['method'] in methods)]


def tbm_start_seeds(job):
    from model.tbm import tbm_start_seeds as start_seeds
    policy = 'paired_masks' if job['family'] == 'legacy' else 'simulation'
    return start_seeds(job['seeds']['fit'], policy=policy)


def expanded_start(values, m, scheme):
    """Duplicate templates and split weights without changing the initial mean."""
    v = {key: np.array(value, copy=True) for key, value in values.items()}
    minimum = v['W'].shape[1]
    if minimum != 2 or m not in (3, 4, 5):
        raise ValueError('The reported sensitivity analysis expands a two-template fit to 3, 4 or 5')
    mapping = list(range(minimum)) + [i % minimum for i in range(m - minimum)]
    fractions = np.zeros(m)
    for parent in range(minimum):
        columns = [j for j, p in enumerate(mapping) if p == parent]
        if scheme == 'equal_split':
            fractions[columns] = 1. / len(columns)
        elif scheme == 'small_extras':
            fractions[columns[0]] = 1. if len(columns) == 1 else .99
            if len(columns) > 1:
                fractions[columns[1:]] = .01 / (len(columns) - 1)
        else:
            raise ValueError(f'Unknown expansion: {scheme}')
    v['V'] = v['V'][mapping].copy()
    v['gamma'] = v['gamma'][mapping].copy()
    v['W'] = v['W'][:, mapping] * fractions
    return v
