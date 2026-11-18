"""Posterior recovery, paired comparisons and grayscale simulation figures."""
from __future__ import annotations

import itertools
import hashlib
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm, t
from sklearn.metrics import adjusted_rand_score, roc_auc_score

from simulation.designs import all_jobs, seed_for
from simulation.run import (case_masks, digest_arrays, file_lock, fit_signature, jsonable, load_npz,
                            prepare_dataset, read_json, write_json)


def align_draws(q, weights, reference):
    matched_q, matched_w = np.empty_like(q), np.empty_like(weights)
    for draw, current in enumerate(q):
        order = linear_sum_assignment(((reference[:, None] - current[None, :]) ** 2).mean(-1))[1]
        matched_q[draw] = current[order]
        matched_w[draw] = weights[draw][:, order]
    return matched_q, matched_w


def recovery(q, weights, truth):
    if q.shape[1] != truth['q'].shape[0]:
        return {}
    q, weights = align_draws(q, weights, truth['q'])
    qmean, wmean = q.mean(0), weights.mean(0)
    qlo, qhi = np.quantile(q, [.025, .975], axis=0)
    wlo, whi = np.quantile(weights, [.025, .975], axis=0)
    return dict(Q_RMSE=float(np.sqrt(np.mean((qmean - truth['q']) ** 2))),
                Q_min_correlation=float(min(np.corrcoef(x, y)[0, 1] for x, y in zip(qmean, truth['q']))),
                W_MAE=float(np.mean(abs(wmean - truth['W']))),
                W_RMSE=float(np.sqrt(np.mean((wmean - truth['W']) ** 2))),
                Q_coverage=float(np.mean((qlo <= truth['q']) & (truth['q'] <= qhi))),
                W_coverage=float(np.mean((wlo <= truth['W']) & (truth['W'] <= whi))),
                Q_median_interval_width=float(np.median(qhi - qlo)),
                W_median_interval_width=float(np.median(whi - wlo)))


def evaluate_case(job, output):
    with file_lock(Path(output) / 'locks' / ('evaluate_' + job['id'] + '.lock')):
        return _evaluate_case(job, output)


def _evaluate_case(job, output):
    from model.balm import FitResult, predict
    folder = Path(output) / 'fits' / job['fit_key']
    complete = read_json(folder / 'complete.json')
    data, truth, base_masks, metadata = prepare_dataset(job, output)
    masks = case_masks(job, base_masks)
    expected = fit_signature(job, metadata, masks, complete['signature']['sampler'],
                             complete['signature']['initialization'])
    if complete['signature'] != expected:
        raise ValueError(f'Fit and evaluation settings do not match: {job["id"]}')
    cache = Path(output) / 'evaluations' / job['id']
    cache_key = dict(complete=complete, masks=digest_arrays(masks),
                     analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (cache / 'summary.json').exists():
        cached = read_json(cache / 'summary.json')
        if cached['key'] == cache_key:
            return cached['record'], load_npz(cache / 'scores.npz')
    test = masks['test']
    record = dict(case=job['id'], fit_key=job['fit_key'], dataset=job['dataset'],
                  experiments=','.join(map(str, job['experiments'])), setting=job['setting'],
                  rep=job['rep'], method=job['method'], mask=job['mask'], phase=job['phase'],
                  M=job['M'], true_M=job['config']['M'], density=job['config']['density'],
                  true_a1=job['config']['a1'] if job['config']['kind'] != 'independent' else None,
                  sensitivity=bool(job.get('sensitivity')),
                  wall_seconds=complete['wall_seconds'])
    score_arrays = dict(test_indices=np.flatnonzero(test))
    if job['method'] in ('alma', 'tbm'):
        result = load_npz(folder / 'estimates.npz')
        probabilities, labels = result['topology_score'], result['labels']
        if job['method'] == 'tbm':
            record['Brier'] = float(np.mean((probabilities[test] - data['z'][test]) ** 2))
        record['optimizer_converged'] = read_json(folder / 'optimization.json').get('converged')
    else:
        result = FitResult.load(folder)
        evaluation_mask = test | masks.get('validation', False)
        prediction = predict(result, data['z'], data['y'], mask=evaluation_mask)
        q, weights = result.q, result.samples['W']
        aligned_q, aligned_w = align_draws(q, weights, q[0])
        recovery_weights = align_draws(q, weights, truth['q'])[1] if q.shape[1] == truth['q'].shape[0] else aligned_w
        labels = recovery_weights.mean(0).argmax(1)
        score_arrays.update(q_mean=aligned_q.mean(0), w_mean=aligned_w.mean(0))
        if job['method'] != 'weighted':
            probabilities = prediction['presence_probability']
            record.update(recovery(q, weights, truth))
            record['Brier'] = float(np.mean((probabilities[test] - data['z'][test]) ** 2))
            record['log_score'] = float(np.mean(prediction['log_density'][test]))
            score_arrays['test_subject'] = np.sum(np.where(test, prediction['log_density'], 0.), axis=1) / test.sum(1)
            positive = test & (data['z'] == 1)
            strength = prediction['score_mean'] if job['setting'] == 'normal' and 1 in job['experiments'] else prediction['strength_mean']
            record['present_score_MSE'] = float(np.mean((strength[positive] - data['y'][positive]) ** 2))
            if 'validation' in masks:
                val = masks['validation']
                score_arrays['validation_indices'] = np.flatnonzero(val)
                score_arrays['validation_subject'] = np.sum(np.where(val, prediction['log_density'], 0.), axis=1) / val.sum(1)
                record['validation_log_score'] = float(score_arrays['validation_subject'].mean())
            if 2 in job['experiments'] or 3 in job['experiments']:
                blocks = []
                for indices in np.array_split(np.arange(len(q)), min(10, len(q))):
                    block = FitResult({key: value[indices] for key, value in result.samples.items()},
                                      result.log_weights[indices] if result.log_weights is not None else None,
                                      {}, {}, dict(result.metadata, draws=len(indices)))
                    bp = predict(block, data['z'], data['y'], mask=evaluation_mask)
                    blocks.append(bp['log_density'])
                block_ll = np.asarray(blocks)
                for split in ('test', 'validation'):
                    if split not in masks:
                        continue
                    mask = masks[split]
                    score_arrays[split + '_blocks'] = np.sum(np.where(mask, block_ll, 0.), axis=2) / mask.sum(1)
                if len(blocks) > 1:
                    record['log_score_MCSE'] = float(score_arrays['test_blocks'].mean(1).std(ddof=1) / np.sqrt(len(blocks)))
            if 'a1' in result.samples:
                coupling = np.asarray(result.samples['a1']).reshape(-1)
                record.update(a1_mean=float(coupling.mean()), a1_upper95=float(np.quantile(coupling, .95)),
                              a1_lower025=float(np.quantile(coupling, .025)),
                              a1_upper975=float(np.quantile(coupling, .975)))
                if record['true_a1'] is not None:
                    record['a1_bias'] = float(coupling.mean() - record['true_a1'])
                    record['a1_coverage'] = float(record['a1_lower025'] <= record['true_a1'] <= record['a1_upper975'])
        else:
            probabilities = None
        record['B_MSE'] = float(np.mean((prediction['zero_filled_mean'][test] - data['y'][test]) ** 2))
        if 'diverging' in result.diagnostics:
            record['divergences'] = int(np.sum(result.diagnostics['diverging']))
    record['ARI'] = float(adjusted_rand_score(truth['W'].argmax(1), labels))
    if probabilities is not None:
        record['AUC'] = float(roc_auc_score(data['z'][test], probabilities[test]))
    cache.mkdir(parents=True, exist_ok=True)
    with (cache / 'scores.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **score_arrays)
    os.replace(cache / 'scores.npz.tmp', cache / 'scores.npz')
    record = jsonable(record)
    write_json(cache / 'summary.json', dict(key=cache_key, record=record))
    return record, score_arrays


def choose_template_number(records, candidates=(2, 3, 4, 5)):
    """Paired one-SE rule; diagnostics do not veto the selected dimension."""
    subjects = np.asarray([r['validation_subject'] for r in records])
    best = subjects.mean(1).argmax()
    gaps = subjects[best] - subjects
    means = gaps.mean(1)
    se = gaps.std(1, ddof=1) / np.sqrt(subjects.shape[1])
    chosen = np.flatnonzero(means <= se + 1e-12)[0]
    result = dict(selected=candidates[chosen], best=candidates[best],
                  scores=subjects.mean(1).tolist(), gaps=means.tolist(), paired_se=se.tolist())
    uncertain, errors = [], {}
    for index in range(chosen + 1):
        if index == best:
            continue
        a = records[best]['validation_blocks'] - subjects[index]
        b = subjects[best] - records[index]['validation_blocks']
        ma = a.mean(1) - a.std(1, ddof=1) / np.sqrt(subjects.shape[1])
        mb = b.mean(1) - b.std(1, ddof=1) / np.sqrt(subjects.shape[1])
        error = np.hypot(ma.std(ddof=1), mb.std(ddof=1)) / np.sqrt(len(ma))
        errors[str(candidates[index])] = float(error)
        if abs(means[index] - se[index]) <= 2 * error:
            uncertain.append(candidates[index])
    result.update(numerically_uncertain_candidates=uncertain, margin_MCSE=errors)
    return result


def selection_tables(jobs, scores, destination, prefix='selection'):
    selected, combinations, stability = [], [], []
    keys = sorted({(j['rep'], j['config']['M']) for j in jobs if 2 in j['experiments']})
    for rep, true_m in keys:
        group = { (j['M'], j['phase']): scores[j['id']] for j in jobs
                  if j['rep'] == rep and j['config']['M'] == true_m and j['id'] in scores
                  and 2 in j['experiments'] and not j.get('sensitivity') }
        for phase in ('primary', 'restart'):
            if all((m, phase) in group for m in (2, 3, 4, 5)):
                decision = choose_template_number([group[m, phase] for m in (2, 3, 4, 5)])
                selected.append(dict(rep=rep, true_M=true_m, phase=phase, **decision))
        if len(group) == 8:
            for phases in itertools.product(('primary', 'restart'), repeat=4):
                decision = choose_template_number([group[m, phase] for m, phase in zip((2, 3, 4, 5), phases)])
                combinations.append(dict(rep=rep, true_M=true_m, phases=','.join(phases), **decision))
            for m in (2, 3, 4, 5):
                a, b = group[m, 'primary'], group[m, 'restart']
                order = linear_sum_assignment(((a['q_mean'][:, None] - b['q_mean'][None, :]) ** 2).mean(-1))[1]
                stability.append(dict(rep=rep, true_M=true_m, M=m,
                    Q_between_start_RMSE=float(np.sqrt(np.mean((a['q_mean'] - b['q_mean'][order]) ** 2))),
                    W_between_start_MAE=float(np.mean(abs(a['w_mean'] - b['w_mean'][:, order])))))
    for name, records in [(prefix + '_datasets', selected), (prefix + '_start_combinations', combinations),
                           (prefix + '_latent_stability', stability)]:
        pd.DataFrame(records).to_csv(destination / (name + '.csv'), index=False)
    if selected:
        counts = pd.DataFrame(selected).groupby(['true_M', 'phase', 'selected']).size().unstack(fill_value=0)
        counts.reindex(columns=[2, 3, 4, 5], fill_value=0).to_csv(destination / (prefix + '_counts.csv'))


def coupling_tables(jobs, records, scores, destination, prefix='coupling'):
    table = {r['case']: r for r in records}
    pairs = []
    for j in jobs:
        if 3 not in j['experiments'] or j['method'] != 'coupled' or j['id'] not in scores:
            continue
        null_id = j['dataset'] + '__null'
        if null_id not in scores:
            continue
        a, b = scores[j['id']], scores[null_id]
        if not np.array_equal(a['test_indices'], b['test_indices']):
            raise ValueError('Coupled and fixed-zero scores use different test entries')
        delta = a['test_subject'] - b['test_subject']
        rng = np.random.default_rng(seed_for('S4_bootstrap_v1', j['dataset']))
        boot = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
        lower = float(np.quantile(boot, .05))
        error = float(np.hypot(table[j['id']]['log_score_MCSE'], table[null_id]['log_score_MCSE']))
        pairs.append(dict(dataset=j['dataset'], rep=j['rep'], density=j['config']['density'], true_a1=j['config']['a1'],
                          delta_log_score=float(delta.mean()), delta_AUC=table[j['id']]['AUC'] - table[null_id]['AUC'],
                          bootstrap_lower95=lower, positive=lower > 0, combined_MCSE=error,
                          MC_boundary=abs(lower) <= 2 * error))
    pd.DataFrame(pairs).to_csv(destination / (prefix + '_pairs.csv'), index=False)
    rows = []
    if pairs:
        for (density, coupling), group in pd.DataFrame(pairs).groupby(['density', 'true_a1']):
            count, n = int(group['positive'].sum()), len(group)
            z = norm.ppf(.975)
            fraction = count / n
            center = (fraction + z * z / (2 * n)) / (1 + z * z / n)
            half = z * np.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            lo, hi = max(0., center - half), min(1., center + half)
            coupled = [r for r in records if r['method'] == 'coupled' and r['density'] == density
                       and r['true_a1'] == coupling and r['experiments'] == '3']
            rows.append(dict(density=density, true_a1=coupling, n=n, positive=count,
                             fraction=count / n, lower95=lo, upper95=hi,
                             MC_boundary=int(group['MC_boundary'].sum()),
                             delta_log_score_mean=group['delta_log_score'].mean(),
                             delta_log_score_sd=group['delta_log_score'].std(ddof=1),
                             delta_AUC_mean=group['delta_AUC'].mean(), delta_AUC_sd=group['delta_AUC'].std(ddof=1),
                             a1_bias_mean=np.mean([r['a1_bias'] for r in coupled]),
                             a1_RMSE=np.sqrt(np.mean([r['a1_bias'] ** 2 for r in coupled])),
                             a1_interval_width=np.mean([r['a1_upper975'] - r['a1_lower025'] for r in coupled]),
                             a1_coverage=np.mean([r['a1_coverage'] for r in coupled]) if coupling > 0 else None,
                             a1_upper95_mean=np.mean([r['a1_upper95'] for r in coupled])))
    pd.DataFrame(rows).to_csv(destination / (prefix + '_summary.csv'), index=False)


def aggregate_table(records, destination):
    frame = pd.DataFrame(records)
    groups = ['setting', 'method', 'mask', 'M', 'true_M', 'density', 'true_a1', 'phase', 'sensitivity']
    metrics = [c for c in ('AUC', 'ARI', 'Brier', 'B_MSE', 'present_score_MSE', 'log_score', 'Q_RMSE',
               'W_MAE', 'W_RMSE', 'Q_coverage', 'W_coverage', 'Q_min_correlation',
               'Q_median_interval_width', 'W_median_interval_width', 'a1_mean', 'a1_bias', 'a1_upper95',
               'a1_coverage', 'wall_seconds') if c in frame]
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            rows.append(dict(zip(groups, key), metric=metric, n=len(values), mean=float(values.mean()), sd=sd,
                             se=sd / np.sqrt(len(values))))
    pd.DataFrame(rows).to_csv(destination / 'group_summary.csv', index=False)


def paired_table(records, destination):
    frame = pd.DataFrame(records)
    rows = []
    for (setting, mask), group in frame.loc[~frame.sensitivity].groupby(['setting', 'mask']):
        if setting == 'normal':
            group = group.loc[group.phase == '']
        for left, right in itertools.combinations(('coupled', 'alma', 'tbm'), 2):
            a, b = group.loc[group.method == left], group.loc[group.method == right]
            if a.empty or b.empty:
                continue
            paired = a.merge(b, on=['dataset', 'rep'], suffixes=('_a', '_b'), validate='one_to_one')
            for metric in ('AUC', 'ARI'):
                diff = (paired[metric + '_a'] - paired[metric + '_b']).dropna()
                if not len(diff):
                    continue
                se = diff.std(ddof=1) / np.sqrt(len(diff))
                half = t.ppf(.975, len(diff) - 1) * se if len(diff) > 1 else np.nan
                rows.append(dict(setting=setting, mask=mask, contrast=left + '-' + right, metric=metric,
                                 n=len(diff), mean=diff.mean(), se=se, lower95=diff.mean() - half, upper95=diff.mean() + half))
    pd.DataFrame(rows).to_csv(destination / 'paired_comparisons.csv', index=False)


def summarize(jobs, output, *, allow_partial=False):
    output = Path(output)
    destination = output / ('sensitivity_summary' if any(j.get('sensitivity') for j in jobs) else 'summary')
    destination.mkdir(parents=True, exist_ok=True)
    missing = [j['id'] for j in jobs if not (output / 'fits' / j['fit_key'] / 'complete.json').exists()]
    write_json(destination / 'coverage.json', dict(requested_cases=len(jobs), completed_cases=len(jobs) - len(missing),
                missing=missing, manuscript_replicates_per_setting=50, selected_replicates=sorted({j['rep'] for j in jobs})))
    if missing and not allow_partial:
        raise ValueError(f'{len(missing)} requested fits are missing; see {destination / "coverage.json"}. Use --allow-partial for a labelled partial summary.')
    records, scores = [], {}
    for job in jobs:
        if job['id'] in missing:
            continue
        print(f'Summarizing {job["id"]}', flush=True)
        record, arrays = evaluate_case(job, output)
        records.append(record)
        scores[job['id']] = arrays
    if not records:
        raise ValueError('No completed requested fits to summarize')
    pd.DataFrame(records).to_csv(destination / 'per_fit.csv', index=False)
    aggregate_table(records, destination)
    paired_table(records, destination)
    selection_tables(jobs, scores, destination)
    coupling_tables(jobs, records, scores, destination)
    if any(j.get('sensitivity') for j in jobs):
        sensitivity_tables(jobs, records, scores, output, destination)
    print(f'Saved summaries: {destination}')


def sensitivity_tables(jobs, records, scores, output, destination):
    lookup = {j['id']: j for j in all_jobs(include_sensitivity=True)}
    comparison, unavailable, stability = [], [], []
    table = {r['case']: r for r in records}
    old_records, old_scores = {}, {}
    def original_score(case):
        original = lookup[case]
        if not (Path(output) / 'fits' / original['fit_key'] / 'complete.json').exists():
            unavailable.append(case)
            return False
        if case not in old_scores:
            old_records[case], old_scores[case] = evaluate_case(original, output)
        return True
    for job in jobs:
        if job['id'] not in scores:
            continue
        if job['start_scheme'] == 'fixed_zero_draw199':
            original_id = job['dataset'] + '__null'
        else:
            phase = 'primary' if job['start_scheme'] == 'small_extras' else 'restart'
            original_id = f'r{job["rep"]:03d}_t2_m{job["M"]}_{phase}'
        if not original_score(original_id):
            continue
        old = old_records[original_id]
        row = dict(case=job['id'], original_case=original_id, scheme=job['start_scheme'])
        for metric in ('Q_RMSE', 'W_MAE', 'log_score', 'AUC'):
            row[metric + '_original'] = old.get(metric)
            row[metric + '_sensitivity'] = table[job['id']].get(metric)
        comparison.append(row)
    pd.DataFrame(comparison).to_csv(destination / 'initialization_comparison.csv', index=False)
    changed_groups = sorted({(j['rep'], j['M']) for j in jobs if 2 in j['experiments'] and j['id'] in scores})
    selected_jobs, replacements = [], {}
    for rep, m in changed_groups:
        ids = [f'r{rep:03d}_t2_m{m}_{phase}' for phase in ('primary', 'restart')]
        new_ids = [f'r{rep:03d}_t2_m{m}_{phase}' for phase in ('small_extras', 'equal_split')]
        if not all(original_score(case) for case in ids) or not all(case in scores for case in new_ids):
            continue
        row = dict(rep=rep, true_M=2, M=m)
        for stage, pair in [('original', [old_scores[case] for case in ids]),
                            ('expanded', [scores[case] for case in new_ids])]:
            a, b = pair
            order = linear_sum_assignment(((a['q_mean'][:, None] - b['q_mean'][None, :]) ** 2).mean(-1))[1]
            row['Q_between_start_RMSE_' + stage] = float(np.sqrt(np.mean((a['q_mean'] - b['q_mean'][order]) ** 2)))
            row['W_between_start_MAE_' + stage] = float(np.mean(abs(a['w_mean'] - b['w_mean'][:, order])))
        stability.append(row)
        replacements.update({old: scores[new] for old, new in zip(ids, new_ids)})
    for rep in sorted({rep for rep, _ in changed_groups}):
        for m, phase in itertools.product((2, 3, 4, 5), ('primary', 'restart')):
            case = f'r{rep:03d}_t2_m{m}_{phase}'
            if original_score(case):
                selected_jobs.append(lookup[case])
    pd.DataFrame(stability).to_csv(destination / 'expanded_start_stability.csv', index=False)
    if selected_jobs:
        selection_tables(selected_jobs, old_scores, destination, prefix='targeted_selection_original')
        selection_tables(selected_jobs, dict(old_scores, **replacements), destination, prefix='targeted_selection_expanded')
    if any(3 in j['experiments'] for j in jobs):
        pair_jobs = []
        new_scores, new_records = {}, {}
        for rep in range(50):
            for method in ('coupled', 'null'):
                case = f'cell00_rep{rep:03d}__{method}'
                if original_score(case):
                    pair_jobs.append(lookup[case])
                    new_scores[case], new_records[case] = old_scores[case], old_records[case]
        for job in jobs:
            if 3 in job['experiments'] and job['id'] in scores:
                original_id = job['dataset'] + '__null'
                if original_id in new_records:
                    new_scores[original_id] = scores[job['id']]
                    new_records[original_id] = dict(table[job['id']], case=original_id)
        coupling_tables(pair_jobs, list(old_records.values()), old_scores, destination, prefix='targeted_null_original')
        coupling_tables(pair_jobs, list(new_records.values()), new_scores, destination, prefix='targeted_null_reinitialized')
    write_json(destination / 'sensitivity_coverage.json', dict(missing_original_cases=sorted(set(unavailable)),
               complete_original_context=not unavailable,
               interpretation='Targeted initialization comparisons; original fits and aggregate tables remain separate.'))


def plot_results(output):
    """Plot aggregate method comparisons from summaries without rerunning fits."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    table = pd.read_csv(output / 'summary/group_summary.csv')
    table['phase'] = table['phase'].fillna('')
    subset = table.loc[table['method'].isin(['coupled', 'alma', 'tbm']) & (table['phase'] == '')]
    settings = [s for s in ('normal_fresh', 'block', 'alpha01', 'alpha3', 'n30', 'L200', 'K2', 'weak_signal', 'high_noise')
                if s in set(subset.setting)]
    if not settings:
        settings = list(dict.fromkeys(subset.setting))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(settings) * 1.1), 4), constrained_layout=True)
    for axis, metric in zip(axes, ('AUC', 'ARI')):
        for offset, (method, color, marker) in enumerate([('coupled', '.15', 'o'), ('alma', '.5', 's'), ('tbm', '.75', '^')]):
            part = subset.loc[(subset.method == method) & (subset.metric == metric) & (subset['mask'] == 'reference')]
            part = part.drop_duplicates('setting').set_index('setting').reindex(settings)
            axis.errorbar(np.arange(len(settings)) + (offset - 1) * .12, part['mean'], yerr=part['sd'],
                          color=color, marker=marker, linestyle='none', capsize=2,
                          label={'coupled': 'BALM', 'alma': 'ALMA', 'tbm': 'TBM'}[method])
        axis.set_xticks(np.arange(len(settings)), settings, rotation=45, ha='right')
        axis.set_ylabel(metric)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False)
    target = output / 'figures'
    target.mkdir(exist_ok=True)
    fig.savefig(target / 'method_comparison.png', dpi=180)
    fig.savefig(target / 'method_comparison.pdf')
    plt.close(fig)
    print(f'Saved figures: {target}')
