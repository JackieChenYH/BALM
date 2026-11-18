"""HCP aggregate prediction, exploratory associations and grayscale figures."""
from pathlib import Path
from collections import defaultdict
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, t
from sklearn.metrics import roc_auc_score

from .data import COHORTS, ROI_LABELS, load_people, read_npz


TRANSFER_PAIRS = {
    'F31_35': [(3, 'Language_Task_Acc'), (4, 'Language_Task_Acc'), (4, 'WM_Task_0bk_Face_Median_RT_Target')],
    'M22_25': [(1, 'DDisc_SV_3yr_200'), (2, 'Language_Task_Story_Avg_Difficulty_Level'),
              (3, 'Emotion_Task_Shape_Median_RT'), (4, 'Emotion_Task_Shape_Median_RT')],
}
PANELS = [('F31_35', 3, 'WM_Task_0bk_Face_Median_RT_Target', 'Face-target reaction time (ms)'),
          ('F31_35', 4, 'WM_Task_0bk_Face_Median_RT_Target', 'Face-target reaction time (ms)'),
          ('M22_25', 2, 'Language_Task_Story_Avg_Difficulty_Level', 'Story difficulty'),
          ('M22_25', 3, 'Emotion_Task_Shape_Median_RT', 'Shape-matching reaction time (ms)'),
          ('M26_30', 3, 'WM_Task_2bk_Acc', '2-back accuracy (%)')]


def standard_ranks(x):
    ranks = rankdata(x, axis=0, method='average')
    ranks -= ranks.mean(axis=0)
    return ranks / np.sqrt((ranks * ranks).sum(axis=0))


def correlations(x, y):
    x, y = rankdata(x, axis=-1), rankdata(y, axis=-1)
    x, y = x - x.mean(-1, keepdims=True), y - y.mean(-1, keepdims=True)
    return np.sum(x * y, axis=-1) / np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))


def permutation_p(x, y, seed, permutations=99999):
    rng = np.random.default_rng(seed)
    xr, yr = standard_ranks(np.asarray(x)), standard_ranks(np.asarray(y))
    observed, exceed = float(xr @ yr), 0
    for offset in range(0, permutations, 1000):
        size = min(1000, permutations - offset)
        null = rng.permuted(np.tile(yr, (size, 1)), axis=1) @ xr
        exceed += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-14))
    return (exceed + 1) / (permutations + 1)


def bootstrap_rho(x, y, seed, resamples=20000):
    rng = np.random.default_rng(seed)
    parts = []
    for offset in range(0, resamples, 1000):
        indices = rng.integers(len(x), size=(min(1000, resamples - offset), len(x)))
        parts.append(correlations(x[indices], y[indices]))
    samples = np.concatenate(parts)
    if not np.isfinite(samples).all():
        raise ValueError('A bootstrap correlation is undefined; do not silently discard resamples')
    return np.quantile(samples, [.025, .975])


def mean_interval(values, seed, resamples=20000):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    parts = []
    for offset in range(0, resamples, 1000):
        indices = rng.integers(len(values), size=(min(1000, resamples - offset), len(values)))
        parts.append(values[indices].mean(1))
    return np.quantile(np.concatenate(parts), [.025, .975])


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order] * (len(p) - np.arange(len(p)))))
    return result


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1])
    return result


def summarize_transfer(folder):
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text())
    records = []
    source = config['source']
    for index in range(config['subjects']):
        record = read_npz(folder / 'subjects' / f'{index:04d}.npz')
        if record['W'].shape != (config['draws'], 4) or not np.isfinite(record['W']).all():
            raise ValueError(f'Invalid membership draws at participant {index}')
        np.testing.assert_allclose(record['W'].sum(1), 1., atol=1e-12)
        records.append(record)
    cohorts = (['M26_30', 'M31_35', 'M36plus'] if source == 'M22_25'
               else ['F26_30', 'F22_25', 'M22_25', 'M26_30', 'M31_35', 'M36plus'])
    summaries = []
    for number, cohort in enumerate(cohorts):
        chosen = [record for record in records if str(record['cohort']) == cohort]
        if not chosen:
            continue
        cat = lambda key: np.concatenate([record[key] for record in chosen])
        z = cat('z')
        gain = np.array([(record['log_score'] - record['fixed_log_score']).mean() for record in chosen])
        seed = 65000 + number if source == 'M22_25' else 66000 + number
        low, high = mean_interval(gain, seed) if len(chosen) > 3 else (np.nan, np.nan)
        score_auc = lambda key: float(roc_auc_score(z, cat(key))) if len(np.unique(z)) == 2 else np.nan
        summaries.append(dict(source=source, cohort=cohort, subjects=len(chosen),
                              pooled_AUC=score_auc('probability'), fixed_W_AUC=score_auc('fixed_probability'),
                              source_empirical_AUC=score_auc('empirical_probability'),
                              log_score_gain=float(gain.mean()), gain_bootstrap_lower=low, gain_bootstrap_upper=high))
    output = pd.DataFrame(summaries)
    output.to_csv(folder / 'network_summary.csv', index=False)
    np.savez_compressed(folder / 'memberships.npz',
                        subject_ids=np.array([str(record['subject_id']) for record in records]),
                        cohorts=np.array([str(record['cohort']) for record in records]),
                        W_mean=np.stack([record['W'].mean(0) for record in records]))
    return output


def score_values(frame, field):
    if field.startswith('NEORAW_'):
        allowed = {'SD': 1., 'D': 2., 'N': 3., 'A': 4., 'SA': 5.}
        if not set(frame[field].dropna().unique()).issubset(allowed):
            raise ValueError(f'Unrecognized agreement response in {field}')
        return frame[field].map(allowed).to_numpy(dtype=float, na_value=np.nan)
    converted = pd.to_numeric(frame[field], errors='coerce')
    if converted.notna().sum() != frame[field].notna().sum():
        raise ValueError(f'Non-numeric values in {field}')
    return converted.to_numpy(dtype=float, na_value=np.nan)


def screen(args):
    people = load_people(args.people)
    fields = pd.read_csv(args.fields).field.tolist()
    if len(fields) != 293 or len(set(fields)) != 293:
        raise ValueError('The manuscript screen uses exactly 293 distinct fields')
    missing = set(fields) - set(people.columns)
    if missing:
        raise ValueError(f'The full screen requires all 293 fields; missing {sorted(missing)}')
    records = []
    for cohort_index, cohort in enumerate(COHORTS):
        source = read_npz(Path(args.fits) / f'{cohort}_M4' / 'source.npz')
        w = source['W_mean']
        frame = people.loc[source['subject_ids'].astype(str)]
        values = {field: score_values(frame, field) for field in fields}
        groups, lookup = defaultdict(list), {}
        for field in fields:
            y = values[field]
            mask = np.isfinite(y)
            n = int(mask.sum())
            varying = len(np.unique(y[mask])) >= 2
            testable = n >= 20 and varying
            rho = (standard_ranks(w[mask]).T @ standard_ranks(y[mask, None])).ravel() if n >= 4 and varying else np.full(4, np.nan)
            for m in range(4):
                record = dict(subset=cohort, score=field, template=m + 1, n=n, rho=float(rho[m]),
                              p_permutation=np.nan if testable else 1., status='tested' if testable else 'untestable')
                records.append(record)
                lookup[(field, m)] = record
            if testable:
                groups[mask.tobytes()].append(field)
        for group_index, (key, shared_fields) in enumerate(groups.items()):
            mask = np.frombuffer(key, dtype=bool)
            xr = standard_ranks(w[mask])
            yr = standard_ranks(np.column_stack([values[field][mask] for field in shared_fields]))
            observed = xr.T @ yr
            counts = np.zeros(observed.shape, dtype=np.int64)
            rng = np.random.default_rng(np.random.SeedSequence([67000, cohort_index, group_index]))
            for offset in range(0, args.permutations, 1000):
                size = min(1000, args.permutations - offset)
                permutations = rng.permuted(np.tile(np.arange(mask.sum()), (size, 1)), axis=1)
                null = (xr[permutations].transpose(0, 2, 1).reshape(size * 4, mask.sum()) @ yr).reshape(size, 4, len(shared_fields))
                counts += np.sum(np.abs(null) >= np.abs(observed)[None, :, :] - 1e-12, axis=0)
            for j, field in enumerate(shared_fields):
                for m in range(4):
                    lookup[(field, m)]['p_permutation'] = (int(counts[m, j]) + 1) / (args.permutations + 1)
        print(f'{cohort}: all 293 fields analyzed', flush=True)
    result = pd.DataFrame(records)
    result['q_BH_global'] = bh(result.p_permutation)
    result['permutations'] = args.permutations
    result.to_csv(args.out, index=False)


def behavioral_transfer(args):
    people = load_people(args.people)
    reference = pd.read_csv(args.screen)
    records = []
    for source_name, transfer_folder in [('F31_35', args.female_transfer), ('M22_25', args.male_transfer)]:
        source = read_npz(Path(args.fits) / f'{source_name}_M4' / 'source.npz')
        target = read_npz(Path(transfer_folder) / 'memberships.npz')
        groups = [source_name] + list(dict.fromkeys(target['cohorts']))
        for group_index, cohort in enumerate(groups):
            is_source = cohort == source_name
            mask = target['cohorts'] == cohort
            ids = source['subject_ids'] if is_source else target['subject_ids'][mask]
            w = source['W_mean'] if is_source else target['W_mean'][mask]
            frame = people.loc[ids.astype(str)]
            for j, (template, field) in enumerate(TRANSFER_PAIRS[source_name]):
                score = score_values(frame, field)
                valid = np.isfinite(score)
                x, y = w[valid, template - 1], score[valid]
                rho = float(spearmanr(x, y).statistic) if len(x) >= 3 else np.nan
                row = dict(source=source_name, cohort=cohort, template=template, field=field, n=len(x), rho=rho,
                           status='source_reference' if is_source else 'descriptive_only' if len(x) <= 3 else 'transfer_test',
                           bootstrap_lower=np.nan, bootstrap_upper=np.nan, p_permutation=np.nan, source_original_q_BH=np.nan)
                seed = 68000 + (0 if source_name == 'F31_35' else 1000) + group_index * 10 + j
                permutation_seed = seed + 10000
                if source_name == 'M22_25' and template in (2, 3):
                    original_group = ['M22_25', 'M26_30', 'M31_35', 'M36plus'].index(cohort)
                    seed = 71000 + original_group * 10 + (template - 2)
                    permutation_seed = seed + 100
                if len(x) > 3:
                    row['bootstrap_lower'], row['bootstrap_upper'] = bootstrap_rho(x, y, seed, args.resamples)
                    if not is_source:
                        row['p_permutation'] = permutation_p(x, y, permutation_seed, args.permutations)
                if is_source:
                    match = reference[(reference.subset == cohort) & (reference.template == template) & (reference.score == field)]
                    if len(match) != 1:
                        raise ValueError('Missing source association in the full screen')
                    ref = match.iloc[0]
                    np.testing.assert_allclose(rho, ref.rho, atol=1e-12, rtol=0)
                    row.update(p_permutation=float(ref.p_permutation), source_original_q_BH=float(ref.q_BH_global))
                records.append(row)
    result = pd.DataFrame(records)
    valid = result.status.eq('transfer_test')
    for source_name in TRANSFER_PAIRS:
        source_mask = valid & result.source.eq(source_name)
        result.loc[source_mask, 'p_Holm_source_family'] = holm(result.loc[source_mask, 'p_permutation'])
        for cohort in result.loc[source_mask, 'cohort'].unique():
            subset = source_mask & result.cohort.eq(cohort)
            result.loc[subset, 'p_Holm_within_cohort'] = holm(result.loc[subset, 'p_permutation'])
    result.loc[valid, 'p_Holm_all_tests'] = holm(result.loc[valid, 'p_permutation'])
    result['permutations'], result['bootstrap_resamples'] = args.permutations, args.resamples
    result.to_csv(args.out, index=False)


def regional_intensity(q):
    m, edges = q.shape
    if edges != 2278:
        raise ValueError('Expected 68-region off-diagonal templates')
    matrices = np.zeros((m, 68, 68))
    ii, jj = np.triu_indices(68, 1)
    matrices[:, ii, jj] = q
    matrices[:, jj, ii] = q
    intensity = np.abs(matrices).sum(2) / 67
    relative = intensity / np.maximum(intensity.max(1, keepdims=True), np.finfo(float).tiny)
    return matrices, intensity, relative


def grouped_scores(w, score):
    cuts = np.quantile(w, [1 / 3, 2 / 3], method='linear')
    groups = np.searchsorted(cuts, w, side='left')
    rows = []
    for group, label in enumerate(['Low', 'Moderate', 'High']):
        members = groups == group
        valid = members & np.isfinite(score)
        y = score[valid]
        mean = y.mean() if len(y) else np.nan
        half = t.ppf(.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
        rows.append(dict(group=label, n_members=int(members.sum()), n_scores=len(y),
                         W_mean=w[members].mean() if members.any() else np.nan,
                         score_mean=mean, ci_lower=mean - half, ci_upper=mean + half,
                         lower_cut=cuts[0], upper_cut=cuts[1]))
    return pd.DataFrame(rows)


def surface_map(ax, relative, roi_labels, hemisphere, surfaces, figure):
    from nibabel.freesurfer import read_annot, read_geometry, read_morph_data
    from nilearn.plotting import plot_surf_stat_map
    root = Path(surfaces)
    vertex_labels, _, names = read_annot(root / 'label' / f'{hemisphere}.aparc.annot')
    names = [name.decode('utf-8') for name in names]
    mapping = {name: k for k, name in enumerate(roi_labels)}
    texture = np.zeros(len(vertex_labels))
    for index, name in enumerate(names):
        key = f'ctx-{hemisphere}-{name}'
        if key in mapping:
            texture[vertex_labels == index] = relative[mapping[key]]
    plot_surf_stat_map(read_geometry(root / 'surf' / f'{hemisphere}.inflated'), texture,
                       bg_map=read_morph_data(root / 'surf' / f'{hemisphere}.sulc'),
                       hemi='left' if hemisphere == 'lh' else 'right', view='lateral',
                       axes=ax, figure=figure, cmap='Greys', colorbar=False, threshold=None,
                       vmin=0, vmax=1, symmetric_cbar=False, bg_on_data=False, avg_method='median', alpha=1)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    people = load_people(args.people)
    sources = {cohort: read_npz(Path(args.fits) / f'{cohort}_M4' / 'source.npz') for cohort in COHORTS}
    regions, groups = [], []
    for cohort, source in sources.items():
        _, intensity, relative = regional_intensity(source['Q_mean'])
        for m in range(4):
            order = np.argsort(-intensity[m], kind='stable')
            for rank, index in enumerate(order, 1):
                regions.append(dict(cohort=cohort, template=m + 1, rank=rank, roi=source['roi_labels'][index],
                                    intensity=intensity[m, index], relative=relative[m, index],
                                    representative=relative[m, index] >= .8))
    fig = plt.figure(figsize=(11.5, 16))
    grid = fig.add_gridspec(5, 3, width_ratios=[1, 1, 1.25], hspace=.55, wspace=.14)
    for panel, (cohort, template, field, label) in enumerate(PANELS):
        source = sources[cohort]
        matrices, _, relative = regional_intensity(source['Q_mean'])
        frame = people.loc[source['subject_ids'].astype(str)]
        table = grouped_scores(source['W_mean'][:, template - 1], score_values(frame, field))
        groups.append(table.assign(panel=chr(97 + panel), cohort=cohort, template=template, field=field))
        if args.surfaces:
            for column, hemisphere in enumerate(('lh', 'rh')):
                ax = fig.add_subplot(grid[panel, column], projection='3d')
                surface_map(ax, relative[template - 1], source['roi_labels'], hemisphere, args.surfaces, fig)
                ax.set_title(f'{cohort} T{template} — {"left" if column == 0 else "right"}', fontsize=10)
        else:
            ax = fig.add_subplot(grid[panel, :2])
            matrix = matrices[template - 1].copy()
            np.fill_diagonal(matrix, np.nan)
            limit = np.max(np.abs(matrix[np.isfinite(matrix)]))
            im = ax.imshow(matrix, cmap='gray_r', vmin=-limit, vmax=limit)
            fig.colorbar(im, ax=ax, shrink=.8, label='Signed template strength')
            ax.set_title(f'{cohort}, template {template}')
        ax = fig.add_subplot(grid[panel, 2])
        means = table.score_mean.to_numpy()
        error = table.ci_upper.to_numpy() - means
        valid = np.isfinite(means) & np.isfinite(error)
        ax.errorbar(np.arange(3)[valid], means[valid], yerr=error[valid], color='.15', fmt='o', capsize=4)
        ax.set_xticks(range(3), table.group)
        ax.set_ylabel(label)
        ax.set_title(f'({chr(97 + panel)}) {cohort}, T{template}', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output / ('hcp_five_panels.png' if args.surfaces else 'hcp_template_matrices.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(regions).to_csv(output / 'regional_intensities.csv', index=False)
    pd.concat(groups, ignore_index=True).to_csv(output / 'behavior_groups.csv', index=False)


def plot_summary(args):
    """Render the archived aggregate figure values without participant data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder, output = Path(args.results), Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    regions = pd.read_csv(folder / 'regional_intensities.csv')
    groups = pd.read_csv(folder / 'behavior_groups.csv')
    fig = plt.figure(figsize=(11.5, 16))
    grid = fig.add_gridspec(5, 3, width_ratios=[1, 1, 1.25], hspace=.55, wspace=.14)
    for panel, (cohort, template, field, label) in enumerate(PANELS):
        r = regions[(regions.cohort == cohort) & (regions.template == template)].set_index('roi')
        relative = r.loc[ROI_LABELS, 'relative'].to_numpy()
        if args.surfaces:
            for column, hemisphere in enumerate(('lh', 'rh')):
                ax = fig.add_subplot(grid[panel, column], projection='3d')
                surface_map(ax, relative, ROI_LABELS, hemisphere, args.surfaces, fig)
                ax.set_title(f'{cohort} T{template} — {"left" if column == 0 else "right"}', fontsize=10)
        else:
            ax = fig.add_subplot(grid[panel, :2])
            ax.bar(np.arange(68), relative, color='.25', width=.9)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('Cortical ROI index (left, then right hemisphere)')
            ax.set_ylabel('Relative incident strength')
            ax.set_title(f'{cohort}, template {template}')
            ax.spines[['top', 'right']].set_visible(False)
        table = groups[(groups.cohort == cohort) & (groups.template == template) & (groups.field == field)]
        table = table.set_index('group').loc[['Low', 'Moderate', 'High']]
        ax = fig.add_subplot(grid[panel, 2])
        mean = table.score_mean.to_numpy()
        error = table.ci_upper.to_numpy() - mean
        ax.errorbar(np.arange(3), mean, yerr=error, fmt='o', color='.15', capsize=4)
        ax.set_xticks(range(3), ['Low', 'Moderate', 'High'])
        ax.set_ylabel(label)
        ax.set_title(f'({chr(97 + panel)}) {cohort}, T{template}', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    filename = 'hcp_five_panels.png' if args.surfaces else 'hcp_regional_summary.png'
    fig.savefig(output / filename, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('screen', help='All 293 behavioral fields, all four memberships, all three source cohorts')
    p.add_argument('--fits', required=True)
    p.add_argument('--people', required=True)
    p.add_argument('--fields', default=str(Path(__file__).with_name('behavior_fields.csv')))
    p.add_argument('--permutations', type=int, default=99999)
    p.add_argument('--out', required=True)
    p = commands.add_parser('behavior', help='Seven source-specific pairings with source-family and joint Holm correction')
    p.add_argument('--fits', required=True)
    p.add_argument('--people', required=True)
    p.add_argument('--screen', required=True)
    p.add_argument('--female-transfer', required=True)
    p.add_argument('--male-transfer', required=True)
    p.add_argument('--permutations', type=int, default=99999)
    p.add_argument('--resamples', type=int, default=20000)
    p.add_argument('--out', required=True)
    p = commands.add_parser('network', help='Summarize completed target membership inferences')
    p.add_argument('--transfer', required=True)
    p = commands.add_parser('plot', help='Plot five grayscale template/behavior panels from saved posterior means')
    p.add_argument('--fits', required=True)
    p.add_argument('--people', required=True)
    p.add_argument('--surfaces', help='Local fsaverage5 folder containing surf/ and label/; no download')
    p.add_argument('--out', required=True)
    p = commands.add_parser('plot-summary', help='Plot public aggregate ROI and behavioral values without participant data')
    p.add_argument('--results', default='results/hcp')
    p.add_argument('--surfaces', help='Local fsaverage5 folder containing surf/ and label/; no download')
    p.add_argument('--out', required=True)
    args = parser.parse_args()
    if getattr(args, 'permutations', 1) < 1 or getattr(args, 'resamples', 1) < 1:
        parser.error('Resampling counts must be positive')
    if args.command == 'network':
        summarize_transfer(args.transfer)
    else:
        {'screen': screen, 'behavior': behavioral_transfer, 'plot': plot, 'plot-summary': plot_summary}[args.command](args)


if __name__ == '__main__':
    main()
