# HCP inputs and analysis

Run the commands below from the repository root. All generated HCP inputs, fits and memberships contain participant-level information; keep them under a private output directory such as `outputs/hcp/`. The distributed `results/hcp/` directory contains aggregate summaries only.

The release uses fixed integer seeds distinct from the original study seeds. New runs therefore produce different splits, sampled estimates and resampling results. Files in `results/hcp/` retain the original study summaries and have not been regenerated with these release seeds. Existing masks can be supplied explicitly when a particular split must be retained.

## Network and metadata inputs

Supply either:

- The processed [`scs_desikan.mat`](https://github.com/maxwass/brain_data_processing/blob/main/data/scs_desikan.mat) file from `maxwass/brain_data_processing`. Required MATLAB variables are `scs` (87 × 87 × participants streamline counts) and `subject_list`. The conversion retains cortical indices `19:87`, takes the upper triangle and adds its transpose, yielding symmetric zero-diagonal 68-region networks.
- An NPZ containing `networks` (participants × 68 × 68 untransformed counts), `subject_ids` (unique strings or integers), and `roi_labels` (the 68 `ctx-lh-*`, then `ctx-rh-*` labels in the original Desikan--Killiany order). Arrays must load with `allow_pickle=False`.

The original matrix download had SHA256 `fbb4985853c400f000bcca1d4c0382a33a2a5013bfa053bba63d1f4df023ea76`. A later upstream download may differ; the preparation command records the actual input hashes.

Supply a separate HCP metadata CSV with unique `Subject`, `Gender` (`F`/`M`) and `Age` (`22-25`, `26-30`, `31-35`, `36+`) columns. Rows are joined by subject ID, preserving network order. Network fitting needs these three columns; behavioral analyses additionally need the named assessment columns. Obtain metadata through the applicable HCP access process and follow its data-use terms. This repository does not download or redistribute it.

```bash
python -m hcp.run prepare --networks /data/scs_desikan.mat --people /data/behavioral_data.csv --out outputs/hcp/prepared
```

For each subject, the split seed is the first 16 hexadecimal digits of SHA256(`12345:<Subject>`). NumPy's default generator permutes the 2,278 unordered edges; the first 228 are validation, the next 228 test, and the remaining 1,822 training. This split is independent of network values and is preserved across cohorts and model dimensions. `--masks <file.npz>` optionally supplies existing Boolean `train`, `validation`, and `test` arrays in network row order.

The three fitted cohorts are all women aged 31–35 (`F31_35`, originally 233), men aged 22–25 (`M22_25`, 143), and men aged 26–30 (`M26_30`, 213). Missing behavioral scores do not remove participants from network fitting. Positive counts are transformed using `log(count/c)`, where `c` is the cohort's median positive **training** count. The original scales were 66, 73, and 73. No count normalization, `log1p`, or additional presence threshold is used.

## Cohort fits and comparisons

```bash
python -m hcp.run fit --prepared outputs/hcp/prepared --cohort F31_35 --templates 4 --out outputs/hcp/fits
python -m hcp.run compare --prepared outputs/hcp/prepared --cohort F31_35 --method alma --out outputs/hcp/fits
python -m hcp.run compare --prepared outputs/hcp/prepared --cohort F31_35 --method tbm --out outputs/hcp/fits
```

Repeat `fit` for the three cohorts and `--templates 2`, `3`, and `4`. Each job has one chain with 500 warm-up and 1,500 retained draws, rank four, total membership concentration one, Student-t5 strength likelihood, target acceptance 0.85 and maximum tree depth nine. The compact initializer screens 12 candidates for 60 optimization iterations and refines four for up to 600 iterations. Initialization, NUTS and GIG seeds use base `41000` plus the cohort/dimension job index; comparator seeds use base `52000`. These fixed seeds are recorded with each fit.

Each fit creates `fit/` (native posterior draws and diagnostics), `source.npz` (aligned posterior means and fixed-source prediction parameters), `validation.npz`, and a completion summary. Template alignment uses the retained draw with median total squared template norm, then descending template norm within that draw and signed-distance assignment for all draws.

ALMA and TBM use the same binary training entries with training-only missing-entry imputation. ALMA starts with 50-start k-means, uses rank four and a `1e-5` loading-change tolerance. A capped ALMA run can continue its saved trajectory with the same command plus `--resume --iterations 5000`. TBM uses the original R `tensorsparse` implementation through `model/tbm.py`, with 20 starts, 1,000 iterations and tolerance `1e-10`. See the root README for the R dependency.

After fitting candidates:

```bash
python -m hcp.run select --fits outputs/hcp/fits --out outputs/hcp/selection.csv
```

The paired one-standard-error rule uses subject-level mean validation log scores. An incomplete candidate grid is reported without a selected dimension. Archived results retain the two originally interrupted M=2 chains; the clean runner targets complete new chains rather than reproducing interruptions. The original three-method comparison uses validation AUC, not the reserved test edges.

## Fixed-source transfer

```bash
python -m hcp.run transfer --prepared outputs/hcp/prepared --source outputs/hcp/fits/M22_25_M4/source.npz --workers 8 --out outputs/hcp/transfer_male
python -m hcp.run transfer --prepared outputs/hcp/prepared --source outputs/hcp/fits/F31_35_M4/source.npz --workers 8 --out outputs/hcp/transfer_female
```

The male source transfers to men aged 26–30, 31–35 and 36+. The female source transfers to women aged 22–25 and 26–30, and all male age groups. Female participants aged 36+ were not part of the reported transfer design. Source and target IDs must be disjoint. ROI order and the source transformation are preserved.

Only a new participant's four memberships are sampled, using their prepared training edges, fixed source template and scalar posterior means, and a Dirichlet(0.25, 0.25, 0.25, 0.25) prior. Each inference has 500 warm-up and 1,500 retained draws, a dense three-dimensional NUTS mass matrix, target acceptance 0.9, maximum tree depth ten, and source-average membership initialization. No global source parameter is refitted. The validation edges are unused; reserved test edges evaluate prediction. The male and female source runs use seed bases `63000` and `64000`, respectively. Add `--resume` to continue unfinished target inferences with identical inputs, settings and seed base. Outputs created with a different seed policy require a separate output directory.

`network_summary.csv` compares pooled test AUC, a source-average-membership baseline, and a Jeffreys-smoothed source training edge-frequency baseline. Joint presence/strength log-score gains have 20,000 participant-bootstrap intervals, conditional on source estimates. Three-person target groups are descriptive only. `memberships.npz` and `subjects/` are private participant-level outputs.

## Behavioral analysis and figures

`behavior_fields.csv` lists the exact 293 fields in the original exploratory screen, in original order. It contains field names and categories only. The screen covers four memberships in each of the three cohorts; all 3,516 comparisons enter BH adjustment, with `p=1` for untestable pairs. Spearman correlations use average-tie ranks, score-complete participants and 99,999 two-sided permutations. Raw `NEORAW_*` agreement items are encoded SD=1, D=2, N=3, A=4, SA=5 without reverse scoring.

```bash
python -m hcp.analysis screen --fits outputs/hcp/fits --people /data/behavioral_data.csv --out outputs/hcp/behavior_screen.csv
python -m hcp.analysis behavior --fits outputs/hcp/fits --people /data/behavioral_data.csv --screen outputs/hcp/behavior_screen.csv --female-transfer outputs/hcp/transfer_female --male-transfer outputs/hcp/transfer_male --out outputs/hcp/behavior_transfer.csv
python -m hcp.analysis plot --fits outputs/hcp/fits --people /data/behavioral_data.csv --surfaces /data/fsaverage5 --out outputs/hcp/figures
```

The seven transfer pairings and their fixed resampling seeds are specified in `analysis.py`. Bootstrap intervals use 20,000 resamples; Holm adjustment is reported within each target cohort, across each source family, and across the combined target tests. In the original data those families contain 15 female-source and eight male-source tests. The male T1 and T4 pairings were added after inspecting the initial transfer results. The analyses are exploratory, conditional on estimated memberships, and do not adjust for unavailable family relationships, exact age or quantitative diffusion-MRI motion.

Surface plotting requires local `fsaverage5/surf/{lh,rh}.inflated`, `surf/{lh,rh}.sulc`, and `label/{lh,rh}.aparc.annot` files, plus `nibabel` and `nilearn`. Nothing is downloaded automatically. It maps mean absolute incident template strength to the cortical parcels, normalizes each template to its peak, and uses a common grayscale from zero to one. Without `--surfaces`, the command produces signed template heatmaps alongside the same behavioral summaries.

Membership tertiles use linear-interpolated one-third and two-third quantiles before excluding missing scores; values exactly at a cut belong to the lower group. Error bars are nominal 95% t intervals for the observed score means. `regional_intensities.csv` and `behavior_groups.csv` contain aggregate figure values.

The shorter `--warmup`, `--draws`, `--permutations` and `--resamples` options support software smoke checks. Manuscript defaults are retained unless these options are supplied.

The archived figure values can also be plotted without participant data or model fits:

```bash
python -m hcp.analysis plot-summary --results results/hcp --out outputs/hcp/archived_figure
```

This creates regional-intensity and behavioral panels. Add `--surfaces /data/fsaverage5` to render the same archived regional values on cortical surfaces. The regional intensities and group means/intervals retain their full saved precision.
