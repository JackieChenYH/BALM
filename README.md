# BALM

Bayesian Adaptive Latent Mixture models for populations of aligned, zero-inflated weighted networks.

BALM represents each network using simplex memberships over shared low-rank, signed templates. A hurdle likelihood models edge presence and the continuous score of a present edge. This repository contains the model, ALMA and tensor block model comparisons, the eight simulation experiments, and the HCP cohort and transfer analyses accompanying **Bayesian Hurdle Template Mixtures for Zero-Inflated Weighted Network Populations**, by Hsin-Hsiung Huang, Yuh-Haur Chen and Teng Zhang.

## Start with the notebook

Use **Python 3.11** and run commands from this directory. The CPU dependencies are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab demo.ipynb
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead. The release was checked in a fresh Python 3.11 environment on Windows; the original large experiments ran on Stokes. For other accelerator configurations, follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) and [NumPyro documentation](https://num.pyro.ai/en/stable/getting_started.html).

[`demo.ipynb`](demo.ipynb) generates 24 networks on 12 nodes, fits two rank-two templates, compares fitted predictions with the true generating parameters, and displays true versus estimated template and membership matrices alongside recovery scatter plots. The complete notebook takes about two minutes on one CPU on the validation machine, including first-call compilation. Its small initialization search and short chain are separate from the manuscript settings.

## Where the code lives

| Location | Purpose |
|---|---|
| `model/balm.py` | Likelihoods, priors, joint NUTS, exact scale updates, posterior summaries and predictions |
| `model/initialization.py` | Training-based simulation, HCP and demo initialization |
| `model/alma.py` | ALMA and its training-data adapter |
| `model/tbm.py`, `model/tbm.R` | Python/R interface to `tensorsparse` |
| `simulation/designs.py` | Eight experiment definitions, data generation, masks and fixed random seeds |
| `simulation/run.py`, `analysis.py` | Run selected cases and summarize new fits |
| `simulation/plots.py` | Recreate the simulation figures from the included numerical summaries |
| `simulation/whole_network.py` | Separate supplementary whole-network predictive assessment |
| `hcp/` | Prepare data, fit cohorts, transfer templates, analyze behavior and plot results |
| `results/` | Saved aggregate results and synthetic illustration data used by the manuscript |

Generated data, fits and figures belong under `outputs/`. These outputs are excluded from version control. The release does not require the original project directory, Stokes account paths or archived deployment packages.

## Use BALM in your own code

```python
from model import fit_balm, FitResult

fit = fit_balm(
    Z, Y, train_mask, n=n, M=3, K=3,
    alpha=1.0, warmup=300, draws=1000, seed=123,
)
summary = fit.aligned_summary()
prediction = fit.predict(Z, Y, mask=test_mask)
fit.save("outputs/my_fit")
restored = FitResult.load("outputs/my_fit")
```

- `Z`, `Y` and the Boolean mask have shape `(L, n*(n-1)//2)`, with columns ordered by `numpy.triu_indices(n, 1)`. `Z` is binary; `Y` contains transformed, signed scores for present edges. An absent edge has no continuous observation.
- `M` is the template number, `K` is the spectral rank, and `alpha` is the **total** Dirichlet concentration, giving component concentration `alpha/M`.
- `summary["Q_mean"]` and `summary["W_mean"]` contain aligned posterior means; the corresponding `Q_interval` and `W_interval` entries contain equal-tail intervals. Synthetic recovery may pass generating templates through `reference=Q_true` after fitting.
- Predictions include `presence_probability`, `strength_mean` conditional on presence, `score_mean`, `zero_filled_mean`, and pointwise `log_density`. These predictions use the fitted participants' memberships. Fixed-template inference for new participants is implemented in `hcp/run.py`.
- Saved fits contain `posterior.npz` and `metadata.json`, including sampling diagnostics and initialization settings. Existing fits are protected from overwrite.

The default model is coupled Gaussian BALM. The manuscript variants are `null` (fixed zero coupling), `hard` (hard membership), `weighted` (zero-filled Gaussian scores), `independent` (separate presence and strength templates), `student3`, and `student5`. Their likelihoods and prediction targets differ; the simulation runner applies the appropriate comparisons.

## ALMA and TBM

ALMA is implemented in Python. TBM requires R and **tensorsparse 3.0**:

```bash
Rscript -e "install.packages('tensorsparse', repos='https://cloud.r-project.org')"
Rscript -e "stopifnot(as.character(packageVersion('tensorsparse')) == '3.0')"
```

`Rscript` must be on the executable path, or pass its path to the TBM API/HCP command. The adapters retain the original training-only missing-entry imputation. ALMA can continue from a saved state using the unchanged convergence tolerance. TBM uses 20 starts and averages predictions for the two edge directions. Neither comparator's loadings are interpreted as BALM continuous membership posteriors. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for method citations and dependency licensing.

## Eight simulation experiments

| Number | Experiment | Main variation |
|---|---|---|
| 1 | Template and membership recovery | Two observation masks; BALM, ALMA and TBM |
| 2 | Template-number selection | True M=2,3,4; candidates M=2,3,4,5; two initialization sets |
| 3 | Coupling and density | Coupling 0,0.5,1 crossed with density 0.15,0.30,0.50 |
| 4 | Block-structured networks | Pure memberships; hard hurdle, BALM, ALMA and TBM |
| 5 | Membership overlap | Total concentration 0.1,1,3 |
| 6 | Network size and rank | Nodes, participants and spectral rank |
| 7 | Signal and noise | Reduced template amplitude and increased score noise |
| 8 | Components and distribution | Hard/weighted/separate variants, heavy tails and different channel templates |

The default reference has 68 nodes, 100 participants, three rank-three templates, template RMS 0.6 and Gaussian noise SD 0.3. Each setting uses 50 datasets. Bayesian fits use 300 warm-up iterations and 1,000 retained draws. Exact per-experiment settings and seeds are centralized in `simulation/designs.py`.

Inspect a case list before executing full-sized fits:

```bash
python -m simulation.run list --experiment 1 --rep 0
python -m simulation.run prepare --experiment 1 --rep 0 --output outputs/simulation
python -m simulation.run fit --experiment 1 --rep 0 --method coupled --output outputs/simulation
```

`--rep` accepts zero-based replicate numbers; `--experiment` and `--method` accept multiple values. Omit these filters to request all eight experiments and all 50 replicates. The common reference is reused across experiments, and shared fits are cached. The complete primary design has 4,150 case descriptions corresponding to 4,100 distinct fits. These are substantial compute jobs; the notebook is the short example.

After the requested cases finish:

```bash
python -m simulation.run summarize --experiment 1 --rep 0 --method coupled --output outputs/simulation
python -m simulation.run plot --output outputs/simulation
```

Summary CSVs contain per-fit results, aggregate recovery/prediction, paired comparisons, selection frequencies and coupling contrasts as applicable. A partial collection requires `--allow-partial` and remains explicitly labeled. The paired one-standard-error selection rule is the rule used in the 1110 manuscripts; numerical diagnostics do not replace it.

The 17 targeted initialization analyses are separate from the primary experiment:

```bash
python -m simulation.run list --sensitivity
python -m simulation.run fit --sensitivity --output outputs/simulation
python -m simulation.run summarize --sensitivity --output outputs/simulation
```

Dependency fits are reused when available. The block BALM initialization uses the specified hard-model training draw; targeted fixed-zero and expanded-template starts use the specified training-draw transfers. These targeted results do not overwrite the primary comparisons. The 17 case indices were selected in the original study; under the new seed schedule they illustrate the same initialization procedures, but need not reproduce the numerical issues that motivated the original selection.

### Recreate the saved simulation figures

```bash
python -m simulation.plots --data results/simulation --output outputs/simulation_figures
```

This command uses the included saved numerical summaries and synthetic illustration arrays. It does not refit models. `simulation.run plot` instead plots summaries of new executions.

### Stokes execution

`simulation/submit.sh` is a one-CPU Slurm template. It runs one case selected by `--task` from the same filtered case list; account, partition, memory and scheduler time limits are supplied for the user's Stokes allocation. Completed matching fits are reused, and shared dependency fits are protected by locks. Scheduler submissions are explicit user actions.

For example, Experiment 1, replicate 0 contains six cases (two masks and three methods):

```bash
python -m simulation.run list --experiment 1 --rep 0
sbatch --array=0-5 simulation/submit.sh --experiment 1 --rep 0 --output outputs/simulation
```

Activate the Stokes Python environment before submitting. Supply any account/resource flags required by the allocation, and derive the array range from the case list when changing filters. A cohort fit can use the same environment through a single Slurm job:

```bash
sbatch --cpus-per-task=1 --wrap="python -m hcp.run fit --prepared outputs/hcp/prepared --cohort F31_35 --templates 4 --out outputs/hcp/fits"
```

### Supplementary whole-network assessment

```bash
python -m simulation.whole_network run --rep 0 --fold 0 --output outputs/whole_network
python -m simulation.whole_network summarize --output outputs/whole_network
```

Each `run` processes one fold. Run both folds for replicates 0–4 before summarizing. The stages `prepare`, `fit`, `integrate`, `auxiliary` and `score` can also be called separately. This earlier study retains its original ordinary-NUTS training sampler, 500 warm-up and 2,000 retained draws, plus the auxiliary-chain, quadrature and corrected bridge calculation. Its sampler and whole-network prediction target are distinct from the eight main experiments. Saved fold summaries are included under `results/simulation/`.

## HCP analysis

The [HCP data guide](hcp/DATA.md) gives the network/metadata schemas, exact behavioral fields, preprocessing, cohort definitions, masks and full commands. Data are supplied by the user under the applicable HCP access terms. The public results contain aggregate summaries; participant-level networks, IDs, behavioral rows, memberships and posterior draws are not distributed.

The main workflow is:

```bash
python -m hcp.run prepare --networks data/scs_desikan.mat --people data/behavior.csv --out outputs/hcp/prepared
python -m hcp.run fit --prepared outputs/hcp/prepared --cohort M22_25 --templates 4 --out outputs/hcp/fits
python -m hcp.run compare --prepared outputs/hcp/prepared --cohort M22_25 --method alma --out outputs/hcp/fits
python -m hcp.run compare --prepared outputs/hcp/prepared --cohort M22_25 --method tbm --out outputs/hcp/fits
python -m hcp.run transfer --prepared outputs/hcp/prepared --source outputs/hcp/fits/M22_25_M4/source.npz --out outputs/hcp/transfer_male
```

Repeat the cohort fit for `F31_35` and `M26_30`, and fit M=2 and M=3 to perform the dimension comparison. The historical candidate summaries retain the two incomplete M=2 runs; new commands target 1,500 retained draws. Cohort fitting uses the training-median log-count transformation, rank four, concentration one and Student-t5 strengths. The three-method cohort comparison reports validation AUC.

Transfer holds source templates, scalar estimates and transformation constants fixed and estimates only new participants' memberships. Its reserved test edges assess prediction. Behavioral screening, transfer correlations and grayscale figures have separate commands in `hcp/analysis.py`; see the data guide for their original multiple-comparison and resampling settings.

The included aggregate values also support a figure preview without participant data:

```bash
python -m hcp.analysis plot-summary --results results/hcp --out outputs/hcp/figure
```

This draws regional-intensity and behavioral panels. Add `--surfaces <fsaverage5-directory>` for cortical-surface views; the data guide lists the required local surface files.

## Reproduction and validation

The model kernels and analysis routines were checked against the frozen implementations and saved results used by the current manuscripts. Checks cover model densities and gradients, initialization, predictive mixtures, simulation generators, ALMA continuation, the R TBM adapter, HCP conversion and masks, and whole-network integration. The toy notebook and small execution checks run in the pinned environment. The saved results are the original study summaries.

The current runners use ordinary fixed integer seeds, with separate streams for data generation, initialization, sampling and resampling. The notebook and HCP mask default use `12345`; simulation seeds are defined in `simulation/designs.py`, and HCP seeds in `hcp/run.py` and `hcp/analysis.py`. These replace the date-based seed constants used in the original study. Model specifications and experiment settings are unchanged, but rerunning with the new seeds produces different random realizations and numerical results. The files in `results/` retain the original study values; use the plotting commands above to redraw those saved results.

Use a new output directory when changing seeds. Cached simulation data and fits are checked against their generating settings to prevent reuse across seed schedules. Numerical optimization and MCMC paths can also vary with hardware and numerical-library versions. For new runs, retain the output settings and diagnostics alongside reported estimates.

## Citation and license

Please cite the BALM manuscript and the corresponding comparator papers when using these analyses. The manuscript title and authors are given at the top of this file; a journal citation or DOI should be added when available.

The supplied BALM source directories did not include a project license. No new license has been assigned to the project in this release draft. The authors should select a license before describing the repository as open-source. Third-party dependencies retain their own licenses, as noted in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
