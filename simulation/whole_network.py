"""Whole-network prediction for the separate five-dataset, two-fold study.

Run one fold with ``python -m simulation.whole_network run --rep 0 --fold 0``.
The stages can also be run separately: prepare, fit, integrate, auxiliary,
score, summarize. The historical study used ordinary NUTS for the training
fit, followed by a balanced auxiliary chain and proposal-corrected integration.
It predates the GIG sampler used by the main eight simulation experiments.

The predictive target integrates a fresh membership under its Dirichlet prior.
Auxiliary draws enter only through the bridge correction, never as additional
training-posterior draws. Product quadrature here is implemented for M=3.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import time

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.special import logsumexp, roots_jacobi


CONFIG = dict(n=20, L=100, M=3, K=3, alpha=1.0, template_rms=0.6,
              sigma=0.3, a1=2.0, density=0.5, holdout_fraction=0.15,
              kind="coupled")
AUXILIARY_SEEDS = ((75401, 75402), (75403, 75404), (75405, 75406),
                   (75400, 75407), (75408, 75409))


def logmeanexp(values, axis=None):
    values = np.asarray(values)
    count = values.size if axis is None else values.shape[axis]
    return logsumexp(values, axis=axis) - np.log(count)


def simplex_rule(order=64, alpha=1.0):
    """Product Gauss--Jacobi rule for Dirichlet(alpha/3, alpha/3, alpha/3)."""
    if order < 2 or alpha <= 0:
        raise ValueError("order must be at least 2 and alpha must be positive")
    concentration = alpha / 3
    x, wx = roots_jacobi(order, 2 * concentration - 1, concentration - 1)
    y, wy = roots_jacobi(order, concentration - 1, concentration - 1)
    first, second = np.meshgrid((x + 1) / 2, (y + 1) / 2, indexing="ij")
    weights = np.stack([first, (1 - first) * second,
                        (1 - first) * (1 - second)], axis=-1).reshape(-1, 3)
    log_weights = np.log((wx[:, None] * wy[None, :] / (wx.sum() * wy.sum())).ravel())
    return weights, log_weights


def integrated_log_likelihood(q, a0, a1, variance, z, y, weights, log_weights,
                              mask=None):
    """Integrate each network's joint Gaussian hurdle likelihood over fresh W."""
    z, y = np.asarray(z), np.asarray(y)
    q = np.asarray(q)
    if q.shape[0] != 3 or variance <= 0 or z.shape != y.shape:
        raise ValueError("Expected three templates, positive variance and matching Z/Y")
    mask = np.ones_like(z, dtype=bool) if mask is None else np.asarray(mask, bool)
    z = np.where(mask, z, 0)
    y = np.where(mask & (z == 1), y, 0.0)
    means = weights @ q
    softplus = np.logaddexp(0, a0 + a1 * means)
    normalizer = (softplus[:, mask[0]].sum(1)[None, :] if np.all(mask == mask[0])
                  else mask @ softplus.T)
    linear = a1 * (z @ q.T) + (z * y) @ q.T / variance
    quadratic = np.einsum("le,ie,je->lij", z, q, q, optimize=True) / variance
    base = z.sum(1) * (a0 - 0.5 * np.log(2 * np.pi * variance))
    base -= 0.5 * np.sum(z * y * y, axis=1) / variance
    terms = linear @ weights.T
    terms -= 0.5 * np.einsum("ni,lij,nj->ln", weights, quadratic, weights, optimize=True)
    return base + logsumexp(terms - normalizer + log_weights, axis=1)


def bridge_normalizer(log_h_training, log_h_auxiliary, tolerance=1e-11,
                      max_iterations=2000):
    """Meng--Wong fixed point for the balanced auxiliary normalizing constant."""
    first, second = map(np.asarray, (log_h_training, log_h_auxiliary))
    if any(x.ndim != 1 or not len(x) or not np.isfinite(x).all() for x in (first, second)):
        raise ValueError("Bridge inputs must be finite, nonempty one-dimensional arrays")
    n0, n1 = len(first), len(second)
    log_s0, log_s1 = np.log(n0 / (n0 + n1)), np.log(n1 / (n0 + n1))
    estimate = float(logmeanexp(first))
    for iteration in range(max_iterations):
        numerator = first - np.logaddexp(log_s0, log_s1 + first - estimate)
        denominator = -np.logaddexp(log_s0, log_s1 + second - estimate)
        updated = float(logmeanexp(numerator) - logmeanexp(denominator))
        if abs(updated - estimate) < tolerance:
            return updated, iteration + 1
        estimate = updated
    raise RuntimeError("Bridge fixed point did not converge")


def bridge_scores(log_g_training, log_g_auxiliary, centers):
    """Correct both endpoint samples before averaging each target likelihood."""
    first, second, centers = map(np.asarray, (log_g_training, log_g_auxiliary, centers))
    if (first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]
            or centers.shape != (first.shape[1],)):
        raise ValueError("Likelihood matrices must have one column per center/network")
    log_h0 = logsumexp(first - centers, axis=1) - np.log(len(centers))
    log_h1 = logsumexp(second - centers, axis=1) - np.log(len(centers))
    log_normalizer, iterations = bridge_normalizer(log_h0, log_h1)
    n0, n1 = len(first), len(second)
    pooled = np.concatenate([first, second])
    denominator = np.logaddexp(np.log(n0 / (n0 + n1)),
                              np.log(n1 / (n0 + n1))
                              + np.r_[log_h0, log_h1] - log_normalizer)
    contributions = pooled - denominator[:, None]
    scores = logmeanexp(contributions, axis=0)
    weights = np.exp(contributions - logsumexp(contributions, axis=0))
    normalization = float(np.exp(logmeanexp(-denominator)))
    if not np.isclose(normalization, 1, atol=1e-8, rtol=0):
        raise RuntimeError("Bridge mixture failed the density-ratio normalization check")
    return scores, dict(log_normalizer=log_normalizer, iterations=iterations,
                        normalization_check=normalization,
                        contribution_ess=1 / np.sum(weights * weights, axis=0),
                        max_draw_weight=weights.max(axis=0))


def block_bootstrap(first, second, centers, *, seed=6301, replicates=160, block=100):
    """Re-estimate the bridge normalizer in every circular block bootstrap."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(replicates):
        resampled = []
        for values in (first, second):
            count = len(values)
            starts = rng.integers(0, count, size=int(np.ceil(count / block)))
            indices = ((starts[:, None] + np.arange(block)) % count).ravel()[:count]
            resampled.append(values[indices])
        scores.append(bridge_scores(*resampled, centers)[0])
    scores = np.asarray(scores)
    return dict(block=block, replicates=replicates,
                network_mcse=scores.std(axis=0, ddof=1),
                mean_score_mcse=float(scores.mean(axis=1).std(ddof=1)))


def _load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n",
                    encoding="utf-8")


def _samples(path):
    samples = _load(path)
    if samples["V"].ndim == 5:
        samples = {key: value.reshape((-1,) + value.shape[2:])
                   for key, value in samples.items()}
    return samples


def _q_draws(samples):
    vectors = np.linalg.qr(samples["V"])[0]
    n = vectors.shape[-2]
    first, second = np.triu_indices(n, 1)
    return samples["tau"][:, None, None] * np.einsum(
        "smik,smk,smik->smi", vectors[:, :, first], samples["gamma"],
        vectors[:, :, second])


def integrate_draws(samples, z, y, *, alpha=1.0, endpoint="training",
                    full_check=False, low_order=32, order=64, high_order=128,
                    tolerance=1e-7):
    q = _q_draws(samples)
    count = len(q)
    results = {}
    for current in (low_order, order):
        weights, log_weights = simplex_rule(current, alpha)
        results[current] = np.array([
            integrated_log_likelihood(q[index], samples["a0"][index],
                                      samples["a1"][index], samples["sigma2"][index],
                                      z, y, weights, log_weights)
            for index in range(count)
        ])
    low, middle = results[low_order], results[order]
    if full_check:
        selected = np.arange(count)
    elif endpoint == "training":
        relative_weight = np.exp(middle - logsumexp(middle, axis=0))
        influence = np.abs(np.expm1(low - middle)) * relative_weight
        selected = np.unique(np.r_[np.flatnonzero(influence.max(1) > 1e-5),
                                   np.argsort(middle, axis=0)[-5:].ravel(),
                                   np.linspace(0, count - 1, min(64, count), dtype=int)])
    else:
        selected = np.unique(np.r_[np.linspace(0, count - 1, min(64, count), dtype=int),
                                   np.argsort(abs(middle - low).max(1))[-32:],
                                   np.argsort(middle, axis=0)[-3:].ravel()])
    weights, log_weights = simplex_rule(high_order, alpha)
    high = np.array([
        integrated_log_likelihood(q[index], samples["a0"][index],
                                  samples["a1"][index], samples["sigma2"][index],
                                  z, y, weights, log_weights)
        for index in selected
    ])
    error = float(abs(high - middle[selected]).max())
    if error > tolerance:
        raise RuntimeError(f"Quadrature check failed: maximum log-integral change {error:.3g}")
    corrected = middle.copy()
    corrected[selected] = high
    return corrected, dict(order=order, high_order=high_order, checked_draws=selected,
                           maximum_checked_log_difference=error,
                           all_draws_in_final_average=count)


def prepare(output, rep, config=None):
    from simulation.designs import generate, generation_seeds
    config = dict(CONFIG if config is None else config)
    if config["M"] != 3 or config["kind"] != "coupled":
        raise ValueError("The whole-network implementation uses three shared Gaussian templates")
    folder = Path(output) / f"rep{rep:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "config.json"
    seed_path = folder / "seeds.json"
    seeds = generation_seeds("whole_network", "normal", rep)
    if config_path.exists():
        if json.loads(config_path.read_text("utf-8")) != config:
            raise ValueError("Existing dataset uses a different configuration")
        if not seed_path.exists() or json.loads(seed_path.read_text("utf-8")) != seeds:
            raise ValueError("Existing dataset uses different or unrecorded seeds; choose a new --output directory")
        return folder
    data, truth, masks, _ = generate(config, rep, family="whole_network")
    np.savez_compressed(folder / "data.npz", **data, **masks)
    np.savez_compressed(folder / "truth.npz", **truth)
    _json(config_path, config)
    _json(seed_path, seeds)
    return folder


def _fold(output, rep, fold):
    from simulation.designs import generation_seeds
    data_folder = Path(output) / f"rep{rep:03d}"
    seed_path = data_folder / "seeds.json"
    seeds = generation_seeds("whole_network", "normal", rep)
    if not seed_path.exists() or json.loads(seed_path.read_text("utf-8")) != seeds:
        raise ValueError("Existing dataset uses different or unrecorded seeds; choose a new --output directory")
    config = json.loads((data_folder / "config.json").read_text("utf-8"))
    data = _load(data_folder / "data.npz")
    folder = data_folder / f"fold{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    train_rows = np.flatnonzero(data["fold"] != fold)
    eval_rows = np.flatnonzero(data["fold"] == fold)
    return folder, config, data, train_rows, eval_rows


def fit_training(output, rep, fold, *, warmup=500, draws=2000, quick=False):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS, init_to_value
    from numpyro.infer.util import constrain_fn
    from model.balm import balm_hurdle_model
    from simulation.designs import seed_for

    folder, config, data, train_rows, eval_rows = _fold(output, rep, fold)
    if (folder / "training.npz").exists():
        raise FileExistsError("Training samples exist; use the next stage or another output folder")
    case = f"mixed__rep{rep:03d}__newnetwork_fold{fold}"
    seed = seed_for(16180, case, "fit")
    init_seed = seed_for(16180, case, "initialization")
    model = partial(balm_hurdle_model, n=config["n"], M=3, K=config["K"],
                    hyper={"alpha": config["alpha"]})
    kwargs = dict(Z=jnp.asarray(data["z"][train_rows]),
                  Y=jnp.asarray(data["y"][train_rows]),
                  mask=jnp.asarray(data["train"][train_rows]))
    values, initial_info = _initialize(config, kwargs, init_seed, quick=quick)
    np.savez_compressed(folder / "initial_values.npz", **values)
    kernel = NUTS(model, init_strategy=init_to_value(values=values),
                  max_tree_depth=8, target_accept_prob=0.8)
    sampler = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=1,
                   progress_bar=True,
                   postprocess_fn=partial(constrain_fn, model, (), kwargs,
                                          return_deterministic=False))
    start = time.perf_counter()
    sampler.run(jax.random.PRNGKey(seed), **kwargs,
                extra_fields=("diverging", "num_steps", "accept_prob"))
    np.savez_compressed(folder / "training.npz",
                        **{key: np.asarray(value) for key, value in sampler.get_samples().items()})
    np.savez_compressed(folder / "training_diagnostics.npz",
                        **{key: np.asarray(value) for key, value in sampler.get_extra_fields().items()})
    _json(folder / "training.json", dict(rep=rep, fold=fold, seed=seed,
          init_seed=init_seed, warmup=warmup, draws=draws, sampler="ordinary_NUTS",
          target_accept=0.8, max_tree_depth=8,
          initialization={key: value for key, value in initial_info.items() if key != "records"},
          training_rows=train_rows, evaluation_rows=eval_rows,
          seconds=time.perf_counter() - start))


def _initialize(config, kwargs, seed, *, quick=False):
    from model.initialization import initialize
    options = dict(profile="simulation", screen_iterations=80, refine_candidates=12, maxiter=1000,
                   ftol=1e-10, gtol=1e-5, maxls=30, maxfun=50000, families=5)
    if quick:
        options.update(profile="demo", screen_iterations=2, refine_candidates=1, maxiter=5)
    return initialize(np.asarray(kwargs["Z"]), np.asarray(kwargs["Y"]),
                      np.asarray(kwargs["mask"]), n=config["n"], M=3, K=config["K"],
                      alpha=config["alpha"], hyper={"alpha": config["alpha"]},
                      method="coupled", seed=seed,
                      options=options)


def integrate_training(output, rep, fold):
    folder, config, data, _, eval_rows = _fold(output, rep, fold)
    values, precision = integrate_draws(_samples(folder / "training.npz"),
                                        data["z"][eval_rows], data["y"][eval_rows],
                                        alpha=config["alpha"],
                                        full_check=(rep == 0 and fold == 0))
    centers = logmeanexp(values, axis=0)
    np.savez_compressed(folder / "training_integrals.npz", log_g=values, centers=centers)
    _json(folder / "training_quadrature.json", precision)


def _endpoint_model(config, new_z, new_y, centers):
    import jax.numpy as jnp
    import jax.scipy as jsp
    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers
    from model.balm import balm_hurdle_model

    new_z, new_y, centers = map(jnp.asarray, (new_z, new_y, centers))

    def model(Z, Y, mask):
        trace = handlers.trace(balm_hurdle_model).get_trace(
            Z, Y, n=config["n"], M=3, K=config["K"],
            hyper={"alpha": config["alpha"]}, mask=mask)
        weights = numpyro.sample("_bridge_W", dist.Dirichlet(jnp.ones(3) * config["alpha"] / 3))
        means = weights @ trace["Q_ut"]["value"]
        variance = trace["sigma2"]["value"]
        logits = trace["a0"]["value"] + trace["a1"]["value"] * means
        likelihood = jnp.sum(new_z * logits - jnp.logaddexp(0, logits)
                            - 0.5 * new_z * (jnp.log(2 * jnp.pi * variance)
                                             + (new_y - means) ** 2 / variance), axis=1)
        numpyro.factor("_bridge_balanced_likelihood",
                       jsp.special.logsumexp(likelihood - centers) - jnp.log(len(centers)))
    return model


def _refresh_bridge(key, coordinates, values, z, y, centers, alpha, moves=32):
    import jax
    import jax.numpy as jnp
    from model.balm import helmert_basis

    q, a0, a1, variance = (values[name] for name in ("Q_ut", "a0", "a1", "sigma2"))
    linear = a1 * (z @ q.T) + (z * y) @ q.T / variance
    quadratic = jnp.einsum("le,ie,je->lij", z, q, q) / variance
    base = z.sum(1) * (a0 - 0.5 * jnp.log(2 * jnp.pi * variance))
    base -= 0.5 * jnp.sum(z * y * y, axis=1) / variance + centers

    def log_h(weights):
        likelihood = base + jnp.einsum("lm,...m->...l", linear, weights)
        likelihood -= 0.5 * jnp.einsum("...i,lij,...j->...l", weights, quadratic, weights)
        return (jax.scipy.special.logsumexp(likelihood, axis=-1) - jnp.log(len(z))
                - jnp.logaddexp(0, a0 + a1 * (weights @ q)).sum(-1))

    proposal_key, uniform_key = jax.random.split(key)
    log_weights = jax.random.loggamma(proposal_key, jnp.full((moves, 3), alpha / 3))
    log_weights -= jax.scipy.special.logsumexp(log_weights, axis=-1, keepdims=True)
    basis = helmert_basis(3)
    proposed = log_weights @ basis
    log_values = log_h(jnp.exp(log_weights))
    log_uniform = jnp.log(jax.random.uniform(uniform_key, (moves,)))
    current_value = log_h(jax.nn.softmax(coordinates @ basis.T))

    def step(state, proposal):
        current, current_log_h = state
        candidate, candidate_log_h, log_u = proposal
        accept = log_u < jnp.minimum(0.0, candidate_log_h - current_log_h)
        return (jnp.where(accept, candidate, current),
                jnp.where(accept, candidate_log_h, current_log_h)), accept

    (result, _), accepted = jax.lax.scan(step, (coordinates, current_value),
                                       (proposed, log_values, log_uniform))
    return result, jnp.mean(accepted)


def fit_auxiliary(output, rep, fold, *, warmup=500, draws=2000):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from numpyro.infer import NUTS
    from model.balm import build_reparameterized

    folder, config, data, train_rows, eval_rows = _fold(output, rep, fold)
    if (folder / "auxiliary.npz").exists():
        raise FileExistsError("Auxiliary samples exist; use score or another output folder")
    centers = _load(folder / "training_integrals.npz")["centers"]
    initial = {key: value[-1] for key, value in _samples(folder / "training.npz").items()}
    initial["_bridge_W"] = np.ones(3) / 3
    kwargs = dict(Z=jnp.asarray(data["z"][train_rows]),
                  Y=jnp.asarray(data["y"][train_rows]),
                  mask=jnp.asarray(data["train"][train_rows]))
    z, y = jnp.asarray(data["z"][eval_rows]), jnp.asarray(data["y"][eval_rows])
    seed = AUXILIARY_SEEDS[rep][fold]
    key = jax.random.PRNGKey(seed)
    adapter = build_reparameterized(_endpoint_model(config, z, y, centers), kwargs,
                                    initial, key, site_name="_bridge_W")
    kernel = NUTS(potential_fn=adapter.potential_fn, target_accept_prob=0.8,
                  max_tree_depth=9)
    value_gradient = jax.value_and_grad(adapter.potential_fn)
    state = kernel.init(key, warmup, init_params=adapter.init_params)
    centers = jnp.asarray(centers)

    def step(state, _):
        state = kernel.sample(state, (), {})
        values = adapter.constrained_values(state.z, True)
        next_key, proposal_key = jax.random.split(state.rng_key)
        coordinates, accepted = _refresh_bridge(proposal_key, state.z["_bridge_W"],
                                                 values, z, y, centers, config["alpha"])
        parameters = dict(state.z, _bridge_W=coordinates)
        value, gradient = value_gradient(parameters)
        updated = state._replace(z=parameters, z_grad=gradient, potential_energy=value,
                                 energy=value, r=None, rng_key=next_key)
        diagnostics = {name: getattr(state, name) for name in
                       ("diverging", "num_steps", "accept_prob")}
        diagnostics["bridge_mh_accept"] = accepted
        return updated, (adapter.postprocess_fn(parameters), diagnostics)

    start = time.perf_counter()
    sample_chunk = jax.jit(lambda state, length: jax.lax.scan(step, state, None, length),
                           static_argnums=1)
    for offset in range(0, warmup, 100):
        state, _ = sample_chunk(state, min(100, warmup - offset))
        jax.block_until_ready(state.z)
    kept = []
    for offset in range(0, draws, 100):
        state, samples = sample_chunk(state, min(100, draws - offset))
        jax.block_until_ready(state.z)
        kept.append(jax.tree.map(np.asarray, samples))
        print(f"Auxiliary draws: {min(offset + 100, draws)}/{draws}", flush=True)
    posterior, diagnostics = jax.tree.map(lambda *arrays: np.concatenate(arrays), *kept)
    np.savez_compressed(folder / "auxiliary.npz", **posterior)
    np.savez_compressed(folder / "auxiliary_diagnostics.npz", **diagnostics)
    _json(folder / "auxiliary.json", dict(rep=rep, fold=fold, seed=seed,
          warmup=warmup, draws=draws, max_tree_depth=9, target_accept=0.8,
          membership_mh_moves=32, seconds=time.perf_counter() - start))


def score_fold(output, rep, fold, *, bootstrap_replicates=160):
    folder, config, data, _, eval_rows = _fold(output, rep, fold)
    first = _load(folder / "training_integrals.npz")
    second, precision = integrate_draws(_samples(folder / "auxiliary.npz"),
                                        data["z"][eval_rows], data["y"][eval_rows],
                                        alpha=config["alpha"], endpoint="auxiliary")
    np.savez_compressed(folder / "auxiliary_integrals.npz", log_g=second)
    _json(folder / "auxiliary_quadrature.json", precision)
    scores, info = bridge_scores(first["log_g"], second, first["centers"])
    bootstraps = [block_bootstrap(first["log_g"], second, first["centers"],
                                  seed=AUXILIARY_SEEDS[rep][fold] + block,
                                  replicates=bootstrap_replicates, block=block)
                  for block in (100, 200)]
    report = dict(rep=rep, fold=fold, networks=len(eval_rows),
                  network_log_scores=scores, mean_joint_logscore=float(scores.mean()),
                  mean_score_mcse=max(item["mean_score_mcse"] for item in bootstraps),
                  bridge=info, bootstrap=bootstraps)
    _json(folder / "score.json", report)
    return report


def summarize(output):
    rows = []
    for rep in range(5):
        for fold in range(2):
            path = Path(output) / f"rep{rep:03d}" / f"fold{fold}" / "score.json"
            rows.append(json.loads(path.read_text("utf-8")))
    means = [np.mean([item["mean_joint_logscore"] for item in rows if item["rep"] == rep])
             for rep in range(5)]
    report = dict(independent_datasets=5, folds=10, networks=sum(item["networks"] for item in rows),
                  per_dataset_twofold_mean=means, grand_mean_joint_logscore=float(np.mean(means)),
                  between_dataset_se=float(np.std(means, ddof=1) / np.sqrt(5)),
                  conservative_aggregate_mcse_bound=float(np.mean([item["mean_score_mcse"] for item in rows])))
    _json(Path(output) / "summary.json", report)
    return report


def check_math():
    """Fast analytic and brute-force checks; no model fitting."""
    from scipy.stats import norm
    rng = np.random.default_rng(1367)
    noise = 0.8
    observations = np.array([-2.0, 0.5, 2.5])
    truth = norm.logpdf(observations, 0, np.sqrt(1 + noise * noise))
    centers = truth + np.array([0.2, -0.3, 0.5])
    mixing = np.exp(truth - centers - logsumexp(truth - centers))
    first = rng.normal(size=100000)
    component = rng.choice(3, 100000, p=mixing)
    second = rng.normal(observations[component] / (1 + noise * noise),
                        noise / np.sqrt(1 + noise * noise))
    first = norm.logpdf(observations[None], first[:, None], noise)
    second = norm.logpdf(observations[None], second[:, None], noise)
    estimates, _ = bridge_scores(first, second, centers)
    error = float(abs(estimates - truth).max())
    if error > 0.02:
        raise AssertionError(f"Analytic bridge check failed: {error}")
    shifted, _ = bridge_scores(first - 1000, second - 1000, centers - 1000)
    np.testing.assert_allclose(shifted, estimates - 1000, rtol=0, atol=1e-8)
    weights, log_weights = simplex_rule(16)
    q = rng.normal(size=(3, 7))
    z = rng.binomial(1, 0.5, size=(4, 7))
    y = rng.normal(size=z.shape)
    mask = rng.random(z.shape) > 0.3
    mean = weights @ q
    logits = 0.2 + 1.4 * mean
    direct = z[:, None] * logits - np.logaddexp(0, logits)
    direct -= 0.5 * z[:, None] * (np.log(2 * np.pi * 0.3) + (y[:, None] - mean) ** 2 / 0.3)
    direct = logsumexp(np.where(mask[:, None], direct, 0).sum(-1) + log_weights, axis=1)
    compressed = integrated_log_likelihood(q, 0.2, 1.4, 0.3, z, y, weights, log_weights, mask)
    np.testing.assert_allclose(compressed, direct, atol=1e-11, rtol=0)
    return dict(analytic_bridge_max_error=error,
                compressed_likelihood_max_error=float(abs(compressed - direct).max()),
                log_offset_invariance=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "fit", "integrate", "auxiliary", "score",
                                           "run", "summarize", "check"])
    parser.add_argument("--output", type=Path, default=Path("outputs/whole_network"))
    parser.add_argument("--rep", type=int, choices=range(5), default=0)
    parser.add_argument("--fold", type=int, choices=(0, 1), default=0)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--quick-initialization", action="store_true",
                        help="Reduced optimization for software checks, not the paper experiment")
    parser.add_argument("--bootstrap-replicates", type=int, default=160)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.draws < 2 or args.bootstrap_replicates < 2:
        parser.error("warmup must be nonnegative; draws and bootstrap replicates must be at least 2")
    if args.stage == "check":
        print(json.dumps(check_math(), indent=2))
        return
    if args.stage == "summarize":
        print(json.dumps(summarize(args.output), indent=2))
        return
    if args.stage in ("prepare", "run"):
        prepare(args.output, args.rep)
    if args.stage in ("fit", "run"):
        fit_training(args.output, args.rep, args.fold, warmup=args.warmup, draws=args.draws,
                     quick=args.quick_initialization)
    if args.stage in ("integrate", "run"):
        integrate_training(args.output, args.rep, args.fold)
    if args.stage in ("auxiliary", "run"):
        fit_auxiliary(args.output, args.rep, args.fold, warmup=args.warmup, draws=args.draws)
    if args.stage in ("score", "run"):
        report = score_fold(args.output, args.rep, args.fold,
                            bootstrap_replicates=args.bootstrap_replicates)
        print(json.dumps({key: report[key] for key in
                          ("rep", "fold", "mean_joint_logscore", "mean_score_mcse")}, indent=2))


if __name__ == "__main__":
    main()
