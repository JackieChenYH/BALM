"""Training-only starting values used by the simulation and HCP analyses."""

from __future__ import annotations

import time
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.infer.util import constrain_fn, potential_energy, unconstrain_fn
from scipy.optimize import minimize
from scipy.special import expit, softmax
from sklearn.cluster import KMeans

from .balm import _coerce_hyper, _orthonormalize_qr, model_for


def problem(z, y, mask, *, n, M, K, alpha, method, seed, hyper=None):
    mask = np.asarray(mask, bool)
    z = np.where(mask, z, 0).astype(np.int32)
    obs = mask & (z == 1)
    y = np.where(obs, y, 0.0)
    model = model_for(n, M, K, alpha, method, hyper=hyper)
    kw = dict(Z=jnp.array(z), Y=jnp.array(y), mask=jnp.array(mask))
    rng = np.random.default_rng(seed)
    vals = dict(
        V=rng.normal(size=(M, n, K)),
        gamma=np.ones((M, K)),
        tau=np.array(2.0),
        sigma2=np.array(max(float(np.var(y[obs])) if obs.any() else 0.0, 0.1)),
    )
    if method != "hard":
        vals["W"] = softmax(rng.normal(0, 0.5, (len(z), M)), axis=1)
    if method != "weighted":
        vals["a0"] = np.array(
            np.log((z[mask].sum() + 0.5) / ((1 - z[mask]).sum() + 0.5))
        )
    if (
        method in ("coupled", "hard", "student3", "student5")
        and _coerce_hyper(hyper).a1_fixed is None
    ):
        vals["a1"] = np.array(0.8)
    if method == "independent":
        vals.update(
            V_binary=rng.normal(size=(M, n, K)),
            gamma_binary=np.ones((M, K)),
            tau_binary=np.array(2.0),
        )
    vkeys = [k for k in vals if k == "V" or k.startswith("V_")]

    def flat(v):
        return np.asarray(
            ravel_pytree(
                unconstrain_fn(model, (), kw, {k: jnp.array(a) for k, a in v.items()})
            )[0]
        )

    x = flat(vals)
    _, unravel = ravel_pytree(
        unconstrain_fn(model, (), kw, {k: jnp.array(a) for k, a in vals.items()})
    )

    def normalize(u):
        return dict(
            u, **{k: jnp.sqrt(float(n)) * _orthonormalize_qr(u[k]) for k in vkeys}
        )

    def exact(x):
        return potential_energy(model, (), kw, unravel(x))

    def profiled(x):
        u = unravel(x)
        penalty = sum(
            0.5 * jnp.sum((jnp.swapaxes(u[k], -1, -2) @ u[k] / n - jnp.eye(K)) ** 2)
            for k in vkeys
        )
        return potential_energy(model, (), kw, normalize(u)) + penalty

    def values(x):
        return {
            k: np.asarray(a)
            for k, a in constrain_fn(
                model,
                (),
                kw,
                normalize(unravel(jnp.array(x))),
                return_deterministic=False,
            ).items()
        }

    return dict(
        vals=vals,
        flat=flat,
        unravel=unravel,
        normalize=normalize,
        exact=exact,
        profiled=profiled,
        values=values,
        model=model,
        kw=kw,
        rng=rng,
        x=x,
    )


def spectral_edges(z, y, mask, M, method, seed):
    if M > len(z):
        raise ValueError(
            "The spectral initializer requires at least M networks; supply init_values otherwise"
        )
    obs = mask if method == "weighted" else mask & (z == 1)
    a = np.where(obs, z * y if method == "weighted" else y, 0.0)
    means = a.sum(0) / np.maximum(obs.sum(0), 1)
    filled = np.where(obs, a, means)
    km = KMeans(M, n_init=10, random_state=seed).fit(filled)
    d = ((filled[:, None] - km.cluster_centers_[None]) ** 2).mean(-1)
    soft = softmax(-d / max(float(np.median(d)), 1e-06) * 3.0, axis=1)
    hard = np.full_like(soft, 0.02 / M)
    hard[np.arange(len(z)), km.labels_] += 0.98
    binary_mean = (np.where(mask, z, 0).sum(0) + 0.5) / (mask.sum(0) + 1.0)
    binary = np.where(mask, 2.0 * (z - 0.5), np.log(binary_mean / (1 - binary_mean)))
    result = []
    for w in (soft, hard):
        edges = []
        b = []
        for e in range(z.shape[1]):
            ww = w[obs[:, e]]
            edges.append(
                np.linalg.solve(ww.T @ ww + 0.001 * np.eye(M), ww.T @ a[obs[:, e], e])
            )
            ww = w[mask[:, e]]
            b.append(
                np.linalg.solve(
                    ww.T @ ww + 0.001 * np.eye(M), ww.T @ binary[mask[:, e], e]
                )
            )
        result.append((w, np.array(edges).T, np.array(b).T))
    return result


def spectral_factor(edges, pattern, n, K):
    ij = np.triu_indices(n, 1)
    us = []
    gs = []
    for row, neg in zip(edges, pattern):
        a = np.zeros((n, n))
        a[ij] = row
        a += a.T
        for _ in range(12):
            ev, vec = np.linalg.eigh(a)
            idx = list(range(neg)) + list(range(n - (K - neg), n))
            lam = np.r_[
                -np.maximum(abs(ev[idx][:neg]), 0.2),
                np.maximum(abs(ev[idx][neg:]), 0.2),
            ]
            a = vec[:, idx] * lam @ vec[:, idx].T
            a[ij] = row
            a[ij[1], ij[0]] = row
        us.append(vec[:, idx])
        gs.append(lam / 2.0)
    return (np.sqrt(n) * np.array(us), np.array(gs))


def candidates(p, z, y, mask, n, M, K, method, seed):
    pats = list(product(range(K + 1), repeat=M))
    rng = p["rng"]
    sp = spectral_edges(z, y, mask, M, method, seed)
    for family in range(5):
        directions = {
            k: rng.normal(size=np.shape(v))
            for k, v in p["vals"].items()
            if k.startswith("V")
        }
        for index, pat in enumerate(pats):
            vals = dict(p["vals"])
            patb = pats[(index + family * 17 + 1) % len(pats)]
            for suffix, pattern in [("", pat)] + (
                [("_binary", patb)] if method == "independent" else []
            ):
                if family < 3:
                    vals["V" + suffix] = directions["V" + suffix]
                    vals["gamma" + suffix] = np.array(
                        [[-1.0] * r + [1.0] * (K - r) for r in pattern]
                    )
                else:
                    w, e, b = sp[family - 3]
                    vals["V" + suffix], vals["gamma" + suffix] = spectral_factor(
                        b if suffix else e, pattern, n, K
                    )
                    if method != "hard":
                        vals["W"] = w
            yield (
                dict(
                    id=f"f{family}_p{index:04d}",
                    family=family,
                    negative_start=list(pat),
                    binary_negative_start=list(patb)
                    if method == "independent"
                    else None,
                ),
                p["flat"](vals),
            )


DEFAULT_OPTIONS = {
    "profile": "simulation",
    "screen_iterations": 80,
    "refine_candidates": 12,
    "maxiter": 1000,
    "ftol": 1e-10,
    "gtol": 1e-5,
    "maxls": 30,
    "maxfun": 50000,
}


def _compact_candidates(p, z, y, mask, n, M, K, method, seed):
    spectral = spectral_edges(z, y, mask, M, method, seed)
    patterns = [
        np.full(M, K // 2),
        np.full(M, max(0, K // 2 - 1)),
        np.full(M, min(K, K // 2 + 1)),
        np.arange(M) % (K + 1),
    ]
    for family in range(3):
        for index, pattern in enumerate(patterns):
            values = {
                name: np.array(value, copy=True) for name, value in p["vals"].items()
            }
            if family < 2:
                w, edges, _ = spectral[family]
                values["V"], values["gamma"] = spectral_factor(edges, pattern, n, K)
                values["W"] = w.copy()
            else:
                values["V"] = p["rng"].normal(size=(M, n, K))
                values["gamma"] = np.array(
                    [[-1.0] * int(r) + [1.0] * (K - int(r)) for r in pattern]
                )
                values["W"] = softmax(p["rng"].normal(size=(len(z), M)), axis=1)
            yield (
                {
                    "id": f"f{family}_p{index}",
                    "family": family,
                    "negative_start": pattern.tolist(),
                },
                p["flat"](values),
            )


def initialize(
    Z, Y, mask, *, n, M, K, alpha, method, seed, options=None, hyper=None, progress=None
):
    """Return constrained starting values and optimizer summaries.

    ``profile='simulation'`` uses the complete five-family sign-count search.
    ``profile='hcp'`` uses the twelve proposals from the HCP subset analysis.
    ``profile='demo'`` uses two spectral proposals for a short illustration.
    The selection objective uses only observed training entries.
    """
    options = dict(options or {})
    profile = options.get("profile", "simulation")
    settings = dict(DEFAULT_OPTIONS)
    if profile == "hcp":
        settings.update(screen_iterations=60, refine_candidates=4, maxiter=600)
    elif profile == "demo":
        settings.update(screen_iterations=20, refine_candidates=1, maxiter=60)
    elif profile != "simulation":
        raise ValueError("Unknown initialization profile")
    settings.update(options)
    if any(
        int(settings[k]) != settings[k] or settings[k] < 1
        for k in (
            "screen_iterations",
            "refine_candidates",
            "maxiter",
            "maxls",
            "maxfun",
        )
    ):
        raise ValueError("Initialization iteration counts must be positive integers")
    z = np.where(mask, Z, 0).astype(np.int32)
    y = np.where(mask & (z == 1), Y, 0.0)
    p = problem(
        z, y, mask, n=n, M=M, K=K, alpha=alpha, method=method, seed=seed, hyper=hyper
    )
    if profile in ("hcp", "demo"):
        if method in ("hard", "independent"):
            raise ValueError(
                "Compact initialization supports shared-template continuous memberships"
            )
        proposed = _compact_candidates(p, z, y, mask, n, M, K, method, seed)
        if profile == "demo":
            proposed = (
                (meta, x) for meta, x in proposed if meta["id"] in ("f0_p0", "f1_p0")
            )
    else:
        proposed = candidates(p, z, y, mask, n, M, K, method, seed)
    profiled = jax.jit(jax.value_and_grad(p["profiled"]))
    exact = jax.jit(jax.value_and_grad(p["exact"]))

    def objective(x):
        value, gradient = profiled(jnp.asarray(x))
        return float(value), np.asarray(gradient, dtype=float)

    records, pool = [], []
    start = time.perf_counter()
    optimizer_options = {
        key: settings[key] for key in ("ftol", "gtol", "maxls", "maxfun")
    }

    def optimize(meta, x, phase, iterations):
        result = minimize(
            objective,
            x,
            jac=True,
            method="L-BFGS-B",
            options=dict(optimizer_options, maxiter=iterations),
        )
        normalized, _ = ravel_pytree(
            p["normalize"](p["unravel"](jnp.asarray(result.x)))
        )
        value, gradient = exact(normalized)
        values = p["values"](result.x)
        finite = bool(
            np.isfinite(value)
            and np.isfinite(gradient).all()
            and all(np.isfinite(a).all() for a in values.values())
        )
        record = dict(
            meta,
            phase=phase,
            potential=float(value),
            eligible=finite,
            optimizer_success=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
        )
        records.append(record)
        if progress:
            progress({"stage": "initialization", **record})
        if finite:
            pool.append((float(value), meta, result.x, values))

    for meta, x in proposed:
        optimize(meta, x, "screen", settings["screen_iterations"])
    if not pool:
        raise RuntimeError("No finite training initialization candidate")
    screened = len(pool)
    for _, meta, x, _ in sorted(pool, key=lambda item: item[0])[
        : settings["refine_candidates"]
    ]:
        optimize(meta, x, "refine", settings["maxiter"])
    best = min(pool, key=lambda item: item[0])
    values = best[3]
    info = dict(
        profile=profile,
        selected=best[1]["id"],
        selected_potential=best[0],
        screened=screened,
        records=records,
        seconds=time.perf_counter() - start,
    )
    if method == "independent":
        values, extra = refine_independent(
            z, y, mask, values, n=n, M=M, K=K, alpha=alpha, seed=seed, progress=progress
        )
        info["binary_refinement"] = extra
    return values, info


def refine_independent(z, y, mask, old, *, n, M, K, alpha, seed, progress=None):
    """Binary-channel refinement used for every separate-channel simulation fit."""
    p = problem(z, y, mask, n=n, M=M, K=K, alpha=alpha, method="independent", seed=seed)
    profiled = jax.jit(jax.value_and_grad(p["profiled"]))
    exact = jax.jit(jax.value_and_grad(p["exact"]))

    def objective(x):
        value, gradient = profiled(jnp.asarray(x))
        return float(value), np.asarray(gradient, dtype=float)

    def normalized(x):
        return ravel_pytree(p["normalize"](p["unravel"](jnp.asarray(x))))[0]

    records = []

    def optimize(name, x, iterations, keys=None):
        x = np.asarray(x).copy()
        if keys is None:
            indices = np.arange(len(x))
        else:
            flags = {
                k: np.ones_like(v) if k in keys else np.zeros_like(v)
                for k, v in p["unravel"](jnp.asarray(x)).items()
            }
            indices = np.flatnonzero(np.asarray(ravel_pytree(flags)[0]))

        def partial_objective(v):
            full = x.copy()
            full[indices] = v
            value, gradient = objective(full)
            return value, gradient[indices]

        result = minimize(
            partial_objective,
            x[indices],
            jac=True,
            method="L-BFGS-B",
            options=dict(
                maxiter=iterations, ftol=1e-11, gtol=1e-5, maxls=30, maxfun=50000
            ),
        )
        x[indices] = result.x
        values = p["values"](x)
        value, gradient = exact(normalized(x))
        profile, pg = objective(x)
        finite = bool(
            np.isfinite(value)
            and np.isfinite(gradient).all()
            and np.isfinite(profile)
            and np.isfinite(pg).all()
            and np.isfinite(x).all()
            and all(np.isfinite(a).all() for a in values.values())
            and all(float(values[k]) > 0 for k in ("tau", "tau_binary", "sigma2"))
            and (values["W"] > 0).all()
            and np.allclose(values["W"].sum(-1), 1.0, rtol=0, atol=1e-10)
        )
        record = dict(
            id=name,
            potential=float(value),
            eligible=finite,
            iterations=int(result.nit),
            optimizer_success=bool(result.success),
        )
        records.append(record)
        if progress:
            progress({"stage": "initialization", **record})
        return (float(value) if finite else float("inf"), name, x, values)

    initial = p["flat"](old)
    baseline = float(exact(normalized(initial))[0])
    w = old["W"]
    edges = np.zeros((M, z.shape[1]))
    intercept = float(old["a0"])
    for edge in range(z.shape[1]):
        use = mask[:, edge]
        weights, observed = w[use], z[use, edge]
        coefficients = np.zeros(M)
        for _ in range(20):
            probability = expit(intercept + weights @ coefficients)
            gradient = weights.T @ (probability - observed) + 0.05 * coefficients
            hessian = weights.T @ (
                (probability * (1 - probability))[:, None] * weights
            ) + 0.05 * np.eye(M)
            step = np.linalg.solve(hessian, gradient)
            coefficients -= np.clip(step, -3, 3)
            if np.abs(step).max() < 1e-7:
                break
        edges[:, edge] = coefficients
    pool = [
        (baseline, "baseline_original", initial, old),
        optimize("baseline_refine", initial, 500),
    ]
    for index, pattern in enumerate(product(range(K + 1), repeat=M)):
        values = dict(old)
        vectors, eigenvalues = spectral_factor(edges, pattern, n, K)
        values["V_binary"] = vectors
        values["tau_binary"] = np.asarray(max(float(old["tau_binary"]), 0.5))
        values["gamma_binary"] = eigenvalues * 2 / float(values["tau_binary"])
        pool.append(
            optimize(
                f"binary_screen_{index:02d}",
                p["flat"](values),
                80,
                {"V_binary", "gamma_binary", "tau_binary", "a0"},
            )
        )
    pool = sorted(
        (item for item in pool if np.isfinite(item[0])), key=lambda item: item[0]
    )
    refined = [optimize(item[1] + "_joint", item[2], 600) for item in pool[:8]]
    refined = sorted(
        (item for item in refined if np.isfinite(item[0])), key=lambda item: item[0]
    )
    final = [optimize(item[1] + "_final", item[2], 1400) for item in refined[:2]]
    best = min(pool + refined + final, key=lambda item: item[0])
    return best[3], dict(
        selected=best[1],
        selected_potential=best[0],
        baseline_potential=baseline,
        records=records,
    )
