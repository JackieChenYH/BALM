"""BALM likelihoods, product-coordinate NUTS, and exact scale updates.

Inputs are upper-triangular edge arrays Z and Y with shape (subjects, edges).
The observation mask is explicit; absent edges have no continuous observation.
Use fit_balm for fitting and predict for posterior predictive summaries.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import gammaln
from numpyro import handlers
from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform
from numpyro.distributions.util import sum_rightmost
from numpyro.infer import NUTS, init_to_value
from numpyro.infer.util import constrain_fn, initialize_model, potential_energy
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, logsumexp
from scipy.stats import geninvgauss, norm
from scipy.stats import t as student_t

Array = jnp.ndarray


@dataclass(frozen=True)
class BALMHyperParams:
    alpha: float = 1.0
    a_sigma: float = 2.0
    b_sigma: float = 1.0
    gamma_scale: float = 1.0
    tau_scale: float = 2.0
    a0_scale: float = 2.0
    a1_scale: float = 1.0
    a1_fixed: float | None = None
    use_student_t: bool = False
    t_df: float = 5.0
    sigma_beta: float = 1.0
    sigma_epsilon: float = 1.0


def _coerce_hyper(hyper: object | None) -> BALMHyperParams:
    if hyper is None:
        return BALMHyperParams()
    if isinstance(hyper, BALMHyperParams):
        return hyper
    if isinstance(hyper, dict):
        return BALMHyperParams(
            alpha=float(hyper.get("alpha", 1.0)),
            a_sigma=float(hyper.get("a_sigma", 2.0)),
            b_sigma=float(hyper.get("b_sigma", 1.0)),
            gamma_scale=float(hyper.get("gamma_scale", 1.0)),
            tau_scale=float(hyper.get("tau_scale", 2.0)),
            a0_scale=float(hyper.get("a0_scale", 2.0)),
            a1_scale=float(hyper.get("a1_scale", 1.0)),
            a1_fixed=None
            if hyper.get("a1_fixed", None) is None
            else float(hyper["a1_fixed"]),
            use_student_t=bool(hyper.get("use_student_t", False)),
            t_df=float(hyper.get("t_df", 5.0)),
            sigma_beta=float(hyper.get("sigma_beta", 1.0)),
            sigma_epsilon=float(hyper.get("sigma_epsilon", 1.0)),
        )
    raise TypeError(f"hyper must be None, BALMHyperParams or dict; got {type(hyper)}")


def _upper_triangle_indices(n: int) -> tuple[Array, Array]:
    return jnp.triu_indices(n, k=1)


def _vectorize_upper(A: Array) -> Array:
    if A.ndim == 2:
        iu = _upper_triangle_indices(A.shape[0])
        return A[iu]
    if A.ndim == 3:
        iu = _upper_triangle_indices(A.shape[1])
        return A[:, iu[0], iu[1]]
    raise ValueError(f"Expected 2D or 3D array, got shape {A.shape}")


def _orthonormalize_qr(V: Array) -> Array:
    Q, R = jnp.linalg.qr(V, mode="reduced")
    diag_sign = jnp.sign(jnp.diagonal(R, axis1=-2, axis2=-1))
    diag_sign = jnp.where(diag_sign == 0.0, 1.0, diag_sign)
    return Q * diag_sign[..., None, :]


def _logit(x: Array) -> Array:
    return jnp.log(x) - jnp.log1p(-x)


def preprocess_hurdle_logit(
    A: Array, eps: float = 1e-06, z_thresh: float = 0.0
) -> tuple[Array, Array]:
    if A.ndim != 3:
        raise ValueError(f"A must have shape (L,n,n). Got {A.shape}.")
    A_ut = _vectorize_upper(A)
    Z = (A_ut > z_thresh).astype(jnp.int32)
    A_clip = jnp.clip(A_ut, eps, 1.0 - eps)
    Y_full = _logit(A_clip)
    Y = jnp.where(Z == 1, Y_full, 0.0)
    return (Z, Y)


def assert_inputs_host(Z, Y, X=None) -> tuple[int, int]:
    Znp = np.asarray(Z)
    Ynp = np.asarray(Y)
    if Znp.ndim != 2 or Ynp.ndim != 2:
        raise ValueError(
            f"Z and Y must have shape (L,P). Got {Znp.shape}, {Ynp.shape}."
        )
    if Znp.shape != Ynp.shape:
        raise ValueError(
            f"Z and Y must have the same shape. Got {Znp.shape} vs {Ynp.shape}."
        )
    if not np.issubdtype(Znp.dtype, np.integer) and Znp.dtype != np.bool_:
        raise ValueError(f"Z should be integer/bool. Got dtype {Znp.dtype}.")
    if not np.all((Znp == 0) | (Znp == 1)):
        raise ValueError("Z must be binary {0,1}.")
    L, P = Znp.shape
    if L == 0 or P == 0:
        raise ValueError("Invalid input shapes: empty Z/Y.")
    if X is not None:
        Xnp = np.asarray(X)
        if Xnp.ndim != 2:
            raise ValueError(f"X must be a 2D array. Got shape {Xnp.shape}.")
        if Xnp.shape[0] != L:
            raise ValueError(f"X must have {L} subjects. Got {Xnp.shape[0]}.")
    return (int(L), int(P))


def assert_mask_host(Z, Y, n, mask):
    """Strict host validation prevents accidental broadcast across edges/layers."""
    shape = np.asarray(Z).shape
    if len(shape) != 2 or shape[1] != n * (n - 1) // 2:
        raise ValueError("P does not match n*(n-1)/2")
    if mask is None:
        out = np.ones(shape, dtype=bool)
    else:
        out = np.asarray(mask)
        if out.shape != shape or out.dtype != np.dtype(bool):
            raise ValueError("mask must be a boolean array with exactly the Z/Y shape")
    if not np.all(np.isfinite(np.asarray(Y)[out & (np.asarray(Z) == 1)])):
        raise ValueError("Observed present-edge Y must be finite")
    return out


def balm_hurdle_model(
    Z: Array,
    Y: Array,
    n: int,
    M: int,
    K: int,
    X: Array | None = None,
    hyper: object = None,
    mask: Array | None = None,
) -> None:
    hyper = _coerce_hyper(hyper)
    L, P = Z.shape
    alpha_vec = jnp.full((M,), hyper.alpha / M)
    sigma2 = numpyro.sample("sigma2", dist.InverseGamma(hyper.a_sigma, hyper.b_sigma))
    sigma = jnp.sqrt(sigma2)
    tau = numpyro.sample("tau", dist.HalfNormal(hyper.tau_scale))
    if X is not None:
        p_covariates = X.shape[1]
        beta_raw = numpyro.sample(
            "beta_raw",
            dist.Normal(0.0, hyper.sigma_beta)
            .expand((M - 1, p_covariates))
            .to_event(2),
        )
        beta_zero = jnp.zeros((1, p_covariates))
        beta = numpyro.deterministic("beta", jnp.vstack([beta_zero, beta_raw]))
    with numpyro.plate("layers", L):
        if X is None:
            W = numpyro.sample("W", dist.Dirichlet(alpha_vec))
        else:
            epsilon = numpyro.sample(
                "epsilon",
                dist.Normal(0.0, hyper.sigma_epsilon).expand((M,)).to_event(1),
            )
            eta = jnp.matmul(X, beta.T) + epsilon
            W = numpyro.deterministic("W", jax.nn.softmax(eta, axis=-1))
    with numpyro.plate("templates", M):
        V = numpyro.sample("V", dist.Normal(0.0, 1.0).expand((n, K)).to_event(2))
        U = numpyro.deterministic("U", _orthonormalize_qr(V))
        gamma = numpyro.sample(
            "gamma", dist.Normal(0.0, hyper.gamma_scale).expand((K,)).to_event(1)
        )
    U_gamma = U * gamma[..., None, :]
    U_transpose = jnp.swapaxes(U, -1, -2)
    Q = tau * jnp.matmul(U_gamma, U_transpose)
    Q_ut = numpyro.deterministic("Q_ut", _vectorize_upper(Q))
    mu_y = jnp.matmul(W, Q_ut)
    a0 = numpyro.sample("a0", dist.Normal(0.0, hyper.a0_scale))
    if hyper.a1_fixed is None:
        a1 = numpyro.sample("a1", dist.HalfNormal(hyper.a1_scale))
    else:
        a1 = numpyro.deterministic("a1", jnp.asarray(hyper.a1_fixed))
    logits_p = a0 + a1 * mu_y
    observed = (
        jnp.ones_like(Z, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    )
    Z_safe = jnp.where(observed, Z, 0)
    Y_safe = jnp.where(observed & (Z_safe == 1), Y, 0.0)
    numpyro.sample(
        "Z", dist.Bernoulli(logits=logits_p).mask(observed).to_event(1), obs=Z_safe
    )
    if hyper.use_student_t:
        base_dist = dist.StudentT(df=hyper.t_df, loc=mu_y, scale=sigma)
    else:
        base_dist = dist.Normal(loc=mu_y, scale=sigma)
    log_prob_Y = base_dist.log_prob(Y_safe)
    numpyro.factor(
        "Y_obs", jnp.sum(jnp.where(observed & (Z_safe == 1), log_prob_Y, 0.0))
    )


def templates(n, M, K, suffix=""):
    tau = numpyro.sample("tau" + suffix, dist.HalfNormal(2.0))
    V = numpyro.sample("V" + suffix, dist.Normal(0, 1).expand((M, n, K)).to_event(3))
    gamma = numpyro.sample(
        "gamma" + suffix, dist.Normal(0, 1).expand((M, K)).to_event(2)
    )
    U = _orthonormalize_qr(V)
    i, j = jnp.triu_indices(n, 1)
    return tau * jnp.einsum("mik,mk,mik->mi", U[:, i, :], gamma, U[:, j, :])


def hard_layer_scores(z, y, mask, q, a0, a1, sigma2):
    zz = jnp.where(mask, z, 0)
    yy = jnp.where(mask & (zz == 1), y, 0.0)
    lg = a0 + a1 * q[None, :, :]
    ll = dist.Bernoulli(logits=lg).log_prob(zz[:, None, :]) + jnp.where(
        zz[:, None, :] == 1,
        dist.Normal(q[None, :, :], jnp.sqrt(sigma2)).log_prob(yy[:, None, :]),
        0.0,
    )
    return jnp.where(mask[:, None, :], ll, 0.0).sum(-1)


def comparator(Z, Y, mask, n, M, K, alpha, method):
    observed = jnp.asarray(mask)
    z = jnp.where(observed, Z, 0)
    y = jnp.where(observed & (z == 1), Y, 0.0)
    q = templates(n, M, K)
    s = numpyro.sample("sigma2", dist.InverseGamma(2.0, 1.0))
    if method != "hard":
        W = numpyro.sample(
            "W", dist.Dirichlet(jnp.repeat(alpha / M, M)).expand((len(Z),))
        )
        mu = W @ q
    if method == "weighted":
        numpyro.factor(
            "weighted_observations",
            jnp.where(
                observed, dist.Normal(mu, jnp.sqrt(s)).log_prob(z * y), 0.0
            ).sum(),
        )
        return
    a0 = numpyro.sample("a0", dist.Normal(0, 2.0))
    if method == "independent":
        lg = a0 + W @ templates(n, M, K, "_binary")
    else:
        a1 = numpyro.sample("a1", dist.HalfNormal(1.0))
        if method == "hard":
            scores = hard_layer_scores(z, y, observed, q, a0, a1, s)
            numpyro.factor(
                "whole_layer_mixture",
                (jax.scipy.special.logsumexp(scores, axis=1) - jnp.log(float(M))).sum(),
            )
            return
        lg = a0 + a1 * mu
    numpyro.factor(
        "hurdle",
        jnp.where(
            observed,
            dist.Bernoulli(logits=lg).log_prob(z)
            + jnp.where(z == 1, dist.Normal(mu, jnp.sqrt(s)).log_prob(y), 0.0),
            0.0,
        ).sum(),
    )


def helmert_basis(size, dtype=None):
    """Fixed M x (M-1) orthonormal basis of the sum-zero subspace.

    Column j has 1/sqrt((j+1)(j+2)) in its first j+1 entries and
    -(j+1)/sqrt((j+1)(j+2)) in entry j+1 (zero-based j).
    """
    if int(size) != size or size < 1:
        raise ValueError("Simplex size must be a positive integer")
    h = np.zeros((size, size - 1), dtype=np.float64)
    for j in range(size - 1):
        den = np.sqrt((j + 1) * (j + 2))
        h[: j + 1, j] = 1.0 / den
        h[j + 1, j] = -(j + 1) / den
    return jnp.asarray(h, dtype=dtype)


class HelmertSoftmaxTransform(Transform):
    """Bijection R^(M-1) -> interior simplex via W=softmax(H z).

    The Jacobian of the first M-1 components is sqrt(M)*prod(W).
    The ambient surface Jacobian would instead be M*prod(W); using that
    measure would introduce an incorrect constant into a Dirichlet density.
    """

    domain = constraints.real_vector
    codomain = constraints.simplex

    def __init__(self, size):
        self.size = int(size)
        if self.size != size or self.size < 1:
            raise ValueError("Simplex size must be a positive integer")

    def log_softmax(self, z):
        z = jnp.asarray(z)
        self.forward_shape(z.shape)
        logits = z @ helmert_basis(self.size, z.dtype).T
        return jax.nn.log_softmax(logits, axis=-1)

    def __call__(self, z):
        return jnp.exp(self.log_softmax(z))

    def _inverse(self, w):
        w = jnp.asarray(w)
        self.inverse_shape(w.shape)
        log_w = jnp.log(w)
        return (log_w - jnp.mean(log_w, axis=-1, keepdims=True)) @ helmert_basis(
            self.size, w.dtype
        )

    def log_abs_det_jacobian(self, z, w, intermediates=None):
        return 0.5 * jnp.log(float(self.size)) + self.log_softmax(z).sum(axis=-1)

    def forward_shape(self, shape):
        if not shape or shape[-1] != self.size - 1:
            raise ValueError(
                f"Expected final coordinate dimension {self.size - 1}, got {shape}"
            )
        return shape[:-1] + (self.size,)

    def inverse_shape(self, shape):
        if not shape or shape[-1] != self.size:
            raise ValueError(
                f"Expected final simplex dimension {self.size}, got {shape}"
            )
        return shape[:-1] + (self.size - 1,)

    def tree_flatten(self):
        return ((), ((), {"size": self.size}))

    def __eq__(self, other):
        return isinstance(other, HelmertSoftmaxTransform) and self.size == other.size


@dataclass
class WGeometryAdapter:
    potential_fn: Callable
    postprocess_fn: Callable
    init_params: dict
    constrained_values: Callable
    transform: HelmertSoftmaxTransform


def build_reparameterized(
    model, model_kwargs, values, key, *, model_args=(), site_name="W"
):
    """Use Helmert coordinates for a Dirichlet site, including its Jacobian.

    The potential evaluates log memberships directly to avoid underflow at
    simplex boundaries. Other latent sites keep their NumPyro coordinates.
    """
    model_kwargs = dict(model_kwargs or {})
    values = {k: jnp.asarray(v) for k, v in values.items()}
    if site_name not in values:
        raise ValueError(f"A constrained initialization for {site_name!r} is required")
    w_initial = np.asarray(values[site_name])
    if (
        w_initial.ndim < 1
        or not np.isfinite(w_initial).all()
        or (not (w_initial > 0).all())
    ):
        raise ValueError(
            "W initialization must be finite and strictly inside the simplex"
        )
    if not np.allclose(w_initial.sum(-1), 1.0, rtol=0, atol=1e-06):
        raise ValueError("W initialization rows must sum to one")
    original = initialize_model(
        key,
        model,
        init_strategy=init_to_value(values=values),
        model_args=model_args,
        model_kwargs=model_kwargs,
    )
    site = original.model_trace.get(site_name)
    if site is None or site["type"] != "sample" or site["is_observed"]:
        raise ValueError(f"{site_name!r} must be an unobserved simplex sample site")
    support = site["fn"].support
    base_support = support
    while isinstance(base_support, constraints.independent):
        base_support = base_support.base_constraint
    if base_support is not constraints.simplex:
        raise ValueError(f"{site_name!r} must have simplex support")
    prior = site["fn"]
    while isinstance(prior, (dist.ExpandedDistribution, dist.Independent)):
        prior = prior.base_dist
    if not isinstance(prior, dist.Dirichlet):
        raise ValueError(
            "The stable W geometry adapter requires the original Dirichlet prior"
        )
    transform = HelmertSoftmaxTransform(w_initial.shape[-1])
    original_init = dict(original.param_info.z)
    initial = dict(original_init, **{site_name: transform.inv(values[site_name])})

    def split(params):
        return {k: v for k, v in params.items() if k != site_name}

    def w_model(params, add_jacobian):

        def substitute_w(site):
            if site["type"] == "sample" and site["name"] == site_name:
                z = params[site_name]
                w = transform(z)
                if add_jacobian:
                    log_w = transform.log_softmax(z)
                    a = jnp.broadcast_to(prior.concentration, w.shape)
                    log_norm = gammaln(a.sum(-1)) - gammaln(a).sum(-1)
                    target = (
                        log_norm
                        + (a * log_w).sum(-1)
                        + 0.5 * jnp.log(float(transform.size))
                    )
                    event_dim = len(site["fn"].event_shape)
                    target = sum_rightmost(
                        target, jnp.ndim(target) - jnp.ndim(w) + event_dim
                    )
                    site["fn"] = site["fn"].mask(False)
                    numpyro.factor(f"_{site_name}_dirichlet_helmert_density", target)
                return w

        return handlers.substitute(model, substitute_fn=substitute_w)

    def potential(params):
        return potential_energy(
            w_model(params, True), model_args, model_kwargs, split(params)
        )

    def constrained(params, return_deterministic=False):
        result = constrain_fn(
            w_model(params, False),
            model_args,
            model_kwargs,
            split(params),
            return_deterministic=return_deterministic,
        )
        result[site_name] = transform(params[site_name])
        return result

    return WGeometryAdapter(
        potential_fn=potential,
        postprocess_fn=partial(constrained, return_deterministic=False),
        init_params=initial,
        constrained_values=constrained,
        transform=transform,
    )


@dataclass
class SamplingGeometry:
    potential_fn: Callable
    postprocess_fn: Callable
    init_params: dict
    metadata: dict
    constrained_values: Callable
    log_w_fn: Callable
    dense_mass: bool | list = False


def build_sampling_geometry(
    model, model_kwargs, values, key, *, geometry="original", model_args=()
):
    """Construct a fixed-data potential in Helmert and optional product coordinates.

    For lambda=tau*gamma and t=log(tau), the potential adds M*K*t
    per template channel. log_w_fn retains finite log weights even when
    the corresponding probabilities underflow.
    """
    if geometry not in ("original", "lambda_logtau_v1"):
        raise ValueError(f"Unsupported sampling geometry: {geometry!r}")
    kwargs = dict(model_kwargs or {})
    values = {name: jnp.asarray(value) for name, value in values.items()}
    if "W" in values:
        base = build_reparameterized(model, kwargs, values, key, model_args=model_args)
        base_potential = base.potential_fn
        base_constrained = base.constrained_values
        base_initial = dict(base.init_params)

        def log_w(params):
            return base.transform.log_softmax(params["W"])
    else:
        original = initialize_model(
            key,
            model,
            init_strategy=init_to_value(values=values),
            model_args=model_args,
            model_kwargs=kwargs,
        )
        base_potential = original.potential_fn
        base_initial = dict(original.param_info.z)

        def base_constrained(params, return_deterministic=False):
            return constrain_fn(
                model,
                model_args,
                kwargs,
                params,
                return_deterministic=return_deterministic,
            )


        def log_w(params):
            return None

    channels = []
    for gamma_name in sorted(
        name for name in base_initial if name == "gamma" or name.startswith("gamma_")
    ):
        suffix = gamma_name[len("gamma") :]
        tau_name, lambda_name = ("tau" + suffix, "lambda" + suffix)
        if tau_name not in base_initial or lambda_name in base_initial:
            raise ValueError(f"Invalid paired scale sites for {gamma_name!r}")
        gamma, t = (base_initial[gamma_name], base_initial[tau_name])
        if gamma.ndim != 2 or t.ndim != 0:
            raise ValueError("Each channel requires gamma[M,K] and scalar log-tau")
        if not np.isfinite(np.asarray(gamma)).all() or not np.isfinite(np.asarray(t)):
            raise ValueError("Initial scale coordinates must be finite")
        if not np.allclose(
            np.asarray(gamma), np.asarray(values[gamma_name]), rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Gamma must use the original real-valued transform")
        if not np.allclose(
            np.exp(np.asarray(t)), np.asarray(values[tau_name]), rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Tau must use the original log-scale transform")
        channels.append(
            dict(
                gamma=gamma_name,
                tau=tau_name,
                product=lambda_name,
                shape=list(gamma.shape),
                dimensions=prod(gamma.shape),
            )
        )
    if not channels or "gamma" not in base_initial:
        raise ValueError("A continuous gamma/tau channel is required")
    if geometry == "original":

        def to_base(params):
            return dict(params)

        def from_base(params):
            return dict(params)

        def correction(params):
            return jnp.asarray(0.0, dtype=params[channels[0]["tau"]].dtype)
    else:

        def to_base(params):
            result = dict(params)
            for channel in channels:
                lam = result.pop(channel["product"])
                result[channel["gamma"]] = lam * jnp.exp(-params[channel["tau"]])
            return result

        def from_base(params):
            result = dict(params)
            for channel in channels:
                gamma = result.pop(channel["gamma"])
                result[channel["product"]] = gamma * jnp.exp(params[channel["tau"]])
            return result

        def correction(params):
            return sum(
                channel["dimensions"] * params[channel["tau"]] for channel in channels
            )

    def potential(params):
        return base_potential(to_base(params)) + correction(params)

    def constrained(params, return_deterministic=False):
        return base_constrained(to_base(params), return_deterministic)

    metadata = dict(geometry=geometry, channels=channels)
    initial = from_base(base_initial)
    if not all(np.isfinite(np.asarray(value)).all() for value in initial.values()):
        raise ValueError("Mapped initialization contains nonfinite coordinates")
    return SamplingGeometry(
        potential_fn=potential,
        postprocess_fn=partial(constrained, return_deterministic=False),
        init_params=initial,
        metadata=metadata,
        constrained_values=constrained,
        log_w_fn=log_w,
    )


def build_w_block_geometry(base_geometry):
    """Give each network's Helmert membership coordinates a dense mass block."""
    original = base_geometry.init_params
    if base_geometry.metadata["geometry"] != "lambda_logtau_v1":
        raise ValueError("Membership blocks require product coordinates")
    shape = original["W"].shape
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("Membership coordinates must have shape (subjects, M-1), M > 1")
    rows = tuple(f"W_row_{index:04d}" for index in range(shape[0]))

    def to_unsplit(params):
        result = {name: value for name, value in params.items() if name not in rows}
        result["W"] = jnp.stack([params[name] for name in rows], axis=0)
        return result

    initial = {name: value for name, value in original.items() if name != "W"}
    initial.update({name: original["W"][index] for index, name in enumerate(rows)})

    def potential(params):
        return base_geometry.potential_fn(to_unsplit(params))

    def constrained(params, return_deterministic=False):
        return base_geometry.constrained_values(to_unsplit(params), return_deterministic)

    def log_w(params):
        return base_geometry.log_w_fn(to_unsplit(params))

    return SamplingGeometry(
        potential_fn=potential,
        postprocess_fn=partial(constrained, return_deterministic=False),
        init_params=initial,
        metadata=dict(base_geometry.metadata, geometry="lambda_logtau_W_row_blocks_v1"),
        constrained_values=constrained,
        log_w_fn=log_w,
        dense_mass=[(name,) for name in rows],
    )


def conditional_parameters(lam, s_tau, s_gamma):
    """Map a finite, nonzero channel product to SciPy's GIG convention."""
    lam = np.asarray(lam, dtype=np.float64)
    if lam.ndim != 2 or lam.size == 0 or (not np.isfinite(lam).all()):
        raise ValueError("Each channel requires a finite lambda[M,K] array")
    s_tau, s_gamma = (float(s_tau), float(s_gamma))
    if not all(math.isfinite(x) and x > 0 for x in (s_tau, s_gamma)):
        raise ValueError("Prior scales must be finite and strictly positive")
    largest = float(np.max(np.abs(lam)))
    if largest == 0:
        raise ValueError("r=0 is not supported by this positive-b GIG kernel")
    r = largest * float(np.sqrt(np.sum(np.square(lam / largest))))
    p = (1.0 - lam.size) / 2.0
    scipy_b = r / s_tau / s_gamma
    scale = r / s_gamma * s_tau
    if not all(math.isfinite(x) and x > 0 for x in (r, scipy_b, scale)):
        raise FloatingPointError(
            "GIG parameter mapping is outside finite positive float64 range"
        )
    return dict(
        dimensions=int(lam.size),
        r=r,
        p=p,
        scipy_b=scipy_b,
        scale=scale,
        s_tau=s_tau,
        s_gamma=s_gamma,
    )


@dataclass(frozen=True)
class Channel:
    tau: str
    product: str
    s_tau: float
    s_gamma: float


class GIGRefresh:
    """Refresh tau squared from its GIG conditional after each retained NUTS step.

    Lambda stays fixed; gamma=lambda/tau is reconstructed by postprocessing.
    Potential energy and gradients are updated for the next NUTS step.
    """

    def __init__(self, geometry, channel_scales, warmup_steps=500):
        if not jax.config.x64_enabled:
            raise ValueError("The GIG update requires JAX float64 mode")
        if geometry.metadata["geometry"] not in (
            "lambda_logtau_v1", "lambda_logtau_W_row_blocks_v1"
        ):
            raise ValueError("The GIG update requires product coordinates")
        self.geometry = geometry
        self.warmup_steps = warmup_steps
        self.channels = []
        for channel in geometry.metadata["channels"]:
            scales = channel_scales[channel["tau"]]
            conditional_parameters(geometry.init_params[channel["product"]], **scales)
            self.channels.append(Channel(channel["tau"], channel["product"], **scales))
        self.value_and_grad = jax.jit(jax.value_and_grad(geometry.potential_fn))

    def refresh(self, state, rng):
        if state.r is not None or int(state.i) < self.warmup_steps:
            raise ValueError("GIG updates require post-warmup NUTS with refreshed momentum")
        if not math.isfinite(float(state.potential_energy)) or not math.isfinite(float(state.energy)):
            raise FloatingPointError("Input potential and energy must be finite")
        before_rng = deepcopy(rng.bit_generator.state)
        z = dict(state.z)
        try:
            for channel in self.channels:
                parameters = conditional_parameters(
                    z[channel.product], channel.s_tau, channel.s_gamma
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    x = float(geninvgauss.rvs(
                        parameters["p"], parameters["scipy_b"], loc=0.0,
                        scale=parameters["scale"], random_state=rng,
                    ))
                if not math.isfinite(x) or x <= 0:
                    raise FloatingPointError("GIG draw is not finite and strictly positive")
                t = 0.5 * math.log(x)
                tau = math.exp(t)
                gamma = np.asarray(z[channel.product]) / tau
                if not math.isfinite(tau) or tau <= 0 or not np.isfinite(gamma).all():
                    raise FloatingPointError("Reconstructed tau/gamma is not finite")
                z[channel.tau] = jnp.asarray(t, dtype=z[channel.tau].dtype)
            potential, gradient = self.value_and_grad(z)
            jax.block_until_ready((potential, gradient))
            if not math.isfinite(float(potential)) or any(
                not np.isfinite(np.asarray(value)).all()
                for value in jax.tree_util.tree_leaves(gradient)
            ):
                raise FloatingPointError("Refreshed potential or gradient is nonfinite")
            return state._replace(
                z=z, potential_energy=potential, z_grad=gradient,
                energy=state.energy + (potential - state.potential_energy),
            )
        except BaseException:
            rng.bit_generator.state = before_rng
            raise


def model_for(n, M, K, alpha=1.0, method="coupled", hyper=None):
    """Return the NumPyro likelihood used by a named manuscript model."""
    method = {
        "balm": "coupled",
        "fixed_zero": "null",
        "separate": "independent",
        "weighted_only": "weighted",
    }.get(method, method)
    if method in ("coupled", "null", "student3", "student5"):
        settings = replace(_coerce_hyper(hyper), alpha=float(alpha))
        if method == "null":
            settings = replace(settings, a1_fixed=0.0)
        if method.startswith("student"):
            settings = replace(settings, use_student_t=True, t_df=float(method[-1]))
        return partial(balm_hurdle_model, n=n, M=M, K=K, hyper=settings)
    if method not in ("hard", "weighted", "independent"):
        raise ValueError(f"Unknown model: {method}")
    if (
        hyper is not None
        and replace(_coerce_hyper(hyper), alpha=1.0) != BALMHyperParams()
    ):
        raise ValueError(
            "The manuscript comparator variants use their fixed published priors"
        )
    return partial(comparator, n=n, M=M, K=K, alpha=float(alpha), method=method)


def template_draws(samples, suffix=""):
    """Reconstruct signed off-diagonal templates, shape (draws, M, P)."""
    vectors = np.asarray(samples["V" + suffix])
    if vectors.ndim != 4:
        raise ValueError("Posterior V must have shape (draws, M, n, K)")
    orthogonal, triangular = np.linalg.qr(vectors)
    signs = np.where(np.diagonal(triangular, axis1=-2, axis2=-1) < 0.0, -1.0, 1.0)
    orthogonal *= signs[..., None, :]
    rows, columns = np.triu_indices(vectors.shape[-2], 1)
    eigenvalues = np.asarray(samples["tau" + suffix])[:, None, None] * np.asarray(
        samples["gamma" + suffix]
    )
    return np.einsum(
        "smek,smk,smek->sme",
        orthogonal[:, :, rows],
        eigenvalues,
        orthogonal[:, :, columns],
    )


def _json_values(value):
    if isinstance(value, dict):
        return {key: _json_values(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_values(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass
class FitResult:
    """Posterior draws and numerical diagnostics from one chain."""

    samples: dict
    log_weights: np.ndarray
    diagnostics: dict
    initialization: dict
    metadata: dict

    @property
    def q(self):
        return template_draws(self.samples)

    def save(self, directory):
        """Save portable arrays and metadata; an existing fit is not overwritten."""
        import json
        from pathlib import Path

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if (path / "posterior.npz").exists() or (path / "metadata.json").exists():
            raise FileExistsError(f"A fit already exists in {path}")
        arrays = dict(self.samples, log_weights=self.log_weights)
        arrays.update(
            {
                "diagnostic__" + name: np.asarray(value)
                for name, value in self.diagnostics.items()
            }
        )
        np.savez_compressed(path / "posterior.npz", **arrays)
        (path / "metadata.json").write_text(
            json.dumps(
                _json_values(
                    {
                        "schema": 1,
                        "metadata": self.metadata,
                        "initialization": self.initialization,
                    }
                ),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory):
        """Load a saved fit without unpickling executable Python objects."""
        import json
        from pathlib import Path

        path = Path(directory)
        document = json.loads((path / "metadata.json").read_text("utf-8"))
        if document.get("schema") != 1:
            raise ValueError("Unsupported fit schema")
        with np.load(path / "posterior.npz", allow_pickle=False) as archive:
            samples = {
                key: archive[key]
                for key in archive.files
                if key != "log_weights" and not key.startswith("diagnostic__")
            }
            diagnostics = {
                key[len("diagnostic__") :]: archive[key]
                for key in archive.files
                if key.startswith("diagnostic__")
            }
            log_weights = archive["log_weights"]
        return cls(
            samples,
            log_weights,
            diagnostics,
            document["initialization"],
            document["metadata"],
        )

    def predict(self, Z, Y, mask=None, **kwargs):
        return predict(self, Z, Y, mask, **kwargs)

    def aligned_summary(self, reference=None, **kwargs):
        return aligned_summary(self, reference, **kwargs)


def _validate_data(Z, Y, mask, n, M, K):
    if any(int(value) != value or value < 1 for value in (n, M, K)) or n < 2 or K > n:
        raise ValueError("Require n >= 2, M >= 1 and 1 <= K <= n")
    Z, Y = np.asarray(Z), np.asarray(Y)
    assert_inputs_host(Z, Y)
    observed = assert_mask_host(Z, Y, int(n), mask)
    if not observed.any(axis=1).all():
        raise ValueError("Every network must have an observed training entry")
    return (
        np.where(observed, Z, 0).astype(np.int32),
        np.where(observed & (Z == 1), Y, 0.0),
        observed,
    )


def fit_balm(
    Z,
    Y,
    mask,
    n,
    M,
    K,
    *,
    method="coupled",
    alpha=None,
    hyper=None,
    warmup=500,
    draws=2000,
    seed=0,
    init_seed=None,
    gig_seed=None,
    init_values=None,
    initialization=None,
    target_accept=0.8,
    max_tree_depth=9,
    progress=None,
):
    """Fit BALM or a manuscript variant with joint NUTS and exact scale updates.

    Z, Y and the boolean training mask have shape (L, n*(n-1)//2). Y is already
    on the chosen signed score scale. The function enables JAX float64.
    One dense kinetic block is adapted per membership vector. After warm-up,
    each NUTS step is followed by the exact conditional draw of tau squared.
    ``init_values`` accepts a constrained native draw for transfer starts.
    ``initialization`` controls the training-only search; see initialization.py.
    A callback receives dictionaries describing initialization or iteration.
    """
    from .initialization import initialize

    jax.config.update("jax_enable_x64", True)
    method = {
        "balm": "coupled",
        "fixed_zero": "null",
        "separate": "independent",
        "weighted_only": "weighted",
    }.get(method, method)
    if int(warmup) != warmup or warmup < 0 or int(draws) != draws or draws < 1:
        raise ValueError("warmup must be nonnegative and draws positive integers")
    if (
        not 0 < target_accept < 1
        or int(max_tree_depth) != max_tree_depth
        or max_tree_depth < 1
    ):
        raise ValueError("Require 0 < target_accept < 1 and positive max_tree_depth")
    settings = _coerce_hyper(hyper)
    alpha = settings.alpha if alpha is None else float(alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    settings = replace(settings, alpha=alpha)
    if method == "null":
        settings = replace(settings, a1_fixed=0.0)
    if method.startswith("student"):
        settings = replace(settings, use_student_t=True, t_df=float(method[-1]))
    z, y, observed = _validate_data(Z, Y, mask, n, M, K)
    model_hyper = (
        settings if method in ("coupled", "null", "student3", "student5") else hyper
    )
    model = model_for(n, M, K, alpha, method, hyper=model_hyper)
    init_seed = int(seed) if init_seed is None else int(init_seed)
    gig_seed = int(seed) + 1 if gig_seed is None else int(gig_seed)
    started = time.perf_counter()
    if init_values is None:
        values, init_info = initialize(
            z,
            y,
            observed,
            n=n,
            M=M,
            K=K,
            alpha=alpha,
            method=method,
            seed=init_seed,
            options=initialization,
            hyper=model_hyper,
            progress=progress,
        )
    else:
        values = {key: np.asarray(value) for key, value in init_values.items()}
        init_info = {"profile": "supplied_constrained_values"}
    kwargs = dict(Z=jnp.asarray(z), Y=jnp.asarray(y), mask=jnp.asarray(observed))
    trace = handlers.trace(
        handlers.seed(model, jax.random.PRNGKey(int(seed)))
    ).get_trace(**kwargs)
    latent_names = {
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site["is_observed"]
    }
    values = {
        key: jnp.asarray(value) for key, value in values.items() if key in latent_names
    }
    missing = latent_names - values.keys()
    if missing:
        raise ValueError(f"Initial values lack latent sites: {sorted(missing)}")
    geometry = build_sampling_geometry(
        model,
        kwargs,
        values,
        jax.random.PRNGKey(int(seed)),
        geometry="lambda_logtau_v1",
    )
    if "W" in geometry.init_params and M > 1:
        geometry = build_w_block_geometry(geometry)
    kernel = NUTS(
        potential_fn=geometry.potential_fn,
        dense_mass=geometry.dense_mass,
        max_tree_depth=int(max_tree_depth),
        target_accept_prob=target_accept,
    )
    state = kernel.init(
        jax.random.PRNGKey(int(seed)),
        num_warmup=int(warmup),
        init_params=geometry.init_params,
        model_args=(),
        model_kwargs={},
    )
    step = jax.jit(lambda state: kernel.sample(state, (), {}))
    postprocess = jax.jit(geometry.postprocess_fn)
    log_weights_fn = jax.jit(geometry.log_w_fn) if method != "hard" else None
    scales = {}
    for channel in geometry.metadata["channels"]:
        channel_scales = {}
        for name, field in ((channel["tau"], "s_tau"), (channel["gamma"], "s_gamma")):
            distribution = trace[name]["fn"]
            while hasattr(distribution, "base_dist"):
                distribution = distribution.base_dist
            scale = np.asarray(distribution.scale)
            if not np.all(scale == scale.flat[0]):
                raise ValueError(
                    "GIG refresh requires a common coefficient scale per channel"
                )
            channel_scales[field] = float(scale.flat[0])
        scales[channel["tau"]] = channel_scales
    refresh = GIGRefresh(geometry, scales, warmup_steps=int(warmup))
    rng = np.random.Generator(np.random.PCG64(gig_seed))
    posterior, log_weights, diagnostics = [], [], []

    def hard_log_weights(sample):
        orthogonal = _orthonormalize_qr(sample["V"])
        i, j = jnp.triu_indices(n, 1)
        q = jnp.einsum(
            "mek,mk,mek->me",
            orthogonal[:, i],
            sample["tau"] * sample["gamma"],
            orthogonal[:, j],
        )
        scores = hard_layer_scores(
            kwargs["Z"],
            kwargs["Y"],
            kwargs["mask"],
            q,
            sample["a0"],
            sample["a1"],
            sample["sigma2"],
        )
        return jax.nn.log_softmax(scores, axis=1)

    hard_log_weights = jax.jit(hard_log_weights)
    for iteration in range(int(warmup) + int(draws)):
        before = step(state)
        jax.block_until_ready(before)
        if iteration < warmup:
            state = before
        else:
            state = refresh.refresh(before, rng)
            sample = {
                key: np.asarray(value)
                for key, value in jax.device_get(postprocess(state.z)).items()
            }
            if method == "hard":
                lw = np.asarray(hard_log_weights(sample))
                sample["W"] = np.exp(lw)
            else:
                lw = np.asarray(log_weights_fn(state.z))
            if settings.a1_fixed is not None and method not in (
                "weighted",
                "independent",
            ):
                sample["a1"] = np.asarray(settings.a1_fixed)
            posterior.append(sample)
            log_weights.append(lw)
            diagnostics.append(
                {
                    name: np.asarray(getattr(before, name))
                    for name in (
                        "num_steps",
                        "diverging",
                        "accept_prob",
                        "potential_energy",
                        "energy",
                    )
                }
            )
        if progress:
            progress(
                {
                    "stage": "warmup" if iteration < warmup else "sampling",
                    "iteration": iteration + 1,
                    "total": int(warmup) + int(draws),
                }
            )
    samples = {key: np.stack([draw[key] for draw in posterior]) for key in posterior[0]}
    diagnostic_arrays = {
        key: np.stack([row[key] for row in diagnostics]) for key in diagnostics[0]
    }
    metadata = dict(
        n=int(n),
        M=int(M),
        K=int(K),
        method=method,
        alpha=alpha,
        hyper=vars(settings),
        warmup=int(warmup),
        draws=int(draws),
        seed=int(seed),
        init_seed=init_seed,
        gig_seed=gig_seed,
        target_accept=target_accept,
        max_tree_depth=int(max_tree_depth),
        seconds=time.perf_counter() - started,
        sampler="joint_NUTS_then_exact_scale_GIG",
        energy_diagnostics="pre_refresh_NUTS",
        membership="training_conditional_class_probability"
        if method == "hard"
        else "continuous",
    )
    return FitResult(
        samples, np.stack(log_weights), diagnostic_arrays, init_info, metadata
    )


def aligned_summary(
    result, reference=None, *, anchor="first", probability=0.95, return_draws=False
):
    """Align signed template distances, permuting W identically for every draw.

    A supplied reference is intended for simulation recovery only. Without a
    reference, anchor='median_norm' applies the HCP anchor and ordering rule.
    """
    if not 0 < probability < 1:
        raise ValueError("probability must lie between zero and one")
    q = result.q
    w = np.asarray(result.samples["W"])
    if reference is None:
        index = 0
        if anchor == "median_norm":
            index = np.argsort(np.sum(q * q, axis=(1, 2)), kind="stable")[len(q) // 2]
        elif anchor != "first":
            raise ValueError("anchor must be first or median_norm")
        reference = q[index].copy()
        if anchor == "median_norm":
            reference = reference[
                np.argsort(-np.sum(reference * reference, axis=1), kind="stable")
            ]
    reference = np.asarray(reference)
    if reference.shape != q.shape[1:]:
        raise ValueError("Reference templates must have shape (M,P)")
    aligned_q, aligned_w = np.empty_like(q), np.empty_like(w)
    permutations = np.empty((len(q), q.shape[1]), dtype=int)
    for draw in range(len(q)):
        cost = np.mean((reference[:, None, :] - q[draw][None, :, :]) ** 2, axis=-1)
        _, permutation = linear_sum_assignment(cost)
        permutations[draw] = permutation
        aligned_q[draw] = q[draw, permutation]
        aligned_w[draw] = w[draw][:, permutation]
    tail = (1 - probability) / 2
    summary = dict(
        Q_mean=aligned_q.mean(0),
        W_mean=aligned_w.mean(0),
        Q_interval=np.quantile(aligned_q, [tail, 1 - tail], axis=0),
        W_interval=np.quantile(aligned_w, [tail, 1 - tail], axis=0),
        permutations=permutations,
        reference=reference,
    )
    if return_draws:
        summary.update(Q_draws=aligned_q, W_draws=aligned_w)
    return summary


def predict(result, Z, Y, mask=None, *, draw_batch=16):
    """Posterior means and pointwise hurdle log predictive densities.

    Returns arrays with shape Z.shape. ``strength_mean`` conditions on presence;
    ``score_mean`` is E[mu]. For the weighted-only model the only prediction is
    ``zero_filled_mean``. Log scores are NaN outside the requested mask.
    """
    Z, Y = np.asarray(Z), np.asarray(Y)
    n = result.metadata["n"]
    assert_inputs_host(Z, Y)
    mask = assert_mask_host(Z, Y, n, mask)
    if Z.shape[0] != result.samples["W"].shape[1]:
        raise ValueError(
            "Prediction requires the fitted participants; infer new memberships separately"
        )
    if int(draw_batch) != draw_batch or draw_batch < 1:
        raise ValueError("draw_batch must be a positive integer")
    samples, q = result.samples, result.q
    method = result.metadata["method"]
    count = len(q)
    mean_mu, mean_probability, mean_product = [
        np.zeros_like(Y, dtype=float) for _ in range(3)
    ]
    log_sum = np.full(Y.shape, -np.inf)
    observed_y = np.where(mask & (Z == 1), Y, 0.0)
    binary_q = template_draws(samples, "_binary") if method == "independent" else None
    hyper = result.metadata.get("hyper", {})
    use_student = bool(hyper.get("use_student_t", method.startswith("student")))
    df = float(
        hyper.get("t_df", int(method[-1]) if method.startswith("student") else 5)
    )
    for start in range(0, count, int(draw_batch)):
        stop = min(start + int(draw_batch), count)
        weights = samples["W"][start:stop]
        templates_batch = q[start:stop]
        mu = np.einsum("slm,smp->slp", weights, templates_batch)
        mean_mu += mu.sum(0)
        if method == "weighted":
            continue
        intercept = samples["a0"][start:stop, None, None]
        scale = np.sqrt(samples["sigma2"][start:stop, None, None])
        if method == "hard":
            logits = intercept + samples["a1"][start:stop, None, None] * templates_batch
            probability = expit(logits)
            mean_probability += np.einsum("slm,smp->lp", weights, probability)
            mean_product += np.einsum(
                "slm,smp->lp", weights, probability * templates_batch
            )
            density = np.where(
                Z[None, :, None, :] == 1,
                -np.logaddexp(0.0, -logits[:, None, :, :]),
                -np.logaddexp(0.0, logits[:, None, :, :]),
            )
            density += np.where(
                Z[None, :, None, :] == 1,
                norm.logpdf(
                    observed_y[None, :, None, :],
                    templates_batch[:, None, :, :],
                    scale[:, None, :, :],
                ),
                0.0,
            )
            lw = result.log_weights[start:stop]
            log_likelihood = logsumexp(lw[:, :, :, None] + density, axis=2)
        else:
            logits = intercept + (
                np.einsum("slm,smp->slp", weights, binary_q[start:stop])
                if method == "independent"
                else samples["a1"][start:stop, None, None] * mu
            )
            probability = expit(logits)
            mean_probability += probability.sum(0)
            mean_product += (probability * mu).sum(0)
            continuous = (
                student_t.logpdf(observed_y[None], df, loc=mu, scale=scale)
                if use_student
                else norm.logpdf(observed_y[None], mu, scale)
            )
            log_likelihood = np.where(
                Z[None] == 1,
                -np.logaddexp(0.0, -logits) + continuous,
                -np.logaddexp(0.0, logits),
            )
        log_sum = np.logaddexp(log_sum, logsumexp(log_likelihood, axis=0))
    mean_mu /= count
    if method == "weighted":
        return {"zero_filled_mean": mean_mu}
    mean_probability /= count
    mean_product /= count
    conditional = np.divide(
        mean_product,
        mean_probability,
        out=np.full_like(Y, np.nan, dtype=float),
        where=mean_probability > 0,
    )
    return dict(
        presence_probability=mean_probability,
        strength_mean=conditional,
        score_mean=mean_mu,
        zero_filled_mean=mean_product,
        log_density=np.where(mask, log_sum - np.log(count), np.nan),
    )
