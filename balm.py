from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import random

import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

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
    a1_fixed: Optional[float] = None
    use_student_t: bool = False
    t_df: float = 5.0
    sigma_beta: float = 1.0
    sigma_epsilon: float = 1.0

def _coerce_hyper(hyper: Optional[object]) -> BALMHyperParams:
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
            a1_fixed=(None if hyper.get("a1_fixed", None) is None else float(hyper["a1_fixed"])),
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

def preprocess_hurdle_logit(A: Array, eps: float = 1e-6, z_thresh: float = 0.0) -> Tuple[Array, Array]:
    if A.ndim != 3:
        raise ValueError(f"A must have shape (L,n,n). Got {A.shape}.")
    A_ut = _vectorize_upper(A)
    Z = (A_ut > z_thresh).astype(jnp.int32)
    A_clip = jnp.clip(A_ut, eps, 1.0 - eps)
    Y_full = _logit(A_clip)
    Y = jnp.where(Z == 1, Y_full, 0.0)
    return Z, Y

def assert_inputs_host(Z, Y, X=None) -> tuple[int, int]:
    Znp = np.asarray(Z)
    Ynp = np.asarray(Y)

    if Znp.ndim != 2 or Ynp.ndim != 2:
        raise ValueError(f"Z and Y must have shape (L,P). Got {Znp.shape}, {Ynp.shape}.")
    if Znp.shape != Ynp.shape:
        raise ValueError(f"Z and Y must have the same shape. Got {Znp.shape} vs {Ynp.shape}.")
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

    return int(L), int(P)

def balm_hurdle_model(
    Z: Array,
    Y: Array,
    n: int,
    M: int,
    K: int,
    X: Optional[Array] = None,
    hyper: object = None,
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
            dist.Normal(0.0, hyper.sigma_beta).expand((M - 1, p_covariates)).to_event(2)
        )
        beta_zero = jnp.zeros((1, p_covariates))
        beta = numpyro.deterministic("beta", jnp.vstack([beta_zero, beta_raw]))

    with numpyro.plate("layers", L):
        if X is None:
            W = numpyro.sample("W", dist.Dirichlet(alpha_vec))
        else:
            epsilon = numpyro.sample(
                "epsilon", 
                dist.Normal(0.0, hyper.sigma_epsilon).expand((M,)).to_event(1)
            )
            eta = jnp.matmul(X, beta.T) + epsilon
            W = numpyro.deterministic("W", jax.nn.softmax(eta, axis=-1))

    with numpyro.plate("templates", M):
        V = numpyro.sample("V", dist.Normal(0.0, 1.0).expand((n, K)).to_event(2))
        U = numpyro.deterministic("U", _orthonormalize_qr(V))
        gamma = numpyro.sample(
            "gamma",
            dist.Normal(0.0, hyper.gamma_scale).expand((K,)).to_event(1),
        )

    U_gamma = U * gamma[..., None, :] 
    U_transpose = jnp.swapaxes(U, -1, -2)
    Q = tau * jnp.matmul(U_gamma, U_transpose)
    
    # Q_ut = _vectorize_upper(Q)
    Q_ut = numpyro.deterministic("Q_ut", _vectorize_upper(Q))
    mu_y = jnp.matmul(W, Q_ut)

    a0 = numpyro.sample("a0", dist.Normal(0.0, hyper.a0_scale))
    if hyper.a1_fixed is None:
        # a1 = numpyro.sample("a1", dist.Normal(0.0, hyper.a1_scale))
        a1 = numpyro.sample("a1", dist.HalfNormal(hyper.a1_scale))
    else:
        a1 = numpyro.deterministic("a1", jnp.asarray(hyper.a1_fixed))

    logits_p = a0 + a1 * mu_y
    numpyro.sample("Z", dist.Bernoulli(logits=logits_p).to_event(1), obs=Z)

    if hyper.use_student_t:
        base_dist = dist.StudentT(df=hyper.t_df, loc=mu_y, scale=sigma)
    else:
        base_dist = dist.Normal(loc=mu_y, scale=sigma)

    log_prob_Y = base_dist.log_prob(Y)
    numpyro.factor("Y_obs", jnp.sum(log_prob_Y * Z))

def run_balm_hurdle_nuts_from_ZY(
    rng_key: Array,
    Z: Array,
    Y: Array,
    n: int,
    M: int,
    K: int,
    X: Optional[Array] = None,
    hyper: Optional[BALMHyperParams] = None,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    progress_bar: bool = True,
    target_accept_prob: float = 0.8,
    max_tree_depth=7
) -> dict[str, Array]:
    hyper = _coerce_hyper(hyper)
    _ = assert_inputs_host(Z, Y, X)

    kernel = NUTS(
        lambda Z, Y, X: balm_hurdle_model(Z=Z, Y=Y, n=n, M=M, K=K, X=X, hyper=hyper),
        target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    mcmc.run(rng_key, Z=Z, Y=Y, X=X)
    return mcmc.get_samples()