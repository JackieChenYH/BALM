"""ALMA with the training-only adapter used in the BALM experiments.

The alternating updates implement Algorithm 1 of Fan et al. (JMLR, 2022).
ALMA loadings are orthogonal factors, not simplex membership probabilities.
"""
from __future__ import annotations

import hashlib
import warnings
from collections.abc import Callable

import numpy as np
from sklearn.cluster import KMeans


def _training_arrays(z, train_mask, n, M, K):
    z = np.asarray(z, dtype=float)
    mask = np.asarray(train_mask)
    if mask.dtype != np.bool_:
        if not np.isin(mask, [0, 1]).all():
            raise ValueError("train_mask must contain only boolean values")
        mask = mask.astype(bool)
    if z.ndim != 2 or z.shape != mask.shape or z.shape[1] != n * (n - 1) // 2:
        raise ValueError("Expected z and train_mask with shape (subjects, n*(n-1)//2)")
    if not 1 <= M <= len(z) or not 1 <= K <= n:
        raise ValueError("Require 1 <= M <= subjects and 1 <= K <= nodes")
    if not mask.any() or not np.isin(z[mask], [0, 1]).all():
        raise ValueError("Observed training entries must be binary and nonempty")
    return z, mask


def _training_tensor(z, mask, n):
    observed = np.where(mask, z, 0.0)
    counts = mask.sum(axis=0)
    global_mean = observed.sum() / mask.sum()
    means = np.divide(observed.sum(axis=0), counts,
                      out=np.full(z.shape[1], global_mean), where=counts > 0)
    filled = np.where(mask, z, means[None, :])
    i, j = np.triu_indices(n, 1)
    tensor = np.zeros((len(z), n, n))
    tensor[:, i, j] = filled
    tensor[:, j, i] = filled
    return tensor, means


def _orthogonal_procrustes(matrix):
    left, _, right = np.linalg.svd(matrix, full_matrices=False)
    return left @ right


def _rank_projection(templates, rank):
    projected = np.zeros_like(templates)
    for m, template in enumerate(templates):
        values, vectors = np.linalg.eigh(0.5 * (template + template.T))
        order = np.argsort(np.abs(values))[::-1][:rank]
        selected = vectors[:, order]
        projected[m] = selected @ np.diag(values[order]) @ selected.T
    return projected


def fit_alma(z, train_mask, n, M, K, seed=0, *, max_iter=5000,
             tol=1e-5, state=None, callback: Callable | None = None,
             callback_every=100):
    """Fit ALMA to upper-triangle binary observations.

    Missing entries are filled by training means for the corresponding edge,
    with the overall training mean when an edge is never observed. The
    diagonal is zero. Both initialization and final loading-row clustering
    use 50 k-means starts with ``seed``.

    ``max_iter`` is the number of additional updates, or None to continue
    until the loading-change tolerance is met. Passing a previous returned
    dictionary as ``state`` continues its exact trajectory without a new
    initialization. The caller can persist the callback dictionary and use
    it as state. A capped result has optimization['converged'] == False.
    """
    z, mask = _training_arrays(z, train_mask, n, M, K)
    if max_iter is not None and (int(max_iter) != max_iter or max_iter < 1):
        raise ValueError("max_iter must be a positive integer or None")
    if not np.isfinite(tol) or tol <= 0 or callback_every < 1:
        raise ValueError("tol and callback_every must be positive")
    tensor, means = _training_tensor(z, mask, n)
    unfolded = tensor.reshape(len(z), n * n)
    digest = hashlib.sha256()
    digest.update(np.asarray([n, M, K, seed], dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(mask).tobytes())
    digest.update(np.ascontiguousarray(np.where(mask, z, 0.0)).tobytes())
    input_digest = digest.hexdigest()
    if state is None:
        labels = KMeans(n_clusters=M, n_init=50, random_state=seed).fit_predict(unfolded)
        one_hot = np.zeros((len(z), M))
        one_hot[np.arange(len(z)), labels] = 1.0
        loading = _orthogonal_procrustes(one_hot)
        templates = np.zeros((M, n, n))
        iteration, error = 0, float("inf")
    else:
        if state.get("input_digest") != input_digest:
            raise ValueError("Saved ALMA state does not match the training input and settings")
        loading = np.array(state["loading"], dtype=float, copy=True)
        templates = np.array(state["Q_binary"], dtype=float, copy=True)
        iteration = int(state["optimization"]["iterations"])
        error = float(state["optimization"]["final_error"])
        if loading.shape != (len(z), M) or templates.shape != (M, n, n):
            raise ValueError("Saved ALMA state has inconsistent dimensions")
    stop = None if max_iter is None else iteration + int(max_iter)

    def current_state():
        return {"loading": loading.copy(), "Q_binary": templates.copy(),
                "input_digest": input_digest,
                "optimization": {"iterations": iteration, "final_error": error,
                                 "tolerance": tol, "converged": error <= tol}}

    while error > tol and (stop is None or iteration < stop):
        old = loading.copy()
        templates = _rank_projection((loading.T @ unfolded).reshape(M, n, n), K)
        loading = _orthogonal_procrustes(unfolded @ templates.reshape(M, n * n).T)
        error = float(np.linalg.norm(loading - old, ord="fro"))
        iteration += 1
        if not np.isfinite(error):
            raise FloatingPointError("ALMA produced a nonfinite loading update")
        if callback is not None and (iteration % callback_every == 0 or error <= tol or iteration == stop):
            callback(current_state())
    result = current_state()
    i, j = np.triu_indices(n, 1)
    result.update(topology_score=loading @ templates[:, i, j],
                  labels=KMeans(n_clusters=M, n_init=50, random_state=seed).fit_predict(loading),
                  training_imputation_mean=means)
    if not result["optimization"]["converged"]:
        warnings.warn("ALMA reached its update budget; continue with state=result.", RuntimeWarning)
    return result
