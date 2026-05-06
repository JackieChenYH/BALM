# ALMA.py
#
# Mathematically correct implementation of Algorithm 1
# from "ALternating Minimization Algorithm for Clustering
# Mixture Multilayer Network" (ALMA).
#
# Model:
#   A in R^{L x n x n} is a tensor of L adjacency matrices.
#   We approximate
#       A ≈ Q ×_1 W
#   where
#       Q in R^{M x n x n} are M template matrices (rank K_m),
#       W in R^{L x M} has orthonormal columns (W^T W = I_M).
#
# Algorithm 1 alternates:
#   1) Q-update: Q = Pi_K( A ×_1 W^T )
#   2) W-update: W = Pi_o( A ×_{2,3} Q )
#
# where:
#   ×_1 is mode-1 product,
#   ×_{2,3} is mode-(2,3) product,
#   Pi_K is rank-K projection (best rank K approximation),
#   Pi_o is orthogonal Procrustes projection (columns orthonormal).
#
# The main entry point is:
#   Q_hat, W_hat = AltMin(A, params)
#
# params must contain:
#   'K': 1D array of length M with ranks K_m
#   'L': number of layers
#   'M': number of templates
#   'n': number of nodes
#
# Optional keys (if provided) are:
#   'max_iter', 'tol', 'random_state', 'verbose'

import numpy as np
from sklearn.cluster import KMeans


def _mode1_product(A, Wt):
    """
    Mode-1 product of tensor A with matrix Wt.

    A : (L, n, n)
    Wt: (M, L)  (this is W^T in the paper)

    Returns:
        Y : (M, n, n) = A ×_1 W^T
    """
    L, n, _ = A.shape
    M = Wt.shape[0]
    A_unfold = A.reshape(L, n * n)      # (L, n*n)
    Y_mat = Wt @ A_unfold               # (M, n*n)
    Y = Y_mat.reshape(M, n, n)
    return Y


def _mode23_product(A, Q):
    """
    Mode-(2,3) product A ×_{2,3} Q, as defined in the paper.

    A : (L, n, n)
    Q : (M, n, n)

    Returns:
        X : (L, M), where
            X[l, m] = sum_{i2,i3} A[l,i2,i3] * Q[m,i2,i3]
    """
    L, n, _ = A.shape
    M, n2, n3 = Q.shape
    assert n == n2 and n == n3

    A_unfold = A.reshape(L, n * n)      # (L, n*n)
    Q_unfold = Q.reshape(M, n * n)      # (M, n*n)
    X = A_unfold @ Q_unfold.T           # (L, M)
    return X


def _orthogonal_procrustes(X):
    """
    Pi_o(X) as in the paper.

    Given X in R^{L x M}, returns W in R^{L x M} with orthonormal columns.

    If SVD(X) = U S V^T with U in R^{L x M}, V in R^{M x M},
    then Pi_o(X) = U V^T.
    """
    L, M = X.shape
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  # U: (L, M), Vt: (M, M)
    W = U @ Vt                                        # (L, M)
    return W


def _rankK_projection(Q, K_vec):
    """
    Pi_K(Q): best rank-K_m approximation for each slice Q[m,:,:].

    Q     : (M, n, n)
    K_vec : array-like of shape (M,) with ranks K_m

    Returns:
        Q_proj : (M, n, n) with rank(Q_proj[m]) <= K_vec[m]
    """
    Q = np.asarray(Q)
    M, n, _ = Q.shape
    K_vec = np.asarray(K_vec, dtype=int)
    assert K_vec.shape[0] == M

    Q_proj = np.zeros_like(Q)
    for m in range(M):
        Qm = Q[m]
        # Enforce symmetry (since adjacency is symmetric)
        Qm_sym = 0.5 * (Qm + Qm.T)
        # Best rank-K approximation via eigen-decomposition
        # (equivalent to SVD for symmetric matrices)
        eigvals, eigvecs = np.linalg.eigh(Qm_sym)     # eigenvalues in ascending order
        # Sort eigenvalues by absolute value descending
        idx = np.argsort(np.abs(eigvals))[::-1]
        k = min(K_vec[m], n)
        idx_k = idx[:k]
        lam_k = eigvals[idx_k]
        vec_k = eigvecs[:, idx_k]
        Qm_k = vec_k @ np.diag(lam_k) @ vec_k.T
        Q_proj[m] = Qm_k

    return Q_proj


def AltMin(A, params):
    """
    ALMA main alternating minimization algorithm (Algorithm 1 in the paper).

    Parameters
    ----------
    A : ndarray, shape (L, n, n)
        Tensor of L adjacency matrices.
    params : dict
        Must contain:
            'K': 1D array of length M (ranks K_m)
            'L': int, number of layers
            'M': int, number of templates
            'n': int, number of nodes
        Optional:
            'max_iter': int, maximum iterations (default 200)
            'tol': float, tolerance on ||W_new - W_old||_F (default 1e-5)
            'random_state': int or None for KMeans
            'verbose': bool, print progress (default True)

    Returns
    -------
    Q_hat : ndarray, shape (M, n, n)
        Estimated template matrices.
    W_hat : ndarray, shape (L, M)
        Estimated mixing weights with orthonormal columns.
    """
    A = np.asarray(A, dtype=float)
    L, n1, n2 = A.shape
    K_vec = np.asarray(params["K"], dtype=int)
    L_param = int(params["L"])
    M = int(params["M"])
    n = int(params["n"])

    assert L == L_param, "Mismatch: L in A vs params['L']"
    assert n1 == n2 == n, "A must be (L, n, n) with square layers"
    assert K_vec.shape[0] == M, "K must have length M"

    max_iter = int(params.get("max_iter", 200))
    tol = float(params.get("tol", 1e-5))
    random_state = params.get("random_state", 0)
    verbose = bool(params.get("verbose", True))

    # ==========================================
    # 1. Initialization of W via KMeans on layers
    # ==========================================
    # A_unfold: (L, n*n), each row = vec(A_l)
    A_unfold = A.reshape(L, n * n)

    kmeans = KMeans(n_clusters=M, n_init=50, random_state=random_state)
    labels = kmeans.fit_predict(A_unfold)  # labels in {0,...,M-1}

    Z = np.zeros((L, M))
    Z[np.arange(L), labels] = 1.0          # one-hot indicator of cluster

    # Initial W = Pi_o(Z), so that W^T W = I_M
    W = _orthogonal_procrustes(Z)

    # ==========================================
    # 2. Alternating minimization
    # ==========================================
    err = np.inf
    iter_count = 0

    while err > tol and iter_count < max_iter:
        iter_count += 1
        W_old = W.copy()

        # ---- Q-update: Q = Pi_K(A ×_1 W^T) ----
        Q_tmp = _mode1_product(A, W.T)     # (M, n, n)
        Q = _rankK_projection(Q_tmp, K_vec)

        # ---- W-update: W = Pi_o(A ×_{2,3} Q) ----
        X = _mode23_product(A, Q)          # (L, M)
        W = _orthogonal_procrustes(X)      # (L, M), W^T W = I

        # Convergence diagnostic
        diff = W - W_old
        err = np.linalg.norm(diff, ord="fro")

        # if verbose:
        #     print(f"[ALMA] iter {iter_count}, ||W_new - W_old||_F = {err:.6e}")

    if verbose:
        print(f"[ALMA] finished in {iter_count} iterations, final error = {err:.6e}")

    return Q, W
