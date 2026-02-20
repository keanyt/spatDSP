"""Graph construction for spatial coordinates.

We build a symmetric kNN graph with a locally-adaptive Gaussian kernel and
return the symmetric normalized Laplacian:

    L = I - D^{-1/2} W D^{-1/2}

This Laplacian is then used to compute heat-diffusion signatures.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def build_graph_laplacian(
    coords: np.ndarray,
    k: int,
    *,
    sigma_k: Optional[int] = None,
    eps: float = 1e-12,
) -> sp.csr_matrix:
    """Build a symmetric normalized Laplacian from spatial coordinates.

    Parameters
    ----------
    coords:
        Array of shape (n_cells, n_dim). Typically, adata.obsm['spatial'].
    k:
        Number of nearest neighbors per node.
    sigma_k:
        Neighbor rank used to define local bandwidth sigma_i. Defaults to k.
        Must satisfy 1 <= sigma_k <= k.
    eps:
        Numerical stability constant.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric normalized Laplacian (n_cells x n_cells).
    """
    coords = np.asarray(coords)
    if coords.ndim != 2:
        raise ValueError("coords must be 2D (n_cells, n_dim)")
    n = coords.shape[0]
    if n == 0:
        return sp.csr_matrix((0, 0))
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if k >= n:
        k = max(1, n - 1)

    if sigma_k is None:
        sigma_k = k
    sigma_k = int(np.clip(int(sigma_k), 1, k))

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(coords)
    dists, indices = nbrs.kneighbors(coords)
    dists, indices = dists[:, 1:], indices[:, 1:]

    sigma = np.maximum(dists[:, sigma_k - 1].copy(), eps)

    rows = np.repeat(np.arange(n), k)
    cols = indices.reshape(-1)
    dij2 = (dists.reshape(-1) ** 2)

    sigma_i = np.repeat(sigma, k)
    sigma_j = sigma[cols]
    denom = np.maximum(sigma_i * sigma_j, eps)

    vals = np.exp(-dij2 / denom)

    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)

    d = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    D_inv_sqrt = sp.diags(d_inv_sqrt)

    L = sp.eye(n, format="csr") - (D_inv_sqrt @ W @ D_inv_sqrt)
    return L
