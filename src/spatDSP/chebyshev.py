"""Chebyshev approximation for heat diffusion on graphs.

We approximate the heat kernel exp(-tau * L) acting on a signal X using
Chebyshev polynomials on the scaled Laplacian L' in [-1, 1].

This implementation is optimized for evaluating many taus at once using
Bessel-function based coefficients.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import scipy.sparse as sp
from scipy import special as spspecial


def _to_torch_sparse_csr(L: sp.csr_matrix, device: str) -> torch.Tensor:
    if not sp.isspmatrix_csr(L):
        L = L.tocsr()
    return torch.sparse_csr_tensor(
        torch.from_numpy(L.indptr).to(device),
        torch.from_numpy(L.indices).to(device),
        torch.from_numpy(L.data).to(device=device, dtype=torch.float32),
        size=L.shape,
    )


def _Lprime_mv(L_t: torch.Tensor, V: torch.Tensor, lmax: float) -> torch.Tensor:
    # L' = (2/lmax) L - I
    return (2.0 / lmax) * torch.sparse.mm(L_t, V) - V


def chebyshev_heat_scores_multi_tau(
    L: sp.csr_matrix,
    X: np.ndarray,
    *,
    order: int,
    taus: Sequence[float],
    lmax: float = 2.0,
    device: str = "cuda",
) -> np.ndarray:
    """Compute heat-diffusion signature scores for many taus.

    The returned scores match the statistic used in your notebook:
        score(g, tau) = || exp(-tau L) x_g ||_2^2

    where x_g is the z-scored expression vector for gene g.

    Parameters
    ----------
    L:
        Symmetric normalized Laplacian (CSR).
    X:
        Dense array (n_cells, n_genes) of per-cell signals.
        You should z-score columns before calling (we do not do it here).
    order:
        Chebyshev polynomial order.
    taus:
        Iterable of tau values.
    lmax:
        Spectral upper bound used for scaling (2.0 for normalized Laplacian).
    device:
        "cuda" or "cpu".

    Returns
    -------
    np.ndarray
        Array of shape (n_genes, n_taus).
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    taus = np.asarray(list(taus), dtype=float)
    if taus.ndim != 1 or len(taus) == 0:
        raise ValueError("taus must be a non-empty 1D sequence")

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_cells, n_genes)")


    gamma = 0.5 * taus * float(lmax)

    # I_m(gamma) for m=0..order, for each tau
    I = np.stack([spspecial.iv(np.arange(order + 1), g) for g in gamma], axis=1).astype(np.float32)

    coeffs = np.empty_like(I)
    coeffs[0] = I[0]
    for m in range(1, order + 1):
        coeffs[m] = 2.0 * ((-1.0) ** m) * I[m]

    scales = np.exp(-gamma).astype(np.float32)

    # Move to torch
    L_t = _to_torch_sparse_csr(L, device)
    X_t = torch.from_numpy(X).to(device)

    coeffs_t = torch.from_numpy(coeffs).to(device)
    scales_t = torch.from_numpy(scales).to(device)

    # Chebyshev recursion
    T0 = X_t
    T1 = _Lprime_mv(L_t, T0, float(lmax))

    Y = T0.unsqueeze(2) * coeffs_t[0] + T1.unsqueeze(2) * coeffs_t[1]
    for m in range(2, order + 1):
        T2 = 2.0 * _Lprime_mv(L_t, T1, float(lmax)) - T0
        Y = Y + T2.unsqueeze(2) * coeffs_t[m]
        T0, T1 = T1, T2

    Y = Y * scales_t


    scores = (Y * Y).sum(dim=0).T

    return scores.detach().cpu().numpy()
