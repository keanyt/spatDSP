"""Compute heat-diffusion curves for genes."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .chebyshev import chebyshev_heat_scores_multi_tau


def _dense_X(adata_X) -> np.ndarray:
    if sp.issparse(adata_X):
        return adata_X.toarray()
    return np.asarray(adata_X)


def zscore_columns(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def compute_curves_for_genes(
    adata,
    L,
    genes: Sequence[str],
    taus: Sequence[float],
    *,
    cheb_order: int = 125,
    batch_size: int = 8,
    device: str = "cuda",
    zscore_eps: float = 1e-8,
) -> pd.DataFrame:
    """Compute diffusion curves for a set of genes within one sample.

    Parameters
    ----------
    adata:
        AnnData for one sample.
    L:
        CSR Laplacian for this sample.
    genes:
        Genes to compute curves for (must be in adata.var_names).
    taus:
        Tau values.
    cheb_order:
        Chebyshev order.
    batch_size:
        Number of genes to process per batch.
    device:
        Torch device string.

    Returns
    -------
    pandas.DataFrame
        Index: genes, Columns: taus, Values: diffusion score per tau.
    """
    if len(genes) == 0:
        return pd.DataFrame(index=[], columns=list(taus), dtype=float)

    taus_list = list(map(float, taus))
    n_cells = int(adata.n_obs)


    var_names = adata.var_names
    gene_indices = [var_names.get_loc(g) for g in genes]

    out_blocks: List[np.ndarray] = []
    out_gene_names: List[str] = []

    for start in range(0, len(genes), int(batch_size)):
        batch_genes = genes[start : start + int(batch_size)]
        batch_idx = gene_indices[start : start + int(batch_size)]

        X_sub = _dense_X(adata.X[:, batch_idx])
        X_sub = zscore_columns(X_sub, eps=zscore_eps)

        scores = chebyshev_heat_scores_multi_tau(
            L,
            X_sub,
            order=int(cheb_order),
            taus=taus_list,
            lmax=2.0,
            device=device,
        )


        scores = scores / max(n_cells, 1)

        out_blocks.append(scores)
        out_gene_names.extend(batch_genes)

    full = np.vstack(out_blocks)  # (n_genes, n_taus)
    return pd.DataFrame(full, index=out_gene_names, columns=taus_list)
