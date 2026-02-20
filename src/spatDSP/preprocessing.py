"""Preprocessing helpers.
DSP is a sample-level test: we compute diffusion curves within each sample
(e.g., donor), then test for differences between groups across samples.
"""

from __future__ import annotations

from anndata import AnnData

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def split_adata_by_sample(
    adata,
    *,
    sample_key: str,
    group_key: str,
    groups: Tuple[str, str] | Tuple[int, int] | None = None,
    spatial_key: str = "spatial",
    ensure_unique_obs_names: bool = True,
) -> Tuple[Dict[str, "anndata.AnnData"], pd.DataFrame]:
    """Split a single AnnData into a dict of per-sample AnnDatas.

    Parameters
    ----------
    adata:
        Input AnnData with all cells.
    sample_key:
        adata.obs column that identifies sample/donor/section.
    group_key:
        adata.obs column that identifies group labels.
    groups:
        Optional tuple (group0, group1). If given, only those samples are kept.
        Group labels are assumed constant within sample; if not, majority vote is used.
    spatial_key:
        Key in adata.obsm for spatial coordinates.

    Returns
    -------
    (adatas, sample_meta)
        adatas: dict mapping sample_id -> AnnData (subset of cells)
        sample_meta: DataFrame indexed by sample_id with columns ['group']
    """
    if sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key '{sample_key}' not found in adata.obs")
    if group_key not in adata.obs.columns:
        raise KeyError(f"group_key '{group_key}' not found in adata.obs")
    if spatial_key not in adata.obsm:
        raise KeyError(f"spatial_key '{spatial_key}' not found in adata.obsm")

    sample_ids = adata.obs[sample_key].astype(str).values
    unique_samples = pd.unique(sample_ids)

    out: Dict[str, any] = {}
    rows: List[dict] = []

    for sid in unique_samples:
        mask = sample_ids == sid
        A = adata[mask].copy()
        if A.n_obs == 0:
            continue

        g = A.obs[group_key]
        g_nonnull = g.dropna()
        if len(g_nonnull) == 0:
            continue
        g_major = g_nonnull.value_counts().idxmax()

        if groups is not None:
            g0, g1 = groups
            if g_major not in {g0, g1}:
                continue

        if ensure_unique_obs_names:
            A.obs_names = [f"{sid}|{x}" for x in A.obs_names]

        out[str(sid)] = A
        rows.append({"sample": str(sid), "group": g_major})

    if len(out) == 0:
        raise ValueError("No samples left after splitting/filtering.")

    meta = pd.DataFrame(rows).set_index("sample")
    return out, meta
