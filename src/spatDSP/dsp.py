from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .graph import build_graph_laplacian
from .curves import compute_curves_for_genes
from .permtest import varweighted_permutation_test
from .preprocessing import split_adata_by_sample
from .results import DSPResult
from .utils import choose_device


def run_dsp(
    adata,
    *,
    sample_key: str,
    group_key: str,
    groups: Tuple[object, object],
    spatial_key: str = "spatial",
    genes: Optional[Sequence[str]] = None,
    taus: Sequence[float] = tuple(np.geomspace(0.01, 50.0, 51)),
    cheb_order: int = 125,
    k: int = 50,
    sigma_k: Optional[int] = None,
    graph_k_by_sample: Optional[Dict[str, int]] = None,
    batch_size: int = 8,
    device: str | None = None,
    # permutation params
    exact: Optional[bool] = None,
    n_perms: int = 1000,
    max_exact: int = 200_000,
    midp: bool = True,
    eps: float = 1e-6,
    seed: int = 0,
) -> DSPResult:
    """Run differential spatial patterning (DSP) between two groups.

    Parameters
    ----------
    adata:
        AnnData containing all cells across samples.
    sample_key:
        Column in ``adata.obs`` indicating sample / donor / section id.
    group_key:
        Column in ``adata.obs`` indicating group label.
    groups:
        Tuple ``(group0, group1)`` defining the comparison.
    spatial_key:
        Key in ``adata.obsm`` with spatial coordinates.
    genes:
        Genes to analyze. If None, uses the intersection of genes present in all
        per-sample curve matrices.
    taus:
        Tau grid for the heat diffusion curve.
    cheb_order:
        Chebyshev polynomial order.
    k:
        Default k for kNN graph building.
    sigma_k:
        Neighbour rank for adaptive bandwidth; defaults to k.
    graph_k_by_sample:
        Optional mapping sample_id -> k to override the default per sample.
    batch_size:
        Genes per batch for curve calculation.
    device:
        Torch device string ('cuda' or 'cpu'). Defaults to 'cuda'.

    Permutation testing
    -------------------
    exact:
        True: exact permutations, False: Monte Carlo, None: auto.
    n_perms:
        Number of Monte Carlo permutations (ignored if exact).
    max_exact:
        Maximum exact permutations allowed.

    Returns
    -------
    DSPResult
        Contains per-gene p-values, q-values, test statistic, and per-sample curves.
    """

    device = choose_device(device)

    group0, group1 = groups

    # Split into samples
    adatas, sample_meta = split_adata_by_sample(
        adata,
        sample_key=sample_key,
        group_key=group_key,
        groups=(group0, group1),
        spatial_key=spatial_key,
    )

    # Determine gene list if provided; otherwise keep all genes for curve calc,
    # and let the permutation test use the intersection.
    if genes is None:
        genes_to_compute = None
    else:
        genes_to_compute = list(genes)

    # Graph + curve computation
    curves_by_sample: Dict[str, pd.DataFrame] = {}
    taus_list = list(map(float, taus))

    for sid, A in adatas.items():
        coords = A.obsm[spatial_key]
        k_i = graph_k_by_sample.get(sid, k) if graph_k_by_sample is not None else k

        L = build_graph_laplacian(coords, k=int(k_i), sigma_k=sigma_k)

        # choose genes for this sample
        if genes_to_compute is None:
            gene_list = list(A.var_names)
        else:
            missing = [g for g in genes_to_compute if g not in A.var_names]
            if len(missing) > 0:
                raise KeyError(f"Sample {sid} is missing {len(missing)} requested genes (e.g. {missing[:5]}).")
            gene_list = genes_to_compute

        curves_by_sample[sid] = compute_curves_for_genes(
            A,
            L,
            gene_list,
            taus_list,
            cheb_order=int(cheb_order),
            batch_size=int(batch_size),
            device=device,
        )

    # Permutation test
    res_table = varweighted_permutation_test(
        curves_by_sample,
        sample_meta,
        group0=group0,
        group1=group1,
        taus=taus_list,
        genes=genes,
        exact=exact,
        n_perms=int(n_perms),
        max_exact=int(max_exact),
        midp=bool(midp),
        eps=float(eps),
        device=device,
        seed=int(seed),
    )

    params = {
        "sample_key": sample_key,
        "group_key": group_key,
        "groups": (group0, group1),
        "spatial_key": spatial_key,
        "taus": taus_list,
        "cheb_order": int(cheb_order),
        "k": int(k),
        "sigma_k": sigma_k if sigma_k is None else int(sigma_k),
        "batch_size": int(batch_size),
        "device": device,
        "perm_exact": exact,
        "perm_n_perms": int(n_perms),
        "perm_max_exact": int(max_exact),
        "perm_midp": bool(midp),
        "perm_eps": float(eps),
        "seed": int(seed),
    }

    return DSPResult(table=res_table, curves_by_sample=curves_by_sample, params=params)
