"""Permutation tests for Differential Spatial Patterning (DSP)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests

from .utils import normalize_curve_l2, choose_device


@dataclass
class PermutationPlan:
    mode: str  # 'exact' or 'montecarlo'
    n_perms: int


def plan_permutations(
    n: int,
    n1: int,
    *,
    n_perms: int = 1000,
    exact: Optional[bool] = None,
    max_exact: int = 200_000,
) -> PermutationPlan:
    """Decide whether to use exact or Monte Carlo permutations."""
    total = comb(n, n1)
    if exact is True:
        if total > max_exact:
            raise ValueError(
                f"Exact permutations would require {total} permutations, exceeding max_exact={max_exact}."
            )
        return PermutationPlan("exact", int(total))
    if exact is False:
        return PermutationPlan("montecarlo", int(n_perms))


    if total <= max_exact and total <= max(n_perms, 1_000_000_000):
        return PermutationPlan("exact", int(total))
    return PermutationPlan("montecarlo", int(n_perms))


def _make_label_matrix_exact(n: int, n1: int, obs_ones: np.ndarray) -> np.ndarray:
    """(P,n) matrix with observed labeling as row 0."""
    obs_tuple = tuple(obs_ones.tolist())
    combs = list(combinations(range(n), n1))
    if obs_tuple in combs:
        combs.remove(obs_tuple)
    combs = [obs_tuple] + combs

    P = len(combs)
    G = np.zeros((P, n), dtype=np.float32)
    for p, ones_idx in enumerate(combs):
        G[p, list(ones_idx)] = 1.0
    return G


def _make_label_matrix_montecarlo(
    n: int,
    n1: int,
    obs_ones: np.ndarray,
    *,
    n_perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """(P,n) matrix with observed labeling as row 0."""
    if n_perms < 1:
        raise ValueError("n_perms must be >= 1")

    G = np.zeros((n_perms, n), dtype=np.float32)
    G[0, obs_ones] = 1.0

    all_idx = np.arange(n)
    for p in range(1, n_perms):
        ones = rng.choice(all_idx, size=n1, replace=False)
        G[p, ones] = 1.0
    return G


def varweighted_permutation_test(
    curves_by_sample: Dict[str, pd.DataFrame],
    sample_meta: pd.DataFrame,
    *,
    group0,
    group1,
    taus: Sequence[float],
    genes: Optional[Sequence[str]] = None,
    exact: Optional[bool] = None,
    n_perms: int = 1000,
    max_exact: int = 200_000,
    midp: bool = True,
    eps: float = 1e-6,
    device: str | None = None,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Variance-weighted (Welch-style) multivariate permutation DSP test.

    This is the library version of your `run_perm_mean_curve_dsp_gpu_varweighted`.

    Parameters
    ----------
    curves_by_sample:
        Dict sample_id -> DataFrame with index genes and columns taus.
    sample_meta:
        DataFrame indexed by sample_id with at least column 'group'.
    group0, group1:
        Labels defining the two groups (e.g., 'Healthy', 'Lesional').
        group0 is treated as the reference group in the statistic.
    taus:
        Tau grid (used only to define dimensionality and ordering).
    genes:
        Optional list of genes to test. If None, uses intersection across samples.
    exact:
        True for exact permutations, False for Monte Carlo, None for auto.
    n_perms:
        Number of Monte Carlo permutations (ignored for exact).
    max_exact:
        Maximum number of exact permutations allowed.
    midp:
        If True, use mid-p value for discrete permutation p-values.
    eps:
        Floor for standard errors.
    device:
        Torch device: 'cuda' or 'cpu'. If None, defaults to 'cuda'.
    seed:
        RNG seed for Monte Carlo.

    Returns
    -------
    pandas.DataFrame
        Index genes, columns: T_obs, P, q.
    """
    device = choose_device(device)

    if "group" not in sample_meta.columns:
        raise KeyError("sample_meta must include a 'group' column")

    group0_samples = sample_meta.index[sample_meta["group"] == group0].astype(str).tolist()
    group1_samples = sample_meta.index[sample_meta["group"] == group1].astype(str).tolist()

    if len(group0_samples) < 2 or len(group1_samples) < 2:
        raise ValueError("Need at least 2 samples per group for variance-weighted test.")

    all_samples = group0_samples + group1_samples
    n0 = len(group0_samples)
    n1 = len(group1_samples)
    n = n0 + n1

    if genes is None:
        gene_sets = [set(curves_by_sample[s].index) for s in all_samples]
        genes = sorted(set.intersection(*gene_sets))
    else:
        genes = list(genes)

    base = np.array([0] * n0 + [1] * n1, dtype=np.float32)
    obs_ones = np.where(base == 1.0)[0]

    plan = plan_permutations(n, n1, n_perms=n_perms, exact=exact, max_exact=max_exact)
    rng = np.random.default_rng(seed)

    if plan.mode == "exact":
        G = _make_label_matrix_exact(n, n1, obs_ones)
    else:
        G = _make_label_matrix_montecarlo(n, n1, obs_ones, n_perms=plan.n_perms, rng=rng)

    P = G.shape[0]

    taus_list = list(map(float, taus))
    Ttau = len(taus_list)
    sqrtT = float(np.sqrt(Ttau) + 1e-12)

    G_t = torch.from_numpy(G).to(device)
    ones_t = torch.ones((n,), device=device, dtype=torch.float32)
    g_obs_t = torch.from_numpy(base).to(device)

    n1_f = float(n1)
    n0_f = float(n0)
    denom1 = float(max(n1 - 1, 1))
    denom0 = float(max(n0 - 1, 1))

    rows: List[dict] = []

    for gi, gene in enumerate(genes):
        # assemble (n,T)
        curves = []
        ok = True
        for sid in all_samples:
            df = curves_by_sample[sid]
            if gene not in df.index:
                ok = False
                break
            v = df.loc[gene, taus_list].to_numpy(dtype=float, copy=False)
            curves.append(normalize_curve_l2(v))
        if not ok:
            continue

        X = np.vstack(curves).astype(np.float32)
        X_t = torch.from_numpy(X).to(device)
        X2_t = X_t * X_t

        # observed
        S1_obs = (g_obs_t @ X_t)
        S1sq_obs = (g_obs_t @ X2_t)
        S_tot = (ones_t @ X_t)
        S_tot2 = (ones_t @ X2_t)

        S0_obs = S_tot - S1_obs
        S0sq_obs = S_tot2 - S1sq_obs

        mean1_obs = S1_obs / n1_f
        mean0_obs = S0_obs / n0_f

        var1_obs = (S1sq_obs - (S1_obs * S1_obs) / n1_f) / denom1
        var0_obs = (S0sq_obs - (S0_obs * S0_obs) / n0_f) / denom0

        se_obs = torch.sqrt(torch.clamp(var1_obs / n1_f + var0_obs / n0_f, min=eps))
        z_obs = (mean1_obs - mean0_obs) / se_obs
        T_obs = float(torch.linalg.norm(z_obs).item() / sqrtT)

        # null
        S1_null = G_t @ X_t
        S1sq_null = G_t @ X2_t

        S0_null = S_tot.unsqueeze(0) - S1_null
        S0sq_null = S_tot2.unsqueeze(0) - S1sq_null

        mean1_null = S1_null / n1_f
        mean0_null = S0_null / n0_f

        var1_null = (S1sq_null - (S1_null * S1_null) / n1_f) / denom1
        var0_null = (S0sq_null - (S0_null * S0_null) / n0_f) / denom0

        se_null = torch.sqrt(torch.clamp(var1_null / n1_f + var0_null / n0_f, min=eps))
        z_null = (mean1_null - mean0_null) / se_null

        T_null = (torch.linalg.norm(z_null, dim=1) / sqrtT).detach().cpu().numpy()

        if midp:
            gt = float(np.sum(T_null > T_obs))
            eq = float(np.sum(T_null == T_obs))
            pval = (gt + 0.5 * eq) / float(P)
        else:
            pval = (float(np.sum(T_null >= T_obs)) + 1.0) / (float(P) + 1.0)

        rows.append({"Gene": gene, "T_obs": T_obs, "P_Spatial": float(pval), "N_Perm": int(P), "Perm_Mode": plan.mode})

        del X_t, X2_t
        if device == "cuda" and (gi % 200 == 0):
            torch.cuda.empty_cache()

    res = pd.DataFrame(rows).set_index("Gene")
    if len(res) == 0:
        return res

    _, q, _, _ = multipletests(res["P_Spatial"].values, method="fdr_bh")
    res["FDR_Spatial"] = q
    return res
