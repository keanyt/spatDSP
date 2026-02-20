"""Result containers for spatDSP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class DSPResult:
    """Results returned by :func:`spatDSP.run_dsp`.

    Attributes
    ----------
    table:
        Per-gene test table with columns ['T_obs','P_Spatial','FDR_Spatial',...].
    curves_by_sample:
        Dict sample_id -> curves DataFrame (index genes, columns taus).
    params:
        Dictionary of run parameters for reproducibility.
    """

    table: pd.DataFrame
    curves_by_sample: Dict[str, pd.DataFrame]
    params: dict

    def top(self, n: int = 20, by: str = "T_obs") -> pd.DataFrame:
        if by not in self.table.columns:
            raise KeyError(f"'{by}' not in result table")
        return self.table.sort_values(by, ascending=False).head(n)
