"""spatDSP: Differential Spatial Patterning with heat diffusion curves.

Main entry point:
    - :func:`spatDSP.run_dsp`

The library expects a single AnnData containing all cells, with:
    - ``adata.obsm['spatial']`` (or user-provided key) containing coordinates
    - ``adata.obs[sample_key]`` identifying samples (donor/section)
    - ``adata.obs[group_key]`` identifying groups to compare

DSP is performed at the sample level by:
    1) Building a kNN graph per sample
    2) Computing diffusion curves per gene across a tau grid
    3) Running a variance-weighted permutation test across samples
"""

from .dsp import run_dsp
from .results import DSPResult

__all__ = ["run_dsp", "DSPResult"]
