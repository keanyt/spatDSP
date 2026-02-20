"""Utility helpers for spatDSP.

This package is focused on Differential Spatial Patterning (DSP) using
heat-diffusion signatures approximated by Chebyshev polynomials.

Public functions here are intentionally small and dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RNGWrapper:
    """Wrapper to provide a consistent NumPy Generator interface."""

    seed: Optional[int] = None

    def generator(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


def as_1d_float_array(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D sequence.")
    return arr


def normalize_curve_l2(c: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise a non-negative curve, keeping shape only."""
    c = np.asarray(c, dtype=float)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.maximum(c, 0.0)
    return c / (np.linalg.norm(c) + eps)


def choose_device(device: str | None = None) -> str:
    """Pick torch device string.

    Parameters
    ----------
    device:
        "cuda", "cpu", or None.

    Returns
    -------
    str
        Device string.
    """
    if device is None:
        return "cuda"
    device = str(device).lower()
    if device not in {"cuda", "cpu"}:
        raise ValueError("device must be 'cuda', 'cpu', or None")
    return device
