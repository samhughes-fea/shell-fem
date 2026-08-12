"""Assembled-system primary results (K, f, d, reactions) for the station shell mesh.

``d_global`` / ``f`` / ``K`` are **assembled FE system** arrays on the station
shell model (mesh geometry in **B**). They are not blade XYZ components and not
frame **R**. Prefer “assembled” over bare “global” when speaking about frames —
see ``geometry_utilities.coordinate_frames`` and shell ``TAXONOMY.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class ShellGlobalResults:
    """Assembled FE system payloads for one station shell solve."""

    K: sp.spmatrix | None = None
    f: np.ndarray | None = None
    T: sp.csr_matrix | None = None
    d_global: np.ndarray | None = None
    """Assembled DOF solution vector (station mesh); name is FE-system, not blade frame."""
    r_full: np.ndarray | None = None
    reactions: np.ndarray | None = None
    residual: np.ndarray | None = None
    K_rr_factor: spla.SuperLU | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


__all__ = ["ShellGlobalResults"]
