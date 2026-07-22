"""Global-resolution primary results (system-level K, f, d, reactions)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass
class ShellGlobalResults:
    K: sp.spmatrix | None = None
    f: np.ndarray | None = None
    T: sp.csr_matrix | None = None
    d_global: np.ndarray | None = None
    r_full: np.ndarray | None = None
    reactions: np.ndarray | None = None
    residual: np.ndarray | None = None
    K_rr_factor: spla.SuperLU | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


__all__ = ["ShellGlobalResults"]
