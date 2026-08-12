"""Solve-state DTOs for the linear buckling eigenproblem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass
class PanelPrestressState:
    """Membrane prestress extracted from static secondary resultants for one panel."""

    panel_index: int
    panel_label: str
    N11: float
    N22: float
    N12: float
    is_compressive: bool
    source_diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class BucklingEigenSolveState:
    """Raw eigen solve output before primary packaging.

    ``status`` is ``\"ok\"`` when modes were found, ``\"skipped\"`` for
    non-compressive / missing inputs, or ``\"failed\"`` on solver errors.

    Full-section solves use ``panel_label == "section"`` with ``modeshapes``
    shaped ``(n_modes, n_gdof)`` in the global DOF basis.
    """

    panel_index: int
    panel_label: str
    lambdas: np.ndarray
    modeshapes: np.ndarray
    n_modes_found: int
    is_compressive: bool
    status: str = "ok"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    K: sp.spmatrix | None = None
    Kg: sp.spmatrix | None = None

    @property
    def is_section_solve(self) -> bool:
        return str(self.panel_label) == "section" or bool(
            self.diagnostics.get("eigen_track") == "section"
        )


@dataclass
class SectionBucklingSolveState:
    """Full-section geometric-stiffness eigenproduct before primary packaging."""

    lambdas: np.ndarray
    modeshapes_global: np.ndarray
    n_modes_found: int
    is_compressive: bool
    status: str = "ok"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    K: sp.spmatrix | None = None
    Kg: sp.spmatrix | None = None
    T: sp.spmatrix | None = None

    def as_eigen_state(self) -> BucklingEigenSolveState:
        diag = dict(self.diagnostics)
        diag.setdefault("eigen_track", "section")
        return BucklingEigenSolveState(
            panel_index=0,
            panel_label="section",
            lambdas=np.asarray(self.lambdas, dtype=float),
            modeshapes=np.asarray(self.modeshapes_global, dtype=float),
            n_modes_found=int(self.n_modes_found),
            is_compressive=bool(self.is_compressive),
            status=str(self.status),
            diagnostics=diag,
            K=self.K,
            Kg=self.Kg,
        )
