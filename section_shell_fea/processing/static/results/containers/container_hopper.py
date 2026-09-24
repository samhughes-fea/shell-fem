"""Primary / secondary / tertiary / quaternary result sets and index-map collection.

Frame contract (``geometry_utilities.coordinate_frames``):

* Primary / secondary kinematics & resultants → **E** (element = laminate),
  on a station mesh authored in **B**.
* Tertiary face stresses / strains → **M** (ply material); mid-surface
  ``eps0``/``kappa`` and ``tau_iface`` remain **E**.
* Quaternary FI → consumes **M** face stresses (Hashin / Von Mises / extreme-fibre).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .elemental_results import ShellElementalResults
from .gaussian_results import ShellGaussianResults
from .global_results import ShellGlobalResults
from .map_results import ShellMapEntry
from .nodal_results import ShellNodalResults
from .ply_results import ShellPlyResults


@dataclass
class ShellPrimaryResultSet:
    """Tier 1: assembled solve + per-element DOF disassembly (B mesh, E DOFs)."""

    global_results: ShellGlobalResults | None = None
    elemental_results: ShellElementalResults | None = None
    nodal_results: Optional[ShellNodalResults] = None
    gaussian_results: Optional[ShellGaussianResults] = None


@dataclass
class ShellSecondaryResultSet:
    """Tier 2: shell resultants in **E** (``N11…`` = \\(N_{11}\\ldots\\))."""

    global_results: ShellGlobalResults | None = None
    elemental_results: ShellElementalResults | None = None
    nodal_results: Optional[ShellNodalResults] = None
    gaussian_results: Optional[ShellGaussianResults] = None


@dataclass
class ShellHashinFiResult:
    """Per-mode Hashin failure-index breakdown (Phase 6 quaternary tightening).

    In-plane modes require **M**-frame mid-face stress. ``interface_shear`` is
    a separate FSDT screening criterion, not a Hashin mode; ``interlaminar`` is
    retained as its compatibility alias.

    ``in_plane`` is the envelope max of the four in-plane modes (governing /
    optimiser scalar). The four mode arrays are stored separately when computed.
    """

    in_plane: np.ndarray | None = None
    """(n_elem, n_ply) envelope max of fibre/matrix tension/compression modes."""
    interlaminar: np.ndarray | None = None
    """Deprecated compatibility alias of ``interface_shear``."""
    interface_shear: np.ndarray | None = None
    """(n_elem,) two-sided material-frame interface-shear criterion."""
    interface_shear_recovery_mode: np.ndarray | None = None
    interface_shear_strength_provenance: np.ndarray | None = None
    interface_shear_governing_interface: np.ndarray | None = None
    interface_shear_governing_side: np.ndarray | None = None
    interface_shear_component_ratios: np.ndarray | None = None
    fi_ft: np.ndarray | None = None
    """(n_elem, n_ply) fibre-tension Hashin mode."""
    fi_fc: np.ndarray | None = None
    """(n_elem, n_ply) fibre-compression Hashin mode."""
    fi_mt: np.ndarray | None = None
    """(n_elem, n_ply) matrix-tension Hashin mode."""
    fi_mc: np.ndarray | None = None
    """(n_elem, n_ply) matrix-compression Hashin mode."""


@dataclass
class ShellQuaternaryResultSet:
    """Failure indices, governing element, and review flags (tier 4).

    Pure consumer of :class:`ShellTertiaryResultSet`. Criteria act on **M**
    stresses: Hashin (composite outer/inner faces), Von Mises (isotropic mid-face),
    and material-agnostic extreme-fibre (opt-in) at laminate free surfaces.
    Interface shear is diagnostic-only and is not a governing criterion.
    """

    hashin_fi: ShellHashinFiResult | None = None
    von_mises_fi: np.ndarray | None = None
    """(n_elem,) Von Mises FI = σ_vm / σ_y (isotropic mid-face, **M**≡**E**)."""
    sig_extreme_fibre: np.ndarray | None = None
    """(n_elem, 2, 3) free-surface plane stress [outer, inner] in **M**
    ``[σ11, σ22, τ12]``."""
    extreme_fibre_fi: np.ndarray | None = None
    """(n_elem,) extreme-fibre envelope FI vs σ_uts / σ_ucs / τ_us."""
    governing: dict[str, object] = field(default_factory=dict)
    needs_review: np.ndarray | None = None
    meta: dict[str, object] = field(default_factory=dict)
    max_utilization_governing: float | None = None
    formulation_id: str = "MITC4-FSDT"
    canonical_layout_version: str = "v1"


@dataclass
class ShellTertiaryResultSet:
    """Tier 3 CLT recovery: **M** face stresses + **E** mid-surface / shear pass-through.

    Carries the fields tier 4 needs (``transverse_shear``, ``thickness_m``,
    ``element_index``) so quaternary never reaches back to tier 2. Optionally
    carries recovered interface transverse shears on ``ply_results.tau_iface``
    (**E**). Contains no failure-criterion logic.
    """

    ply_results: ShellPlyResults | None = None
    transverse_shear: np.ndarray | None = None
    """(n_elem, 2) **E**-frame \\(Q_1,Q_2\\) (secondary ``Q1``, ``Q2``) for
    interlaminar FI fallback."""
    thickness_m: np.ndarray | None = None
    """(n_elem,) panel thickness, carried forward for interlaminar peak stress."""
    element_index: np.ndarray | None = None
    """(n_elem,) element / station identity for envelopes."""
    panel_index: np.ndarray | None = None
    """(n_elem,) panel identity, carried forward so tier 4 can resolve per-row
    strengths from ``cache.panel_laminates[panel_index]`` without re-deriving the
    row -> panel grouping tier 3 already computed."""
    meta: dict[str, object] = field(default_factory=dict)
    """Includes ``shear_recovery_mode`` when FSDT interface shear recovery ran."""


@dataclass
class ShellIndexMapSet:
    assembly_map: list[ShellMapEntry] | None = None
    constraint_map: list[ShellMapEntry] | None = None
    reduction_map: list[ShellMapEntry] | None = None
    reconstruction_map: list[ShellMapEntry] | None = None


__all__ = [
    "ShellHashinFiResult",
    "ShellIndexMapSet",
    "ShellPrimaryResultSet",
    "ShellQuaternaryResultSet",
    "ShellSecondaryResultSet",
    "ShellTertiaryResultSet",
]
