"""Primary / secondary / tertiary / quaternary result sets and index-map collection."""

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
    global_results: ShellGlobalResults | None = None
    elemental_results: ShellElementalResults | None = None
    nodal_results: Optional[ShellNodalResults] = None
    gaussian_results: Optional[ShellGaussianResults] = None


@dataclass
class ShellSecondaryResultSet:
    global_results: ShellGlobalResults | None = None
    elemental_results: ShellElementalResults | None = None
    nodal_results: Optional[ShellNodalResults] = None
    gaussian_results: Optional[ShellGaussianResults] = None


@dataclass
class ShellHashinFiResult:
    """Per-mode Hashin failure-index breakdown (Phase 6 quaternary tightening).

    Absorbs the standalone ``interlaminar_fi`` array: the transverse-shear /
    delamination mode is theory-independent (works from ``Qx/Qy`` + thickness
    regardless of FSDT vs HSDT origin), so it is exposed here as a second mode
    of the Hashin computer rather than a separate quaternary field. The two
    arrays are kept distinct (not flattened) so optimiser KS-aggregation call
    sites can read each mode explicitly. See PROCESSING_ARCHITECTURE_PROPOSAL.md
    Phase 6.

    ``in_plane`` is the envelope max of the four in-plane modes (governing /
    optimiser scalar). The four mode arrays are stored separately when computed.
    """

    in_plane: np.ndarray | None = None
    """(n_elem, n_ply) envelope max of fibre/matrix tension/compression modes."""
    interlaminar: np.ndarray | None = None
    """(n_elem,) transverse-shear/delamination mode."""
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

    Pure consumer of :class:`ShellTertiaryResultSet`; the home of
    ``compute_quaternary`` outputs. Scope-tightened in Phase 6 to strictly two
    criteria: Hashin FI (``hashin_fi: ShellHashinFiResult``, in-plane +
    interlaminar modes) and Von Mises FI (``von_mises_fi``, isotropic
    constituents only). See PROCESSING_ARCHITECTURE_PROPOSAL.md.
    """

    hashin_fi: ShellHashinFiResult | None = None
    von_mises_fi: np.ndarray | None = None
    """(n_elem, n_ply) or (n_elem,) for isotropic skins — Phase 6 addition."""
    governing: dict[str, object] = field(default_factory=dict)
    needs_review: np.ndarray | None = None
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class ShellTertiaryResultSet:
    """CLT recovery at the material frame (tier 3): ply strains/stresses only.

    Carries the pass-through fields tier 4 needs (``transverse_shear``,
    ``thickness_m``, ``element_index``) so quaternary never reaches back to
    tier 2. Optionally carries recovered interface transverse shears on
    ``ply_results.tau_iface``. Contains no failure-criterion logic.
    """

    ply_results: ShellPlyResults | None = None
    transverse_shear: np.ndarray | None = None
    """(n_elem, 2) Qx, Qy carried forward from secondary for interlaminar FI fallback."""
    thickness_m: np.ndarray | None = None
    """(n_elem,) panel thickness, carried forward for interlaminar peak stress."""
    element_index: np.ndarray | None = None
    """(n_elem,) local element/station identity for envelopes."""
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
