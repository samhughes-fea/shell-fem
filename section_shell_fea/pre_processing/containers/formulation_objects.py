"""Agnostic formulation-cache DTOs (mirror 1d-fem gauss_point_data / FormulationResultSet)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from blade_structure_utilities.material_utilities.laminate import LaminateDefinition

PanelConstitutiveKind = Literal["isotropic", "orthotropic"]


@dataclass(frozen=True)
class ShellStiffnessGaussPointData:
    xi: float
    eta: float
    weight: float
    B_matrix: NDArray[np.float64]
    D_matrix: NDArray[np.float64]
    jacobian: float
    shape_functions: NDArray[np.float64] | None = None
    shape_derivatives: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ShellForceGaussPointData:
    xi: float
    eta: float
    weight: float
    shape_functions: NDArray[np.float64]
    membrane_nx: float = 0.0
    membrane_nxy: float = 0.0
    donnell_qn: float = 0.0


@dataclass
class ShellStiffnessObject:
    element_id: int
    element_type: str
    K_e: NDArray[np.float64]
    gauss_data: list[ShellStiffnessGaussPointData] = field(default_factory=list)
    global_dof_indices: NDArray[np.int64] | None = None
    panel_index: int = -1
    local_element_index: int = -1


@dataclass
class ShellForceObject:
    element_id: int
    element_type: str
    F_e: NDArray[np.float64]
    gauss_data: list[ShellForceGaussPointData] = field(default_factory=list)
    global_dof_indices: NDArray[np.int64] | None = None
    panel_index: int = -1
    local_element_index: int = -1


@dataclass
class ShellFormulationCache:
    stiffness_objects: list[ShellStiffnessObject]
    force_objects: list[ShellForceObject]
    n_dof_node: int = 6
    panel_load_totals: list[dict[str, float]] = field(default_factory=list)
    panel_abd_stacks: list[tuple[NDArray[np.float64], float, float, float, float]] = field(
        default_factory=list
    )
    # Full per-panel laminate objects (plies, Q_bar, ply depths, strengths), indexed by
    # panel_index. Retained post-solve alongside panel_abd_stacks so tiers 3/4 (CLT ply
    # recovery, failure indices) resolve laminates from the cache instead of taking a
    # laminate side argument. See PROCESSING_ARCHITECTURE_PROPOSAL.md.
    panel_laminates: list["LaminateDefinition"] = field(default_factory=list)
    # Per-panel constitutive kind and textbook strengths (TAXONOMY.md). Length matches
    # panel_laminates. Orthotropic panels use Hashin; isotropic use Von Mises with
    # panel_sigma_y_pa[pi] (yield). Extreme-fibre uses panel_sigma_uts/ucs/tau_us (ultimates).
    panel_constitutive_kind: list[PanelConstitutiveKind] = field(default_factory=list)
    panel_sigma_y_pa: list[float] = field(default_factory=list)
    panel_sigma_uts_pa: list[float] = field(default_factory=list)
    panel_sigma_ucs_pa: list[float] = field(default_factory=list)
    panel_tau_us_pa: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class AssemblyDiagnostics:
    n_global_dof: int
    k_norm: float
    f_norm: float
    nnz: int
    symmetry_max_abs_diff: float


__all__ = [
    "AssemblyDiagnostics",
    "PanelConstitutiveKind",
    "ShellForceGaussPointData",
    "ShellForceObject",
    "ShellFormulationCache",
    "ShellStiffnessGaussPointData",
    "ShellStiffnessObject",
]
