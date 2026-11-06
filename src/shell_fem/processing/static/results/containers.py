"""Result containers of a static solve.

``SolutionResult`` is the primary result of one load case: global displacements,
reactions and the load vector. The field dataclasses are what ``ResultsExtractor``
returns at each fidelity level (Gauss point, ply point, element, solid point, node,
global totals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from shell_fem.pre_processing.containers.mesh import FEAMesh
from shell_fem.pre_processing.load_library.load_case import LoadCase

PLY_POSITIONS = ("bottom", "mid", "top")


@dataclass
class SolutionResult:
    mesh: FEAMesh
    displacements: np.ndarray
    reactions: np.ndarray
    load_vector: np.ndarray
    solve_time: float
    load_case: Optional[LoadCase] = None
    info: Dict[str, object] = field(default_factory=dict)

    def node_displacement(self, node_id: int) -> np.ndarray:
        """[u, v, w, theta_x, theta_y, theta_z] of a node (global axes)."""
        base = self.mesh.dof_index(node_id, 0)
        return self.displacements[base : base + 6].copy()

    def node_enrichment(self, node_id: int) -> np.ndarray:
        """Enrichment amplitudes of a node, [a_1x, a_1y, a_2x, ...] (element local axes)."""
        base = self.mesh.dof_index(node_id, 0)
        return self.displacements[base + 6 : base + self.mesh.node_dof_count(node_id)].copy()

    def node_reaction(self, node_id: int) -> np.ndarray:
        base = self.mesh.dof_index(node_id, 0)
        return self.reactions[base : base + 6].copy()

    def displacement_array(self) -> np.ndarray:
        """(n_nodes, 6) displacements in mesh node order."""
        return self.mesh.standard_array(self.displacements).copy()

    def reaction_array(self) -> np.ndarray:
        return self.mesh.standard_array(self.reactions).copy()

    def load_array(self) -> np.ndarray:
        return self.mesh.standard_array(self.load_vector).copy()

    def displacement_map(self) -> Dict[int, np.ndarray]:
        return {n: self.node_displacement(n) for n in self.mesh.node_ids}

    def element_displacements(self, element_id: int) -> np.ndarray:
        return self.displacements[self.mesh.element_dofs(element_id)].copy()

    def max_abs(self, dof: int) -> float:
        return float(np.max(np.abs(self.displacement_array()[:, int(dof)])))

    @property
    def external_work(self) -> float:
        return 0.5 * float(self.displacements @ self.load_vector)



# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GaussPointField:
    """Section-level results at every integration point (one row per point)."""

    element_ids: np.ndarray
    point_index: np.ndarray
    natural: np.ndarray  # (n, 2) xi, eta
    xyz: np.ndarray  # (n, 3)
    weight: np.ndarray  # quadrature weight x det J (area share)
    eps0: np.ndarray  # (n, 3) membrane strains
    kappa: np.ndarray  # (n, 3) curvatures
    gamma0: np.ndarray  # (n, 2) transverse shear strains [yz, xz]
    N: np.ndarray  # (n, 3) force resultants
    M: np.ndarray  # (n, 3) moment resultants
    Q: np.ndarray  # (n, 2) shear force resultants [Q_yz, Q_xz]


@dataclass(frozen=True)
class PlyPointField:
    """Ply stresses at Gauss points (one row per point and sampled height)."""

    element_ids: np.ndarray
    point_index: np.ndarray
    ply_index: np.ndarray
    z: np.ndarray
    xyz: np.ndarray
    stress_global: np.ndarray  # (m, 3) [sx, sy, txy]
    stress_material: np.ndarray  # (m, 3) [s11, s22, t12]
    shear_global: np.ndarray  # (m, 2) [t_yz, t_xz]
    shear_material: np.ndarray  # (m, 2) [t23, t13]
    failure_index: np.ndarray  # (m,) NaN when no criterion was given
    failure_mode: np.ndarray  # (m,) mode names ('' when no criterion)


@dataclass(frozen=True)
class ElementField:
    element_ids: np.ndarray
    centroid: np.ndarray
    area: np.ndarray
    N: np.ndarray
    M: np.ndarray
    Q: np.ndarray
    energy_membrane: np.ndarray
    energy_bending: np.ndarray
    energy_coupling: np.ndarray
    energy_enrichment: np.ndarray
    energy_shear: np.ndarray
    energy_drilling: np.ndarray
    energy_total: np.ndarray
    max_failure_index: np.ndarray  # NaN when no criterion was given
    critical_ply: np.ndarray  # -1 when no criterion was given


@dataclass(frozen=True)
class SolidPointField:
    """Ply stresses in layered solids (one row per point; Voigt order [11, 22, 33, 23, 13, 12])."""

    element_ids: np.ndarray
    ply_index: np.ndarray
    natural: np.ndarray  # (n, 3) xi, eta, zeta
    xyz: np.ndarray
    stress_global: np.ndarray  # (n, 6) [sx, sy, sz, tyz, txz, txy]
    stress_material: np.ndarray  # (n, 6) [s1, s2, s3, t23, t13, t12]
    failure_index: np.ndarray
    failure_mode: np.ndarray


@dataclass(frozen=True)
class NodalField:
    node_ids: np.ndarray
    xyz: np.ndarray
    displacement: np.ndarray  # (n, 6)
    reaction: np.ndarray  # (n, 6)
    N: np.ndarray  # (n, 3) extrapolated and averaged
    M: np.ndarray
    Q: np.ndarray
    element_count: np.ndarray  # elements sharing the node


@dataclass(frozen=True)
class GlobalTotals:
    strain_energy: float
    energy_split: Dict[str, float]
    external_work: float
    applied_resultant: np.ndarray  # [Fx, Fy, Fz, Mx, My, Mz] about the origin
    reaction_resultant: np.ndarray
    equilibrium_residual: float  # |applied + reactions| / sum|nodal forces| (moments over model size)
    max_displacement: Dict[str, Tuple[float, int]]  # dof -> (signed value, node)
    max_failure: Optional[Tuple[float, int, int, str]]  # (index, element, ply, mode)
