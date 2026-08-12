"""Results containers stratified by tier and resolution (mirror 1d-fem container_hopper).

Coordinate frames follow
:mod:`blade_structure_utilities.geometry_utilities.coordinate_frames`
(see that package README). Avoid bare “local/global” for structural axes.

| Symbol | Frame | What lives here in these containers |
|--------|-------|-------------------------------------|
| **B** | Blade (twisted section mesh) | Station shell mesh geometry |
| **C** | Curvilinear strip (≡ **E** on each facet) | MITC4 nodal DOFs in ``U_e`` / ``d_global`` |
| **E** | Element = laminate \\((x_1,x_2,x_3)\\) | ``N11…M12`` / ``Q1,Q2`` (= \\(N_{11}\\ldots\\), \\(Q_1,Q_2\\)); ``eps0``, ``kappa``; ``tau_iface`` |
| **M** | Ply material (E rotated by ply θ) | ``eps_ply_mat``, ``sig_ply_mat``; quaternary FI inputs |

``d_global`` / ``global_dof`` mean **assembled FE system** indices on the station
mesh — not blade XYZ and not frame **R**.
"""

from __future__ import annotations

from .container_hopper import (
    ShellHashinFiResult,
    ShellIndexMapSet,
    ShellPrimaryResultSet,
    ShellQuaternaryResultSet,
    ShellSecondaryResultSet,
    ShellTertiaryResultSet,
)
from .diagnostics_bin import DiagnosticsBin
from .elemental_results import ShellElementalResults
from .gaussian_results import ShellGaussianResults
from .global_results import ShellGlobalResults
from .map_results import ShellMapEntry
from .nodal_results import ShellNodalResults
from .ply_results import ShellPlyResults

__all__ = [
    "DiagnosticsBin",
    "ShellElementalResults",
    "ShellGaussianResults",
    "ShellGlobalResults",
    "ShellHashinFiResult",
    "ShellIndexMapSet",
    "ShellMapEntry",
    "ShellNodalResults",
    "ShellPlyResults",
    "ShellPrimaryResultSet",
    "ShellQuaternaryResultSet",
    "ShellSecondaryResultSet",
    "ShellTertiaryResultSet",
]
