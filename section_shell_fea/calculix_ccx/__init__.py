"""External CalculiX (``ccx``) S4R shell backend.

The CalculiX backend is *not* a native Python element: it writes a CalculiX ``.inp`` deck for the
section strip mesh, shells out to the external ``ccx`` binary on ``PATH``, and parses the ``.dat`` /
``.frd`` output back into ``ShellPanelResultants``. This package is the single home for that
backend (mesh export, deck writers, result parsers, solve drivers).

Peers in :mod:`..` are the two *native* Python element libraries (``mitc4``, ``hsdt``).
"""

from .mesh import CcxElement, CcxSectionMesh, build_ccx_section_mesh
from .orientation import (
    local_to_global_transform,
    quad_coords_from_strip,
    shell_basis_from_quad,
)
from .solve import solve_global_coupled_ccx
from .cap_skin_ties import CcxCapSkinTieSurface, build_cap_skin_tie_surfaces
from .joint_validation import run_joint_validation
from .ccx_cascade_mesh_convergence import run_ccx_cascade_at_frac
from .sfv_panel_loads import sfv_to_n11_n12, sfv_to_nx_nxy

__all__ = [
    "CcxElement",
    "CcxSectionMesh",
    "CcxCapSkinTieSurface",
    "build_ccx_section_mesh",
    "build_cap_skin_tie_surfaces",
    "local_to_global_transform",
    "quad_coords_from_strip",
    "run_ccx_cascade_at_frac",
    "shell_basis_from_quad",
    "run_joint_validation",
    "sfv_to_n11_n12",
    "sfv_to_nx_nxy",
    "solve_global_coupled_ccx",
]
