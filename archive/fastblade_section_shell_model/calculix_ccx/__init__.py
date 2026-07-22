"""External CalculiX (``ccx``) S4R shell backend.

The CalculiX backend is *not* a native Python element: it writes a CalculiX ``.inp`` deck for the
section strip mesh, shells out to the external ``ccx`` binary on ``PATH``, and parses the ``.dat`` /
``.frd`` output back into ``ShellPanelResultants``. This package is the single home for that
backend (mesh export, deck writers, result parsers, clamped + periodic solve drivers).

Peers in :mod:`..` are the two *native* Python element libraries (``mitc4``, ``hsdt``).
"""

from .mesh import CcxElement, CcxSectionMesh, build_ccx_section_mesh
from .orientation import (
    local_to_global_transform,
    quad_coords_from_strip,
    shell_basis_from_quad,
)
from .periodic import (
    assemble_k6_from_steps,
    run_section_periodic_bc_ccx,
    superpose_stresses,
)
from .solve import solve_global_coupled_ccx

__all__ = [
    "CcxElement",
    "CcxSectionMesh",
    "assemble_k6_from_steps",
    "build_ccx_section_mesh",
    "local_to_global_transform",
    "quad_coords_from_strip",
    "run_section_periodic_bc_ccx",
    "shell_basis_from_quad",
    "solve_global_coupled_ccx",
    "superpose_stresses",
]
