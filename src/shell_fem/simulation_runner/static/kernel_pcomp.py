"""Native static solve of a kernel shell mesh from one PCOMP per element.

The mesh and the assignment are built by ``KernelMeshBladeBuilder`` in
``pre_processing``. This runner only sequences that build and the static solve.
"""

from __future__ import annotations

from typing import Sequence

from laminate_constitutive_model.laminate_kinematics import KinematicModel

from shell_fem.pre_processing.mesh_library.blade_kernel_mesh import PcompAssignment


def solve_kernel_pcomp(shell_mesh, assignments: Sequence[PcompAssignment], boundary_conditions,
                       load_cases, kinematics: KinematicModel | None = None, material_axes=None,
                       **element_options):
    """Solve the kernel shell mesh with the given PCOMP assignment.

    Returns ``(blade, results)``. ``results`` is one :class:`~shell_fem.KernelResult`
    per load case, from the native static solver. One factorisation serves every case.
    """
    from laminate_constitutive_model.laminate_kinematics import FSDTModel

    from shell_fem.pre_processing.load_library.load_case import LoadCase
    from shell_fem.pre_processing.mesh_library.blade_kernel_mesh import KernelMeshBladeBuilder
    from shell_fem.simulation_runner.backends.base import KernelResult, KernelStatus
    from shell_fem.simulation_runner.static.linear_static_simulation import StructuralGlobalSolver

    if isinstance(load_cases, LoadCase):
        cases = [load_cases]
    else:
        cases = list(load_cases)
    if not cases:
        raise ValueError("solve_kernel_pcomp needs at least one load case")
    blade = KernelMeshBladeBuilder(shell_mesh).build_pcomp(
        assignments, FSDTModel() if kinematics is None else kinematics,
        material_axes=material_axes, **element_options)
    solutions = StructuralGlobalSolver().solve_many(blade.mesh, boundary_conditions, cases)
    results = [
        KernelResult("native", KernelStatus.SUCCESS, displacements=sol.displacement_map(),
                     info={"solve_time": sol.solve_time, **sol.info})
        for sol in solutions
    ]
    return blade, results


__all__ = ["solve_kernel_pcomp"]
