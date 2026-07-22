"""Main-track linear-static section solve through the full results cascade.

Production FI path:

``assemble → SectionLinearStaticRunner.solve()`` (always primary→secondary→tertiary→quaternary)

Optional face traction is superimposed on the beam-resultant nodal RHS and solved
under the same ``contour_support`` BCs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from blade_structure_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.section_shell_model.pre_processing.loading.beam_resultants_to_nodal_load import (
    beam_resultants_to_nodal_load,
)
from blade_structure_utilities.section_shell_model.pre_processing.loading.traction_face_load import (
    traction_face_loads,
)
from blade_structure_utilities.section_shell_model.pre_processing.mesh.api.pipeline_b import PipelineBSection
from blade_structure_utilities.section_shell_model.pre_processing.mesh.mesh_topology import NElements
from blade_structure_utilities.section_shell_model.pre_processing.mesh.section_strip_mesh import (
    MIN_ELEMENTS_PER_PANEL,
)
from blade_structure_utilities.section_shell_model.processing.static.operations.connected_assembly_core import (
    build_clamp_supports,
)
from blade_structure_utilities.section_shell_model.processing.static.operations.section_assembly import (
    SectionAssemblyResult,
    assemble_section_from_panels,
)
from blade_structure_utilities.section_shell_model.processing.static.results.containers.container_hopper import (
    ShellQuaternaryResultSet,
)
from blade_structure_utilities.section_shell_model.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticResult,
    SectionLinearStaticRunner,
)


@dataclass(frozen=True)
class LinearStaticCascadeResult:
    """Full primary→quaternary cascade output for one station solve.

    ``resultants`` is the exclusive beam vector. ``f_nodal`` is the
    complete solved RHS (beam ± face traction).
    """

    runner: SectionLinearStaticRunner
    assembly: SectionAssemblyResult
    solve_result: SectionLinearStaticResult
    quaternary: ShellQuaternaryResultSet
    resultants: BeamResultantVector
    f_nodal: np.ndarray
    used_traction: bool = False


def _resolve_traction_overlay(
    *,
    traction_field: Any | None,
    traction_points_yz: np.ndarray | None,
) -> tuple[Any | None, np.ndarray | None, bool]:
    """Validate optional traction kwargs; reject half-specified input."""
    has_field = traction_field is not None
    has_points = traction_points_yz is not None
    if has_field ^ has_points:
        raise ValueError(
            "traction_field and traction_points_yz must be supplied together "
            "(got one without the other)."
        )
    if not has_field:
        return None, None, False
    pts = np.asarray(traction_points_yz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(
            f"traction_points_yz must have shape (n, 2); got {pts.shape!r}"
        )
    return traction_field, pts, True


def run_section_linear_static_from_resultants(
    pb: PipelineBSection,
    *,
    resultants: BeamResultantVector | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    about_yz: tuple[float, float] | None = None,
    traction_field: Any | None = None,
    traction_points_yz: np.ndarray | None = None,
    n_elements_per_panel: NElements = 10,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = MIN_ELEMENTS_PER_PANEL,
    L_x: float = 1.0,
    interface_constraint_mode: str = "shared",
    bc_mode: str = "contour_support",
    shell_theory: ShellTheory = ShellTheory.FSDT,
    compute_section_deformation_1a: bool = False,
) -> LinearStaticCascadeResult:
    """Beam resultants (± face traction) → nodal ``f`` → full results cascade.

    BC policy: ``contour_support`` clamps via :func:`build_clamp_supports`
    (panel contour endpoints + RBM pin). Face traction uses the same free set
    (superimposed on the beam nodal load). Callers must keep beam shears and
    face traction exclusive — when traction is applied, beam shears ``Vy``/``Vz``
    should already be zeroed by the load-field adapter.
    """
    if shell_theory is not ShellTheory.FSDT:
        raise ValueError(
            "run_section_linear_static_from_resultants currently supports "
            f"ShellTheory.FSDT only; got {shell_theory!r}"
        )
    tf, pts, used_traction = _resolve_traction_overlay(
        traction_field=traction_field,
        traction_points_yz=traction_points_yz,
    )

    panels = list(pb.panels)
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    nxy = [np.zeros_like(a) for a in nx]

    asm = assemble_section_from_panels(
        panels,
        nx,
        nxy,
        L_x=L_x,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
        shell_theory=shell_theory,
    )
    ndof = int(asm.cache.n_dof_node)
    fixed = build_clamp_supports(asm.panel_maps, ndof=ndof, bc_mode=bc_mode)
    constraints = dict(asm.node_meta.constraints or {})

    if resultants is None:
        origin = about_yz if about_yz is not None else BEAM_LOAD_AXIS_YZ
        resultants = BeamResultantVector.from_named(
            N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
            frame=FRAME_SECTION_YZ,
            about_yz=origin,
        )
    elif about_yz is not None:
        resultants = resultants.transported_to(float(about_yz[0]), float(about_yz[1]))

    f = beam_resultants_to_nodal_load(
        asm.node_meta,
        resultants,
        about_yz=resultants.about_yz,
        ndof=ndof,
        L_x=L_x,
    )
    if used_traction:
        forces = np.asarray(tf.total_forces, dtype=float)
        f = f + traction_face_loads(
            asm.node_meta,
            forces,
            pts,
            ndof=ndof,
            L_x=L_x,
        )

    provenance_tag = (
        "simulation_runner.api.linear_static + face traction"
        if used_traction
        else "simulation_runner.api.linear_static"
    )

    runner = SectionLinearStaticRunner.from_cache(
        asm.cache, asm.n_gdof, constraints, fixed,
    )
    solve_result = runner.solve(
        f_override=f,
        L_x=L_x,
        theory="fsdt",
        panel_maps=asm.panel_maps,
        panels=panels,
        provenance_tag=provenance_tag,
    )
    if compute_section_deformation_1a:
        runner.compute_section_deformation_average_1a(
            node_meta=asm.node_meta,
            about_yz=resultants.about_yz,
            panels=panels,
            panel_maps=asm.panel_maps,
        )

    quaternary = solve_result.quaternary_results
    if quaternary is None:
        raise RuntimeError("solve() did not populate quaternary_results")

    return LinearStaticCascadeResult(
        runner=runner,
        assembly=asm,
        solve_result=solve_result,
        quaternary=quaternary,
        resultants=resultants,
        f_nodal=f,
        used_traction=used_traction,
    )


def governing_fi_from_quaternary(quaternary: ShellQuaternaryResultSet) -> tuple[float, bool]:
    """Return ``(governing_fi, needs_review_any)`` from a tier-4 result set."""
    gov = quaternary.governing or {}
    fi = float(gov.get("fi", 0.0) or 0.0)
    nr = quaternary.needs_review
    needs = bool(np.any(nr)) if nr is not None else False
    return fi, needs


__all__ = [
    "LinearStaticCascadeResult",
    "governing_fi_from_quaternary",
    "run_section_linear_static_from_resultants",
]
