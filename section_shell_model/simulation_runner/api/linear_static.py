"""Linear-static section solve API (thin shims over SectionLinearStaticRunner).

Preferred orchestration path::

    SectionLinearStaticRunner(panel_topology=..., resultants=...).execute().get_results()

Deprecated function entry points below construct the runner and return a
:class:`LinearStaticCascadeResult` view for transitional callers (buckling /
legacy tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BeamResultantVector,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import ShellPanelTopology
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.mesh_topology import NElements
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.section_strip_mesh import (
    MIN_ELEMENTS_PER_PANEL,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.section_assembly import (
    SectionAssemblyResult,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.results.containers.container_hopper import (
    ShellQuaternaryResultSet,
)
from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticResult,
    SectionLinearStaticRunner,
)


@dataclass(frozen=True)
class LinearStaticCascadeResult:
    """Transitional view over an executed :class:`SectionLinearStaticRunner`.

    Prefer ``runner.execute().get_results()`` and ``runner.assembly`` directly.
    """

    runner: SectionLinearStaticRunner
    assembly: SectionAssemblyResult
    solve_result: SectionLinearStaticResult
    quaternary: ShellQuaternaryResultSet
    resultants: BeamResultantVector
    f_nodal: np.ndarray
    used_traction: bool = False

    @classmethod
    def from_executed_runner(cls, runner: SectionLinearStaticRunner) -> LinearStaticCascadeResult:
        """Build a cascade view from an executed station runner."""
        results = runner.get_results()
        quaternary = results.quaternary_results
        if quaternary is None:
            raise RuntimeError("executed runner missing quaternary_results")
        if runner.assembly is None:
            raise RuntimeError("executed runner missing assembly")
        if runner.resultants is None or runner.f_nodal is None:
            raise RuntimeError("executed runner missing resultants / f_nodal")
        return cls(
            runner=runner,
            assembly=runner.assembly,
            solve_result=results,
            quaternary=quaternary,
            resultants=runner.resultants,
            f_nodal=runner.f_nodal,
            used_traction=bool(runner.used_traction),
        )


def run_section_linear_static_from_resultants(
    panel_topology: ShellPanelTopology,
    *,
    resultants: BeamResultantVector | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    about_yz: tuple[float, float] | None = None,
    torsion_about_shear_centre: bool = False,
    shear_centre_E_mean: float = 70e9,
    traction_field: Any | None = None,
    traction_points_yz: np.ndarray | None = None,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = MIN_ELEMENTS_PER_PANEL,
    L_x: float = 1.0,
    interface_constraint_mode: str = "shared",
    bc_mode: str = "contour_support",
    shell_theory: ShellTheory = ShellTheory.FSDT,
    compute_section_deformation_1a: bool = False,
    load_channel: str | None = None,
) -> LinearStaticCascadeResult:
    """Deprecated shim: construct → :meth:`~SectionLinearStaticRunner.execute` → cascade view."""
    # Face-traction overlay is only combined with Channel A (equipollent).
    # Thinwall (default) does not stack matching aero shears.
    channel = load_channel
    if channel is None:
        channel = "equipollent" if traction_field is not None else "thinwall"
    runner = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        resultants=resultants,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        about_yz=about_yz,
        torsion_about_shear_centre=torsion_about_shear_centre,
        shear_centre_E_mean=shear_centre_E_mean,
        traction_field=traction_field,
        traction_points_yz=traction_points_yz,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        L_x=L_x,
        interface_constraint_mode=interface_constraint_mode,
        bc_mode=bc_mode,
        shell_theory=shell_theory,
        compute_section_deformation_1a=compute_section_deformation_1a,
        load_channel=channel,
    ).execute()
    return LinearStaticCascadeResult.from_executed_runner(runner)


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
