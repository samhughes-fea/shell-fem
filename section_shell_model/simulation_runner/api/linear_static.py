"""Linear-static cascade view over :class:`SectionLinearStaticRunner`.

Preferred orchestration path::

    SectionLinearStaticRunner(panel_topology=..., section_strip_mesh=..., ...).execute().get_results()

:class:`LinearStaticCascadeResult` is a thin view for buckling / recovery callers
that need ``assembly`` / ``f_nodal`` / quaternary without owning the runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BeamResultantVector,
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
    """View over an executed :class:`SectionLinearStaticRunner`.

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
]
