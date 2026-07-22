"""Quarantined CCX per-station dispatch adapter.

Moved from ``engine/station_theory_dispatch.py`` (Phase 6 CalculiX quarantine,
mirroring the Phase 8 HSDT quarantine pattern — see
docs/PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md). Not imported by
``station_theory_dispatch.py`` at module load; only reachable via
:func:`register_calculix_ccx_backend` (``calculix_ccx/register.py``).

Historically this path called the vendor thin-wall section oracle then
``ccx_global_coupled_solve_from_thin_wall``. The station seam now uses the same
Pipeline-B geometry as FSDT and the package periodic CCX unit-cell solve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from blade_precompute.section_shell_model.engine.station_theory_dispatch import (
        StationTheorySolveRequest,
        StationTheorySolveResult,
    )


def ccx_station_solve_from_thin_wall(request: "StationTheorySolveRequest") -> dict[str, Any]:
    """Solve one station with external CalculiX via Pipeline B + periodic BC.

    Name retained for registry/API stability; implementation no longer depends on
    the vendor thin-wall ``run_section`` oracle.
    """
    from blade_precompute.section_shell_model.api.pipeline_b import (
        pipeline_b_section_from_shell_inputs,
    )
    from .periodic import run_section_periodic_bc_ccx

    opts = request.external_backend_options or {}
    work_subdir = str(opts.get("work_subdir", "ccx"))
    exe = str(opts.get("exe", "ccx"))
    ccx_dir = (request.station_dir / work_subdir).resolve()

    pb = pipeline_b_section_from_shell_inputs(
        request.section_view,
        request.shell_inputs,
    )
    bundle = run_section_periodic_bc_ccx(
        pb,
        N=request.N,
        Vy=request.Vy,
        Vz=request.Vz,
        My=request.My,
        Mz=request.Mz,
        T=request.T,
        n_elements_per_panel=request.n_elements_per_panel,
        target_element_length_m=request.target_element_length_m,
        min_elements_per_panel=request.min_elements_per_panel,
        ccx_work_dir=str(ccx_dir),
        ccx_exe=exe,
    )
    ref = bundle.reference_resultants
    return {
        "solver": "periodic_ccx",
        "n_panels": len(bundle.panels),
        "has_reference_resultants": ref is not None,
    }


def dispatch_ccx_station_solve(request: "StationTheorySolveRequest") -> "StationTheorySolveResult":
    from blade_precompute.section_shell_model.engine.station_theory_dispatch import (
        StationTheorySolveResult,
    )

    return StationTheorySolveResult(backend_extra=ccx_station_solve_from_thin_wall(request))


__all__ = ["ccx_station_solve_from_thin_wall", "dispatch_ccx_station_solve"]
