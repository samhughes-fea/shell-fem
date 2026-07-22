"""Quarantined CCX per-station dispatch adapter.

Moved from ``engine/station_theory_dispatch.py`` (Phase 6 CalculiX quarantine,
mirroring the Phase 8 HSDT quarantine pattern — see
docs/PRE_PROCESSING_ARCHITECTURE_PROPOSAL.md). Not imported by
``station_theory_dispatch.py`` at module load; only reachable via
:func:`register_calculix_ccx_backend` (``calculix_ccx/register.py``).

Periodic CCX station solve has been removed with Channel-A.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from blade_structure_utilities.section_shell_model.pre_processing.dispatch.station_theory_dispatch import (
        StationTheorySolveRequest,
        StationTheorySolveResult,
    )


def ccx_station_solve_from_thin_wall(request: "StationTheorySolveRequest") -> dict[str, Any]:
    """Raise: periodic CCX station path removed with Channel-A."""
    raise NotImplementedError(
        "CCX station dispatch via periodic BC (removed CCX periodic path) has been "
        "removed with Channel-A. Use native FSDT linear-static "
        "(run_section_linear_static_from_resultants) or call solve_global_coupled_ccx "
        "directly for validation."
    )


def dispatch_ccx_station_solve(request: "StationTheorySolveRequest") -> "StationTheorySolveResult":
    from blade_structure_utilities.section_shell_model.pre_processing.dispatch.station_theory_dispatch import (
        StationTheorySolveResult,
    )

    return StationTheorySolveResult(backend_extra=ccx_station_solve_from_thin_wall(request))


__all__ = ["ccx_station_solve_from_thin_wall", "dispatch_ccx_station_solve"]
