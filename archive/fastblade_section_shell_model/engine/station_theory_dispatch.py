"""Per-station shell theory dispatch for the strip-mesh orchestration path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blade_precompute.section_shell_model.pre_processing.containers.types import (
    ShellTheory,
    require_shell_theory_supports,
)


@dataclass(frozen=True)
class StationTheorySolveRequest:
    """Inputs shared by FSDT / CCX station solves."""

    theory: ShellTheory
    section_view: Any
    shell_inputs: Any
    airfoil: np.ndarray
    spars_m: list[float]
    station_dir: Path
    station_tag: str
    N: float
    Vy: float
    Vz: float
    My: float
    Mz: float
    T: float
    n_elements_per_panel: int | None
    target_element_length_m: float
    min_elements_per_panel: int
    load_field: Any | None = None
    external_backend_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class StationTheorySolveResult:
    backend_extra: dict[str, Any] | None = None
    fsdt_periodic_extra: dict[str, Any] | None = None


def fsdt_periodic_station_hashin(
    section_view: Any,
    shell_inputs: Any,
    *,
    N: float,
    Vy: float,
    Vz: float,
    My: float,
    Mz: float,
    T: float,
    n_elements_per_panel: int | None,
    target_element_length_m: float | None,
    min_elements_per_panel: int,
    load_field: Any | None = None,
) -> dict[str, Any]:
    """Periodic unit-cell FSDT solve + Hashin FI summary for orchestration JSON."""
    from blade_precompute.section_shell_model.api.fi_recovery import select_governing_hashin_element
    from blade_precompute.section_shell_model.api.station_recovery import run_station_periodic_hashin

    from blade_precompute.section_shell_model.api.pipeline_b import pipeline_b_section_from_shell_inputs
    from blade_precompute.section_shell_model.interface.traction_load_adapter import (
        traction_points_from_load_field,
    )

    pb = pipeline_b_section_from_shell_inputs(section_view, shell_inputs)
    traction_field = None
    traction_points = None
    used_traction = False
    if load_field is not None:
        traction_points = traction_points_from_load_field(load_field)
        if traction_points is not None:
            traction_field = load_field.traction_field
            used_traction = True

    out = run_station_periodic_hashin(
        pb,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        traction_field=traction_field,
        traction_points_yz=traction_points,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
    )
    gov = select_governing_hashin_element(out["bundle"])
    return {
        "solver": "fsdt_periodic_unit_cell",
        "used_traction": bool(used_traction),
        "max_fi_hashin": float(out["max_fi_hashin"]),
        "needs_review": bool(out["needs_review"]),
        "fi_by_panel": [float(x) for x in out["fi_by_panel"]],
        "provenance_breakdown": dict(out["provenance_breakdown"]),
        "governing_element": gov,
    }


def _dispatch_fsdt(request: StationTheorySolveRequest) -> StationTheorySolveResult:
    return StationTheorySolveResult(
        fsdt_periodic_extra=fsdt_periodic_station_hashin(
            request.section_view,
            request.shell_inputs,
            N=request.N,
            Vy=request.Vy,
            Vz=request.Vz,
            My=request.My,
            Mz=request.Mz,
            T=request.T,
            n_elements_per_panel=request.n_elements_per_panel,
            target_element_length_m=request.target_element_length_m,
            min_elements_per_panel=request.min_elements_per_panel,
            load_field=request.load_field,
        )
    )


_StationSolverFn = Callable[["StationTheorySolveRequest"], "StationTheorySolveResult"]

from .shell_theory_registry import (
    get_station_solver as _get_station_solver,
    register_station_solver as _register_station_solver_unified,
    station_solver_map as _station_solver_map,
)

_register_station_solver_unified(ShellTheory.FSDT, _dispatch_fsdt)


class _StationSolverMap(dict):
    def __getitem__(self, key: ShellTheory) -> _StationSolverFn:
        try:
            return _get_station_solver(key)
        except KeyError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key: object) -> bool:
        return key in _station_solver_map()

    def __setitem__(self, key: ShellTheory, value: _StationSolverFn) -> None:
        _register_station_solver_unified(key, value)

    def keys(self):
        return _station_solver_map().keys()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


_STATION_SOLVERS: dict[ShellTheory, _StationSolverFn] = _StationSolverMap()


def register_station_solver(theory: ShellTheory, solve_fn: _StationSolverFn) -> None:
    """Register ``solve_fn`` as the per-station dispatch handler for ``theory``."""
    _register_station_solver_unified(theory, solve_fn)


def dispatch_station_theory(request: StationTheorySolveRequest) -> StationTheorySolveResult:
    """Dispatch one station solve to the registered shell-theory handler."""
    require_shell_theory_supports(request.theory, feature="station_dispatch")
    try:
        solver = _get_station_solver(request.theory)
    except KeyError:
        raise ValueError(
            f"No station-dispatch handler registered for {request.theory!r}; "
            "opt-in backends (e.g. CalculiX ccx) require calling their "
            "register_*_backend() hook first."
        ) from None
    return solver(request)


__all__ = [
    "StationTheorySolveRequest",
    "StationTheorySolveResult",
    "dispatch_station_theory",
    "fsdt_periodic_station_hashin",
    "register_station_solver",
]
