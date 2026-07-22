"""Unified shell-theory registration seam (R4).

Owns the dispatch maps that previously lived separately:

* station dispatch handlers (``engine/station_theory_dispatch.py``)
* global-coupled backends (``engine/shell_solve_registry.py``)

Capability flags still come from
:data:`pre_processing.containers.types.SHELL_THEORY_CAPABILITIES`; this module
only stores the *callables* once a theory opts in (or for the unconditional
FSDT default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from blade_structure_utilities.section_shell_model.pre_processing.containers.types import ShellTheory

StationSolverFn = Callable[..., Any]
GlobalSolveFn = Callable[..., tuple[list[list[Any]], list[dict]]]


@dataclass
class ShellTheoryRegistration:
    """Callables registered for one :class:`ShellTheory`."""

    theory: ShellTheory
    station_solver: StationSolverFn | None = None
    global_solve: GlobalSolveFn | None = None
    global_label: str | None = None


_REGISTRATIONS: dict[ShellTheory, ShellTheoryRegistration] = {}


def _entry(theory: ShellTheory) -> ShellTheoryRegistration:
    reg = _REGISTRATIONS.get(theory)
    if reg is None:
        reg = ShellTheoryRegistration(theory=theory)
        _REGISTRATIONS[theory] = reg
    return reg


def register_station_solver(theory: ShellTheory, solve_fn: StationSolverFn) -> None:
    _entry(theory).station_solver = solve_fn


def register_global_backend(
    theory: ShellTheory,
    solve_fn: GlobalSolveFn,
    *,
    label: str | None = None,
) -> None:
    e = _entry(theory)
    e.global_solve = solve_fn
    e.global_label = label


def get_station_solver(theory: ShellTheory) -> StationSolverFn:
    reg = _REGISTRATIONS.get(theory)
    if reg is None or reg.station_solver is None:
        raise KeyError(theory)
    return reg.station_solver


def get_global_backend(theory: ShellTheory) -> tuple[GlobalSolveFn, str | None]:
    reg = _REGISTRATIONS.get(theory)
    if reg is None or reg.global_solve is None:
        raise KeyError(theory)
    return reg.global_solve, reg.global_label


def station_solver_map() -> dict[ShellTheory, StationSolverFn]:
    return {
        t: r.station_solver
        for t, r in _REGISTRATIONS.items()
        if r.station_solver is not None
    }


def registered_theories() -> tuple[ShellTheory, ...]:
    return tuple(_REGISTRATIONS.keys())


def global_backend_theories() -> tuple[ShellTheory, ...]:
    return tuple(t for t, r in _REGISTRATIONS.items() if r.global_solve is not None)


__all__ = [
    "ShellTheoryRegistration",
    "get_global_backend",
    "get_station_solver",
    "global_backend_theories",
    "register_global_backend",
    "register_station_solver",
    "registered_theories",
    "station_solver_map",
]
