"""Per-station shell theory dispatch for the strip-mesh orchestration path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
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
    section_strip_mesh: Any,
    section_force_vector: Any | None = None,
    panel_topology: Any | None = None,
    load_field: Any | None = None,
    n_elements_per_panel: int | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int | None = None,
) -> dict[str, Any]:
    """Linear-static FSDT cascade + Hashin FI summary for orchestration JSON.

    Usable face traction is applied on the contour-supported linear-static RHS
    (same free set as beam loads). Beam-only stations use the cascade without
    Channel B overlay. Requires ``section_strip_mesh=`` (density is owned
    outside FE; dispatch builds the mesh once before calling this helper).

    Thinwall path requires a frozen ``section_force_vector=`` shaped outside FE.
    Prefer ``panel_topology=`` with pre-bound zones; omitted topology may
    demo-bind via ``shell_panel_topology_from_shell_inputs``.
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.interface.traction_load_adapter import (
        traction_points_from_load_field,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
        shell_panel_topology_from_shell_inputs,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
        SectionLinearStaticRunner,
    )

    if section_strip_mesh is None:
        raise ValueError(
            "section_strip_mesh is required; build it with "
            "mesh_utilities.section_mesh.build_section_strip_mesh "
            "(density is owned outside FE)."
        )
    if (
        n_elements_per_panel is not None
        or target_element_length_m is not None
        or min_elements_per_panel is not None
    ):
        raise ValueError(
            "Density kwargs are not accepted on fsdt_periodic_station_hashin; "
            "pass a prebuilt section_strip_mesh= instead."
        )

    if panel_topology is None:
        panel_topology = shell_panel_topology_from_shell_inputs(
            shell_inputs, section=section_view
        )
    traction_field = None
    traction_points = None
    used_traction = False
    if load_field is not None:
        traction_points = traction_points_from_load_field(load_field)
        if traction_points is not None:
            traction_field = load_field.traction_field
            used_traction = True

    # Default Channel C (thinwall). When usable face traction is present, use
    # Channel B alone (do not stack tip-integrated V with matching face traction).
    load_channel = "surface_traction" if used_traction else "thinwall"
    sfv = section_force_vector
    if load_channel == "thinwall" and sfv is None:
        raise ValueError(
            "section_force_vector is required for thinwall; shape SFV with "
            "section_loads outside FE (FE applies frozen SFV only)."
        )
    strip_mesh = section_strip_mesh
    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        traction_field=traction_field,
        traction_points_yz=traction_points,
        section_strip_mesh=strip_mesh,
        load_channel=load_channel,
        section_force_vector=sfv,
    ).execute()
    results = static.get_results()
    quaternary = results.quaternary_results
    if quaternary is None:
        raise RuntimeError("execute() did not populate quaternary_results")
    gov = quaternary.governing or {}
    max_fi = float(gov.get("fi", 0.0) or 0.0)
    nr = quaternary.needs_review
    needs_review = bool(np.any(nr)) if nr is not None else False
    hi = None if quaternary.hashin_fi is None else quaternary.hashin_fi.in_plane
    panel_index = (
        None if results.tertiary_results is None else results.tertiary_results.panel_index
    )
    assert static.assembly is not None
    fi_by_panel: list[float] = []
    for pi in range(len(static.assembly.panel_maps)):
        pmax = 0.0
        if hi is not None and panel_index is not None:
            for ri in range(hi.shape[0]):
                if int(panel_index[ri]) != pi:
                    continue
                if hi[ri].size:
                    pmax = max(pmax, float(np.nanmax(hi[ri])))
        fi_by_panel.append(pmax)
    return {
        "solver": "fsdt_linear_static_cascade",
        "used_traction": bool(used_traction or static.used_traction),
        "max_fi_hashin": max_fi,
        "needs_review": needs_review,
        "fi_by_panel": [float(x) for x in fi_by_panel],
        "provenance_breakdown": {},
        "governing_element": dict(gov),
    }


def _dispatch_fsdt(request: StationTheorySolveRequest) -> StationTheorySolveResult:
    from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
        BEAM_LOAD_AXIS_YZ,
        BeamResultantVector,
        FRAME_SECTION_YZ,
    )
    from blade_structure_utilities.load_utilities.section_loads import (
        shape_section_force_vector,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import (
        build_section_strip_mesh,
    )

    # Density ownership: build once at the dispatch boundary, then hand a mesh
    # into the FE shim (no remesh / density kwargs on the solve path).
    strip_mesh = build_section_strip_mesh(
        request.shell_inputs,
        n_elements_per_panel=request.n_elements_per_panel,
        target_element_length_m=request.target_element_length_m,
        min_elements_per_panel=request.min_elements_per_panel,
    )
    # Named FE-PREP convenience: registry path still shapes SFV here so
    # fsdt_periodic_station_hashin itself only applies a frozen vector.
    # Production L4 must shape via SectionLoadsRunner before the runner.
    sfv = None
    if request.load_field is None and request.shell_inputs is not None:
        resultants = BeamResultantVector.from_named(
            N=request.N,
            Vy=request.Vy,
            Vz=request.Vz,
            My=request.My,
            Mz=request.Mz,
            T=request.T,
            frame=FRAME_SECTION_YZ,
            about_xy=BEAM_LOAD_AXIS_YZ,
        )
        sfv = shape_section_force_vector(
            resultants, getattr(request.shell_inputs, "midlines", None)
        )
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
            section_strip_mesh=strip_mesh,
            section_force_vector=sfv,
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
