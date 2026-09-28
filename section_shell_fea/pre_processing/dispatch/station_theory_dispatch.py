"""Per-station shell theory dispatch for the strip-mesh orchestration path.

Production path: accept prebuilt mesh / panel topology (frozen zones) / SFV and
select the registered FE formulation only. Mesh building, SFV shaping, and
demo-bind are not owned here — compose those in BSP L4-EVAL (or the explicit
non-production convenience helper below).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
    require_shell_theory_supports,
)


@dataclass(frozen=True)
class StationTheorySolveRequest:
    """Inputs shared by FSDT / CCX station solves.

    Production dispatch requires ``section_strip_mesh``, ``panel_topology``
    (frozen zones), and ``section_force_vector``. Density kwargs are retained
    only for the non-production convenience helper.
    """

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
    section_strip_mesh: Any | None = None
    panel_topology: Any | None = None
    section_force_vector: Any | None = None


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

    Requires ``section_strip_mesh=`` (density owned outside FE), frozen
    ``panel_topology`` (zones from bind / L4), and frozen
    ``section_force_vector`` shaped outside FE. When ``load_field`` carries a
    usable TractionField, Channel B is selected and SFV is mandatory
    (FEA-L-003: no proximity face-load overlay).
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.interface.traction_load_adapter import (
        traction_points_from_load_field,
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
    if panel_topology is None:
        raise ValueError(
            "panel_topology with frozen zones is required; compose via L4-EVAL / "
            "SectionBindRunner. Demo convenience must use "
            "dispatch_fsdt_demo_convenience (not production dispatch)."
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

    has_traction = (
        load_field is not None
        and traction_points_from_load_field(load_field) is not None
    )
    # Default Channel C. Usable face traction selects Channel B; SFV required.
    load_channel = "surface_traction" if has_traction else "thinwall"
    sfv = section_force_vector
    if sfv is None:
        raise ValueError(
            f"section_force_vector is required for load_channel={load_channel!r}; "
            "shape SFV with section_loads outside FE (FE applies frozen SFV only; "
            "FEA-L-003 refuses proximity TractionField overlay)."
        )

    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        section_strip_mesh=section_strip_mesh,
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
        "load_channel": str(static.load_channel_used),
        "used_traction": False,  # FEA-L-003: proximity overlay removed
        "max_fi_hashin": max_fi,
        "needs_review": needs_review,
        "fi_by_panel": [float(x) for x in fi_by_panel],
        "provenance_breakdown": {},
        "governing_element": dict(gov),
    }


def _require_prebuilt_fsdt_products(request: StationTheorySolveRequest) -> None:
    missing = [
        name
        for name, value in (
            ("section_strip_mesh", request.section_strip_mesh),
            ("panel_topology", request.panel_topology),
            ("section_force_vector", request.section_force_vector),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Production station dispatch requires prebuilt "
            + ", ".join(missing)
            + ". Compose mesh/zones/SFV in BSP L4-EVAL "
            "(station_evaluation_workflow / station_preparation_workflow); "
            "for demos/tests use dispatch_fsdt_demo_convenience()."
        )


def _dispatch_fsdt(request: StationTheorySolveRequest) -> StationTheorySolveResult:
    """Production FSDT handler: formulation select over frozen worker products only."""
    _require_prebuilt_fsdt_products(request)
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
            section_strip_mesh=request.section_strip_mesh,
            section_force_vector=request.section_force_vector,
            panel_topology=request.panel_topology,
            load_field=request.load_field,
        )
    )


def dispatch_fsdt_demo_convenience(
    request: StationTheorySolveRequest,
) -> StationTheorySolveResult:
    """Non-production helper: mesh + demo-bind + SFV shape, then FSDT dispatch.

    Emits a warning. Not an L4 sequencer and not the production ownership home.
    Prefer BSP ``station_evaluation_workflow`` / ``station_preparation_workflow``.
    """
    warnings.warn(
        "dispatch_fsdt_demo_convenience builds mesh, demo-binds zones, and shapes "
        "SFV inside FE dispatch; use only for demos/tests. Production composition "
        "belongs in BSP L4-EVAL.",
        UserWarning,
        stacklevel=2,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.api.shell_panel_topology import (
        shell_panel_topology_from_shell_inputs,
    )
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

    strip_mesh = request.section_strip_mesh
    if strip_mesh is None:
        strip_mesh = build_section_strip_mesh(
            request.shell_inputs,
            n_elements_per_panel=request.n_elements_per_panel,
            target_element_length_m=request.target_element_length_m,
            min_elements_per_panel=request.min_elements_per_panel,
        )

    panel_topology = request.panel_topology
    if panel_topology is None:
        panel_topology = shell_panel_topology_from_shell_inputs(
            request.shell_inputs,
            section=request.section_view,
            allow_demo_bind=True,
        )

    sfv = request.section_force_vector
    if sfv is None and request.load_field is None and request.shell_inputs is not None:
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
            resultants,
            getattr(request.shell_inputs, "midlines", None),
            cell_wall_loops=getattr(request.shell_inputs, "cell_wall_loops", None),
        )

    prepared = replace(
        request,
        section_strip_mesh=strip_mesh,
        panel_topology=panel_topology,
        section_force_vector=sfv,
    )
    return _dispatch_fsdt(prepared)


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
    "dispatch_fsdt_demo_convenience",
    "dispatch_station_theory",
    "fsdt_periodic_station_hashin",
    "register_station_solver",
]
