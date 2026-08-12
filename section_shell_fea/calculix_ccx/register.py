"""Opt-in registration hook for the quarantined CalculiX (ccx) backend.

CCX is a self-contained, external-subprocess **validation** backend that production
dispatch (``engine/shell_solve_registry.py``, ``engine/station_theory_dispatch.py``)
does not import or register at module load. Callers that need CCX through the
theory-dispatched router — the orchestration ``shell_theory="calculix_ccx"`` option
(``orchestration/precompute/shell_runtime.py``) and validation harnesses — must
call :func:`register_calculix_ccx_backend` first.

Consumers that import ``calculix_ccx.solve.solve_global_coupled_ccx`` (or
``calculix_ccx.thin_wall_adapter.ccx_global_coupled_solve_from_thin_wall``)
directly bypass the registry maps entirely and do not need this hook. Thin-wall
CCX is validation-only. CCX has no local element-factory entry (it is an external
subprocess, not a native element formulation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

_REGISTERED = False


def _solve_ccx(
    zones: list[Any],
    nx_panels: list[np.ndarray],
    nxy_panels: list[np.ndarray],
    **kwargs: Any,
) -> tuple[list[list[Any]], list[dict]]:
    from .solve import solve_global_coupled_ccx

    work_dir = kwargs.pop("work_dir", None)
    if work_dir is None:
        raise ValueError("CALCULIX_CCX backend requires work_dir=...")
    ccx_exe = kwargs.pop("ccx_exe", None)
    if ccx_exe is None:
        ccx_exe = kwargs.pop("exe", "ccx")
    job_name = kwargs.pop("job_name", "section")
    kwargs.pop("return_assembly_data", None)
    section_strip_mesh = kwargs.get("section_strip_mesh")
    if section_strip_mesh is None:
        from blade_structure_utilities.mesh_utilities.section_mesh import (
            build_section_strip_mesh_from_panels,
        )

        section_strip_mesh = build_section_strip_mesh_from_panels(
            panels,
            n_elements_per_panel=kwargs.get("n_elements_per_panel", 10),
            target_element_length_m=kwargs.get("target_element_length_m"),
            endpoint_tol=kwargs.get("endpoint_tol", 1e-6),
        )
    res, diag, _inp = solve_global_coupled_ccx(
        panels,
        nx_panels,
        nxy_panels,
        section_strip_mesh=section_strip_mesh,
        work_dir=work_dir,
        ccx_exe=ccx_exe,
        job_name=job_name,
        bc_mode=kwargs.get("bc_mode", "contour_support"),
        interface_constraint_mode=kwargs.get("interface_constraint_mode", "shared"),
        l_x=kwargs.get("l_x", 1.0),
    )
    return res, diag


def register_calculix_ccx_backend() -> None:
    """Populate production dispatch maps with the CCX backend.

    Idempotent — safe to call multiple times (e.g. once per worker process).
    """
    global _REGISTERED
    if _REGISTERED:
        return

    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellTheory

    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.dispatch.station_theory_dispatch import (
        register_station_solver,
    )
    from .station_dispatch_adapter import dispatch_ccx_station_solve

    register_station_solver(ShellTheory.CALCULIX_CCX, dispatch_ccx_station_solve)

    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.dispatch.shell_solve_registry import register_backend

    register_backend(ShellTheory.CALCULIX_CCX, _solve_ccx, label="external_calculix_ccx")

    _REGISTERED = True


__all__ = ["register_calculix_ccx_backend"]
