"""External CalculiX ccx solve orchestration for section strip meshes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from blade_precompute.section_shell_model.pre_processing.mesh.shell_mesh_map import NElements
from .mesh import build_ccx_section_mesh
from .results import map_ccx_stresses_to_panel_resultants, parse_ccx_dat_stresses
from .section_export import write_section_inp


def _require_ccx(ccx_exe: str = "ccx") -> str:
    path = shutil.which(ccx_exe)
    if path is None:
        raise RuntimeError(
            f"CalculiX executable {ccx_exe!r} not found on PATH; "
            "required for ShellTheory.CALCULIX_CCX"
        )
    return path


def solve_global_coupled_ccx(
    panels: list[Any],
    nx_panels: list[np.ndarray],
    nxy_panels: list[np.ndarray],
    *,
    work_dir: str | Path,
    ccx_exe: str = "ccx",
    l_x: float = 1.0,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = 2,
    endpoint_tol: float = 1e-6,
    bc_mode: str = "contour_support",
    interface_constraint_mode: str = "shared",
    job_name: str = "section",
) -> tuple[list[list], list[dict], Path]:
    """Export section mesh, run ccx, parse section forces."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ccx_path = _require_ccx(ccx_exe)

    mesh = build_ccx_section_mesh(
        panels,
        n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        endpoint_tol=endpoint_tol,
        interface_constraint_mode=interface_constraint_mode,
        l_x=l_x,
    )

    inp_path = work / f"{job_name}.inp"
    write_section_inp(
        mesh,
        panels,
        nx_panels,
        nxy_panels,
        inp_path,
        bc_mode=bc_mode,
        job_name=job_name,
    )

    subprocess.run(
        [ccx_path, job_name],
        cwd=str(work),
        check=True,
        capture_output=True,
        text=True,
    )

    dat_path = work / f"{job_name}.dat"
    stress_rows = parse_ccx_dat_stresses(dat_path) if dat_path.is_file() else []
    results, diag = map_ccx_stresses_to_panel_resultants(mesh, panels, stress_rows)

    for d in diag:
        d["inp_path"] = str(inp_path.resolve())
        d["dat_path"] = str(dat_path.resolve()) if dat_path.is_file() else None
        d["n_stress_rows"] = len(stress_rows)

    return results, diag, inp_path


__all__ = ["solve_global_coupled_ccx"]
