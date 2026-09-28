"""External CalculiX ccx solve orchestration for section strip meshes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

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
    zones: list[Any],
    nx_panels: list[np.ndarray],
    nxy_panels: list[np.ndarray],
    *,
    section_strip_mesh: Any,
    work_dir: str | Path,
    ccx_exe: str = "ccx",
    l_x: float = 1.0,
    bc_mode: str = "contour_support",
    interface_constraint_mode: str = "shared",
    cap_tie_mode: str | None = None,
    cap_tie_export: str = "tie",
    job_name: str = "section",
    shell_midlines: Any | None = None,
) -> tuple[list[list], list[dict], Path]:
    """Export section mesh, run ccx, parse section forces.

    ``cap_tie_mode`` defaults to ``skin_offset`` when the strip contains ``cap:*``
    panels (MITC4-FSDT production analogue). Pass ``"none"`` for the untied
    control. ``shell_midlines`` is required when ties resolve to ``skin_offset``.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    ccx_path = _require_ccx(ccx_exe)

    mesh = build_ccx_section_mesh(
        zones,
        section_strip_mesh=section_strip_mesh,
        interface_constraint_mode=interface_constraint_mode,
        l_x=l_x,
        cap_tie_mode=cap_tie_mode,
        shell_midlines=shell_midlines,
    )

    inp_path = work / f"{job_name}.inp"
    write_section_inp(
        mesh,
        zones,
        nx_panels,
        nxy_panels,
        inp_path,
        bc_mode=bc_mode,
        job_name=job_name,
        cap_tie_mode=mesh.cap_tie_mode,
        cap_tie_export=cap_tie_export,
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
    results, diag = map_ccx_stresses_to_panel_resultants(mesh, zones, stress_rows)
    return results, diag, inp_path


__all__ = ["solve_global_coupled_ccx"]
