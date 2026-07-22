"""Shared MITC4 / ccx periodic K6 oracles for cross-solver validation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from blade_precompute.section_shell_model.pre_processing.mesh.shell_mesh_map import NElements
from ..engine.homogenisation import compute_elastic_centroid_from_panels, compute_k6_periodic_bc
from ..engine.global_shell_assembly import assemble_global_shell
from ..calculix_ccx.periodic import assemble_k6_from_steps
from ..calculix_ccx.results import parse_ccx_dat_periodic
from ..calculix_ccx.section_export import (
    periodic_ref_node_ids,
    write_section_inp_periodic,
)


def mitc4_periodic_k6(
    pb: Any,
    *,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = 2,
    interface_constraint_mode: str = "shared",
    dof_frame: str = "local",
) -> np.ndarray:
    """MITC4 periodic unit-cell ``K6`` (same basis as production ``run_section_periodic_bc_mitc4``)."""
    panels = pb.panels
    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    k_global, _f, node_meta, panel_maps, *_rest = assemble_global_shell(
        panels,
        zeros,
        zeros,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
    )
    y_e, z_e = compute_elastic_centroid_from_panels(panels)
    k6, _ = compute_k6_periodic_bc(
        k_global,
        node_meta,
        panel_maps,
        panels,
        y_e,
        z_e,
        dof_frame=dof_frame,
    )
    return np.asarray(k6, dtype=float)


def mitc4_periodic_k6_global(
    pb: Any,
    *,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = 2,
    interface_constraint_mode: str = "shared",
) -> np.ndarray:
    """MITC4 periodic ``K6`` with global translation frame (ccx deck macro basis)."""
    return mitc4_periodic_k6(
        pb,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
        dof_frame="global",
    )


def ccx_periodic_k6(
    pb: Any,
    *,
    n_elements_per_panel: int,
    ccx_work_dir: str | Path,
    interface_constraint_mode: str = "shared",
    ccx_exe: str = "ccx",
) -> np.ndarray:
    """External ccx periodic unit-cell ``K6`` from reference-node reactions."""
    from ..calculix_ccx.mesh import build_ccx_section_mesh

    if shutil.which(ccx_exe) is None:
        raise RuntimeError(f"CalculiX executable {ccx_exe!r} not found on PATH")

    panels = pb.panels
    mesh = build_ccx_section_mesh(
        panels, n_elements_per_panel, interface_constraint_mode=interface_constraint_mode
    )
    y_e, z_e = compute_elastic_centroid_from_panels(panels)
    ref_ids = periodic_ref_node_ids(mesh)

    work = Path(ccx_work_dir)
    work.mkdir(parents=True, exist_ok=True)
    inp = work / "section_periodic.inp"
    write_section_inp_periodic(mesh, panels, inp, y_e=y_e, z_e=z_e)
    subprocess.run(
        [shutil.which(ccx_exe), inp.stem],
        cwd=str(work),
        check=True,
        capture_output=True,
        text=True,
    )
    steps = parse_ccx_dat_periodic(work / f"{inp.stem}.dat")
    return assemble_k6_from_steps(steps, ref_ids)
