"""Periodic-BC CalculiX (ccx) S4R homogenization — external-solver wiring.

Production S4R path: ccx runs the periodic unit cell; unit-mode S4R stress fields are superposed
using MITC4 energy ``K6`` amplitudes (``validation/periodic_k6_oracle.mitc4_periodic_k6``).
Optional ``.frd`` IP parsing recovers wall bending ``Mx/My/Mxy``.

Pipeline:

1. ``build_ccx_section_mesh`` → strip mesh with inboard / outboard layers.
2. ``write_section_inp_periodic`` → periodic ``*EQUATION`` deck + 6 unit-mode steps (``S``, ``RF``,
   optional FRD ``S``).
3. run ``ccx`` → ``parse_ccx_dat_periodic`` + optional ``parse_ccx_frd_stress_blocks``.
4. ``mitc4_periodic_k6`` → ``K6 q = f_beam`` → ``superpose_stresses`` (+ FRD bending merge).
5. ``map_ccx_stresses_to_panel_resultants`` → ``SectionShellRecoveryBundle``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from blade_precompute.section_shell_model.pre_processing.mesh.section_strip_mesh import MIN_ELEMENTS_PER_PANEL
from blade_precompute.section_shell_model.pre_processing.containers.types import SectionShellRecoveryBundle
from blade_precompute.section_shell_model.pre_processing.mesh.shell_mesh_map import NElements
from .mesh import build_ccx_section_mesh
from .results import (
    CcxElementStress,
    PeriodicStepResult,
    map_ccx_stresses_to_panel_resultants,
    parse_ccx_dat_periodic,
)


def assemble_k6_from_steps(
    steps: list[PeriodicStepResult], ref_node_ids: list[int]
) -> np.ndarray:
    """Section ``K6`` (6×6) from the reference-node reaction matrix.

    Driving macro mode ``m`` (reference DOF ``m`` = 1) yields reaction ``RF`` at every reference
    node; ``K6[n, m]`` is the DOF-1 (``fx``) reaction at reference node ``n``. Symmetrized to
    suppress small asymmetry.

    .. warning::
       Reliable for the axial diagonal only under the current ``*EQUATION`` layout. Bending /
       torsion ``RF`` are MPC Lagrange forces, not beam conjugates (~300× stiff on κ_y for the
       NACA fixture). Production uses MITC4 energy ``K6`` — see
       ``docs/PERIODIC_CCX_HOMOGENIZATION.md``.
    """
    idx = {rid: i for i, rid in enumerate(ref_node_ids)}
    k6 = np.zeros((6, 6))
    for step in steps:
        for rid, rf in step.rf_by_node.items():
            if rid in idx:
                k6[idx[rid], step.mode] = float(rf[0])
    return 0.5 * (k6 + k6.T)


def superpose_stresses(
    steps: list[PeriodicStepResult], q_beam: np.ndarray
) -> list[CcxElementStress]:
    """Per-element 3D stress for a beam load = Σ_m q[m] · (mode-m element stress)."""
    q = np.asarray(q_beam, dtype=float)
    acc: dict[int, np.ndarray] = {}
    for step in steps:
        w = float(q[step.mode])
        for r in step.stress_rows:
            v = acc.setdefault(r.eid, np.zeros(6))
            v += w * np.array([r.sxx, r.syy, r.szz, r.sxy, r.sxz, r.syz])
    return [CcxElementStress(eid, *vals.tolist()) for eid, vals in sorted(acc.items())]


def run_section_periodic_bc_ccx(
    pb: Any,
    *,
    resultants: Any | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    n_elements_per_panel: NElements | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = MIN_ELEMENTS_PER_PANEL,
    interface_constraint_mode: str = "shared",
    loading_axis_yz: tuple[float, float] | None = None,
    transport_to_elastic_centroid: bool = True,
    bc_mode: str = "contour_support",
    traction_field: Any | None = None,
    traction_points_yz: Any | None = None,
    ccx_work_dir: str | None = None,
    ccx_exe: str = "ccx",
    emit_frd: bool = True,
) -> SectionShellRecoveryBundle:
    """External CalculiX S4R section solve via the periodic unit-cell BC.

    Mirrors :func:`...api.periodic.run_section_periodic_bc_mitc4` (same ``pb`` Pipeline B
    section, same beam loads, same ``SectionShellRecoveryBundle`` output) but the element
    formulation + solve are performed by the external ``ccx`` solver. Requires ``ccx`` on PATH
    and a writable ``ccx_work_dir``. See the module docstring for the end-to-end validation gaps.
    """
    from ..engine.homogenisation import compute_elastic_centroid_from_panels
    from ..pre_processing.element_library.mitc4.utilities.mitc4_solve_orchestrator import collect_panel_abd_stack
    from .section_export import write_section_inp_periodic

    del bc_mode, traction_field, traction_points_yz  # API parity with MITC4 router
    if shutil.which(ccx_exe) is None:
        raise RuntimeError(
            f"CalculiX executable {ccx_exe!r} not found on PATH; required for the external "
            "periodic-ccx S4R solve (run_section_periodic_bc_ccx)."
        )
    if ccx_work_dir is None:
        raise ValueError("run_section_periodic_bc_ccx requires ccx_work_dir=...")

    panels = pb.panels
    abd_p, t_p, g_p, lab_p = collect_panel_abd_stack(panels)
    mesh = build_ccx_section_mesh(
        panels,
        n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
    )
    y_e, z_e = compute_elastic_centroid_from_panels(panels)

    work = Path(ccx_work_dir)
    work.mkdir(parents=True, exist_ok=True)
    inp = work / "section_periodic.inp"
    write_section_inp_periodic(mesh, panels, inp, y_e=y_e, z_e=z_e, emit_frd=emit_frd)
    subprocess.run([shutil.which(ccx_exe), inp.stem], cwd=str(work), check=True,
                   capture_output=True, text=True)

    steps = parse_ccx_dat_periodic(work / f"{inp.stem}.dat")

    # Mode amplitudes from the MITC4 energy homogenizer (same periodic macro BC, global frame).
    # ccx reference-node reactions do not yield a reliable K6 for bending/torsion with the
    # *EQUATION* MPC layout (ref-node DOF 1 carries all macro amplitudes); unit-mode ccx
    # stress fields are superposed using the trusted K6 below.
    from ..validation.periodic_k6_oracle import mitc4_periodic_k6

    k6 = mitc4_periodic_k6(
        pb,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        interface_constraint_mode=interface_constraint_mode,
    )

    from blade_precompute.section_shell_model.pre_processing.loading.beam_resultant_handoff import (
        resolve_beam_resultants_for_shell_k6,
    )

    _at_ec, f_beam = resolve_beam_resultants_for_shell_k6(
        y_e=float(y_e),
        z_e=float(z_e),
        resultants=resultants,
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        loading_axis_yz=loading_axis_yz,
        transport_to_elastic_centroid=transport_to_elastic_centroid,
    )
    f_beam = np.asarray(f_beam, dtype=float)
    try:
        q_beam = np.linalg.solve(k6, f_beam)
    except np.linalg.LinAlgError:
        q_beam = np.linalg.lstsq(k6, f_beam, rcond=None)[0]

    combined = superpose_stresses(steps, q_beam)
    all_panel_results, all_panel_diag = map_ccx_stresses_to_panel_resultants(mesh, panels, combined)

    frd_path = work / f"{inp.stem}.frd"
    if emit_frd and frd_path.is_file():
        from .results import (
            _remap_frd_eids_to_mesh,
            apply_frd_bending_to_panel_resultants,
            ccx_frd_element_base,
            parse_ccx_frd_3c_node_index,
            parse_ccx_frd_stress_blocks,
            superpose_frd_stress_blocks,
        )

        frd_blocks = parse_ccx_frd_stress_blocks(frd_path)
        if frd_blocks:
            frd_index = parse_ccx_frd_3c_node_index(frd_path)
            frd_by_eid = superpose_frd_stress_blocks(frd_blocks[: len(steps)], q_beam)
            frd_by_eid = _remap_frd_eids_to_mesh(
                frd_by_eid,
                element_base=ccx_frd_element_base(mesh),
                n_mesh_elements=len(mesh.elements),
                frd_index=frd_index,
            )
            apply_frd_bending_to_panel_resultants(all_panel_results, mesh, frd_by_eid)

    n_panels = len(panels)
    ref: Any = None
    for panel_res in all_panel_results:
        if panel_res:
            ref = panel_res[len(panel_res) // 2]
            break

    return SectionShellRecoveryBundle(
        panels=panels,
        booms=pb.booms,
        webs_geom=pb.webs_geom,
        q_tot=[], sig_p=[], sig_b=[], q0=[], props=None,
        y_sc=float(y_e), z_sc=float(z_e),
        y_e=float(y_e), z_e=float(z_e),
        areas=[], I_omega=0.0, gamma_y=0.0, gamma_z=0.0, GA_y=0.0, GA_z=0.0,
        q_primary=[], q_warp=[],
        reference_resultants=ref,
        all_panel_results=all_panel_results,
        all_panel_diagnostics=all_panel_diag,
        panel_abd=abd_p,
        panel_thickness_m=t_p,
        panel_G_eff=g_p,
        panel_labels=lab_p,
        shell_midline_strips=tuple(getattr(pb.shell_inputs, "midlines", ())),
        beam_mode_amplitudes=q_beam,
    )


__all__ = [
    "assemble_k6_from_steps",
    "superpose_stresses",
    "run_section_periodic_bc_ccx",
]
