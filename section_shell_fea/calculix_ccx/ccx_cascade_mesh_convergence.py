"""CCX membrane cascade rung for grid-independence / GCI (quarantined).

Mirrors ``cascade_mesh_convergence.run_cascade_at_frac`` geometry + mesh QA, but
solves with live CalculiX and fills membrane peaks (``peak_abs_n11`` /
``peak_abs_n12``; ``N22`` when present on mapped rows). Bending, tertiary,
and quaternary kind peaks remain empty/NaN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BeamResultantVector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    shell_panel_topology_from_shell_inputs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    DEFAULT_CHORD,
    DEFAULT_LAYOUT,
    DEFAULT_LOAD,
    DEFAULT_THICKNESS,
    DEFAULT_TWIST_DEG,
    ENDPOINT_BAND,
    KINDS,
    CascadeScalars,
    _accumulate_kind_peaks,
    _build_shell,
    _kind_from_label,
    _nan_kind_peaks,
)
from blade_structure_utilities.load_utilities.section_loads import (
    shape_section_force_vector,
)
from blade_structure_utilities.mesh_utilities.section_mesh import (
    assert_strip_mesh_quality,
    build_section_strip_mesh,
    mesh_quality_report,
)

from .sfv_panel_loads import sfv_to_n11_n12

from .solve import solve_global_coupled_ccx


def _panel_resultant_arrays(
    zones: list[Any],
    results: list[list[Any]],
    attr: str,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for p, res_list in zip(zones, results):
        n_guess = max(1, len(getattr(p, "s", [])) - 1)
        if not res_list:
            out.append(np.full(n_guess, np.nan, dtype=float))
            continue
        vals = np.asarray(
            [abs(float(getattr(r, attr, float("nan")))) for r in res_list],
            dtype=float,
        )
        out.append(vals)
    return out


def _n_elem_by_kind(zones: list[Any], results: list[list[Any]]) -> dict[str, int]:
    counts = {k: 0 for k in KINDS}
    for p, res_list in zip(zones, results):
        kind = _kind_from_label(getattr(p, "label", ""))
        if kind in counts:
            counts[kind] += len(res_list) if res_list else 0
    return counts


def run_ccx_cascade_at_frac(
    frac: float,
    *,
    cap_tie_mode: str = "skin_offset",
    interface_constraint_mode: str = "shared",
    layout: str = DEFAULT_LAYOUT,
    chord: float = DEFAULT_CHORD,
    twist_deg: float = DEFAULT_TWIST_DEG,
    thickness: Mapping[str, float] | None = None,
    load: Mapping[str, float] | None = None,
    naca_dat: str | Path | None = None,
    require_mesh_qa: bool = True,
    work_dir: str | Path | None = None,
    ccx_exe: str = "ccx",
    l_x: float = 1.0,
    cap_tie_export: str = "tie",
    junction_band_elements: int | None = None,
    junction_band_step_over_h: float | None = None,
) -> CascadeScalars:
    """Build mesh at ``frac``, solve with CalculiX, return membrane ``CascadeScalars``."""
    thickness = dict(thickness or DEFAULT_THICKNESS)
    load = dict(load or DEFAULT_LOAD)
    shell = _build_shell(
        layout=layout,
        chord=chord,
        twist_deg=twist_deg,
        thickness=thickness,
        naca_dat=naca_dat,
    )
    mesh = build_section_strip_mesh(
        shell,
        target_element_length_frac=float(frac),
        junction_band_elements=junction_band_elements,
        junction_band_step_over_h=junction_band_step_over_h,
    )
    report = mesh_quality_report(mesh, target_element_length_frac=float(frac))
    if require_mesh_qa:
        assert_strip_mesh_quality(mesh, target_element_length_frac=float(frac))

    el_min = {k: report.per_kind[k].el_min_over_h for k in report.per_kind}
    el_max = {k: report.per_kind[k].el_max_over_h for k in report.per_kind}
    n_elem_mesh = {k: report.per_kind[k].n_elements for k in report.per_kind}
    h_m = float(frac) * float(chord)

    topo = shell_panel_topology_from_shell_inputs(shell, allow_demo_bind=True)
    zones = list(topo.zones)
    resultants = BeamResultantVector.from_named(**load)
    sfv = shape_section_force_vector(resultants, shell.midlines, L_x=float(l_x))
    n11_panels, n12_panels = sfv_to_n11_n12(sfv, zones)

    work = Path(work_dir) if work_dir is not None else Path("ccx_cascade_work")
    work = work / str(layout) / f"frac_{float(frac):.4f}".replace(".", "p")
    work.mkdir(parents=True, exist_ok=True)

    export = str(cap_tie_export or "tie")
    results: list[list[Any]] = [[] for _ in zones]
    diag: list[dict] = []
    trust = False
    try:
        results, diag, _inp = solve_global_coupled_ccx(
            zones,
            n11_panels,
            n12_panels,
            section_strip_mesh=mesh,
            work_dir=work / export,
            ccx_exe=ccx_exe,
            l_x=float(l_x),
            interface_constraint_mode=interface_constraint_mode,
            cap_tie_mode=cap_tie_mode,
            cap_tie_export=export,
            job_name="section",
        )
        trust = True
    except Exception:
        if export == "tie" and str(cap_tie_mode) == "skin_offset":
            try:
                results, diag, _inp = solve_global_coupled_ccx(
                    zones,
                    n11_panels,
                    n12_panels,
                    section_strip_mesh=mesh,
                    work_dir=work / "equation",
                    ccx_exe=ccx_exe,
                    l_x=float(l_x),
                    interface_constraint_mode=interface_constraint_mode,
                    cap_tie_mode=cap_tie_mode,
                    cap_tie_export="equation",
                    job_name="section",
                )
                trust = True
            except Exception:
                trust = False
                results = [[] for _ in zones]
        else:
            trust = False
            results = [[] for _ in zones]

    # Coverage: every panel that has strip elements should have mapped rows.
    n_elem_res = _n_elem_by_kind(zones, results)
    if trust:
        for k in KINDS:
            if int(n_elem_mesh.get(k, 0) or 0) > 0 and int(n_elem_res.get(k, 0) or 0) == 0:
                trust = False
                break
        if diag and all(int(d.get("n_parsed_stresses", 0) or 0) == 0 for d in diag):
            # Global parse may still be ok if stresses land in one panel; only fail
            # when every panel diag reports zero parses and all result lists empty.
            if all(len(r) == 0 for r in results):
                trust = False

    labels = [str(getattr(p, "label", "")) for p in zones]
    abs_n12 = _panel_resultant_arrays(zones, results, "N12")
    abs_n11 = _panel_resultant_arrays(zones, results, "N11")
    abs_n22 = _panel_resultant_arrays(zones, results, "N22")
    peak_n12, peak_n12_int = _accumulate_kind_peaks(
        abs_n12, labels=labels, band=ENDPOINT_BAND
    )
    peak_n11, peak_n11_int = _accumulate_kind_peaks(
        abs_n11, labels=labels, band=ENDPOINT_BAND
    )
    peak_n22, peak_n22_int = _accumulate_kind_peaks(
        abs_n22, labels=labels, band=ENDPOINT_BAND
    )

    return CascadeScalars(
        frac=float(frac),
        h_m=h_m,
        mesh_ok=bool(report.ok),
        layout=str(layout),
        el_min_over_h=el_min,
        el_max_over_h=el_max,
        n_elem=dict(n_elem_mesh),
        d_inf=float("nan"),
        residual_rel=float("nan"),
        residual_rel_independent=float("nan"),
        peak_abs_n12=peak_n12,
        peak_abs_n11=peak_n11,
        peak_abs_n22=peak_n22,
        peak_abs_n12_interior=peak_n12_int,
        peak_abs_n11_interior=peak_n11_int,
        peak_abs_n22_interior=peak_n22_int,
        governing_fi=float("nan"),
        cascade_trust_ok=bool(trust),
        solve_residual_rel=None,
        interface_constraint_mode=str(interface_constraint_mode),
        cap_tie_mode=str(cap_tie_mode),
    )


__all__ = ["run_ccx_cascade_at_frac"]
