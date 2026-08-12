"""Mesh-density convergence of M + P→S→T→Q cascade dataclass scalars.

Density is swept exclusively via ``section_mesh.build_section_strip_mesh``
(``target_element_length_frac``). Each rung runs pre-FE mesh QA then a full
``SectionLinearStaticRunner`` cascade and extracts station-level scalars by
component kind (skin / cap / web).

Tier coverage
-------------
* **Primary** — global solve scalars (``d_inf``, residuals).
* **Secondary** — mid-surface resultants ``|N11|…|M12|`` and FSDT ``|Q1|``, ``|Q2|``.
* **Tertiary** — mid-face ply stress norm + interface ``|τ13|``, ``|τ23|``.
* **Quaternary** — Hashin in-plane / interlaminar peaks + ``governing_fi``.

``peak_*`` fields are unrestricted elemental maxima (raw). ``peak_*_interior``
exclude the first/last :data:`ENDPOINT_BAND` elements of each panel strip
(endpoint-band sampling diagnostic; see grid-independence gate policy).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BeamResultantVector,
)
from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
    build_section_view,
    resolve_system_type,
)
from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (
    AirfoilSDF,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    shell_panel_topology_from_shell_inputs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.load_utilities.section_loads import (
    shape_section_force_vector,
)
from blade_structure_utilities.mesh_utilities.section_mesh import (
    assert_strip_mesh_quality,
    build_section_strip_mesh,
    mesh_quality_report,
)
from blade_structure_utilities.mesh_utilities.section_mesh.inputs._section_to_shell_inputs import (
    shell_mesh_inputs_from_section,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.containers.ply_results import (
    FACE_MID,
)

DEFAULT_LAYOUT = "5D-F"
DEFAULT_CHORD = 0.241
DEFAULT_TWIST_DEG = 34.1
DEFAULT_THICKNESS = {"skin": 0.001, "spar": 0.004, "web": 0.0015}
DEFAULT_LOAD = {"T": 5.0e3, "Vy": 2.0e4, "Mz": 1.0e4}
DEFAULT_REFINEMENTS = (0.024, 0.016, 0.012)

KINDS = ("skin", "cap", "web")
ENDPOINT_BAND = 2

# Station-level primary scalars (not kind-keyed).
PRIMARY_SCALAR_METRICS: tuple[str, ...] = (
    "d_inf",
    "residual_rel",
    "residual_rel_independent",
    "solve_residual_rel",
)

# Secondary mid-surface resultants → kind peaks (abs).
SECONDARY_KIND_METRICS: tuple[str, ...] = (
    "peak_abs_n11",
    "peak_abs_n22",
    "peak_abs_n12",
    "peak_abs_m11",
    "peak_abs_m22",
    "peak_abs_m12",
    "peak_abs_q1",
    "peak_abs_q2",
)
# Map CascadeScalars peak name → ShellPanelResultants attribute.
SECONDARY_ROW_ATTR: dict[str, str] = {
    "peak_abs_n11": "N11",
    "peak_abs_n22": "N22",
    "peak_abs_n12": "N12",
    "peak_abs_m11": "M11",
    "peak_abs_m22": "M22",
    "peak_abs_m12": "M12",
    "peak_abs_q1": "Q1",
    "peak_abs_q2": "Q2",
}

TERTIARY_KIND_METRICS: tuple[str, ...] = (
    "peak_mid_sig",
    "peak_abs_tau13",
    "peak_abs_tau23",
)

QUATERNARY_KIND_METRICS: tuple[str, ...] = (
    "peak_hashin",
    "peak_hashin_il",
)

# All kind-keyed peaks written to ladder CSV / GCI report tables.
KIND_PEAK_METRICS: tuple[str, ...] = (
    SECONDARY_KIND_METRICS + TERTIARY_KIND_METRICS + QUATERNARY_KIND_METRICS
)

PEAK_METRIC_LABELS: dict[str, str] = {
    "peak_abs_n11": "peak |N11|",
    "peak_abs_n22": "peak |N22|",
    "peak_abs_n12": "peak |N12|",
    "peak_abs_m11": "peak |M11|",
    "peak_abs_m22": "peak |M22|",
    "peak_abs_m12": "peak |M12|",
    "peak_abs_q1": "peak |Q1|",
    "peak_abs_q2": "peak |Q2|",
    "peak_mid_sig": "peak mid-σ",
    "peak_abs_tau13": "peak |τ13|",
    "peak_abs_tau23": "peak |τ23|",
    "peak_hashin": "peak Hashin (in-plane)",
    "peak_hashin_il": "peak Hashin (interlaminar)",
}


@dataclass(frozen=True)
class CascadeScalars:
    """Station-level scalars at one mesh density (primary + S/T/Q kind peaks)."""

    frac: float
    h_m: float
    mesh_ok: bool
    layout: str = ""
    el_min_over_h: dict[str, float] = field(default_factory=dict)
    el_max_over_h: dict[str, float] = field(default_factory=dict)
    n_elem: dict[str, int] = field(default_factory=dict)
    # Primary
    d_inf: float = float("nan")
    residual_rel: float = float("nan")
    """Unreduced ‖f−Kd‖/‖f‖ on the full DOF basis (MPC/support reactions included)."""
    residual_rel_independent: float = float("nan")
    """‖(f−Kd)[independent]‖/‖f‖ when reduction maps are available."""
    # Secondary membrane / bending / transverse shear
    peak_abs_n11: dict[str, float] = field(default_factory=dict)
    peak_abs_n22: dict[str, float] = field(default_factory=dict)
    peak_abs_n12: dict[str, float] = field(default_factory=dict)
    peak_abs_m11: dict[str, float] = field(default_factory=dict)
    peak_abs_m22: dict[str, float] = field(default_factory=dict)
    peak_abs_m12: dict[str, float] = field(default_factory=dict)
    peak_abs_q1: dict[str, float] = field(default_factory=dict)
    peak_abs_q2: dict[str, float] = field(default_factory=dict)
    # Tertiary ply / interface
    peak_mid_sig: dict[str, float] = field(default_factory=dict)
    peak_abs_tau13: dict[str, float] = field(default_factory=dict)
    peak_abs_tau23: dict[str, float] = field(default_factory=dict)
    # Quaternary FI
    peak_hashin: dict[str, float] = field(default_factory=dict)
    peak_hashin_il: dict[str, float] = field(default_factory=dict)
    # Endpoint-band-excluded maxima (first/last ENDPOINT_BAND elems per panel)
    peak_abs_n11_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_n22_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_n12_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_m11_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_m22_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_m12_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_q1_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_q2_interior: dict[str, float] = field(default_factory=dict)
    peak_mid_sig_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_tau13_interior: dict[str, float] = field(default_factory=dict)
    peak_abs_tau23_interior: dict[str, float] = field(default_factory=dict)
    peak_hashin_interior: dict[str, float] = field(default_factory=dict)
    peak_hashin_il_interior: dict[str, float] = field(default_factory=dict)
    governing_fi: float = float("nan")
    cascade_trust_ok: bool = True
    solve_residual_rel: float | None = None
    interface_constraint_mode: str = "shared"
    cap_tie_mode: str = "skin_offset"


def endpoint_band_mask(n_elem: int, *, band: int = ENDPOINT_BAND) -> np.ndarray:
    """True for first/last ``band`` elements of a strip panel."""
    n = int(n_elem)
    b = max(0, int(band))
    mask = np.zeros(n, dtype=bool)
    if n <= 0:
        return mask
    if 2 * b >= n:
        mask[:] = True
        return mask
    if b > 0:
        mask[:b] = True
        mask[-b:] = True
    return mask


def drift(values: Sequence[float]) -> float:
    """max/min across refinements; 1.0 means perfectly converged."""
    finite = [float(v) for v in values if np.isfinite(v) and float(v) > 0.0]
    if len(finite) != len(values):
        raise AssertionError(f"non-finite or non-positive values in sweep: {list(values)}")
    return max(finite) / min(finite)


def _kind_from_label(label: object) -> str:
    text = str(label)
    return text.split(":", 1)[0] if ":" in text else "panel"


def _build_shell(
    *,
    layout: str,
    chord: float,
    twist_deg: float,
    thickness: Mapping[str, float],
    naca_dat: str | Path | None = None,
):
    twist = float(np.deg2rad(twist_deg))
    if naca_dat is not None:
        # from_dat labels chord metadata only; .scale() resizes the vertices.
        airfoil = AirfoilSDF.from_dat(str(naca_dat)).scale(float(chord))
    else:
        airfoil = AirfoilSDF.from_naca_series(
            6, 63.0, 4.0, 15.0, n_points=200, chord=chord, closed_te=True
        )
    section = build_section_view(
        airfoil,
        resolve_system_type(layout),
        twist_angle_rad=twist,
        skin_thickness=float(thickness["skin"]),
        web_thickness=float(thickness["web"]),
        cap_height=float(thickness["spar"]),
    )
    shell = shell_mesh_inputs_from_section(
        section, twist_rad=twist, layout_key=layout, n_web_samples=20, n_cap_samples=80
    )
    return shell


def _empty_kind_peaks() -> dict[str, float]:
    return {k: 0.0 for k in KINDS}


def _nan_kind_peaks() -> dict[str, float]:
    return {k: float("nan") for k in KINDS}


def _blank_panel_arrays(n_elem_by_panel: Sequence[int]) -> list[np.ndarray]:
    return [np.full(int(n), np.nan, dtype=float) for n in n_elem_by_panel]


def _grow_panel_arrays(
    stores: Mapping[str, list[np.ndarray]],
    n_elem_by_panel: list[int],
    pi: int,
    ei: int,
) -> None:
    """Grow every store at panel ``pi`` so index ``ei`` is addressable."""
    need = int(ei) + 1
    if need <= int(n_elem_by_panel[pi]):
        return
    for arrs in stores.values():
        old = arrs[pi]
        if old.size >= need:
            continue
        grown = np.full(need, np.nan, dtype=float)
        grown[: old.size] = old
        arrs[pi] = grown
    n_elem_by_panel[pi] = need


def _set_panel_value(
    stores: Mapping[str, list[np.ndarray]],
    n_elem_by_panel: list[int],
    metric: str,
    pi: int,
    ei: int,
    value: float,
    *,
    n_panel: int,
) -> None:
    if not (0 <= pi < n_panel) or not np.isfinite(value):
        return
    _grow_panel_arrays(stores, n_elem_by_panel, pi, ei)
    arr = stores[metric][pi]
    if 0 <= ei < arr.size:
        prev = arr[ei]
        if not np.isfinite(prev) or float(value) > float(prev):
            arr[ei] = float(value)


def _accumulate_kind_peaks(
    per_panel: Sequence[np.ndarray],
    *,
    labels: Sequence[str],
    band: int = ENDPOINT_BAND,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (raw_max, interior_max) by kind from per-panel elemental arrays."""
    raw = _empty_kind_peaks()
    interior = _nan_kind_peaks()
    interior_seen = {k: False for k in KINDS}
    for pi, arr in enumerate(per_panel):
        if pi >= len(labels):
            continue
        kind = _kind_from_label(labels[pi])
        if kind not in raw:
            continue
        vals = np.asarray(arr, dtype=float).ravel()
        finite = vals[np.isfinite(vals)]
        if finite.size:
            raw[kind] = max(raw[kind], float(np.nanmax(finite)))
        ep = endpoint_band_mask(vals.size, band=band)
        interior_vals = vals[~ep]
        interior_vals = interior_vals[np.isfinite(interior_vals)]
        if interior_vals.size:
            if not interior_seen[kind]:
                interior[kind] = float(np.nanmax(interior_vals))
                interior_seen[kind] = True
            else:
                interior[kind] = max(interior[kind], float(np.nanmax(interior_vals)))
    # kinds with only endpoint samples keep nan interior; zero raw stays 0
    for k in KINDS:
        if not interior_seen[k] and raw[k] == 0.0:
            interior[k] = 0.0
    return raw, interior


def _kind_peak_kwargs(
    stores: Mapping[str, list[np.ndarray]],
    *,
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Build CascadeScalars kwargs for every KIND_PEAK_METRICS (+ interiors)."""
    out: dict[str, dict[str, float]] = {}
    for metric in KIND_PEAK_METRICS:
        raw, interior = _accumulate_kind_peaks(stores[metric], labels=labels)
        out[metric] = raw
        out[f"{metric}_interior"] = interior
    return out


def run_cascade_at_frac(
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
) -> CascadeScalars:
    """Build mesh at ``frac``, QA, solve cascade, extract dataclass scalars."""
    thickness = dict(thickness or DEFAULT_THICKNESS)
    load = dict(load or DEFAULT_LOAD)
    shell = _build_shell(
        layout=layout,
        chord=chord,
        twist_deg=twist_deg,
        thickness=thickness,
        naca_dat=naca_dat,
    )
    mesh = build_section_strip_mesh(shell, target_element_length_frac=float(frac))
    report = mesh_quality_report(mesh, target_element_length_frac=float(frac))
    if require_mesh_qa:
        assert_strip_mesh_quality(mesh, target_element_length_frac=float(frac))

    el_min = {k: report.per_kind[k].el_min_over_h for k in report.per_kind}
    el_max = {k: report.per_kind[k].el_max_over_h for k in report.per_kind}
    n_elem = {k: report.per_kind[k].n_elements for k in report.per_kind}

    resultants = BeamResultantVector.from_named(**load)
    runner = SectionLinearStaticRunner(
        panel_topology=shell_panel_topology_from_shell_inputs(shell),
        section_strip_mesh=mesh,
        resultants=resultants,
        section_force_vector=shape_section_force_vector(resultants, shell.midlines),
        load_channel="thinwall",
        cap_tie_mode=cap_tie_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    runner.execute()
    results = runner.get_results()

    d = np.asarray(results.d_global, dtype=float)
    d_inf = float(np.linalg.norm(d, ord=np.inf)) if d.size else float("nan")
    residual_rel = float("nan")
    residual_rel_independent = float("nan")
    gr = None if results.primary_results is None else results.primary_results.global_results
    if gr is not None and gr.residual is not None and gr.f is not None:
        r = np.asarray(gr.residual, dtype=float).ravel()
        f = np.asarray(gr.f, dtype=float).ravel()
        f_norm = max(float(np.linalg.norm(f)), 1e-30)
        residual_rel = float(np.linalg.norm(r) / f_norm)
        reduced = getattr(runner, "reduced", None)
        ind = getattr(reduced, "independent", None) if reduced is not None else None
        if ind is not None:
            idx = np.asarray(ind, dtype=int).ravel()
            if idx.size and int(idx.max()) < r.size:
                residual_rel_independent = float(np.linalg.norm(r[idx]) / f_norm)

    asm = runner.assembly
    panel_maps = list(getattr(asm, "panel_maps", None) or [])
    labels = [str(pm.panel_label) for pm in panel_maps] if panel_maps else list(
        getattr(asm, "lab_p", None) or []
    )
    n_panel = len(labels)
    n_elem_by_panel = [
        len(list(pm.elements or [])) if panel_maps else 0 for pm in panel_maps
    ]

    stores: dict[str, list[np.ndarray]] = {
        m: _blank_panel_arrays(n_elem_by_panel) for m in KIND_PEAK_METRICS
    }

    if results.secondary_results is not None and results.secondary_results.elemental_results:
        for row in results.secondary_results.elemental_results.resultant_rows or []:
            pi = int(row.panel_index)
            ei = int(row.station_index)
            for metric, attr in SECONDARY_ROW_ATTR.items():
                raw = getattr(row, attr, None)
                if raw is None:
                    continue
                _set_panel_value(
                    stores,
                    n_elem_by_panel,
                    metric,
                    pi,
                    ei,
                    abs(float(raw)),
                    n_panel=n_panel,
                )

    tertiary = results.tertiary_results
    panel_index = (
        np.asarray(tertiary.panel_index, dtype=int)
        if tertiary is not None and tertiary.panel_index is not None
        else np.zeros(0, dtype=int)
    )
    elem_index = (
        np.asarray(tertiary.element_index, dtype=int)
        if tertiary is not None and tertiary.element_index is not None
        else np.arange(panel_index.size, dtype=int)
    )

    if tertiary is not None and tertiary.ply_results is not None:
        ply = tertiary.ply_results
        sig = np.asarray(ply.sig_ply_mat, dtype=float)
        mask = (
            np.asarray(ply.trusted_mask, dtype=bool)
            if ply.trusted_mask is not None
            else np.ones(sig.shape[0], dtype=bool)
        )
        tau = (
            np.asarray(ply.tau_iface, dtype=float)
            if ply.tau_iface is not None
            else None
        )
        shear_mask = (
            np.asarray(ply.shear_trusted_mask, dtype=bool)
            if ply.shear_trusted_mask is not None
            else None
        )
        for gi in range(panel_index.size):
            pi = int(panel_index[gi])
            ei = int(elem_index[gi])
            if bool(mask[gi]) and sig.size:
                block = sig[gi, :, FACE_MID, :]
                _set_panel_value(
                    stores,
                    n_elem_by_panel,
                    "peak_mid_sig",
                    pi,
                    ei,
                    float(np.linalg.norm(block.reshape(-1))),
                    n_panel=n_panel,
                )
            if tau is None or gi >= tau.shape[0]:
                continue
            if shear_mask is not None and gi < shear_mask.size and not bool(shear_mask[gi]):
                continue
            # tau_iface: (n_elem, n_iface, 2) → peak |τ13|, |τ23| over interfaces
            iface = tau[gi]
            if iface.ndim != 2 or iface.shape[-1] < 2:
                continue
            t13 = float(np.nanmax(np.abs(iface[:, 0])))
            t23 = float(np.nanmax(np.abs(iface[:, 1])))
            _set_panel_value(
                stores, n_elem_by_panel, "peak_abs_tau13", pi, ei, t13, n_panel=n_panel
            )
            _set_panel_value(
                stores, n_elem_by_panel, "peak_abs_tau23", pi, ei, t23, n_panel=n_panel
            )

    governing_fi = float("nan")
    if results.quaternary_results is not None:
        gov = results.quaternary_results.governing or {}
        if gov.get("fi") is not None:
            governing_fi = float(gov["fi"])
        h = results.quaternary_results.hashin_fi
        if h is not None and tertiary is not None:
            if h.in_plane is not None:
                in_plane = np.asarray(h.in_plane, dtype=float)
                for gi in range(min(panel_index.size, in_plane.shape[0])):
                    row = in_plane[gi]
                    val = (
                        float(np.nanmax(row))
                        if np.any(np.isfinite(row))
                        else float("nan")
                    )
                    _set_panel_value(
                        stores,
                        n_elem_by_panel,
                        "peak_hashin",
                        int(panel_index[gi]),
                        int(elem_index[gi]),
                        val,
                        n_panel=n_panel,
                    )
            if h.interlaminar is not None:
                il = np.asarray(h.interlaminar, dtype=float).ravel()
                for gi in range(min(panel_index.size, il.size)):
                    _set_panel_value(
                        stores,
                        n_elem_by_panel,
                        "peak_hashin_il",
                        int(panel_index[gi]),
                        int(elem_index[gi]),
                        abs(float(il[gi])),
                        n_panel=n_panel,
                    )

    peak_kwargs = _kind_peak_kwargs(stores, labels=labels)
    diag = results.diagnostics
    return CascadeScalars(
        frac=float(frac),
        h_m=float(report.h_m),
        mesh_ok=bool(report.ok),
        layout=str(layout),
        el_min_over_h=el_min,
        el_max_over_h=el_max,
        n_elem=n_elem,
        d_inf=d_inf,
        residual_rel=residual_rel,
        residual_rel_independent=residual_rel_independent,
        governing_fi=governing_fi,
        cascade_trust_ok=bool(diag.cascade_trust_ok) if diag is not None else True,
        solve_residual_rel=None if diag is None else diag.solve_residual_rel,
        interface_constraint_mode=str(interface_constraint_mode),
        cap_tie_mode=str(cap_tie_mode),
        **peak_kwargs,
    )


def sweep_cascade_convergence(
    refinements: Sequence[float] = DEFAULT_REFINEMENTS,
    *,
    cap_tie_mode: str = "skin_offset",
    interface_constraint_mode: str = "shared",
    **kwargs: Any,
) -> list[CascadeScalars]:
    """Run the density ladder and return per-rung scalars."""
    return [
        run_cascade_at_frac(
            frac,
            cap_tie_mode=cap_tie_mode,
            interface_constraint_mode=interface_constraint_mode,
            **kwargs,
        )
        for frac in refinements
    ]


def series_by_kind(
    rows: Sequence[CascadeScalars], attr: str
) -> dict[str, list[float]]:
    """Collect a per-kind dict attribute across the ladder."""
    out: dict[str, list[float]] = {k: [] for k in KINDS}
    for row in rows:
        mapping = getattr(row, attr)
        for k in KINDS:
            out[k].append(float(mapping.get(k, float("nan"))))
    return out


def write_convergence_csv(
    path: str | Path,
    rows: Iterable[CascadeScalars],
) -> Path:
    """Write M+P/S/T/Q ladder rows to CSV (Defect K / multi-key studies)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "layout",
        "frac",
        "h_m",
        "interface_constraint_mode",
        "cap_tie_mode",
        "mesh_ok",
        "cascade_trust_ok",
        "d_inf",
        "residual_rel_unreduced",
        "residual_rel_independent",
        "solve_residual_rel",
        "governing_fi",
    ]
    for kind in KINDS:
        fieldnames.extend(
            [
                f"n_elem_{kind}",
                f"el_min_over_h_{kind}",
                f"el_max_over_h_{kind}",
            ]
        )
        for metric in KIND_PEAK_METRICS:
            fieldnames.append(f"{metric}_{kind}")
            fieldnames.append(f"{metric}_interior_{kind}")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            rec: dict[str, Any] = {
                "layout": getattr(row, "layout", "") or "",
                "frac": row.frac,
                "h_m": row.h_m,
                "interface_constraint_mode": row.interface_constraint_mode,
                "cap_tie_mode": row.cap_tie_mode,
                "mesh_ok": row.mesh_ok,
                "cascade_trust_ok": row.cascade_trust_ok,
                "d_inf": row.d_inf,
                "residual_rel_unreduced": row.residual_rel,
                "residual_rel_independent": row.residual_rel_independent,
                "solve_residual_rel": row.solve_residual_rel,
                "governing_fi": row.governing_fi,
            }
            for kind in KINDS:
                rec[f"n_elem_{kind}"] = row.n_elem.get(kind, 0)
                rec[f"el_min_over_h_{kind}"] = row.el_min_over_h.get(kind, float("nan"))
                rec[f"el_max_over_h_{kind}"] = row.el_max_over_h.get(kind, float("nan"))
                for metric in KIND_PEAK_METRICS:
                    mapping = getattr(row, metric)
                    interior = getattr(row, f"{metric}_interior")
                    rec[f"{metric}_{kind}"] = mapping.get(kind, float("nan"))
                    rec[f"{metric}_interior_{kind}"] = interior.get(kind, float("nan"))
            writer.writerow(rec)
    return path


__all__ = [
    "DEFAULT_REFINEMENTS",
    "ENDPOINT_BAND",
    "KIND_PEAK_METRICS",
    "PEAK_METRIC_LABELS",
    "PRIMARY_SCALAR_METRICS",
    "QUATERNARY_KIND_METRICS",
    "SECONDARY_KIND_METRICS",
    "SECONDARY_ROW_ATTR",
    "TERTIARY_KIND_METRICS",
    "CascadeScalars",
    "drift",
    "endpoint_band_mask",
    "run_cascade_at_frac",
    "series_by_kind",
    "sweep_cascade_convergence",
    "write_convergence_csv",
]
