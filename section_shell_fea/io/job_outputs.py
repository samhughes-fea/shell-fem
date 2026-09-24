"""
Precompute job diagnostics helpers: airfoil conversion and station JSON payloads.

Production orchestration routes through ``SectionLinearStaticRunner`` (P→S→T→Q).
This module retains airfoil conversion utilities and JSON payload builders used
by station diagnostics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea._utils.jsonutil import to_jsonable
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellPanelBucklingAssessment,
    ShellPanelBucklingInput,
    ShellStationBucklingBundle,
)


def airfoil_polygon_to_multicell_airfoil(
    xy: NDArray[np.float64],
    *,
    n_each: int,
    chord_spacing: str = "nested_cosine",
) -> NDArray[np.float64]:
    """
    Convert ``airfoil_vertices_from_spanwise`` output to the thin-wall contract.

    Parametric generators stack an **outer polygon** (upper TE→LE, then lower
    LE→TE, unequal row counts). Thin-wall helpers and
    ``open_outline_from_airfoil`` expect ``[upper_LE→TE; lower_LE→TE]`` with
    **equal** row counts on identical chordwise stations.
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.naca_profile import (
        chordwise_stations,
    )

    v = np.asarray(xy, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 4:
        raise ValueError(f"airfoil_polygon_to_multicell_airfoil: need (N,2) with N>=4; got {v.shape}.")
    ne = int(max(4, int(n_each)))

    le_i = int(np.argmin(v[:, 0]))
    upper_te_le = v[: le_i + 1]
    lower_le_te = v[le_i:]
    if len(upper_te_le) < 2 or len(lower_le_te) < 2:
        raise ValueError("airfoil_polygon_to_multicell_airfoil: degenerate upper/lower split at LE.")
    upper = upper_te_le[::-1].copy()
    lower = lower_le_te.copy()

    def _mean_z_on_unique_x(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(x.astype(np.float64))
        xs = x[order].astype(np.float64)
        zs = z[order].astype(np.float64)
        ux, inv = np.unique(xs, return_inverse=True)
        cnt = np.bincount(inv)
        uz = np.bincount(inv, weights=zs) / np.maximum(cnt, 1)
        return ux, uz

    xu, zu = _mean_z_on_unique_x(upper[:, 0], upper[:, 1])
    xl, zl = _mean_z_on_unique_x(lower[:, 0], lower[:, 1])
    x_le = float(max(xu.min(), xl.min()))
    x_te = float(min(xu.max(), xl.max()))
    span = max(x_te - x_le, 1e-18)
    xc = x_le + chordwise_stations(ne, spacing=chord_spacing) * span

    zu_i = np.interp(xc, xu, zu)
    zl_i = np.interp(xc, xl, zl)
    z_mid_le = float(0.5 * (zu_i[0] + zl_i[0]))
    z_mid_te = float(0.5 * (zu_i[-1] + zl_i[-1]))
    zu_i[0] = z_mid_le
    zl_i[0] = z_mid_le
    zu_i[-1] = z_mid_te
    zl_i[-1] = z_mid_te
    return np.vstack(
        [
            np.column_stack([xc.astype(np.float64), zu_i.astype(np.float64)]),
            np.column_stack([xc.astype(np.float64), zl_i.astype(np.float64)]),
        ]
    )


def naca_digits_to_params(naca_m: float, naca_p: float, naca_xx: float) -> tuple[float, float, float]:
    """
    Map spanwise NACA digits (same convention as spanwise NACA label generation)
    to ``naca_four_digit`` fractions: m, p, t_c.
    """
    mi = int(np.clip(int(round(float(naca_m))), 0, 9))
    pi = int(np.clip(int(round(float(naca_p))), 0, 9))
    xxi = int(np.clip(int(round(float(naca_xx))), 0, 99))
    m = float(mi) / 100.0
    p = float(pi) / 10.0
    t_c = float(xxi) / 100.0
    return m, p, t_c


def build_airfoil_for_station(
    naca_m: float,
    naca_p: float,
    naca_xx: float,
    chord_m: float,
    *,
    naca_series: int = 4,
    n_points: int = 120,
) -> NDArray[np.float64]:
    """NACA polyline (metres chord) for shell recovery; respects ``naca_series`` (4/5/6)."""
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.naca import airfoil_vertices_from_spanwise

    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.naca_profile import (
        naca_four_digit,
    )

    c = float(max(chord_m, 1e-9))
    if int(naca_series) == 4:
        m, p, t_c = naca_digits_to_params(naca_m, naca_p, naca_xx)
        return np.asarray(naca_four_digit(m=m, p=p, t_c=t_c, n=n_points), dtype=np.float64) * c
    raw = np.asarray(
        airfoil_vertices_from_spanwise(
            int(naca_series), float(naca_m), float(naca_p), float(naca_xx), int(n_points), c, closed_te=True
        ),
        dtype=np.float64,
    )
    return airfoil_polygon_to_multicell_airfoil(raw, n_each=n_points)


def _serialize_webs_geom(webs_geom: Any) -> list[list[list[float]]]:
    out: list[list[list[float]]] = []
    for item in webs_geom or []:
        if len(item) != 2:
            continue
        u, w = item
        out.append([np.asarray(u, dtype=float).ravel().tolist(), np.asarray(w, dtype=float).ravel().tolist()])
    return out


def _serialize_booms(booms: Any) -> list[Any]:
    rows: list[Any] = []
    for b in booms or []:
        try:
            a = np.asarray(b, dtype=float)
            if a.size and a.ndim <= 2:
                rows.append(a.tolist())
                continue
        except (TypeError, ValueError):
            pass
        if hasattr(b, "__dict__"):
            d: dict[str, Any] = {}
            for k, v in vars(b).items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (float, int, np.floating, np.integer)):
                    d[k] = float(v)
                elif isinstance(v, np.ndarray):
                    d[k] = v.astype(float).tolist()
                else:
                    d[k] = str(v)
            rows.append(d)
        else:
            rows.append(str(b))
    return rows


def _serialize_panel_polylines(zones: Any, q_tot: Any, sig_p: Any, sig_b: Any) -> list[dict[str, Any]]:
    n = len(zones)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        p = zones[i]
        label = getattr(p, "label", None) or f"panel_{i}"
        s = np.asarray(getattr(p, "s", []), dtype=float)
        nodes = np.asarray(getattr(p, "nodes", []), dtype=float)
        q_i = np.asarray(q_tot[i], dtype=float).ravel() if i < len(q_tot) else np.array([])
        sp_i = np.asarray(sig_p[i], dtype=float).ravel() if i < len(sig_p) else np.array([])
        sb_i = np.asarray(sig_b[i], dtype=float).ravel() if i < len(sig_b) else np.array([])
        rows.append(
            {
                "label": str(label),
                "s_m": s.tolist(),
                "nodes_m": nodes.tolist(),
                "q_N_per_m": q_i.tolist(),
                "sigma_panel_Pa": sp_i.tolist(),
                "sigma_boom_Pa": sb_i.tolist(),
            }
        )
    return rows


def _serialize_mitc4_station_resultants(panel_resultants: list[Any] | None) -> list[dict[str, Any]] | None:
    if not panel_resultants:
        return None
    out: list[dict[str, Any]] = []
    for r in panel_resultants:
        out.append(
            {
                "station_index": int(getattr(r, "station_index", 0)),
                "N11": float(getattr(r, "N11", 0.0)),
                "N22": float(getattr(r, "N22", 0.0)),
                "N12": float(getattr(r, "N12", 0.0)),
                "M11": float(getattr(r, "M11", 0.0)),
                "M22": float(getattr(r, "M22", 0.0)),
                "M12": float(getattr(r, "M12", 0.0)),
                "Q1": float(getattr(r, "Q1", None) or 0.0),
                "Q2": float(getattr(r, "Q2", None) or 0.0),
                "sigma_11_Pa": float(getattr(r, "sigma_11_pa", 0.0)),
                "tau_12_Pa": float(getattr(r, "tau_12_pa", 0.0)),
                "q_n_per_m": float(getattr(r, "q_n_per_m", 0.0)),
                "thickness_m": float(getattr(r, "thickness_m", 0.0)),
                "panel_label": str(getattr(r, "panel_label", "")),
                "panel_index": int(getattr(r, "panel_index", 0)),
            }
        )
    return out


def _failure_assessment_json(bundle: Any, *, strengths: dict[str, float]) -> dict[str, Any]:
    """Serialize quaternary failure assessment; do not recompute FI here."""
    del strengths
    prov_counts: dict[str, int] = {}
    all_res = getattr(bundle, "all_panel_results", None) or []
    for panel_res in all_res:
        for row in panel_res or []:
            if row is None:
                continue
            for _f, p in getattr(row, "provenance", {}).items():
                k = p.kind.value
                prov_counts[k] = prov_counts.get(k, 0) + 1
    q = getattr(bundle, "quaternary_results", None)
    if q is None:
        q = getattr(bundle, "quaternary", None)
    if q is None:
        return {
            "max_fi_hashin": None,
            "max_utilization_governing": None,
            "governing_criterion": None,
            "governing_status": "unavailable",
            "needs_review": True,
            "provenance_breakdown": prov_counts,
            "governing_element": {
                "status": "unavailable",
                "reason": "no_quaternary_results",
            },
            "formulation_id": None,
        }
    gov = dict(getattr(q, "governing", None) or {})
    native = gov.get("fi")
    nr = getattr(q, "needs_review", None)
    needs_review = bool(np.any(nr)) if nr is not None else False
    return {
        "max_fi_hashin": None if native is None else float(native),
        "max_utilization_governing": getattr(q, "max_utilization_governing", None),
        "governing_criterion": gov.get("criterion_id") or gov.get("criterion"),
        "governing_status": gov.get("status"),
        "needs_review": needs_review,
        "provenance_breakdown": prov_counts,
        "governing_element": gov,
        "formulation_id": getattr(q, "formulation_id", None),
    }


def build_section_shell_station_json_payload(
    *,
    station_tag: str,
    airfoil: NDArray[np.float64],
    spars: list[float],
    n_elements_per_panel: int,
    bundle: Any,
    strengths: dict[str, float],
    clpt_reference_skin: Any,
    all_panel_results: list[Any],
    all_panel_fi: list[np.ndarray],
    panel_labels_m: list[str],
    sig_omega_reference_panel: np.ndarray | None,
    station_buckling: ShellStationBucklingBundle | None = None,
) -> Any:
    """Structured data backing the section_shell_fea PNG bundle (SI units unless noted)."""
    p0 = bundle.zones[0]
    s_ref = np.asarray(p0.s, dtype=float)
    q_ref = np.asarray(bundle.q_tot[0], dtype=float).ravel()
    sig_ref = np.asarray(bundle.sig_p[0], dtype=float).ravel()
    n = min(len(s_ref), len(q_ref), len(sig_ref))
    s_ref = s_ref[:n]
    q_ref = q_ref[:n]
    sig_ref = sig_ref[:n]
    s_mids: list[float] | None = None
    sig_omega_mpa: list[float] | None = None
    if sig_omega_reference_panel is not None and len(sig_omega_reference_panel) > 0 and n >= 2:
        s_mids_arr = 0.5 * (s_ref[:-1] + s_ref[1:])
        n_om = min(len(s_mids_arr), len(sig_omega_reference_panel))
        s_mids = [float(s_mids_arr[i]) for i in range(n_om)]
        sig_omega_mpa = [float(sig_omega_reference_panel[i]) / 1e6 for i in range(n_om)]

    cr = clpt_reference_skin
    clpt_json: dict[str, Any] = {
        "fi_per_ply": np.asarray(cr.fi, dtype=float).tolist(),
        "eps0": np.asarray(cr.eps0, dtype=float).tolist(),
        "kappa": np.asarray(cr.kappa, dtype=float).tolist(),
        "N_vec_N_per_m": np.asarray(cr.N_vec, dtype=float).tolist(),
        "M_vec_Nmm_per_m": np.asarray(cr.M_vec, dtype=float).tolist(),
        "sig_lam_mid_Pa": [np.asarray(x, dtype=float).tolist() for x in cr.sig_lam_mid],
        "eps_lam_mid": [np.asarray(x, dtype=float).tolist() for x in cr.eps_lam_mid],
    }

    fi_per_panel = [np.asarray(fi, dtype=float).tolist() for fi in all_panel_fi]

    def _serialize_shell_panel_buckling_input(panel: ShellPanelBucklingInput) -> dict[str, Any]:
        return {
            "panel_label": panel.panel_label,
            "panel_index": panel.panel_index,
            "station_index": panel.station_index,
            "length_m": float(panel.length_m),
            "width_m": float(panel.width_m),
            "thickness_m": float(panel.thickness_m),
            "abd": None if panel.abd is None else np.asarray(panel.abd, dtype=float).tolist(),
            "membrane_resultants": np.asarray(panel.membrane_resultants, dtype=float).tolist(),
            "bending_resultants": np.asarray(panel.bending_resultants, dtype=float).tolist(),
            "transverse_shear_resultants": None if panel.transverse_shear_resultants is None else np.asarray(panel.transverse_shear_resultants, dtype=float).tolist(),
            "boundary_condition": panel.boundary_condition,
            "material_label": panel.material_label,
            "provenance": {
                "analysis_regime": panel.provenance.analysis_regime.value,
                "shell_theory": panel.provenance.shell_theory.value,
                "mesh_n_elements_per_panel": panel.provenance.mesh_n_elements_per_panel,
                "solver_version": panel.provenance.solver_version,
                "source": panel.provenance.source,
            },
            "source_resultant": None if panel.source_resultant is None else {
                "N11": float(panel.source_resultant.N11),
                "N22": float(panel.source_resultant.N22),
                "N12": float(panel.source_resultant.N12),
                "M11": float(panel.source_resultant.M11),
                "M22": float(panel.source_resultant.M22),
                "M12": float(panel.source_resultant.M12),
                "Q1": panel.source_resultant.Q1,
                "Q2": panel.source_resultant.Q2,
                "sigma_11_pa": float(panel.source_resultant.sigma_11_pa),
                "tau_12_pa": float(panel.source_resultant.tau_12_pa),
                "q_n_per_m": float(panel.source_resultant.q_n_per_m),
                "thickness_m": float(panel.source_resultant.thickness_m),
            },
        }

    def _serialize_shell_panel_buckling_assessment(panel: ShellPanelBucklingAssessment) -> dict[str, Any]:
        return {
            "panel_label": panel.panel_label,
            "panel_index": panel.panel_index,
            "station_index": panel.station_index,
            "load_multiplier_min": panel.load_multiplier_min,
            "load_multipliers": np.asarray(panel.load_multipliers, dtype=float).tolist(),
            "n_modes_found": int(panel.n_modes_found),
            "is_compressive": bool(panel.is_compressive),
            "buckling_index": panel.buckling_index,
            "governing_mode": panel.governing_mode,
            "governing_resultant": panel.governing_resultant,
            "status": panel.status,
            "diagnostics": dict(panel.diagnostics),
        }

    def _serialize_shell_station_buckling_bundle(bundle: ShellStationBucklingBundle | None) -> dict[str, Any] | None:
        if bundle is None:
            return None
        governing = bundle.governing_panel_result()
        return {
            "station_id": bundle.station_id,
            "station_index": bundle.station_index,
            "z_station": float(bundle.z_station),
            "analysis_regime": bundle.analysis_regime.value,
            "shell_theory": bundle.shell_theory.value,
            "governing_panel": {
                "panel_label": bundle.governing_panel_label,
                "panel_index": bundle.governing_panel_index,
                "buckling_index": bundle.governing_buckling_index,
                "assessment": None if governing is None else _serialize_shell_panel_buckling_assessment(governing),
            },
            "provenance": {
                "analysis_regime": bundle.provenance.analysis_regime.value,
                "shell_theory": bundle.provenance.shell_theory.value,
                "mesh_n_elements_per_panel": bundle.provenance.mesh_n_elements_per_panel,
                "solver_version": bundle.provenance.solver_version,
                "source": bundle.provenance.source,
            },
            "metadata": dict(bundle.metadata),
            "panel_inputs": [_serialize_shell_panel_buckling_input(panel) for panel in bundle.panel_inputs],
            "panel_results": [_serialize_shell_panel_buckling_assessment(panel) for panel in bundle.panel_results],
        }

    audit = _failure_assessment_json(bundle, strengths=strengths)
    payload: dict[str, Any] = {
        "schema": "section_shell_station_v1",
        "station_tag": station_tag,
        "unit_section_resultants": {
            "N_N": 1.0,
            "Vy_N": 1.0,
            "Vz_N": 1.0,
            "My_Nm": 1.0,
            "Mz_Nm": 1.0,
            "T_Nm": 1.0,
            "B_Nm2": 0.0,
            "dB_dx_Nm": 0.0,
        },
        "n_elements_per_panel": int(n_elements_per_panel),
        "reference_panel_index": 0,
        "airfoil_m": np.asarray(airfoil, dtype=float).tolist(),
        "spar_chord_fracs": [float(x) for x in spars],
        "hashin_strengths_Pa": {k: float(v) for k, v in strengths.items()},
        "thin_wall": {
            "y_e_m": float(getattr(bundle, "y_e", bundle.y_sc)),
            "z_e_m": float(getattr(bundle, "z_e", bundle.z_sc)),
            "y_sc_m": float(bundle.y_sc),  # deprecated alias of y_e (elastic centroid)
            "z_sc_m": float(bundle.z_sc),
            "webs_geom": _serialize_webs_geom(bundle.webs_geom),
            "booms": _serialize_booms(bundle.booms),
            "panels": _serialize_panel_polylines(bundle.zones, bundle.q_tot, bundle.sig_p, bundle.sig_b),
        },
        "reference_panel_along_contour": {
            "s_m": s_ref.tolist(),
            "q_N_per_m": q_ref.tolist(),
            "sigma_11_panel_Pa": sig_ref.tolist(),
            "sigma_omega_Vlasov_mids_m": s_mids,
            "sigma_omega_Vlasov_mids_MPa": sig_omega_mpa,
        },
        "clpt_reference_skin_station": clpt_json,
        "mitc4_panel0_resultants_by_element": _serialize_mitc4_station_resultants(
            all_panel_results[0] if all_panel_results else None
        ),
        "mitc4_clpt_hashin_fi_max_over_plies_by_panel": {
            "panel_labels": list(panel_labels_m),
            "fi_by_panel": fi_per_panel,
        },
        "failure_assessment": audit,
        "fsdt_periodic_hashin_audit": audit,
        "shell_station_buckling": _serialize_shell_station_buckling_bundle(station_buckling),
        "panel_buckling": _serialize_shell_station_buckling_bundle(station_buckling),
    }
    return to_jsonable(payload)
