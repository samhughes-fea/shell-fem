"""Panel equilibrium checks and load-reaction audit for MITC4 shell recovery."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ....pre_processing.containers.types import ShellPanelResultants


def _edge_traction_diagnostics_enabled() -> bool:
    """Opt-out via ``SECTION_SHELL_EDGE_TRACTION_DIAG=0`` on hot orchestration paths."""
    return os.environ.get("SECTION_SHELL_EDGE_TRACTION_DIAG", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _panel_endpoint(panel: Any, which: str) -> np.ndarray:
    nodes = np.asarray(getattr(panel, "nodes", []), dtype=float)
    if len(nodes) == 0:
        return np.array([np.nan, np.nan], dtype=float)
    return nodes[0] if which == "start" else nodes[-1]


def _panel_end_tangent(panel: Any, which: str) -> np.ndarray:
    nodes = np.asarray(getattr(panel, "nodes", []), dtype=float)
    if len(nodes) < 2:
        return np.array([1.0, 0.0], dtype=float)
    if which == "start":
        t = nodes[1] - nodes[0]
    else:
        t = nodes[-1] - nodes[-2]
    n = float(np.linalg.norm(t))
    if n < 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return t / n


def _normal_2d(t: np.ndarray) -> np.ndarray:
    """90° CCW rotation of a 2-D tangent unit vector → panel outward normal in (Y,Z)."""
    return np.array([-float(t[1]), float(t[0])], dtype=float)


def _classify_boundary(label_i: str, label_j: str) -> str:
    is_web_i = "web" in label_i.lower()
    is_web_j = "web" in label_j.lower()
    if is_web_i and is_web_j:
        return "web-web"
    if is_web_i or is_web_j:
        return "skin-web"
    return "skin-skin"


def _thresholds_for_boundary(boundary_type: str) -> tuple[float, float]:
    if boundary_type == "skin-skin":
        return 0.05, 0.05
    if boundary_type == "skin-web":
        return 0.10, 0.10
    return 0.10, 0.10


def _secondary_thresholds_for_boundary(boundary_type: str) -> tuple[float, float]:
    if boundary_type == "skin-skin":
        return 0.10, 0.10
    if boundary_type == "skin-web":
        return 0.15, 0.15
    return 0.15, 0.15


def _diag_boundary_force(diag: dict | None, which: str) -> tuple[float, float]:
    if not diag:
        return 0.0, 0.0
    br = diag.get("boundary_reaction_set", diag.get("boundary_reaction", {}))
    side = br.get(which, {})
    return float(side.get("Fx", 0.0)), float(side.get("Fs", 0.0))


def _diag_boundary_field(diag: dict | None, which: str) -> tuple[float, float]:
    if not diag:
        return 0.0, 0.0
    fs = diag.get("interface_field_set", {})
    side = fs.get(which, {})
    return float(side.get("Nx", 0.0)), float(side.get("Nxy", 0.0))


def _diag_boundary_edge_traction(diag: dict | None, which: str) -> tuple[float, float]:
    if not diag:
        return 0.0, 0.0
    es = diag.get("interface_edge_set", {})
    side = es.get(which, {})
    return float(side.get("Tx_int", 0.0)), float(side.get("Ts_int", 0.0))


def _diag_boundary_edge_traction_stats(diag: dict | None, which: str) -> dict[str, float]:
    """
    Return mean and std of Tx near the interface boundary.

    Uses near-boundary element samples (Tx_near) stored by the global solve.
    Falls back to the single Tx_int value when samples are absent.
    """
    if not diag:
        return {"Tx_mean": 0.0, "Tx_std": 0.0, "Tx_std_rel": 0.0, "n_samples": 0}
    es = diag.get("interface_edge_set", {})
    side = es.get(which, {})
    near = side.get("Tx_near", None)
    tx_int = float(side.get("Tx_int", 0.0))
    if near and len(near) > 0:
        arr = np.asarray(near, dtype=float)
    else:
        arr = np.array([tx_int], dtype=float)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr)) if len(arr) > 1 else 0.0
    std_rel = std_v / max(abs(mean_v), 1.0)
    return {"Tx_mean": mean_v, "Tx_std": std_v, "Tx_std_rel": std_rel, "n_samples": len(arr)}


def _is_web_label(label: str) -> bool:
    return "web" in label.lower()


def _interface_traction_residuals(
    tx_i: float,
    ts_i: float,
    tx_j: float,
    ts_j: float,
    t_i: np.ndarray,
    t_j: np.ndarray,
    t_ref: np.ndarray | None = None,
    n_ref: np.ndarray | None = None,
    scale_floor: float = 1.0,
) -> dict[str, float]:
    """
    Compute physically correct interface traction residuals.

    Each panel's local frame has X = span (common to all panels) and ŝ = contour
    tangent (panel-specific, lives in the (Y,Z) cross-section plane).

    ``mitc4_edge_shear_traction_integrated`` already applies the outward-normal sign
    (normal_sign = ±1 for start/end edges), so:

      tx_* = Nxy * normal_sign  — spanwise (X) traction component
      ts_* = Ny  * normal_sign  — contour (ŝ) traction component

    Interface equilibrium (Newton's 3rd law, sum = 0 over all panels at junction):
      X-channel : Tx_i + Tx_j = 0   (X is shared — no rotation needed)
      YZ-channel: Ts_i * ŝ_i + Ts_j * ŝ_j = 0  (vector sum in (Y,Z))

    Parameters
    ----------
    tx_i, tx_j   : spanwise tractions (already outward-normal signed)
    ts_i, ts_j   : contour tractions (already outward-normal signed)
    t_i, t_j     : 2-D (Y,Z) unit contour tangent vectors at the junction endpoint
    t_ref, n_ref : cluster reference tangent/normal (2-D); used for component
                   decomposition in the scaling denominator.  When None, t_i is used.
    scale_floor  : minimum scale for relative normalisation — should be set to the
                   maximum single-panel traction magnitude in the cluster so the
                   denominator is not inflated by nearly-orthogonal tangents.
    """
    dTx = float(tx_i) + float(tx_j)
    t_i_arr = np.asarray(t_i, dtype=float)
    t_j_arr = np.asarray(t_j, dtype=float)
    dT_yz = float(ts_i) * t_i_arr + float(ts_j) * t_j_arr
    dT_yz_mag = float(np.linalg.norm(dT_yz))

    # Use total traction magnitude (sqrt(Tx^2+Ts^2)) for the scaling denominator.
    # This avoids denominator inflation when the two panel tangents are nearly
    # orthogonal: a small YZ residual relative to the panel traction is meaningful
    # even if individual Ts components are large with opposite projections.
    T_i_mag = float(np.sqrt(float(tx_i) ** 2 + float(ts_i) ** 2))
    T_j_mag = float(np.sqrt(float(tx_j) ** 2 + float(ts_j) ** 2))
    scale_Ts = max(T_i_mag, T_j_mag, scale_floor)

    return {
        "Tx_i": float(tx_i),
        "Tx_j": float(tx_j),
        "Ts_i": float(ts_i),
        "Ts_j": float(ts_j),
        "dTx": dTx,
        "dT_yz_y": float(dT_yz[0]),
        "dT_yz_z": float(dT_yz[1]) if len(dT_yz) > 1 else 0.0,
        "dT_yz_mag": dT_yz_mag,
        "scale_Ts": scale_Ts,
    }


def _secondary_traction_thresholds(boundary_type: str) -> tuple[float, float]:
    """Return (tol_dTx, tol_dT_yz) acceptance thresholds for the traction-vector check."""
    if boundary_type == "skin-skin":
        return 0.05, 0.10
    if boundary_type == "skin-web":
        return 0.10, 0.15
    return 0.10, 0.15


def _build_endpoint_clusters_raw(panels: list, tol: float = 1e-6) -> list[list[dict]]:
    """
    Single-source clustering of panel endpoints by geometric proximity.

    Returns list of clusters; each cluster is [{pi, end, pt}, ...].
    Both _build_geometric_interfaces and _build_endpoint_clusters use this
    helper to guarantee topology consistency.
    """
    endpoints: list[dict] = []
    for pi, p in enumerate(panels):
        for which in ("start", "end"):
            pt = _panel_endpoint(p, which)
            if np.any(~np.isfinite(pt)):
                continue
            endpoints.append({"pi": pi, "end": which, "pt": pt})
    clusters: list[list[dict]] = []
    for ep in endpoints:
        placed = False
        for c in clusters:
            if float(np.linalg.norm(ep["pt"] - c[0]["pt"])) <= tol:
                c.append(ep)
                placed = True
                break
        if not placed:
            clusters.append([ep])
    return clusters


def _build_geometric_interfaces(panels: list, tol: float = 1e-6) -> list[dict]:
    """
    Build interface pairs from geometric endpoint proximity.

    Returns list of dicts: {pi, pj, end_i, end_j, point, cluster_id, cluster_size}.
    cluster_size is the total number of panel-endpoints meeting at that junction
    (2 for a 2-way interface, 3 for a T-junction, etc.).
    """
    clusters = _build_endpoint_clusters_raw(panels, tol)

    interfaces: list[dict] = []
    seen_pairs: set[tuple[int, str, int, str]] = set()
    for cid, c in enumerate(clusters):
        cluster_size = len(c)
        if cluster_size < 2:
            continue
        for i in range(cluster_size):
            for j in range(i + 1, cluster_size):
                ei, ej = c[i], c[j]
                if ei["pi"] == ej["pi"]:
                    continue
                key = (int(ei["pi"]), str(ei["end"]), int(ej["pi"]), str(ej["end"]))
                key_rev = (key[2], key[3], key[0], key[1])
                if key in seen_pairs or key_rev in seen_pairs:
                    continue
                seen_pairs.add(key)
                interfaces.append(
                    {
                        "pi": int(ei["pi"]),
                        "pj": int(ej["pi"]),
                        "end_i": str(ei["end"]),
                        "end_j": str(ej["end"]),
                        "point": 0.5 * (ei["pt"] + ej["pt"]),
                        "cluster_id": int(cid),
                        "cluster_size": int(cluster_size),
                    }
                )
    interfaces.sort(key=lambda d: (d["pi"], d["pj"], d["end_i"], d["end_j"]))
    return interfaces


def _build_endpoint_clusters(panels: list, tol: float = 1e-6) -> list[list[dict]]:
    """Cluster coincident panel endpoints: [{pi,end,pt}, ...] per junction."""
    return _build_endpoint_clusters_raw(panels, tol)


def check_panel_equilibrium(
    all_panel_results: list[list[ShellPanelResultants]],
    panels: list,
    *,
    all_panel_diagnostics: list[dict] | None = None,
    endpoint_tol: float = 1e-6,
) -> list[dict]:
    """
    Compare Nx and Nxy at geometrically shared panel boundaries.

    Interface pairs are detected from endpoint proximity rather than index adjacency.
    For each pair, compares resultants at corresponding interface ends.

    Returns
    -------
    list of dicts with dual metrics:
    - reaction-based interface mismatch (authoritative)
    - resultant-based mismatch (secondary diagnostic)
    """
    results: list[dict] = []
    interfaces = _build_geometric_interfaces(panels, tol=endpoint_tol)
    # Build a common traction-frame tangent per geometric junction cluster.
    cluster_tangent: dict[tuple[int, ...], np.ndarray] = {}
    endpoint_sets: list[set[tuple[int, str]]] = []
    for iface in interfaces:
        a = (int(iface["pi"]), str(iface["end_i"]))
        b = (int(iface["pj"]), str(iface["end_j"]))
        merged = False
        for s in endpoint_sets:
            if a in s or b in s:
                s.add(a)
                s.add(b)
                merged = True
                break
        if not merged:
            endpoint_sets.append({a, b})
    for s in endpoint_sets:
        key = tuple(sorted(hash(x) for x in s))
        t_mats: list[np.ndarray] = []
        for pi, end in s:
            t = _panel_end_tangent(panels[pi], end)
            t_mats.append(np.outer(t, t))
        if not t_mats:
            cluster_tangent[key] = np.array([1.0, 0.0], dtype=float)
            continue
        M = np.sum(np.stack(t_mats, axis=0), axis=0)
        evals, evecs = np.linalg.eigh(M)
        t_ref = np.asarray(evecs[:, int(np.argmax(evals))], dtype=float)
        if float(np.linalg.norm(t_ref)) < 1e-12:
            t_ref = np.array([1.0, 0.0], dtype=float)
        if float(t_ref[0]) < 0.0:
            t_ref = -t_ref
        cluster_tangent[key] = t_ref / max(float(np.linalg.norm(t_ref)), 1e-12)

    def _cluster_key(a: tuple[int, str], b: tuple[int, str]) -> tuple[int, ...]:
        for s in endpoint_sets:
            if a in s and b in s:
                return tuple(sorted(hash(x) for x in s))
        return tuple(sorted((hash(a), hash(b))))

    for iface in interfaces:
        pi = int(iface["pi"])
        pj = int(iface["pj"])
        end_i = str(iface["end_i"])
        end_j = str(iface["end_j"])

        if pi >= len(all_panel_results) or pj >= len(all_panel_results):
            continue
        res_i = all_panel_results[pi]
        res_j = all_panel_results[pj]
        if not res_i or not res_j:
            continue

        r_i = res_i[0] if end_i == "start" else res_i[-1]
        r_j = res_j[0] if end_j == "start" else res_j[-1]

        label_i = getattr(panels[pi], "label", None) or f"panel_{pi}"
        label_j = getattr(panels[pj], "label", None) or f"panel_{pj}"

        t_i = _panel_end_tangent(panels[pi], end_i)
        t_j = _panel_end_tangent(panels[pj], end_j)
        key = _cluster_key((pi, end_i), (pj, end_j))
        t_ref = cluster_tangent.get(key, t_i)
        sign_i = 1.0 if float(np.dot(t_i, t_ref)) >= 0.0 else -1.0
        sign_j = 1.0 if float(np.dot(t_j, t_ref)) >= 0.0 else -1.0
        orient = "same" if sign_i == sign_j else "opposite"
        nxy_sign = sign_i / sign_j

        Nx_a = float(r_i.Nx)
        Nx_b = float(r_j.Nx)
        Nxy_a = float(r_i.Nxy)
        Nxy_b = float(r_j.Nxy)

        Nx_a_norm = Nx_a
        Nx_b_norm = Nx_b
        Nxy_a_norm = Nxy_a
        Nxy_b_norm = nxy_sign * Nxy_b

        dNx_abs = abs(Nx_a_norm - Nx_b_norm)
        dNxy_abs = abs(Nxy_a_norm - Nxy_b_norm)
        scale_Nx = max(abs(r_i.Nx), abs(r_j.Nx), 1.0)
        scale_Nxy = max(abs(r_i.Nxy), abs(r_j.Nxy), 1.0)

        dNx_rel_resultant = dNx_abs / scale_Nx
        dNxy_rel_resultant = dNxy_abs / scale_Nxy
        boundary_type = _classify_boundary(str(label_i), str(label_j))
        tol_react_nx, tol_react_nxy = _thresholds_for_boundary(boundary_type)
        tol_res_nx, tol_res_nxy = _secondary_thresholds_for_boundary(boundary_type)

        diag_i = None
        diag_j = None
        if all_panel_diagnostics is not None:
            if pi < len(all_panel_diagnostics):
                diag_i = all_panel_diagnostics[pi]
            if pj < len(all_panel_diagnostics):
                diag_j = all_panel_diagnostics[pj]
        Fx_i, Fs_i = _diag_boundary_force(diag_i, end_i)
        Fx_j, Fs_j = _diag_boundary_force(diag_j, end_j)
        mode_i = str((diag_i or {}).get("interface_constraint_mode", "shared"))
        mode_j = str((diag_j or {}).get("interface_constraint_mode", "shared"))
        _cluster_basis_modes = {"transformed_basis", "shared_rotated"}
        if _cluster_basis_modes & {mode_i, mode_j}:
            # Cluster-basis modes (transformed_basis, shared_rotated) store outward-
            # normal-signed edge tractions as Fx (spanwise) and Fs (contour).
            # Spanwise X-axis is common to all panels → Newton III: Fx_i + Fx_j = 0
            # with NO orientation mapping on the spanwise channel.
            # Contour ŝ is panel-specific → nxy_sign maps slave ŝ onto master ŝ.
            Fs_j_norm = nxy_sign * Fs_j
            dFx_rel = abs(Fx_i + Fx_j) / max(abs(Fx_i), abs(Fx_j), 1.0)
            dFs_rel = abs(Fs_i + Fs_j_norm) / max(abs(Fs_i), abs(Fs_j_norm), 1.0)
        else:
            Fx_j_norm = Fx_j
            Fs_j_norm = nxy_sign * Fs_j
            dFx_rel = abs(Fx_i - Fx_j_norm) / max(abs(Fx_i), abs(Fx_j_norm), 1.0)
            dFs_rel = abs(Fs_i - Fs_j_norm) / max(abs(Fs_i), abs(Fs_j_norm), 1.0)
        pass_react_nx = dFx_rel <= tol_react_nx
        pass_react_nxy = dFs_rel <= tol_react_nxy

        # Secondary field continuity.
        # Nx (spanwise): directly comparable across panels — X is common to all.
        # Nxy/traction: use geometry-based traction-vector residuals when edge data
        # available; fall back to field-set Nxy (orientation-normalised) or centroid.
        fNx_i, fNxy_i = _diag_boundary_field(diag_i, end_i)
        fNx_j, fNxy_j = _diag_boundary_field(diag_j, end_j)
        have_field = (diag_i is not None and diag_j is not None and
                      "interface_field_set" in diag_i and "interface_field_set" in diag_j)
        have_edge = (diag_i is not None and diag_j is not None and
                     "interface_edge_set" in diag_i and "interface_edge_set" in diag_j)

        # Nx: spanwise component, no rotation needed.
        if have_field or have_edge:
            dNx_rel_field = abs(fNx_i - fNx_j) / max(abs(fNx_i), abs(fNx_j), 1.0)
        else:
            dNx_rel_field = dNx_rel_resultant

        trac_res: dict[str, float]
        if have_edge:
            # Geometry-only traction vector residuals (see _interface_traction_residuals).
            tx_i_val, ts_i_val = _diag_boundary_edge_traction(diag_i, end_i)
            tx_j_val, ts_j_val = _diag_boundary_edge_traction(diag_j, end_j)
            # scale_floor: max single-panel traction magnitude so orthogonal-tangent
            # inflation is avoided in the normalisation denominator.
            _scale_floor = max(
                float(np.sqrt(tx_i_val ** 2 + ts_i_val ** 2)),
                float(np.sqrt(tx_j_val ** 2 + ts_j_val ** 2)),
                1.0,
            )
            trac_res = _interface_traction_residuals(
                tx_i_val, ts_i_val, tx_j_val, ts_j_val, t_i, t_j,
                t_ref=t_ref, n_ref=_normal_2d(t_ref),
                scale_floor=_scale_floor,
            )
            scale_Tx = max(abs(trac_res["Tx_i"]), abs(trac_res["Tx_j"]), 1.0)
            scale_Ts = trac_res.get("scale_Ts", max(abs(trac_res["Ts_i"]), abs(trac_res["Ts_j"]), 1.0))
            dTx_rel = abs(trac_res["dTx"]) / scale_Tx
            dT_yz_rel = trac_res["dT_yz_mag"] / scale_Ts
            nxy_source = "traction-vector-strict"
        elif have_field:
            # Field-set Nxy: orientation-normalised comparison.
            fNxy_j_norm = nxy_sign * fNxy_j
            scale_fNxy = max(abs(fNxy_i), abs(fNxy_j_norm), 1.0)
            dTx_rel = abs(fNxy_i - fNxy_j_norm) / scale_fNxy
            dT_yz_rel = 0.0
            nxy_source = "field-fallback"
            trac_res = {
                "Tx_i": float(fNxy_i), "Tx_j": float(fNxy_j),
                "Ts_i": 0.0, "Ts_j": 0.0,
                "dTx": float(fNxy_i - fNxy_j_norm),
                "dT_yz_y": 0.0, "dT_yz_z": 0.0, "dT_yz_mag": 0.0,
            }
        else:
            dTx_rel = dNxy_rel_resultant
            dT_yz_rel = 0.0
            nxy_source = "centroid-fallback"
            trac_res = {
                "Tx_i": float(Nxy_a), "Tx_j": float(Nxy_b),
                "Ts_i": 0.0, "Ts_j": 0.0,
                "dTx": float(Nxy_a - Nxy_b_norm),
                "dT_yz_y": 0.0, "dT_yz_z": 0.0, "dT_yz_mag": 0.0,
            }

        dNxy_rel_field = dTx_rel
        tol_res_tx, tol_res_tyz = _secondary_traction_thresholds(boundary_type)
        pass_res_nx = dNx_rel_field <= tol_res_nx
        pass_res_nxy = dTx_rel <= tol_res_tx
        pass_dT_yz = dT_yz_rel <= tol_res_tyz

        results.append({
            "pi": pi,
            "pj": pj,
            "end_i": end_i,
            "end_j": end_j,
            "label_i": str(label_i),
            "label_j": str(label_j),
            "cluster_id": int(iface.get("cluster_id", -1)),
            "cluster_size": int(iface.get("cluster_size", 2)),
            "orientation": orient,
            "boundary_type": boundary_type,
            "Nx_a": Nx_a,
            "Nx_b": Nx_b,
            "Nxy_a": Nxy_a,
            "Nxy_b": Nxy_b,
            "Nx_a_norm": Nx_a_norm,
            "Nx_b_norm": Nx_b_norm,
            "Nxy_a_norm": Nxy_a_norm,
            "Nxy_b_norm": Nxy_b_norm,
            "reaction_Nx_i": Fx_i,
            "reaction_Nx_j": Fx_j,
            "reaction_Nxy_i": Fs_i,
            "reaction_Nxy_j": Fs_j,
            "reaction_dNx_rel": dFx_rel,
            "reaction_dNxy_rel": dFs_rel,
            "reaction_tol_nx": tol_react_nx,
            "reaction_tol_nxy": tol_react_nxy,
            "reaction_pass_nx": pass_react_nx,
            "reaction_pass_nxy": pass_react_nxy,
            "reaction_pass": bool(pass_react_nx and pass_react_nxy),
            "field_Nx_i": fNx_i,
            "field_Nx_j": fNx_j,
            "field_Nxy_i": fNxy_i,
            "field_Nxy_j": fNxy_j,
            "resultant_dNx_rel": dNx_rel_field,
            "resultant_dNxy_rel": dNxy_rel_field,
            "resultant_nxy_source": nxy_source,
            "nxy_compare_mode": nxy_source,
            # Physics-correct traction-vector continuity (geometry-based).
            "dTx_rel": dTx_rel,
            "dT_yz_rel": dT_yz_rel,
            "dT_yz_mag": trac_res["dT_yz_mag"],
            # Backward-compat aliases for run_example display (Phase D will rename columns).
            "t_t_i": trac_res["Tx_i"],
            "t_t_j": trac_res["Tx_j"],
            "t_n_i": trac_res["Ts_i"],
            "t_n_j": trac_res["Ts_j"],
            "resultant_dTn_rel": dT_yz_rel,
            "resultant_dNx_rel_centroid": dNx_rel_resultant,
            "resultant_dNxy_rel_centroid": dNxy_rel_resultant,
            "resultant_tol_nx": tol_res_nx,
            "resultant_tol_nxy": tol_res_nxy,
            "resultant_tol_dTx": tol_res_tx,
            "resultant_tol_dT_yz": tol_res_tyz,
            "resultant_pass_nx": pass_res_nx,
            "resultant_pass_nxy": pass_res_nxy,
            "resultant_pass_dT_yz": pass_dT_yz,
            "resultant_pass": bool(pass_res_nx and pass_res_nxy),
        })
    return results


def check_cluster_equilibrium(
    panels: list,
    *,
    all_panel_diagnostics: list[dict] | None = None,
    endpoint_tol: float = 1e-6,
) -> list[dict]:
    """
    Cluster-level traction equilibrium (supports 2-way and 3-way junctions).

    Residuals:
      Tx_sum   = Σ Tx_int
      T_yz_sum = Σ Ts_int * s_hat
    """
    clusters = _build_endpoint_clusters(panels, tol=endpoint_tol)
    if not _edge_traction_diagnostics_enabled():
        return []
    if not all_panel_diagnostics:
        return []
    out: list[dict] = []
    for cid, c in enumerate(clusters):
        if len(c) < 2:
            continue
        tx_terms: list[float] = []
        ts_terms: list[float] = []
        t_sum = np.zeros(2, dtype=float)
        members: list[str] = []
        # Compute collinearity of this cluster (5° tolerance).
        _tangents = [_panel_end_tangent(panels[int(ep["pi"])], str(ep["end"])) for ep in c]
        _collinear = True
        if len(_tangents) >= 2:
            _t0 = _tangents[0]
            _tol_rad = 5.0 * np.pi / 180.0
            for _t in _tangents[1:]:
                if float(np.arccos(np.clip(abs(float(np.dot(_t0, _t))), 0.0, 1.0))) > _tol_rad:
                    _collinear = False
                    break
        # Build members first so they appear even when diagnostics are missing.
        for ep in c:
            pi = int(ep["pi"])
            end = str(ep["end"])
            lbl = str(getattr(panels[pi], "label", None) or f"panel_{pi}")
            members.append(f"{lbl}:{end}")
            if pi >= len(all_panel_diagnostics):
                continue
            di = all_panel_diagnostics[pi] or {}
            tx, ts = _diag_boundary_edge_traction(di, end)
            t = _panel_end_tangent(panels[pi], end)
            t_sum = t_sum + ts * t
            tx_terms.append(float(tx))
            ts_terms.append(float(ts))
        if not tx_terms:
            # Cluster exists geometrically but edge traction data is unavailable
            # for all member panels — emit a diagnostic row rather than silently
            # dropping so the missing junction is visible in the output.
            out.append(
                {
                    "cluster_id": int(cid),
                    "n_panels": int(len(c)),
                    "members": members,
                    "cluster_collinear": bool(_collinear),
                    "status": "insufficient_data",
                    "Tx_sum": float("nan"),
                    "T_yz_sum_y": float("nan"),
                    "T_yz_sum_z": float("nan"),
                    "Tx_rel_cluster": float("nan"),
                    "T_yz_rel_cluster": float("nan"),
                }
            )
            continue
        tx_sum = float(np.sum(tx_terms))
        t_yz_mag = float(np.linalg.norm(t_sum))
        # Scale by max single-panel total traction magnitude (Tx^2 + Ts^2)^0.5 so
        # orthogonal-tangent inflation in the denominator is avoided.
        panel_T_mags = [
            float(np.sqrt(tx ** 2 + ts ** 2))
            for tx, ts in zip(tx_terms, ts_terms)
        ]
        cluster_scale = max(max(panel_T_mags), 1.0)
        out.append(
            {
                "cluster_id": int(cid),
                "n_panels": int(len(tx_terms)),
                "members": members,
                "cluster_collinear": bool(_collinear),
                "status": "ok",
                "Tx_sum": tx_sum,
                "T_yz_sum_y": float(t_sum[0]),
                "T_yz_sum_z": float(t_sum[1]),
                "Tx_rel_cluster": abs(tx_sum) / cluster_scale,
                "T_yz_rel_cluster": t_yz_mag / cluster_scale,
            }
        )
    return out


def build_load_reaction_audit(all_panel_diagnostics: list[dict] | None) -> dict[str, float]:
    """
    Summarize panel/global load-reaction consistency from diagnostics.

    Returns two global balance metrics:
    - global_force_balance_rel: panel-sum metric (informational only; double-counts
      shared junction reactions in shared-DOF mode — expect ~0.5 for a 2-panel section).
    - global_force_balance_rel_at_fixed: authoritative metric using the sum of
      r_full at truly fixed (RBM-support) UX DOFs vs total applied load.
      Should be ~0 (< 1e-6) for a correctly solved global assembly.
    """
    _empty = {
        "n_panels": 0.0,
        "max_rel_mismatch_endpoint": 0.0,
        "mean_rel_mismatch_endpoint": 0.0,
        "max_rel_mismatch_target": 0.0,
        "mean_rel_mismatch_target": 0.0,
        "global_force_balance_rel": 0.0,
        "global_force_balance_rel_at_fixed": float("nan"),
        "global_force_balance_rel_delta": float("nan"),
    }
    if not all_panel_diagnostics:
        return _empty
    rel_endpoint: list[float] = []
    rel_target: list[float] = []
    sum_fx = 0.0
    sum_rx = 0.0
    sum_fxt = 0.0
    for di in all_panel_diagnostics:
        if not di:
            continue
        lt = di.get("load_totals", {})
        br = di.get("boundary_reaction_set", {})
        if "start" not in br or "end" not in br:
            continue
        fx = float(lt.get("Fx_total", 0.0))
        fx_t = float(lt.get("Fx_target", fx))
        fs = float(lt.get("Fs_total", 0.0))
        rx = float(br["start"].get("Fx", 0.0) + br["end"].get("Fx", 0.0))
        rs = float(br["start"].get("Fs", 0.0) + br["end"].get("Fs", 0.0))
        ex = abs(rx + fx) / max(abs(rx), abs(fx), 1.0)
        ex_t = abs(rx + fx_t) / max(abs(rx), abs(fx_t), 1.0)
        es = abs(rs + fs) / max(abs(rs), abs(fs), 1.0)
        rel_endpoint.append(max(ex, es))
        rel_target.append(max(ex_t, es))
        sum_fx += fx
        sum_fxt += fx_t
        sum_rx += rx
    if not rel_endpoint:
        return _empty
    global_force_balance_rel = abs(sum_rx + sum_fxt) / max(abs(sum_rx), abs(sum_fxt), 1.0)

    # Authoritative global balance: fixed-DOF UX reactions vs total applied UX load.
    # global_reaction_at_fixed_UX is stored by the global solve in each panel's
    # diagnostics. Extract it from the first panel that has it.
    sum_rx_fixed_ux: float | None = None
    for di in all_panel_diagnostics:
        if di and "global_reaction_at_fixed_UX" in di:
            sum_rx_fixed_ux = float(di["global_reaction_at_fixed_UX"])
            break
    if sum_rx_fixed_ux is not None:
        global_force_balance_rel_at_fixed = abs(sum_rx_fixed_ux + sum_fxt) / max(
            abs(sum_rx_fixed_ux), abs(sum_fxt), 1.0
        )
        modes = {
            str(di.get("interface_constraint_mode", ""))
            for di in all_panel_diagnostics
            if di
        }
        _cluster_basis_modes = {"transformed_basis", "shared_rotated"}
        if modes and modes <= _cluster_basis_modes:
            # In cluster-basis modes panel endpoint reactions are traction-derived
            # panel contributions; use fixed-DOF global balance as the canonical
            # global metric to avoid panel-sum double counting/partition ambiguity.
            global_force_balance_rel = global_force_balance_rel_at_fixed
        global_force_balance_rel_delta = abs(global_force_balance_rel - global_force_balance_rel_at_fixed)
    else:
        global_force_balance_rel_at_fixed = float("nan")
        global_force_balance_rel_delta = float("nan")

    return {
        "n_panels": float(len(rel_endpoint)),
        "max_rel_mismatch_endpoint": float(max(rel_endpoint)),
        "mean_rel_mismatch_endpoint": float(np.mean(rel_endpoint)),
        "max_rel_mismatch_target": float(max(rel_target)),
        "mean_rel_mismatch_target": float(np.mean(rel_target)),
        "global_force_balance_rel": float(global_force_balance_rel),
        "global_force_balance_rel_at_fixed": float(global_force_balance_rel_at_fixed),
        "global_force_balance_rel_delta": float(global_force_balance_rel_delta),
    }


__all__ = [
    "check_panel_equilibrium",
    "check_cluster_equilibrium",
    "build_load_reaction_audit",
]
