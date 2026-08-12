"""Per-station JSON emission for the strip-mesh shell path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea._utils.jsonutil import to_jsonable, write_json


def _median_panel_n_elements(mesh: Any) -> int:
    counts = [int(pm.n_elements) for pm in getattr(mesh, "panel_meshes", ())]
    return int(np.median(counts)) if counts else 0


def write_station_json(
    out_dir: Path,
    *,
    station_tag: str,
    mesh: Any,
    N: float,
    Vy: float,
    Vz: float,
    My: float,
    Mz: float,
    T: float,
    n_elements_per_panel: int | None,
    mesh_element_length_frac: float | None = None,
    target_element_length_m: float | None = None,
    backend_extra: Mapping[str, Any] | None = None,
    fsdt_periodic_extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write per-station JSON for the midline→strip-mesh shell path."""
    panel_rows = [
        {
            "label": pm.panel_label,
            "kind": pm.kind,
            "arc_length_m": float(pm.arc_length_m),
            "thickness_m": float(pm.thickness_m),
            "n_elements": int(pm.n_elements),
        }
        for pm in mesh.panel_meshes
    ]
    payload: dict[str, Any] = {
        "schema": "section_shell_station_v2",
        "station_tag": station_tag,
        "unit_section_resultants": {
            "N_N": float(N),
            "Vy_N": float(Vy),
            "Vz_N": float(Vz),
            "My_Nm": float(My),
            "Mz_Nm": float(Mz),
            "T_Nm": float(T),
        },
        "n_elements_per_panel": (
            int(n_elements_per_panel)
            if n_elements_per_panel is not None
            else _median_panel_n_elements(mesh)
        ),
        "mesh_element_length_frac": (
            None if mesh_element_length_frac is None else float(mesh_element_length_frac)
        ),
        "target_element_length_m": (
            None if target_element_length_m is None else float(target_element_length_m)
        ),
        "layout_key": mesh.layout_key,
        "chord_m": float(mesh.chord_m),
        "twist_rad": float(mesh.twist_rad),
        "thin_wall": {
            "n_panels": int(len(mesh.panels)),
            "n_total_nodes": int(mesh.n_total_nodes),
            "n_total_elements": int(mesh.n_total_elements),
            "n_clusters": int(len(mesh.clusters)),
            "panels": panel_rows,
        },
        "mesh_summary": mesh.summary(),
    }
    if backend_extra:
        payload["backend"] = dict(backend_extra)
    if fsdt_periodic_extra:
        payload["fsdt_periodic_hashin"] = dict(fsdt_periodic_extra)
    json_path = out_dir / f"section_shell_station_{station_tag}.json"
    write_json(json_path, to_jsonable(payload))
    return json_path


__all__ = ["write_station_json"]
