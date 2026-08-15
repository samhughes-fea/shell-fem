"""Residual-4 CI watch diagnostics: peak ownership + cluster flags (shared vs HKC).

Writes a JSON summary under visualisation/output/residual4_diagnosis/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cap_peak_sampling import (
    run_cap_peak_study,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    _build_shell,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.equilibrium_checks import (
    check_cluster_equilibrium,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    load_cotide_station_recipe,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
    cluster_is_kind_crossing,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    shell_panel_topology_from_shell_inputs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.connected_panel_topology import (
    build_connected_panel_maps,
)
from blade_structure_utilities.mesh_utilities.section_mesh import build_section_strip_mesh

KEYS = ("4D-CN", "4D-F", "5D-CN", "5D-F")
FRACS = (0.005, 0.0035)
MODES = ("shared", "hybrid_kind_crossing")
OUT = Path(
    "blade_structure_utilities/fem_utilities/section_shell_fea/"
    "visualisation/output/residual4_diagnosis"
)
GCI_SHARED = Path(
    "blade_structure_utilities/fem_utilities/section_shell_fea/"
    "visualisation/output/gci_study_mitc4/gci_summary.csv"
)
GCI_HKC = Path(
    "blade_structure_utilities/fem_utilities/section_shell_fea/"
    "visualisation/output/gci_study_mitc4_hkc/gci_summary.csv"
)
GCI_TB = Path(
    "blade_structure_utilities/fem_utilities/section_shell_fea/"
    "visualisation/output/gci_study_mitc4_tb_ci_watch/gci_summary.csv"
)


def _gci_row(path: Path, layout: str) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("layout") == layout:
                def f(k: str) -> float:
                    try:
                        return float(row.get(k, "nan"))
                    except ValueError:
                        return float("nan")

                return {
                    "n12_skin": f("peak_abs_n12_gci_fine_skin"),
                    "mid_skin": f("peak_mid_sig_gci_fine_skin"),
                    "hashin_skin": f("peak_hashin_gci_fine_skin"),
                    "gov": f("governing_fi_gci_fine"),
                }
    return {}


def _skin_n12_finest(by_layout: dict, key: str) -> dict:
    rows = by_layout.get(key) or []
    if not rows:
        return {}
    rung = rows[-1]
    for st in rung.stats:
        if st.kind == "skin" and st.metric == "peak_abs_n12":
            loc = st.location
            tags = loc.tags if loc is not None else None
            return {
                "frac": rung.frac,
                "interface": rung.interface_constraint_mode,
                "max": st.max,
                "interior_max": st.interior_max,
                "panel": loc.panel_label if loc else None,
                "elem": loc.local_element_index if loc else None,
                "junction": bool(tags.junction) if tags else None,
                "endpoint_band": bool(tags.endpoint_band) if tags else None,
                "tie_slave": bool(tags.tie_slave) if tags else None,
            }
    return {}


def _cluster_at_panel_end(layout: str, panel_label: str, which: str, frac: float) -> dict:
    recipe = load_cotide_station_recipe()
    kw = recipe.cascade_kwargs()
    shell = _build_shell(
        layout=layout,
        chord=kw["chord"],
        twist_deg=kw["twist_deg"],
        thickness=kw["thickness"],
        naca_dat=kw.get("naca_dat"),
    )
    mesh = build_section_strip_mesh(shell, target_element_length_frac=frac)
    zones = list(shell_panel_topology_from_shell_inputs(shell).zones)
    conn = build_connected_panel_maps(
        zones,
        endpoint_tol=1e-6,
        interface_constraint_mode="shared",
        section_strip_mesh=mesh,
    )
    target_cid = None
    for (pi, w), cid in conn.endpoint_cluster_index.items():
        if str(conn.panel_maps[pi].panel_label) == panel_label and w == which:
            target_cid = int(cid)
            break
    members: list[str] = []
    if target_cid is not None:
        for (pi, w), cid in conn.endpoint_cluster_index.items():
            if int(cid) != target_cid:
                continue
            members.append(f"{conn.panel_maps[pi].panel_label}:{w}")
    cluster = conn.clusters[target_cid] if target_cid is not None else []
    collinear = (
        bool(conn.cluster_is_collinear_flags.get(target_cid, False))
        if target_cid is not None
        else None
    )
    kind_x = (
        cluster_is_kind_crossing(cluster, conn.panel_maps) if cluster else None
    )
    eq = check_cluster_equilibrium(
        zones, all_panel_diagnostics=[{} for _ in zones], endpoint_tol=1e-6
    )
    matched = [
        r
        for r in eq
        if target_cid is not None
        and any(panel_label in str(m) for m in r.get("members", []))
    ]
    return {
        "cluster_id": target_cid,
        "members": members,
        "collinear": collinear,
        "kind_crossing": kind_x,
        "traction_rows": matched[:3],
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {"keys": list(KEYS), "modes": {}, "gci": {}, "clusters": {}}
    for mode in MODES:
        out = OUT / mode
        by_layout, _ = run_cap_peak_study(
            KEYS,
            refinements=FRACS,
            out_dir=out,
            interface_constraint_mode=mode,
            progress=True,
        )
        summary["modes"][mode] = {
            k: _skin_n12_finest(by_layout, k) for k in KEYS
        }
    for k in KEYS:
        summary["gci"][k] = {
            "shared": _gci_row(GCI_SHARED, k),
            "hkc": _gci_row(GCI_HKC, k),
            "tb": _gci_row(GCI_TB, k),
        }
        loc = summary["modes"]["shared"].get(k) or {}
        panel = loc.get("panel")
        if panel:
            # Peak element end: treat junction peaks as start/end via endpoint_band.
            which = "start"
            summary["clusters"][k] = _cluster_at_panel_end(
                k, str(panel), which, float(FRACS[-1])
            )
            # Also try end if start cluster is empty
            if summary["clusters"][k].get("cluster_id") is None:
                summary["clusters"][k] = _cluster_at_panel_end(
                    k, str(panel), "end", float(FRACS[-1])
                )
    path = OUT / "residual4_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
