"""Dump SS:bay_0 junction-cluster membership/collinear under shared (5D-CN)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    _build_shell,
    run_cascade_at_frac,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.equilibrium_checks import (
    check_cluster_equilibrium,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    load_cotide_station_recipe,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.connected_panel_topology import (
    build_connected_panel_maps,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    shell_panel_topology_from_shell_inputs,
)
from blade_structure_utilities.mesh_utilities.section_mesh import build_section_strip_mesh

TARGET = "skin:outer_skin:SS:bay_0"
FRAC = 0.0035
OUT = Path(
    "blade_structure_utilities/fem_utilities/section_shell_fea/"
    "visualisation/output/cap_peak_sampling_5d_cn/ss_bay0_cluster_traction.json"
)


def main() -> int:
    recipe = load_cotide_station_recipe()
    kw = recipe.cascade_kwargs()
    shell = _build_shell(
        layout="5D-CN",
        chord=kw["chord"],
        twist_deg=kw["twist_deg"],
        thickness=kw["thickness"],
        naca_dat=kw.get("naca_dat"),
    )
    mesh = build_section_strip_mesh(shell, target_element_length_frac=FRAC)
    topo_shell = shell_panel_topology_from_shell_inputs(shell, allow_demo_bind=True)
    zones = list(topo_shell.zones)
    conn = build_connected_panel_maps(
        zones,
        endpoint_tol=1e-6,
        interface_constraint_mode="shared",
        section_strip_mesh=mesh,
    )

    target_cid = None
    for (pi, which), cid in conn.endpoint_cluster_index.items():
        lab = str(conn.panel_maps[pi].panel_label)
        if lab == TARGET and which == "start":
            target_cid = int(cid)
            break

    members: list[str] = []
    if target_cid is not None:
        for (pi, which), cid in conn.endpoint_cluster_index.items():
            if int(cid) != target_cid:
                continue
            members.append(f"{conn.panel_maps[pi].panel_label}:{which}")

    collinear = (
        bool(conn.cluster_is_collinear_flags.get(target_cid, False))
        if target_cid is not None
        else None
    )

    cluster_rows = check_cluster_equilibrium(
        zones,
        all_panel_diagnostics=[{} for _ in zones],
        endpoint_tol=1e-6,
    )
    matched = [
        r
        for r in cluster_rows
        if any(TARGET in str(m) for m in r.get("members", []))
    ]

    cascade = run_cascade_at_frac(
        FRAC,
        layout="5D-CN",
        interface_constraint_mode="shared",
        **kw,
    )

    dump = {
        "layout": "5D-CN",
        "frac": FRAC,
        "interface": "shared",
        "target_panel": TARGET,
        "target_end": "start",
        "cluster_id": target_cid,
        "cluster_collinear": collinear,
        "members": sorted(set(members)),
        "n_members": len(set(members)),
        "cluster_equilibrium_rows": matched,
        "reaction_dN12_rel": None,
        "reaction_note": (
            "Live interface_edge_set / reaction_dN12_rel not populated on the "
            "production cascade path; cluster membership + collinear from "
            "topology; Tx_rel_cluster status from check_cluster_equilibrium "
            "(insufficient_data without edge traction samples)."
        ),
        "cascade": {
            "mesh_ok": cascade.mesh_ok,
            "cascade_trust_ok": cascade.cascade_trust_ok,
            "peak_abs_n12_skin": cascade.peak_abs_n12.get("skin"),
            "peak_mid_sig_skin": cascade.peak_mid_sig.get("skin"),
            "residual_rel": cascade.residual_rel,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dump, indent=2), encoding="utf-8")
    print(json.dumps(dump, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
