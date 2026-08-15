"""Precursor: 5D-CN skin |N12| ownership shared vs TB at fine fracs + traction note."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cap_peak_sampling import (
    run_cap_peak_study,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    run_cascade_at_frac,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    load_cotide_station_recipe,
)

OUT_ROOT = Path(
    "blade_structure_utilities/fem_utilities/section_shell_fea/"
    "visualisation/output/cap_peak_sampling_5d_cn"
)
FRACS = (0.005, 0.0035)
MODES = ("shared", "transformed_basis")


def _skin_n12_locs(by_layout: dict) -> list[dict]:
    rows = by_layout.get("5D-CN") or []
    out: list[dict] = []
    for rung in rows:
        for st in rung.stats:
            if st.kind != "skin" or st.metric != "peak_abs_n12":
                continue
            loc = st.location
            tags = loc.tags if loc is not None else None
            out.append(
                {
                    "frac": rung.frac,
                    "interface": rung.interface_constraint_mode,
                    "max": st.max,
                    "interior_max": st.interior_max,
                    "raw_over_interior": (
                        float(st.max) / float(st.interior_max)
                        if st.interior_max and st.interior_max == st.interior_max and st.interior_max > 0
                        else float("nan")
                    ),
                    "panel": loc.panel_label if loc else None,
                    "elem": loc.local_element_index if loc else None,
                    "junction": bool(tags.junction) if tags else None,
                    "tie_slave": bool(tags.tie_slave) if tags else None,
                    "endpoint_band": bool(tags.endpoint_band) if tags else None,
                }
            )
    return out


def main() -> int:
    recipe = load_cotide_station_recipe()
    kw = recipe.cascade_kwargs()
    summary: dict = {"modes": {}, "cascade_trust": {}}
    for mode in MODES:
        out = OUT_ROOT / mode
        by_layout, _ = run_cap_peak_study(
            ("5D-CN",),
            refinements=FRACS,
            out_dir=out,
            interface_constraint_mode=mode,
            progress=True,
        )
        locs = _skin_n12_locs(by_layout)
        summary["modes"][mode] = locs
        trust = {}
        for frac in FRACS:
            row = run_cascade_at_frac(
                float(frac),
                layout="5D-CN",
                interface_constraint_mode=mode,
                **kw,
            )
            trust[str(frac)] = {
                "mesh_ok": row.mesh_ok,
                "cascade_trust_ok": row.cascade_trust_ok,
                "peak_abs_n12_skin": row.peak_abs_n12.get("skin"),
                "peak_mid_sig_skin": row.peak_mid_sig.get("skin"),
                "residual_rel": row.residual_rel,
            }
        summary["cascade_trust"][mode] = trust
        print(f"=== {mode} skin |N12| locations ===", flush=True)
        for loc in locs:
            print(loc, flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "precursor_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
