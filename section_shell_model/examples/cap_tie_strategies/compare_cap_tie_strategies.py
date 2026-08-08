"""Compare spar-cap tie strategies across structural systems and thicknesses (BL-003).

Robustness, not a single happy path: a tie rule that works on 5D-F at one
thickness is worth little. This sweeps every catalogue layout that has spar caps
against a spread of subcomponent thicknesses and scores each strategy on whether
it leaves any cap unconstrained.

Run from the repository root::

    python blade_structure_utilities/fem_utilities/section_shell_model/examples/\\
        cap_tie_strategies/compare_cap_tie_strategies.py

Geometry-only: no FE solve. The question here is *coverage* — does the strategy
produce a load path for every cap, in every layout, at every thickness. A
strategy that fails coverage cannot be fixed by tuning the solver, so this is
the cheap gate to run before touching the assembler.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[5]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from blade_structure_utilities.fem_utilities.section_shell_model.examples.cap_tie_strategies.strategies import (  # noqa: E402
    STRATEGIES,
)
from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (  # noqa: E402
    build_section_view,
    resolve_system_type,
)
from blade_structure_utilities.geometry_utilities.section_geometry.core.system_layout import (  # noqa: E402
    SYSTEM_TYPE_KEYS,
)
from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import (  # noqa: E402
    AirfoilSDF,
)
from blade_structure_utilities.mesh_utilities.section_mesh import (  # noqa: E402
    build_section_strip_mesh,
    build_shell_mesh_inputs,
)

# Root-like, midspan-like and tip-like sections: chord and twist both move.
SECTIONS = [
    ("root", 0.241, 34.1),
    ("mid", 0.199, 5.4),
    ("tip", 0.112, 1.5),
]
# Thin / nominal / thick, spanning the range sizing actually visits.
THICKNESSES = [
    ("thin", {"skin": 0.001, "spar": 0.001, "web": 0.0015}),
    ("nominal", {"skin": 0.001, "spar": 0.004, "web": 0.0015}),
    ("thick", {"skin": 0.003, "spar": 0.012, "web": 0.003}),
]


def _layouts_with_caps() -> list[str]:
    out = []
    for key in SYSTEM_TYPE_KEYS:
        try:
            if bool(resolve_system_type(key).has_spar_caps):
                out.append(key)
        except Exception:
            continue
    return out


def _build_mesh(layout_key: str, chord: float, twist_deg: float, t: dict[str, float]):
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=chord, closed_te=True
    )
    layout = resolve_system_type(layout_key)
    twist_rad = float(np.deg2rad(twist_deg))
    section = build_section_view(
        airfoil,
        layout,
        twist_angle_rad=twist_rad,
        skin_thickness=t["skin"],
        web_thickness=t["web"],
        cap_height=t["spar"],
    )
    shell = build_shell_mesh_inputs(
        section, twist_rad=twist_rad, layout_key=layout_key,
        n_web_samples=20, n_cap_samples=80,
    )
    return build_section_strip_mesh(shell, target_element_length_frac=0.012, endpoint_tol=1e-6)


def main() -> int:
    layouts = _layouts_with_caps()
    print(f"layouts with spar caps: {len(layouts)}")
    print(f"cases: {len(layouts)} layouts x {len(SECTIONS)} sections x {len(THICKNESSES)} thicknesses "
          f"= {len(layouts)*len(SECTIONS)*len(THICKNESSES)}\n")

    # strategy -> [n_cases, n_cases_with_an_untied_cap, total_ties, offsets]
    score = {name: {"cases": 0, "fail": 0, "ties": 0, "offsets": []} for name in STRATEGIES}
    failures: dict[str, list[str]] = {name: [] for name in STRATEGIES}
    build_errors = 0

    for layout_key in layouts:
        for sec_name, chord, twist in SECTIONS:
            for t_name, t in THICKNESSES:
                try:
                    mesh = _build_mesh(layout_key, chord, twist, t)
                except Exception as exc:  # geometry infeasible at this combo
                    build_errors += 1
                    if build_errors <= 3:
                        print(f"  [geometry failed] {layout_key} {sec_name} {t_name}: "
                              f"{type(exc).__name__}: {str(exc)[:90]}")
                    continue
                tag = f"{layout_key}/{sec_name}/{t_name}"
                for name, fn in STRATEGIES.items():
                    try:
                        p = fn(mesh, t)
                    except Exception:
                        failures[name].append(tag + " (raised)")
                        score[name]["cases"] += 1
                        score[name]["fail"] += 1
                        continue
                    score[name]["cases"] += 1
                    score[name]["ties"] += len(p.ties)
                    score[name]["offsets"].extend(ct.offset_m for ct in p.ties)
                    if p.topology_change:
                        continue  # no ties needed by construction
                    if p.caps_with_no_tie or p.caps_seen == 0:
                        score[name]["fail"] += 1
                        failures[name].append(tag)

    print("\n=== coverage (a FAIL means at least one cap left with no load path) ===")
    hdr = f"{'strategy':26s} {'cases':>6s} {'fail':>6s} {'fail%':>7s} {'ties/case':>10s} {'offset mm (min/med/max)':>26s}"
    print(hdr)
    print("-" * len(hdr))
    for name, s in score.items():
        n = max(s["cases"], 1)
        offs = np.array(s["offsets"]) * 1e3 if s["offsets"] else np.zeros(1)
        print(f"{name:26s} {s['cases']:6d} {s['fail']:6d} {100*s['fail']/n:6.1f}% "
              f"{s['ties']/n:10.1f} {offs.min():8.4f}/{np.median(offs):7.4f}/{offs.max():7.4f}")

    print("\n=== where each strategy fails (first 6) ===")
    for name, f in failures.items():
        if not f:
            print(f"  {name}: no coverage failures")
            continue
        uniq = sorted({x.split('/')[0] for x in f})
        print(f"  {name}: {len(f)} case(s) across layouts {uniq[:6]}{' ...' if len(uniq) > 6 else ''}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
