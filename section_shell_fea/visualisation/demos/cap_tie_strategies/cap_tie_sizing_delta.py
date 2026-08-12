"""What changes in sizing when spar caps are actually connected? (BL-003)

Ties are off by default. This is the report that says what turning them on would
cost: it runs the real station sizing loop twice per case — ``cap_tie_mode``
``"none"`` then ``"skin_offset"`` — and diffs everything the loop decides.

The FI values matter, but the *governing design variable* matters more. The FI
loop picks which thickness to grow by worst per-component FI
(``optimize.py``, ties breaking toward ``t_spar``), and with caps disconnected
``fi_spar`` has been unphysical. If ``govern`` flips, ties are not just changing
reported numbers, they are changing which subcomponent the optimiser thickens.

Run from the repository root::

    python blade_structure_utilities/fem_utilities/section_shell_fea/visualisation/demos/\\
        cap_tie_strategies/cap_tie_sizing_delta.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[6]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from blade_structure_precompute.module.section_selection.orchestration.structural_system import (  # noqa: E402
    system_from_layout,
)
from blade_structure_precompute.module.section_selection.processing.shell_station import (  # noqa: E402
    size_station_against_ei,
)
from blade_structure_utilities.geometry_utilities.section_geometry import (  # noqa: E402
    resolve_system_type,
)
from blade_structure_utilities.geometry_utilities.section_geometry.core.system_layout import (  # noqa: E402
    SYSTEM_TYPE_KEYS,
)

LAYOUTS = ["0B", "1C-CN", "2B-CN", "2C-CN", "2D-CN", "3D-F", "5C-F", "5D-F"]
STATIONS = [(0.2, 4.0e7), (0.6, 2.0e7)]
LOADS = {"M_flapwise": 5.0e5, "V_flapwise": 1.0e5, "M_edgewise": 2.0e5}
_FI_KEYS = ("fi_skin", "fi_spar", "fi_web")
_DVS = ("t_skin", "t_spar", "t_web")


class _Iso:
    E = 70e9
    G = 27e9
    density = 2700.0

    def get_E(self) -> float:
        return self.E


def _airfoil(chord: float = 1.0, t_c: float = 0.12, n: int = 120) -> np.ndarray:
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack(
        [0.5 * chord * (1.0 + np.cos(th)), 0.5 * chord * t_c * np.sin(th)]
    )


def _size(layout_key: str, station_idx: int, zL: float, ei_req: float, mode: str):
    system = system_from_layout(resolve_system_type(layout_key))
    mats = {"skin": _Iso(), "spar": _Iso(), "web": _Iso()}
    bin_obj, _ = size_station_against_ei(
        system=system,
        station_idx=station_idx,
        zL=zL,
        coords=_airfoil(),
        metadata={"chord": 1.0, "twist_deg": 0.0},
        EI_req=ei_req,
        component_materials=mats,
        effective_min_thickness=0.002,
        enable_shell_fi=True,
        hashin_fi_allow=1.0,
        cap_tie_mode=mode,
        **LOADS,
    )
    row = bin_obj.best_row()
    if row is None:
        return None
    util = dict(row.utilizations or {})
    dv = dict(row.design_vars or {})
    fi = {k: util.get(k) for k in _FI_KEYS}
    # The DV the FI loop would grow next: worst per-component FI, ties to spar.
    ranked = [(fi.get(f"fi_{c}") or 0.0, f"t_{d}")
              for c, d in (("spar", "spar"), ("web", "web"), ("skin", "skin"))
              if f"t_{d}" in dv]
    govern = max(ranked)[1] if ranked else "-"
    return {
        "max_fi": util.get("max_fi_hashin"),
        "fi": fi,
        "dv": {k: dv.get(k) for k in _DVS},
        "EI_flap": float(getattr(row.props, "EI_flap", float("nan"))),
        "pass": bool(row.pass_flag),
        "govern": govern,
    }


def _fmt(v, w=9, p=4):
    return f"{'-':>{w}}" if v is None or not np.isfinite(v) else f"{v:{w}.{p}g}"


def main() -> int:
    print("Cap-tie sizing delta — real station sizing, ties off vs on")
    print(f"loads {LOADS};  FI gate on, allow=1.0\n")
    hdr = (f"{'case':16s} {'mode':6s} {'max_fi':>9s} {'fi_skin':>9s} {'fi_spar':>9s} "
           f"{'fi_web':>9s} {'t_skin':>9s} {'t_spar':>9s} {'EI_flap':>10s} "
           f"{'govern':>7s} {'pass':>5s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    n_skip = 0
    for layout_key in LAYOUTS:
        if layout_key not in SYSTEM_TYPE_KEYS:
            continue
        for si, (zL, ei_req) in enumerate(STATIONS):
            tag = f"{layout_key}/z{zL:.1f}"
            pair = {}
            for mode in ("none", "skin_offset"):
                try:
                    pair[mode] = _size(layout_key, si, zL, ei_req, mode)
                except Exception as exc:
                    n_skip += 1
                    if n_skip <= 4:
                        print(f"  [skipped] {tag} {mode}: "
                              f"{type(exc).__name__}: {str(exc)[:70]}")
                    pair[mode] = None
            if not (pair.get("none") and pair.get("skin_offset")):
                continue
            for mode in ("none", "skin_offset"):
                r = pair[mode]
                print(f"{tag:16s} {mode[:6]:6s} {_fmt(r['max_fi'])} "
                      f"{_fmt(r['fi']['fi_skin'])} {_fmt(r['fi']['fi_spar'])} "
                      f"{_fmt(r['fi']['fi_web'])} {_fmt(r['dv']['t_skin'])} "
                      f"{_fmt(r['dv']['t_spar'])} {_fmt(r['EI_flap'], 10, 4)} "
                      f"{r['govern']:>7s} {str(r['pass']):>5s}")
            rows.append((tag, pair["none"], pair["skin_offset"]))

    if not rows:
        print("\nno comparable cases")
        return 1

    print(f"\n=== summary over {len(rows)} cases ===")
    flips = [t for t, a, b in rows if a["govern"] != b["govern"]]
    print(f"governing DV flipped in {len(flips)}/{len(rows)} cases"
          + (f": {flips}" if flips else ""))
    passes = [t for t, a, b in rows if a["pass"] != b["pass"]]
    print(f"pass_flag changed in {len(passes)}/{len(rows)} cases"
          + (f": {passes}" if passes else ""))

    for label, get in (
        ("max_fi_hashin", lambda r: r["max_fi"]),
        ("fi_spar", lambda r: r["fi"]["fi_spar"]),
        ("fi_skin", lambda r: r["fi"]["fi_skin"]),
        ("t_spar", lambda r: r["dv"]["t_spar"]),
        ("t_skin", lambda r: r["dv"]["t_skin"]),
        ("EI_flap", lambda r: r["EI_flap"]),
    ):
        ratios = [
            get(b) / get(a)
            for _t, a, b in rows
            if get(a) not in (None, 0.0) and get(b) is not None
            and np.isfinite(get(a)) and np.isfinite(get(b)) and get(a) != 0.0
        ]
        if ratios:
            r = np.array(ratios)
            print(f"  {label:15s} on/off ratio  p5={np.percentile(r,5):7.3f}  "
                  f"p50={np.percentile(r,50):7.3f}  p95={np.percentile(r,95):7.3f}")

    print(
        "\nA governing-DV flip is the load-bearing result: it means ties change "
        "which\nsubcomponent the optimiser thickens, not just what it reports. "
        "fi_spar has been\nunphysical for as long as caps were disconnected, and "
        "it is what breaks that tie."
    )
    print(f"\n{n_skip} run(s) skipped")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
