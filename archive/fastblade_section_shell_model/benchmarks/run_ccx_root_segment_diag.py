"""Diagnostic-only: clamped multi-row root-segment CCX vs periodic RVE at station 0.

Usage:
  python -m blade_precompute.section_shell_model.benchmarks.run_ccx_root_segment_diag

Requires ``ccx`` on PATH. Does not touch production dispatch or FI caches.
The production root path is the prismatic unit-length periodic RVE
(``run_section_periodic_bc``); this benchmark is comparison-only.
See ``docs/ROOT_SEGMENT_CCX_DIAGNOSTIC.md``.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from blade_precompute.section_shell_model.api.fi_recovery import hashin_fi_per_ply_from_rows_batch
from blade_precompute.section_shell_model.api.periodic import run_section_periodic_bc_mitc4
from blade_precompute.section_shell_model.calculix_ccx.root_segment_export import (
    run_root_segment_ccx,
)
from blade_precompute.section_shell_model.calculix_ccx.root_segment_mesh import (
    estimate_root_diameter_m,
)
from blade_precompute.section_shell_model.pre_processing.constitutive.default_laminates import (
    WallLaminate,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
)


_NE = 4
# Target section resultants at the *root* face (same as the periodic RVE comparison).
# Outboard moments are corrected for the shear lever arm so root RF moments match these.
_LOAD_ROOT = dict(N=0.0, Vy=0.0, Vz=5.0e4, My=1.0e5, Mz=0.0, T=0.0)


def _outboard_loads_for_root_target(
    load_root: dict[str, float],
    *,
    segment_length_m: float,
) -> dict[str, float]:
    """Map root-target resultants to outboard face loads on a prismatic free body.

    Force equilibrium: outboard ``(N, Vy, Vz)`` equals the root target (root clamp
    then reacts ``-F``). Moment equilibrium about the root face requires

        My_root = My_out + Vz * L
        Mz_root = Mz_out - Vy * L

    (span ``x`` from root to outboard; sign from ``r × F`` with ``r = L ê_x``).
    Torques/axial force transfer unchanged. So

        My_out = My_root - Vz * L
        Mz_out = Mz_root + Vy * L
    """
    L = float(segment_length_m)
    return {
        "N": float(load_root["N"]),
        "Vy": float(load_root["Vy"]),
        "Vz": float(load_root["Vz"]),
        "T": float(load_root["T"]),
        "My": float(load_root["My"]) - float(load_root["Vz"]) * L,
        "Mz": float(load_root["Mz"]) + float(load_root["Vy"]) * L,
    }


class _Panel:
    def __init__(self, label: str, nodes: list[list[float]], lam: WallLaminate):
        self.label = label
        self.nodes = np.asarray(nodes, dtype=float)
        ds = np.maximum(
            np.hypot(np.diff(self.nodes[:, 0]), np.diff(self.nodes[:, 1])), 1e-30
        )
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.lam = lam
        self.cell_id = 0
        self.end_boom = None


def _l_section_panels() -> list[_Panel]:
    """Trivial two-panel L-section (same falsifier geometry as reaction-K6 diag)."""
    lam = WallLaminate(E=70e9, t=0.01, nu=0.33, name="iso", n_plies=4)
    return [
        _Panel("skin:lower", [[0.0, 0.0], [1.0, 0.0]], lam),
        _Panel("web:1", [[1.0, 0.0], [1.0, 0.5]], lam),
    ]


def _naca_panels(*, naca_code: str = "0012", chord_m: float = 1.0) -> tuple[list[Any], Any]:
    """Closed airfoil root (NACA + one spar web) via Pipeline B."""
    from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section

    pb = build_pipeline_b_section(naca_code, [0.35], chord_m=chord_m)
    return list(pb.panels), pb


def _bundle_from_panels(panels: list[Any], pb: Any | None = None):
    from types import SimpleNamespace

    if pb is not None:
        return pb
    # Minimal Pipeline-B-like bundle for MITC4 periodic comparison.
    return SimpleNamespace(
        panels=panels,
        booms=[],
        webs_geom=[],
        shell_inputs=SimpleNamespace(midlines=()),
    )


def _panel_peak_abs(rows: list[ShellPanelResultants]) -> dict[str, float]:
    if not rows:
        return {"Nx": 0.0, "Ny": 0.0, "Nxy": 0.0, "Qx": 0.0, "Qy": 0.0}
    return {
        "Nx": float(max(abs(r.Nx) for r in rows)),
        "Ny": float(max(abs(r.Ny) for r in rows)),
        "Nxy": float(max(abs(r.Nxy) for r in rows)),
        "Qx": float(max(abs(r.Qx) for r in rows)),
        "Qy": float(max(abs(r.Qy) for r in rows)),
    }


def _to_laminate_definition(lam: Any):
    """Wall laminate / adapter → LaminateDefinition for relative FI."""
    from blade_precompute.materials.laminate import laminate_definition_from_legacy

    ld = laminate_definition_from_legacy(lam)
    if ld is None:
        raise RuntimeError("could not build LaminateDefinition from laminate")
    return ld


def _max_hashin(rows: list[ShellPanelResultants], lam: Any) -> float:
    if not rows:
        return float("nan")
    ld = _to_laminate_definition(lam)
    fi, _nr = hashin_fi_per_ply_from_rows_batch(rows, ld, ld.build_clpt_ply_rows())
    if fi.size == 0:
        return float("nan")
    return float(np.nanmax(fi))


def _equilibrium_report(
    diag: dict[str, Any],
    load_root: dict[str, float],
    load_out: dict[str, float],
) -> None:
    rx, ry, rz = diag["root_rf_sum"]
    print("\n=== Equilibrium (root RF sum vs root-target F) ===")
    print(
        f"  root target N,Vy,Vz = {load_root['N']:.4e}, "
        f"{load_root['Vy']:.4e}, {load_root['Vz']:.4e}"
    )
    print(
        f"  outboard  N,Vy,Vz   = {load_out['N']:.4e}, "
        f"{load_out['Vy']:.4e}, {load_out['Vz']:.4e}"
    )
    print(
        f"  root target My,Mz,T = {load_root['My']:.4e}, "
        f"{load_root['Mz']:.4e}, {load_root['T']:.4e}"
    )
    print(
        f"  outboard  My,Mz,T   = {load_out['My']:.4e}, "
        f"{load_out['Mz']:.4e}, {load_out['T']:.4e}"
    )
    print(f"  root sum RF         = {rx:.4e}, {ry:.4e}, {rz:.4e}")
    # Free-body: root reactions balance outboard/root-target forces (opposite sign).
    err_n = abs(rx + load_root["N"]) / max(abs(load_root["N"]), abs(rx), 1.0)
    err_y = abs(ry + load_root["Vy"]) / max(abs(load_root["Vy"]), abs(ry), 1.0)
    err_z = abs(rz + load_root["Vz"]) / max(abs(load_root["Vz"]), abs(rz), 1.0)
    print(f"  rel force error (N,Vy,Vz) = {err_n:.3e}, {err_y:.3e}, {err_z:.3e}")


def _midspan_flatness(profile: list[dict[str, float]]) -> dict[str, float]:
    """Assess whether max |Nx| plateaus in the middle third of the segment."""
    if len(profile) < 3:
        return {
            "mid_over_root": float("nan"),
            "tip_over_mid": float("nan"),
            "mid_cv": float("nan"),
        }
    nx = np.array([row["max_abs_Nx"] for row in profile], dtype=float)
    n = len(nx)
    i0 = n // 3
    i1 = max(i0 + 1, (2 * n) // 3)
    mid = nx[i0:i1]
    mid_mean = float(np.mean(mid)) if mid.size else float("nan")
    mid_cv = float(np.std(mid) / (mid_mean + 1e-30)) if mid.size else float("nan")
    return {
        "mid_over_root": mid_mean / (float(nx[0]) + 1e-30),
        "tip_over_mid": float(nx[-1]) / (mid_mean + 1e-30),
        "mid_cv": mid_cv,
        "mid_i0": float(i0),
        "mid_i1": float(i1 - 1),
    }


def _decay_report(profile: list[dict[str, float]]) -> dict[str, float]:
    print("\n=== Spanwise decay of max |resultants| ===")
    print(f"{'row':>4} {'x[m]':>10} {'|Nx|':>12} {'|Nxy|':>12} {'|Ny|':>12}")
    for row in profile:
        print(
            f"{int(row['span_row']):4d} {row['x_m']:10.4f} "
            f"{row['max_abs_Nx']:12.4e} {row['max_abs_Nxy']:12.4e} "
            f"{row['max_abs_Ny']:12.4e}"
        )
    flat = _midspan_flatness(profile)
    if len(profile) >= 2:
        root_nx = profile[0]["max_abs_Nx"] + 1e-30
        tip_nx = profile[-1]["max_abs_Nx"]
        print(f"  |Nx|_outboard / |Nx|_root = {tip_nx / root_nx:.3f}")
        print(
            f"  midspan flatness: |Nx|_mid/|Nx|_root = {flat['mid_over_root']:.3f}, "
            f"|Nx|_outboard/|Nx|_mid = {flat['tip_over_mid']:.3f}, "
            f"mid CV = {flat['mid_cv']:.3f}"
        )
        print(
            "  (flat mid-thirds => tip/mid ~1 and mid CV small; "
            "tip/mid >> 1 => outboard cut artefact still dominating)"
        )
    return flat


def diagnose(
    *,
    fixture: str = "l_section",
    n_elem: int | None = None,
    n_span_rows: int = 12,
    diameter_multiples: float = 2.5,
    naca_code: str = "0012",
    chord_m: float = 1.0,
) -> dict[str, Any]:
    """Run one root-segment vs periodic comparison.

    Parameters
    ----------
    fixture
        ``\"l_section\"`` or ``\"naca\"`` (closed airfoil + web via Pipeline B).
    diameter_multiples
        Segment length as a multiple of the estimated root diameter.
    n_span_rows
        Number of element rows along the span (more rows for longer segments).
    """
    pb: Any | None = None
    if fixture == "naca":
        panels, pb = _naca_panels(naca_code=naca_code, chord_m=chord_m)
        if n_elem is None:
            n_elem = 6
    elif fixture == "l_section":
        panels = _l_section_panels()
        if n_elem is None:
            n_elem = _NE
    else:
        raise ValueError(f"unknown fixture {fixture!r}; expected 'l_section' or 'naca'")

    pb_bundle = _bundle_from_panels(panels, pb)
    diam = estimate_root_diameter_m(panels)
    L = float(diameter_multiples) * diam
    load_out = _outboard_loads_for_root_target(_LOAD_ROOT, segment_length_m=L)

    print("\n" + "=" * 72)
    print(
        f"FIXTURE={fixture}  D={diam:.4f} m  L/D={diameter_multiples:.2f}  "
        f"L={L:.4f} m  rows={n_span_rows}  n_elem/panel={n_elem}"
    )
    print(
        f"  load matching: My_out = My_root - Vz*L = "
        f"{_LOAD_ROOT['My']:.3e} - {_LOAD_ROOT['Vz']:.3e}*{L:.3e} = {load_out['My']:.3e}"
    )
    print("=" * 72)

    peri = run_section_periodic_bc_mitc4(pb_bundle, n_elements_per_panel=n_elem, **_LOAD_ROOT)
    peri_results = list(getattr(peri, "all_panel_results", []) or [])

    if shutil.which("ccx") is None:
        print("ccx not on PATH — skipping live root-segment solve.", file=sys.stderr)
        return {"skipped": True, "reason": "ccx missing"}

    with tempfile.TemporaryDirectory(prefix="ccx_root_seg_") as tmp:
        mesh, mapped, diag, inp = run_root_segment_ccx(
            panels,
            work_dir=tmp,
            segment_length_m=L,
            n_span_rows=n_span_rows,
            n_elements_per_panel=n_elem,
            job_name="root_segment",
            **load_out,
        )
        print(f"Wrote {inp}")
        _equilibrium_report(diag, _LOAD_ROOT, load_out)
        flat = _decay_report(mapped.decay_profile)

        print("\n=== Root-adjacent vs periodic RVE (peak |resultants|) ===")
        print(
            f"{'panel':20} {'field':6} {'root_seg':>12} {'periodic':>12} "
            f"{'ratio':>10}"
        )
        amp: dict[str, float] = {}
        max_nx_ratio = 0.0
        max_fi_ratio = 0.0
        for pi, p in enumerate(panels):
            root_rows = mapped.root_row[pi] if pi < len(mapped.root_row) else []
            peri_rows = peri_results[pi] if pi < len(peri_results) else []
            pk_r = _panel_peak_abs(root_rows)
            pk_p = _panel_peak_abs(peri_rows)
            label = str(getattr(p, "label", f"panel_{pi}"))
            for key in ("Nx", "Ny", "Nxy"):
                rr, pp = pk_r[key], pk_p[key]
                ratio = rr / pp if abs(pp) > 1e-30 else float("nan")
                amp[f"{label}:{key}"] = ratio
                if key == "Nx" and np.isfinite(ratio):
                    max_nx_ratio = max(max_nx_ratio, ratio)
                print(f"{label:20} {key:6} {rr:12.4e} {pp:12.4e} {ratio:10.3f}")
            fi_r = _max_hashin(root_rows, p.lam)
            fi_p = _max_hashin(peri_rows, p.lam)
            fi_ratio = fi_r / fi_p if abs(fi_p) > 1e-30 else float("nan")
            amp[f"{label}:Hashin"] = fi_ratio
            if np.isfinite(fi_ratio):
                max_fi_ratio = max(max_fi_ratio, fi_ratio)
            print(f"{label:20} {'Hashin':6} {fi_r:12.4e} {fi_p:12.4e} {fi_ratio:10.3f}")

        print(f"\nn_elements={len(mesh.elements)}, n_nodes={len(mesh.nodes)}")
        print(
            f"SUMMARY: max root/periodic |Nx| ratio = {max_nx_ratio:.3f}; "
            f"max Hashin ratio = {max_fi_ratio:.3f}; "
            f"tip/mid |Nx| = {flat.get('tip_over_mid', float('nan')):.3f}"
        )
        exceeds = max_nx_ratio > 1.15 or max_fi_ratio > 1.15
        print(
            f"Root exceeds RVE (ratio > 1.15)? {'YES' if exceeds else 'NO'}"
        )
        return {
            "skipped": False,
            "fixture": fixture,
            "diameter_multiples": float(diameter_multiples),
            "amplification": amp,
            "max_nx_ratio": max_nx_ratio,
            "max_fi_ratio": max_fi_ratio,
            "flatness": flat,
            "exceeds_rve": exceeds,
            "diag": {k: v for k, v in diag.items() if k != "decay_profile"},
            "decay_profile": mapped.decay_profile,
        }


def main() -> int:
    if shutil.which("ccx") is None:
        print("ERROR: ccx not found on PATH", file=sys.stderr)
        return 2

    # Sweep: short L-section (baseline), longer L-section, closed airfoil at L/D=2.5 and 5.
    cases = [
        dict(fixture="l_section", diameter_multiples=2.5, n_span_rows=8),
        dict(fixture="l_section", diameter_multiples=5.0, n_span_rows=16),
        dict(fixture="naca", diameter_multiples=2.5, n_span_rows=10, naca_code="0012"),
        dict(fixture="naca", diameter_multiples=5.0, n_span_rows=16, naca_code="0012"),
    ]
    results: list[dict[str, Any]] = []
    for kw in cases:
        results.append(diagnose(**kw))

    print("\n" + "=" * 72)
    print("COMPARISON TABLE")
    print("=" * 72)
    print(
        f"{'fixture':10} {'L/D':>6} {'tip/mid':>8} {'mid CV':>8} "
        f"{'max Nx r':>9} {'max FI r':>9} {'>RVE?':>6}"
    )
    for r in results:
        if r.get("skipped"):
            continue
        flat = r.get("flatness") or {}
        print(
            f"{r['fixture']:10} {r['diameter_multiples']:6.2f} "
            f"{flat.get('tip_over_mid', float('nan')):8.3f} "
            f"{flat.get('mid_cv', float('nan')):8.3f} "
            f"{r.get('max_nx_ratio', float('nan')):9.3f} "
            f"{r.get('max_fi_ratio', float('nan')):9.3f} "
            f"{'YES' if r.get('exceeds_rve') else 'no':>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
