"""
Benchmark: model the **same** section with **different shell elements** and compare.

For a small curated set of section types (airfoil + web layout) with specific
**geometric and material** configurations, this builds the section *once* (a
theory-agnostic centerline strip mesh via ``build_pipeline_b_section``) and solves
it through two shell-element formulations on the **identical** mesh:

* **FSDT (MITC4)**       — ``run_section_periodic_bc_mitc4``
* **CalculiX S4R (ccx)** — ``run_section_periodic_bc_ccx``

Both native paths consume the same ``shell_inputs`` (the theory-agnostic seam), so the only
difference is the element kinematics. The ccx peer uses the same periodic unit-cell deck and
MITC4 energy ``K6`` for mode amplitudes (see ``docs/PERIODIC_CCX_HOMOGENIZATION.md``). ``ccx``
must be on ``PATH`` — this benchmark has nothing to compare without its peer element, so it
fails loudly rather than degrading to a single-element report.

The benchmark then compares:

1. **Shell-element resultants** — per-element L2 relative error FSDT↔ccx S4R
   (``compare_resultants``) and the equilibrium section integrals
   ``Fx = ∮ Nx ds``, ``My``, ``Mz`` about the elastic centroid
   (``integrated_section_resultants``) for each element vs the applied loads.
2. **CLT recovery** — ply-level Hashin-envelope failure index per element
   (``sweep_panel_clpt_fi``), reduced to a per-panel max FI for each element
   formulation and the FSDT↔ccx S4R delta.

Expectations (documented, not asserted here):
  * FSDT and ccx S4R **match at the section-integral level** (``Fx/My/Mz`` to
    round-off) but differ **locally per element** — S4R's independent kinematics
    redistribute stress; the redistribution is self-equilibrated.
  * A single prismatic periodic cell produces ~zero transverse-shear /
    interlaminar stress (``Vz = dMy/dx`` needs a spanwise gradient), so this
    benchmark compares **in-plane resultants + CLT FI** only — it makes no
    interlaminar claim.

Outputs (under ``examples/output/benchmark_shell_elements/``):
    benchmark_summary.json            — full machine-readable results
    <config>__resultants_overlay.png  — FSDT vs ccx S4R resultants on the ref panel
    <config>__relerror_bars.png       — per-field L2 rel error, grouped by load
    <config>__panel_maxfi_bars.png    — per-panel max Hashin FI, FSDT vs ccx S4R

Run from repo root (requires ``ccx`` on ``PATH``)::

    python blade_precompute/section_shell_model/examples/benchmark_shell_elements.py

Optional CLI: pass config names to run a subset, e.g.::

    python blade_precompute/section_shell_model/examples/benchmark_shell_elements.py \
        naca2412_twinweb_carbon naca2412_twinweb_glass
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HAS_CCX = shutil.which("ccx") is not None

# Number of elements per panel (matches validation/section_integrals helpers).
NE = 8

# In-plane + bending fields used for the rel-error bar chart (Qx/Qy are ~None in a
# single prismatic periodic cell, so they are excluded from the plot but kept in JSON).
INPLANE_FIELDS = ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy")

# Shared load cases (label -> beam-level section loads).
LOAD_CASES: dict[str, dict[str, float]] = {
    "My": {"My": 1.0e5},
    "T": {"T": 1.0e5},
    "N+My": {"N": 1.0e6, "My": 5.0e4},
}


@dataclass(frozen=True)
class BenchmarkConfig:
    """One section configuration: section type + geometry + per-component materials."""

    name: str
    naca_code: str
    web_x_positions: list[float]
    chord_m: float
    skin_lam: Any
    web_lam: Any
    cap_lam: Any


def _build_configs() -> list[BenchmarkConfig]:
    """Curated 4-row set; #2 vs #3 isolate material, #1/#2/#4 isolate geometry/section."""
    from blade_precompute.section_shell_model.pre_processing.constitutive.default_laminates import (
        WallLaminate,
    )

    # --- material families (representative [0/90/90/0] equal-gauge stacks) -----
    # Carbon-CFRP class.
    carbon_skin = WallLaminate(E=45.0e9, nu=0.30, t=1.0e-3, n_plies=4, name="CFRP skin")
    carbon_web = WallLaminate(E=43.0e9, nu=0.30, t=1.0e-3, n_plies=4, name="CFRP web")
    carbon_cap = WallLaminate(E=120.0e9, t=20.0e-3, n_plies=4, name="CFRP UD cap")
    # Glass-GFRP class (lower modulus, thicker gauge).
    glass_skin = WallLaminate(E=25.0e9, nu=0.28, t=1.8e-3, n_plies=6, name="GFRP skin")
    glass_web = WallLaminate(E=23.0e9, nu=0.28, t=1.8e-3, n_plies=6, name="GFRP web")
    glass_cap = WallLaminate(E=40.0e9, t=22.0e-3, n_plies=4, name="GFRP UD cap")

    return [
        BenchmarkConfig(
            name="naca0012_singleweb_carbon",
            naca_code="0012",
            web_x_positions=[0.35],
            chord_m=1.0,
            skin_lam=carbon_skin, web_lam=carbon_web, cap_lam=carbon_cap,
        ),
        BenchmarkConfig(
            name="naca2412_twinweb_carbon",
            naca_code="2412",
            web_x_positions=[0.25, 0.60],
            chord_m=1.5,
            skin_lam=carbon_skin, web_lam=carbon_web, cap_lam=carbon_cap,
        ),
        BenchmarkConfig(
            name="naca2412_twinweb_glass",
            naca_code="2412",
            web_x_positions=[0.25, 0.60],
            chord_m=1.5,
            skin_lam=glass_skin, web_lam=glass_web, cap_lam=glass_cap,
        ),
        BenchmarkConfig(
            name="naca4415_multiweb_hybrid",
            naca_code="4415",
            web_x_positions=[0.20, 0.45, 0.70],
            chord_m=2.0,
            skin_lam=carbon_skin, web_lam=glass_web, cap_lam=carbon_cap,
        ),
    ]


def _lam_summary(lam: Any) -> dict[str, Any]:
    return {
        "name": getattr(lam, "name", ""),
        "E_pa": float(getattr(lam, "E", float("nan"))),
        "t_m": float(getattr(lam, "t", float("nan"))),
        "nu": float(getattr(lam, "nu", float("nan"))),
        "n_plies": int(getattr(lam, "n_plies", 0)),
    }


def _panel_max_fi(panels: list, panel_results: list, strengths: dict[str, float]) -> list[float]:
    """Per-panel max Hashin FI over its elements (0.0 for empty panels)."""
    import logging
    import numpy as np

    from blade_precompute.section_shell_model.engine.local_clpt_shell import (
        sweep_panel_clpt_fi,
    )

    log = logging.getLogger(__name__)
    _FI_ABSURD = 1.0e6
    out: list[float] = []
    for panel, panel_res in zip(panels, panel_results):
        if not panel_res:
            out.append(0.0)
            continue
        plies = panel.lam.build_plies()
        clpt = sweep_panel_clpt_fi(panel_res, plies, **strengths)
        fis: list[float] = []
        for r in clpt:
            if len(r.fi) == 0:
                fis.append(0.0)
                continue
            fi_val = float(np.max(r.fi))
            if not np.isfinite(fi_val) or fi_val > _FI_ABSURD:
                log.warning("non-physical Hashin FI %.3e; treating as NaN", fi_val)
                fis.append(float("nan"))
            else:
                fis.append(fi_val)
        finite = [f for f in fis if np.isfinite(f)]
        out.append(max(finite) if finite else 0.0)
    return out


# --------------------------------------------------------------------------- #
# Plotting (small, self-contained — Agg backend set in main()).
# --------------------------------------------------------------------------- #

def _save_resultants_overlay(
    path: Path, plt, bm: Any, bc: Any, load_label: str
) -> None:
    """FSDT (MITC4) vs ccx S4R resultants along the reference (skin) panel."""
    import numpy as np

    fields = ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy")
    pm = (bm.all_panel_results or [[]])[0]
    pc = (bc.all_panel_results or [[]])[0]
    n = min(len(pm), len(pc))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    x = np.arange(n)
    for ax, fld in zip(axes.ravel(), fields):
        ym = [float(getattr(pm[i], fld)) for i in range(n)]
        yc = [float(getattr(pc[i], fld)) for i in range(n)]
        ax.plot(x, ym, "-o", ms=3, lw=1.2, label="FSDT (MITC4)")
        ax.plot(x, yc, "-^", ms=3, lw=1.2, label="ccx S4R")
        ax.set_title(fld)
        ax.set_xlabel("element index (ref panel)")
        ax.grid(True, alpha=0.3)
    axes.ravel()[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Reference-panel resultants — load {load_label}", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_relerror_bars(
    path: Path,
    plt,
    per_load_rel: dict[str, dict[str, float]],
) -> None:
    """Grouped bars: per-field L2 rel error, FSDT vs ccx S4R, one subplot."""
    import numpy as np

    loads = list(per_load_rel.keys())
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    nfields = len(INPLANE_FIELDS)
    width = 0.8 / max(len(loads), 1)
    xpos = np.arange(nfields)
    for j, load_label in enumerate(loads):
        rel = per_load_rel[load_label]
        vals = [float(rel.get(f"rel_{f}", float("nan"))) for f in INPLANE_FIELDS]
        ax.bar(xpos + j * width, vals, width=width, label=f"load {load_label}")
    ax.set_xticks(xpos + 0.4 - width / 2)
    ax.set_xticklabels(INPLANE_FIELDS)
    ax.set_ylabel("L2 relative error")
    ax.set_title("Per-field resultant relative error (FSDT vs ccx S4R)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_panel_maxfi_bars(
    path: Path, plt, labels: list[str], per_load_fi: dict[str, dict[str, list[float]]]
) -> None:
    """Per-panel max Hashin FI (FSDT vs ccx S4R), one subplot row per load case."""
    import numpy as np

    loads = list(per_load_fi.keys())
    fig, axes = plt.subplots(len(loads), 1, figsize=(max(8, 0.5 * len(labels) + 4), 3.2 * len(loads)))
    if len(loads) == 1:
        axes = [axes]
    xpos = np.arange(len(labels))
    for ax, load_label in zip(axes, loads):
        fi_m = per_load_fi[load_label]["fsdt"]
        fi_c = per_load_fi[load_label]["ccx"]
        ax.bar(xpos - 0.2, fi_m, width=0.4, label="FSDT (MITC4)")
        ax.bar(xpos + 0.2, fi_c, width=0.4, label="ccx S4R")
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("max FI")
        ax.set_title(f"Per-panel max Hashin FI — load {load_label}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def _run_config(cfg: BenchmarkConfig, out_dir: Path, plt) -> dict[str, Any]:
    import numpy as np

    from blade_precompute.section_shell_model.api.periodic import (
        run_section_periodic_bc_mitc4,
    )
    from blade_precompute.section_shell_model.api.pipeline_b import (
        build_pipeline_b_section,
    )
    from blade_precompute.section_shell_model.calculix_ccx.periodic import (
        run_section_periodic_bc_ccx,
    )
    from blade_precompute.section_shell_model.engine.local_clpt_shell import (
        default_skin_strengths_pa,
    )
    from blade_precompute.section_shell_model.validation.section_integrals import (
        compare_resultants,
        elastic_centroid,
        integrated_section_resultants,
        section_element_geometry,
    )

    pb = build_pipeline_b_section(
        cfg.naca_code,
        cfg.web_x_positions,
        chord_m=cfg.chord_m,
        skin_lam=cfg.skin_lam,
        web_lam=cfg.web_lam,
        cap_lam=cfg.cap_lam,
    )
    geom = section_element_geometry(pb.panels, NE)
    y_e, z_e = elastic_centroid(pb.panels)
    panel_labels = [
        str(getattr(p, "label", None) or f"panel_{i}") for i, p in enumerate(pb.panels)
    ]
    strengths = default_skin_strengths_pa()

    loads_out: dict[str, Any] = {}
    per_load_rel: dict[str, dict[str, float]] = {}
    per_load_fi: dict[str, dict[str, list[float]]] = {}
    overlay_bundles: dict[str, tuple[Any, Any]] = {}

    print(f"\n=== config: {cfg.name}  (NACA{cfg.naca_code}, webs={cfg.web_x_positions}, "
          f"chord={cfg.chord_m} m, n_panels={len(pb.panels)}) ===")
    print(f"  elastic centroid (y_e, z_e) = ({y_e:.5f}, {z_e:.5f}) m")

    for load_label, load in LOAD_CASES.items():
        bm = run_section_periodic_bc_mitc4(pb, n_elements_per_panel=NE, **load)   # FSDT
        with tempfile.TemporaryDirectory(prefix="bench_ccx_") as ccx_tmp:
            bc = run_section_periodic_bc_ccx(
                pb, n_elements_per_panel=NE, ccx_work_dir=ccx_tmp, **load
            )   # ccx S4R
        overlay_bundles[load_label] = (bm, bc)

        rel = compare_resultants(
            bm.all_panel_results, bc.all_panel_results, fields=INPLANE_FIELDS
        )
        per_load_rel[load_label] = rel

        im = integrated_section_resultants(bm, geom, y_e=y_e, z_e=z_e)
        ic = integrated_section_resultants(bc, geom, y_e=y_e, z_e=z_e)

        fi_m = _panel_max_fi(pb.panels, bm.all_panel_results or [], strengths)
        fi_c = _panel_max_fi(pb.panels, bc.all_panel_results or [], strengths)
        per_load_fi[load_label] = {"fsdt": fi_m, "ccx": fi_c}

        max_fi_m = max(fi_m) if fi_m else 0.0
        max_fi_c = max(fi_c) if fi_c else 0.0

        from blade_precompute.section_shell_model.api.station_recovery import (
            run_station_periodic_hashin,
        )

        hashin_audit = run_station_periodic_hashin(pb, n_elements_per_panel=NE, **load)

        load_entry: dict[str, Any] = {
            "applied": load,
            "rel_error_fsdt_vs_ccx": rel,
            "section_integrals": {"fsdt": im, "ccx": ic},
            "max_fi": {
                "fsdt": max_fi_m,
                "ccx": max_fi_c,
                "abs_delta_fsdt_ccx": abs(max_fi_m - max_fi_c),
            },
            "per_panel_max_fi": {"fsdt": fi_m, "ccx": fi_c, "labels": panel_labels},
            "hashin_audit": {
                "max_fi_hashin": float(hashin_audit["max_fi_hashin"]),
                "needs_review": bool(hashin_audit["needs_review"]),
                "provenance_breakdown": dict(hashin_audit["provenance_breakdown"]),
                "governing_element": dict(hashin_audit["governing_element"]),
            },
        }
        loads_out[load_label] = load_entry

        # Console line: section-integral agreement + worst in-plane rel error + max FI.
        scale = max(abs(im["Fx"]), abs(im["My"]), abs(im["Mz"]), 1.0)
        sec_drel = max(abs(im[k] - ic[k]) for k in ("Fx", "My", "Mz")) / scale
        worst_rel = max(
            (float(rel.get(f"rel_{f}", 0.0)) for f in INPLANE_FIELDS
             if np.isfinite(rel.get(f"rel_{f}", float("nan")))),
            default=float("nan"),
        )
        print(
            f"  load {load_label:<5}  section-integral |FSDT-ccx|/scale={sec_drel:.2e}  "
            f"worst per-elem rel={worst_rel:.3e}  "
            f"maxFI FSDT={max_fi_m:.3e} ccx={max_fi_c:.3e}"
        )

    # --- figures (per config) ----------------------------------------------
    # Resultants overlay uses the first load case for a representative panel view.
    first_load = next(iter(LOAD_CASES))
    bm0, bc0 = overlay_bundles[first_load]
    _save_resultants_overlay(
        out_dir / f"{cfg.name}__resultants_overlay.png", plt, bm0, bc0, first_load
    )
    _save_relerror_bars(
        out_dir / f"{cfg.name}__relerror_bars.png",
        plt,
        per_load_rel,
    )
    _save_panel_maxfi_bars(
        out_dir / f"{cfg.name}__panel_maxfi_bars.png", plt, panel_labels, per_load_fi
    )

    return {
        "naca_code": cfg.naca_code,
        "web_x_positions": cfg.web_x_positions,
        "chord_m": cfg.chord_m,
        "materials": {
            "skin": _lam_summary(cfg.skin_lam),
            "web": _lam_summary(cfg.web_lam),
            "cap": _lam_summary(cfg.cap_lam),
        },
        "elastic_centroid": [float(y_e), float(z_e)],
        "n_panels": len(pb.panels),
        "panel_labels": panel_labels,
        "loads": loads_out,
    }


def main() -> None:
    if not _HAS_CCX:
        raise SystemExit(
            "benchmark_shell_elements requires `ccx` on PATH: this benchmark compares "
            "FSDT (MITC4) against CalculiX S4R and has nothing to report without its peer "
            "element. Install CalculiX and ensure `ccx` is on PATH before running."
        )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = _build_configs()
    by_name = {c.name: c for c in configs}

    argv = [a.strip() for a in sys.argv[1:] if a.strip()]
    if argv:
        unknown = [a for a in argv if a not in by_name]
        if unknown:
            raise SystemExit(
                f"Unknown config name(s): {unknown}. Available: {sorted(by_name)}"
            )
        selected = [by_name[a] for a in argv]
    else:
        selected = configs

    out_dir = Path(__file__).resolve().parent / "output" / "benchmark_shell_elements"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark_shell_elements] {len(selected)} config(s), NE={NE}, "
          f"loads={list(LOAD_CASES)}")

    results: dict[str, Any] = {}
    for cfg in selected:
        results[cfg.name] = _run_config(cfg, out_dir, plt)

    summary = {
        "n_elements_per_panel": NE,
        "load_cases": LOAD_CASES,
        "inplane_fields": list(INPLANE_FIELDS),
        "notes": (
            "FSDT (MITC4) vs CalculiX S4R (ccx) on the same theory-agnostic mesh. "
            "Section integrals match to ~10 % (ccx uses MITC4 energy K6 + S4R stress fields). "
            "Per-element resultants differ by element formulation. Single prismatic cell => "
            "no interlaminar."
        ),
        "configs": results,
    }
    summary_path = out_dir / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[benchmark_shell_elements] wrote {summary_path}")


if __name__ == "__main__":
    main()
