"""CLI: systematic M+P→S→T→Q grid independence across structural system keys.

Default load source is Cotide ``new_3.2m_rotor`` + ``NACA_63-415.dat``.

Examples::

    python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_grid_independence_study
    python -m ...run_grid_independence_study --keys smoke
    python -m ...run_grid_independence_study --synthetic
    python -m ...run_grid_independence_study --zL 0.25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    run_grid_independence_study,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mesh-density grid independence study across SYSTEM_TYPE_KEYS."
    )
    parser.add_argument(
        "--keys",
        default="all",
        help=(
            "all | smoke | watch | fast | comma-separated layout keys "
            "(default: all; watch = CI_WATCH_KEYS membrane residual set)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "blade_structure_utilities/fem_utilities/section_shell_fea/"
            "visualisation/output/gci_study_mitc4"
        ),
        help="Directory for gci_ladder.csv, gci_summary.csv, and plots/",
    )
    parser.add_argument(
        "--no-mesh-qa-assert",
        action="store_true",
        help="Record mesh_ok without raising on QA failure (still gates on mesh_ok)",
    )
    parser.add_argument(
        "--gates",
        choices=("ci", "study"),
        default="ci",
        help=(
            "ci: frozen N12+mid-σ production-subladder drift/GCI + mesh/trust; "
            "study: stricter N12/mid-σ/Hashin + governing_fi (Hashin not CI)"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-layout / per-rung progress (summary only at end)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use hard-coded synthetic loads + analytic NACA (not Cotide)",
    )
    parser.add_argument(
        "--zL",
        type=float,
        default=None,
        help="Station z/L for Cotide loads (default: max |m_YB| on worst envelope)",
    )
    parser.add_argument(
        "--interface",
        choices=(
            "shared",
            "transformed_basis",
            "shared_rotated",
            "hybrid_collinear",
            "hybrid_kind_crossing",
        ),
        default="shared",
        help=(
            "Junction constraint mode (default: shared production; "
            "hybrid_collinear = study-only shared collinear / TB non-collinear; "
            "hybrid_kind_crossing = study-only TB on kind-crossing∩non-collinear)"
        ),
    )
    parser.add_argument(
        "--junction-band-elements",
        type=int,
        default=None,
        help=(
            "Override junction-band pin count near strip ends "
            "(default: production JUNCTION_BAND_ELEMENTS)"
        ),
    )
    parser.add_argument(
        "--junction-band-step-over-h",
        type=float,
        default=None,
        help=(
            "Override junction-band pin spacing as a fraction of bulk h "
            "(default: production JUNCTION_BAND_STEP_OVER_H)"
        ),
    )
    parser.add_argument(
        "--junction-band-labels",
        default="",
        help=(
            "Comma-separated StripSourcePanel labels that receive study "
            "junction-band overrides (others stay production 3 / 0.5); "
            "e.g. skin:outer_skin:SS:bay_0"
        ),
    )
    parser.add_argument(
        "--junction-band-which",
        choices=("both", "start", "end"),
        default="both",
        help="Which strip end(s) get denser pins on labeled panels (default: both)",
    )
    args = parser.parse_args(argv)

    labels = tuple(
        x.strip() for x in str(args.junction_band_labels).split(",") if x.strip()
    )
    _by_layout, summaries, failures = run_grid_independence_study(
        args.keys,
        out_dir=Path(args.out_dir),
        require_mesh_qa=not args.no_mesh_qa_assert,
        gates=args.gates,
        progress=not args.quiet,
        load_source="synthetic" if args.synthetic else "cotide",
        zL=args.zL,
        interface_constraint_mode=args.interface,
        junction_band_elements=args.junction_band_elements,
        junction_band_step_over_h=args.junction_band_step_over_h,
        junction_band_panel_labels=labels or None,
        junction_band_which=args.junction_band_which,
    )
    out = Path(args.out_dir)
    print(f"wrote {out / 'gci_ladder.csv'}", flush=True)
    print(f"wrote {out / 'gci_summary.csv'}", flush=True)
    print(f"wrote {out / 'plots'}", flush=True)
    try:
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.stability_scorecard import (
            write_stability_scorecard,
        )

        json_path, md_path, _card = write_stability_scorecard(
            out / "gci_summary.csv", out_dir=out
        )
        print(f"wrote {json_path}", flush=True)
        print(f"wrote {md_path}", flush=True)
    except Exception as exc:  # noqa: BLE001 — scorecard must not abort the study
        print(f"scorecard warning: {exc}", flush=True)
    print(
        "note: regenerate scorecard anytime with "
        "benchmarks.run_stability_scorecard --study-dir <out>",
        flush=True,
    )
    print(f"layouts={len(summaries)} failures={len(failures)}", flush=True)
    for s in summaries:
        gov = s.governing_fi_last_two_drift
        gov_s = f"{gov:.3f}x" if gov == gov else "n/a"
        gci = s.governing_fi_gci_fine
        gci_s = f"{100.0 * gci:.2f}%" if gci == gci else "n/a"
        print(
            f"  {s.layout:8s} family={s.family} trust={s.all_trust_ok} "
            f"mesh={s.all_mesh_ok} gov_last_two={gov_s} gov_gci={gci_s}",
            flush=True,
        )
    if failures:
        print("FAIL table (layout, kind, metric, drift, limit):", flush=True)
        for f in failures:
            d = f"{f.drift:.3f}" if f.drift == f.drift else "nan"
            print(
                f"  {f.layout:8s} {f.kind:6s} {f.metric:28s} {d:>8s} / {f.limit}",
                flush=True,
            )
            if f.detail:
                print(f"           {f.detail}", flush=True)
        return 1
    print("PASS: all gates satisfied", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
