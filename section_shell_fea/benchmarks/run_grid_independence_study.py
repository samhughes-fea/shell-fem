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
        help="all | smoke | fast | comma-separated layout keys (default: all)",
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
            "ci: production-subladder drift≤2 + last-two≤2 + Hashin/gov GCI≤10%%; "
            "study: full ladder≤2 + last-two≤1.25 + skin/web GCI≤5%%"
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
    args = parser.parse_args(argv)

    _by_layout, summaries, failures = run_grid_independence_study(
        args.keys,
        out_dir=Path(args.out_dir),
        require_mesh_qa=not args.no_mesh_qa_assert,
        gates=args.gates,
        progress=not args.quiet,
        load_source="synthetic" if args.synthetic else "cotide",
        zL=args.zL,
    )
    print(f"wrote {Path(args.out_dir) / 'gci_ladder.csv'}", flush=True)
    print(f"wrote {Path(args.out_dir) / 'gci_summary.csv'}", flush=True)
    print(f"wrote {Path(args.out_dir) / 'plots'}", flush=True)
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
