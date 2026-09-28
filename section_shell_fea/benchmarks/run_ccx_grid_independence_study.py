"""CLI: CalculiX-driven membrane GCI study across SYSTEM_TYPE_KEYS.

Gates on ``peak_abs_n12`` only (Hashin / mid_sig / governing_fi unavailable from
live CCX membrane recovery). Requires ``ccx`` on PATH.

Examples::

    python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ccx_grid_independence_study --keys fast
    python -m ...run_ccx_grid_independence_study --keys all --gates ccx
    python -m ...run_ccx_grid_independence_study --keys smoke --gates study
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.convergence.grid_independence_study import (
    run_grid_independence_study,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CCX membrane grid-independence / GCI across SYSTEM_TYPE_KEYS."
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
            "visualisation/output/gci_study_ccx"
        ),
        help="Directory for gci_ladder.csv, gci_summary.csv, plots/, and ccx_work/",
    )
    parser.add_argument(
        "--no-mesh-qa-assert",
        action="store_true",
        help="Record mesh_ok without raising on QA failure (still gates on mesh_ok)",
    )
    parser.add_argument(
        "--gates",
        choices=("ccx", "study"),
        default="ccx",
        help=(
            "ccx: production-subladder nxy drift≤2 + last-two≤2 + nxy GCI≤10%%; "
            "study: full ladder≤2 + last-two≤1.25 + nxy GCI≤5%% (still nxy-only)"
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
    parser.add_argument("--ccx-exe", type=str, default="ccx")
    args = parser.parse_args(argv)

    if shutil.which(str(args.ccx_exe)) is None:
        print(
            f"ERROR: CalculiX executable {args.ccx_exe!r} not found on PATH",
            file=sys.stderr,
            flush=True,
        )
        return 2

    _by_layout, summaries, failures = run_grid_independence_study(
        args.keys,
        out_dir=Path(args.out_dir),
        require_mesh_qa=not args.no_mesh_qa_assert,
        gates=args.gates,
        progress=not args.quiet,
        load_source="synthetic" if args.synthetic else "cotide",
        zL=args.zL,
        backend="ccx",
        ccx_exe=str(args.ccx_exe),
    )
    out = Path(args.out_dir)
    print(f"wrote {out / 'gci_ladder.csv'}", flush=True)
    print(f"wrote {out / 'gci_summary.csv'}", flush=True)
    print(f"wrote {out / 'plots'}", flush=True)
    print(f"layouts={len(summaries)} failures={len(failures)}", flush=True)
    for s in summaries:
        nxy_gci = s.gci_fine.get("peak_abs_n12", {})
        skin = nxy_gci.get("skin", float("nan"))
        web = nxy_gci.get("web", float("nan"))
        skin_s = f"{100.0 * skin:.2f}%" if skin == skin else "n/a"
        web_s = f"{100.0 * web:.2f}%" if web == web else "n/a"
        print(
            f"  {s.layout:8s} family={s.family} trust={s.all_trust_ok} "
            f"mesh={s.all_mesh_ok} nxy_gci_skin={skin_s} nxy_gci_web={web_s}",
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
    print("PASS: all CCX membrane gates satisfied", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
