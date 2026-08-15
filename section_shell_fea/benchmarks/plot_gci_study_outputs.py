"""CLI: regenerate GCI overview + per-layout geometry/convergence plots from CSVs.

No solves — reads ``gci_ladder.csv``, ``gci_summary.csv``, and ``station_recipe.json``.

Examples::

    python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.plot_gci_study_outputs --study-dir visualisation/output/gci_study_mitc4
    python -m ...plot_gci_study_outputs --study-dir visualisation/output/gci_study_ccx --backend-label CCX --no-hashin
    python -m ...plot_gci_study_outputs --study-dir visualisation/output/gci_study_mitc4 --keys 3D-CN,4D-CN,4D-F,5D-CN,5D-F --out-subdir plots/ci_watch_layouts --layouts-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.plot_grid_independence import (
    plot_grid_independence_study,
    plot_study_layout_convergence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate GCI study PNGs (overview + geometry-inclusive layouts)."
    )
    parser.add_argument(
        "--study-dir",
        required=True,
        help="Directory containing gci_ladder.csv / gci_summary.csv / station_recipe.json",
    )
    parser.add_argument(
        "--keys",
        default="",
        help="Optional comma-separated layout keys (default: all layouts in ladder)",
    )
    parser.add_argument(
        "--out-subdir",
        default="plots/layouts",
        help="Per-layout PNG subdirectory under study-dir (default: plots/layouts)",
    )
    parser.add_argument(
        "--layouts-only",
        action="store_true",
        help="Skip aggregate overview plots; write layout composites only",
    )
    parser.add_argument(
        "--backend-label",
        default="",
        help="Label appended to overview plot titles (e.g. MITC4, CCX)",
    )
    parser.add_argument(
        "--no-hashin",
        action="store_true",
        help="Omit Hashin / quaternary overview plots (typical for CCX)",
    )
    parser.add_argument(
        "--mark-ci-watch",
        action="store_true",
        help="Force [CI watch] title tags on every layout in this run",
    )
    args = parser.parse_args(argv)

    study = Path(args.study_dir)
    if not study.is_dir():
        print(f"ERROR: study dir not found: {study}", file=sys.stderr, flush=True)
        return 2

    keys = tuple(k.strip() for k in str(args.keys).split(",") if k.strip()) or None
    written: list[Path] = []

    if not args.layouts_only and keys is None:
        written.extend(
            plot_grid_independence_study(
                study,
                backend_label=str(args.backend_label or ""),
                write_hashin=not bool(args.no_hashin),
            )
        )
        # plot_grid_independence_study already writes plots/layouts; if a custom
        # out-subdir was requested, write that set as well below when keys set.
        if str(args.out_subdir).replace("\\", "/") == "plots/layouts":
            print(f"wrote {len(written)} overview/layout plot(s) under {study / 'plots'}", flush=True)
            for p in written:
                print(f"  {p}", flush=True)
            return 0

    # Custom key subset and/or custom out-subdir (e.g. CI watch gallery).
    layout_paths = plot_study_layout_convergence(
        study,
        keys=keys,
        out_subdir=str(args.out_subdir),
        mark_ci_watch=True if args.mark_ci_watch or keys is not None else None,
    )
    written.extend(layout_paths)
    print(
        f"wrote {len(layout_paths)} layout plot(s) under {study / args.out_subdir}",
        flush=True,
    )
    for p in layout_paths:
        print(f"  {p}", flush=True)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
