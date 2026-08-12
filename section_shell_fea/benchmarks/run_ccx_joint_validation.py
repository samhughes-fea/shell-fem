"""CLI: CalculiX joint-modeling validation (shared nodes + cap *TIE).

Example::

    python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ccx_joint_validation
    python -m ...run_ccx_joint_validation --live-ccx --layouts 3D-F,5D-F
"""

from __future__ import annotations

import argparse
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.joint_validation import (
    run_joint_validation,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("blade_structure_utilities/fem_utilities/section_shell_fea/visualisation/demos/output/ccx_joint_validation"),
        help="Directory for .inp decks and optional ccx workdirs",
    )
    p.add_argument(
        "--layouts",
        type=str,
        default="3D-F",
        help="Comma-separated catalogue layouts with spar caps (e.g. 3D-F,5D-F)",
    )
    p.add_argument(
        "--live-ccx",
        action="store_true",
        help="Run external ccx solves (requires ccx on PATH)",
    )
    p.add_argument(
        "--no-mitc4",
        action="store_true",
        help="Skip MITC4 comparison solves (deck checks only)",
    )
    p.add_argument("--ccx-exe", type=str, default="ccx")
    args = p.parse_args(argv)

    layouts = tuple(s.strip() for s in str(args.layouts).split(",") if s.strip())
    report = run_joint_validation(
        args.work_dir,
        layouts=layouts,
        run_live_ccx=bool(args.live_ccx),
        run_mitc4=not bool(args.no_mitc4),
        ccx_exe=str(args.ccx_exe),
    )
    print(report.summary())
    summary_path = Path(args.work_dir) / "joint_validation_summary.txt"
    summary_path.write_text(report.summary() + "\n", encoding="utf-8")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
