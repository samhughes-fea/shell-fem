"""CLI: write stability_scorecard.json/.md from a GCI study directory.

Examples::

    python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_stability_scorecard
    python -m ...run_stability_scorecard --study-dir path/to/gci_study_mitc4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.stability_scorecard import (
    write_stability_scorecard,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build stability scorecard (JSON + MD) from gci_summary.csv."
    )
    parser.add_argument(
        "--study-dir",
        default=(
            "blade_structure_utilities/fem_utilities/section_shell_fea/"
            "visualisation/output/gci_study_mitc4"
        ),
        help="Directory containing gci_summary.csv",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Override path to gci_summary.csv (default: <study-dir>/gci_summary.csv)",
    )
    args = parser.parse_args(argv)
    study = Path(args.study_dir)
    summary = Path(args.summary_csv) if args.summary_csv else study / "gci_summary.csv"
    if not summary.is_file():
        print(f"error: missing summary CSV: {summary}", file=sys.stderr)
        return 2
    json_path, md_path, card = write_stability_scorecard(summary, out_dir=study)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(
        f"CI membrane magnitude: {card.ci_membrane_pass}/{card.n_layouts}  "
        f"watch cleared: "
        f"{sum(1 for k, v in card.watch_key_status.items() if v.get('ci_magnitude_clear'))}"
        f"/{len(card.watch_key_status)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
