"""CLI: CI-watch kinematics escalate (transformed_basis measurement only).

Runs the five D-family CI watch keys under ``transformed_basis`` with production
junction-band defaults. Does **not** promote TB — Phase B2 Hashin/gov bar is
separate. Compare against ``gci_study_mitc4`` shared baselines.

Example::

    python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_ci_watch_kinematics_spike
"""

from __future__ import annotations

import sys
from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_grid_independence_study import (
    main as _gci_main,
)

CI_WATCH_KEYS = "3D-CN,4D-CN,4D-F,5D-CN,5D-F"
DEFAULT_OUT = (
    "blade_structure_utilities/fem_utilities/section_shell_fea/"
    "visualisation/output/gci_study_mitc4_tb_ci_watch"
)


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not any(a.startswith("--keys") for a in args):
        args = ["--keys", CI_WATCH_KEYS, *args]
    if not any(a.startswith("--out-dir") for a in args):
        args = [*args, "--out-dir", DEFAULT_OUT]
    if not any(a.startswith("--interface") for a in args):
        args = [*args, "--interface", "transformed_basis"]
    if not any(a.startswith("--gates") for a in args):
        args = [*args, "--gates", "ci"]
    print(
        "[ci-watch-kinematics] measurement-only TB spike; "
        "production default remains shared (Phase B2).",
        flush=True,
    )
    return int(_gci_main(args))


if __name__ == "__main__":
    sys.exit(main())
