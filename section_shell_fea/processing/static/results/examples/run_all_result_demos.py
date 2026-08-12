"""Regenerate committed PNG demos for secondary / tertiary / quaternary results docs.

Synthetic / helper-level only — not Shell*ResultsOrchestrator.compute() artefacts.
See ``../docs/README.md`` (demo policy).

Usage (from repository root)::

    python blade_structure_utilities/fem_utilities/section_shell_fea/processing/static/results/examples/run_all_result_demos.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_RESULTS = Path(__file__).resolve().parents[1]
_SCRIPTS = (
    _RESULTS / "compute_secondary" / "examples" / "plot_secondary_demo.py",
    _RESULTS / "compute_tertiary" / "examples" / "plot_x3_recovery_demo.py",
    _RESULTS / "compute_quaternary" / "examples" / "plot_fi_demo.py",
)


def main() -> int:
    for script in _SCRIPTS:
        if not script.is_file():
            print(f"missing: {script}", file=sys.stderr)
            return 1
        rel = script.relative_to(_RESULTS)
        print(f"=== {rel} ===", flush=True)
        proc = subprocess.run([sys.executable, str(script)], check=False)
        if proc.returncode != 0:
            print(f"failed ({proc.returncode}): {rel}", file=sys.stderr)
            return proc.returncode
    print("all result demos ok", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
