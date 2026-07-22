"""CLI for Defect K secondary residual mesh sweep (``SHELL_MODE_SWEEP=1``).

Usage:
  set SHELL_MODE_SWEEP=1
  python -m blade_precompute.section_shell_model.benchmarks.run_defect_k_mesh_sweep

Writes ``examples/output/mesh_sweep_secondary.csv`` (see ``docs/SHELL_MODEL.md``).
"""

from __future__ import annotations

from blade_precompute.section_shell_model.processing.diagnostics.defect_k_mesh_sweep import (
    main,
    shell_mode_sweep_enabled,
)


if __name__ == "__main__":
    if not shell_mode_sweep_enabled():
        print("Set SHELL_MODE_SWEEP=1 to run the Defect K mesh sweep.")
        raise SystemExit(2)
    main()
