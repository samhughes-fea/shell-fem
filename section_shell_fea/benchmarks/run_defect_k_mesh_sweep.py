"""CLI: Defect K cascade mesh sweep (shared vs transformed_basis).

Example::

    python -m blade_structure_utilities.fem_utilities.section_shell_fea.benchmarks.run_defect_k_mesh_sweep
"""

from __future__ import annotations

from pathlib import Path

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.convergence.defect_k_mesh_sweep import (
    run_defect_k_mesh_sweep,
    summarize_hashin_drift,
)


def main() -> None:
    out = Path("blade_structure_utilities/fem_utilities/section_shell_fea/visualisation/demos/output/mesh_sweep_secondary.csv")
    by_mode = run_defect_k_mesh_sweep(out_csv=out)
    drifts = summarize_hashin_drift(by_mode)
    print(f"wrote {out}")
    for mode, per_kind in drifts.items():
        print(f"{mode}: " + ", ".join(f"{k}={v:.3f}x" for k, v in per_kind.items()))


if __name__ == "__main__":
    main()
