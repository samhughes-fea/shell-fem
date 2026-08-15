"""Defect K harness smoke: both junction modes produce trusted cascade rows."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    run_cascade_at_frac,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.defect_k_mesh_sweep import (
    MODES,
)


def test_both_interface_modes_mesh_ok_at_coarse_frac():
    """Production-shaped run: skin_offset + shared / transformed_basis both trust.

    Cap ties compose with cluster-frame MPCs via master expansion (P0).
    Full dual ladder / promote decision lives in the CLI benchmark.
    """
    for mode in MODES:
        row = run_cascade_at_frac(
            0.024,
            cap_tie_mode="skin_offset",
            interface_constraint_mode=mode,
        )
        assert row.mesh_ok, mode
        assert row.cascade_trust_ok, mode
        assert row.peak_hashin["skin"] >= 0.0, mode
        assert row.cap_tie_mode == "skin_offset"
        assert row.interface_constraint_mode == mode
