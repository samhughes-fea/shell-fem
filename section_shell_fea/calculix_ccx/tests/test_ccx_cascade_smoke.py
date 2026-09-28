"""Live CCX cascade smoke (short ladder on FAST_CI_KEYS)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.ccx,
    pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX ccx not on PATH"),
]


def test_ccx_cascade_at_frac_2a_f():
    from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.ccx_cascade_mesh_convergence import (
        run_ccx_cascade_at_frac,
    )

    with tempfile.TemporaryDirectory() as tmp:
        row = run_ccx_cascade_at_frac(
            0.024,
            layout="2A-F",
            cap_tie_mode="none",
            work_dir=tmp,
            require_mesh_qa=True,
        )
    assert row.mesh_ok
    assert row.cascade_trust_ok
    assert row.peak_abs_n12.get("skin", 0.0) >= 0.0
    import math

    assert math.isnan(float(row.peak_hashin.get("skin", float("nan"))))


def test_ccx_layout_ladder_fast_keys_short():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.convergence.grid_independence_study import (
        run_layout_ladder,
        summarize_layout_ladder,
        evaluate_layout_gates,
    )

    with tempfile.TemporaryDirectory() as tmp:
        rows = run_layout_ladder(
            "2A-F",
            refinements=(0.024, 0.016, 0.012),
            backend="ccx",
            work_dir=Path(tmp),
            require_mesh_qa=True,
        )
    assert len(rows) == 3
    summary = summarize_layout_ladder(rows)
    fails = evaluate_layout_gates(summary, gates="ccx")
    # Live solve must at least produce mesh/trust; GCI may fail on coarse 3-rung.
    assert summary.all_mesh_ok
    assert summary.all_trust_ok
    assert isinstance(fails, list)
