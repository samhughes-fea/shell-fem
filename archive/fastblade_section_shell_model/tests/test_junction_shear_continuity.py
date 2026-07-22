"""Junction shear continuity characterization at skin–web interfaces (Defect K)."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.section_shell_model.api.pipeline_b import (
    build_pipeline_b_section,
    build_tapered_pipeline_b_section,
)
from blade_precompute.section_shell_model.api.periodic import run_section_tapered_periodic_bc_mitc4
from blade_precompute.section_shell_model.processing.diagnostics.defect_k_mesh_sweep import (
    run_mesh_sweep,
    secondary_residuals_for_mode,
)


def test_junction_shear_flow_nonzero_under_tapered_shear():
    """Skin–web junctions carry transverse shear under Vy; documents shear-lag residual."""
    pb = build_tapered_pipeline_b_section(
        "2412", [0.35], chord_inboard_m=1.0, chord_outboard_m=0.95, L_x=0.4,
    )
    bundle = run_section_tapered_periodic_bc_mitc4(
        pb, Vy=6.0e3, Vz=2.0e3, n_elements_per_panel=6,
    )
    q_vals: list[float] = []
    for panel_res in bundle.all_panel_results or []:
        for row in panel_res or []:
            if row is None:
                continue
            qx, qy = getattr(row, "Qx", None), getattr(row, "Qy", None)
            if qx is not None and qy is not None:
                q_vals.append(float(np.hypot(qx, qy)))
    assert q_vals, "expected transverse shear resultants at junction panels"
    assert max(q_vals) > 0.0
    # Residual: traction continuity at non-collinear junctions is not fully enforced;
    # this test guards against regressions to zero shear at tapered interfaces.


def test_defect_k_shared_secondary_residual_baseline():
    """Informational Defect K baseline under production ``shared`` junctions.

    Primary reaction metrics remain authoritative. Secondary ``dTx_rel`` /
    ``Tx_rel_cluster`` are elevated at non-collinear skin–web joints under ``shared``
    and are not expected to refine to zero (Spike 1B parked ``transformed_basis`` K6).
    Loose upper bound locks against accidental silence (all-zero / NaN residuals).
    """
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    row = secondary_residuals_for_mode(
        pb.panels,
        interface_constraint_mode="shared",
        n_elements_per_panel=8,
    )
    assert row["n_cluster"] >= 1.0
    # Skin–web secondary residual is the Defect K symptom — finite and typically > 0.5.
    assert np.isfinite(row["sw_dTx"]) or np.isfinite(row["cl_Tx"])
    peak = max(
        v for v in (row["sw_dTx"], row["cl_Tx"]) if np.isfinite(v)
    )
    assert peak > 0.05, f"expected elevated secondary residual under shared, got {peak}"
    # Soft ceiling: residual magnitude is inflated but not astronomical on this fixture.
    assert peak < 50.0, f"unexpectedly huge secondary residual {peak}"


def test_defect_k_mesh_sweep_runs_shared_and_transformed():
    """Harness smoke: two modes × two meshes produce finite CSV-shaped rows."""
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    rows = run_mesh_sweep(
        pb.panels,
        n_elem_list=(8, 12),
        modes=("shared", "transformed_basis"),
        out_csv=None,
    )
    assert len(rows) == 4
    modes = {r["mode"] for r in rows}
    assert modes == {"shared", "transformed_basis"}
    for r in rows:
        assert r["n_elem"] in (8.0, 12.0)
        # At least cluster metric populated for the NACA+spar topology.
        assert r["n_cluster"] >= 1.0
