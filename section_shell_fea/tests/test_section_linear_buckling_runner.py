"""SectionLinearBucklingRunner: P→S cascade + station fold from static secondary."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.buckling_types import (
    PanelBucklingResult,
    ShellPanelBucklingInput,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
    ShellStationProvenance,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    build_shell_panel_topology,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.operations.solve_eigen import (
    solve_panel_linear_buckling_eigen,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.compute_secondary import (
    ShellBucklingSecondaryResultsOrchestrator,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.containers import (
    ShellBucklingPrimaryResultSet,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.linear_buckling import (
    nested_resultants_from_static_cascade,
    run_section_linear_buckling_from_static,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.linear_static import (
    LinearStaticCascadeResult,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.station_recovery import (
    build_fsdt_station_analysis,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_buckling_runner import (
    SectionLinearBucklingRunner,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)
from blade_structure_utilities.mesh_utilities.section_mesh.engine.strip_mesh import (
    build_section_strip_mesh,
)


def _strip_mesh(panel_topology, *, n_elements_per_panel: int = 3):
    return build_section_strip_mesh(
        panel_topology.shell_inputs,
        n_elements_per_panel=n_elements_per_panel,
        min_elements_per_panel=2,
    )


def _isotropic_abd(*, E: float = 70e9, nu: float = 0.3, t: float = 0.005) -> np.ndarray:
    """Simple isotropic ABD for unit tests."""
    G = E / (2.0 * (1.0 + nu))
    A11 = E * t / (1.0 - nu**2)
    A12 = nu * A11
    A66 = G * t
    D11 = E * t**3 / (12.0 * (1.0 - nu**2))
    D12 = nu * D11
    D66 = G * t**3 / 12.0
    abd = np.zeros((6, 6), dtype=float)
    abd[0, 0] = A11
    abd[1, 1] = A11
    abd[0, 1] = abd[1, 0] = A12
    abd[2, 2] = A66
    abd[3, 3] = D11
    abd[4, 4] = D11
    abd[3, 4] = abd[4, 3] = D12
    abd[5, 5] = D66
    return abd


def test_assessment_bi_from_lambda() -> None:
    primary = ShellBucklingPrimaryResultSet(
        panel_results=[
            PanelBucklingResult(
                lambdas=np.array([2.0, 4.0], dtype=float),
                modeshapes=np.zeros((2, 4), dtype=float),
                n_modes_found=2,
                panel_label="p0",
                panel_index=0,
                is_compressive=True,
            )
        ]
    )
    secondary = ShellBucklingSecondaryResultsOrchestrator(primary, station_index=0).compute()
    assert len(secondary.panel_assessments) == 1
    a = secondary.panel_assessments[0]
    assert a.load_multiplier_min == 2.0
    assert a.buckling_index == 0.5
    assert a.governing_mode == 0


def test_eigen_product_compressive_panel() -> None:
    """Direct compressive N11 smoke: real positive λ and BI = 1/λ_min."""
    abd = _isotropic_abd()
    inp = ShellPanelBucklingInput(
        panel_label="comp",
        panel_index=0,
        station_index=0,
        length_m=1.0,
        width_m=0.4,
        thickness_m=0.005,
        abd=abd,
        membrane_resultants=np.array([-1.0e5, 0.0, 0.0], dtype=float),
        bending_resultants=np.zeros(3, dtype=float),
        provenance=ShellStationProvenance(),
    )
    state = solve_panel_linear_buckling_eigen(inp, n_modes=2, n_elements=4)
    assert state.status == "ok"
    assert state.is_compressive
    assert state.n_modes_found >= 1
    assert np.all(np.isfinite(state.lambdas))
    assert float(state.lambdas[0]) > 0.0
    # Default provenance is FSDT; eigen path must resolve the registered strip kernel.
    assert inp.provenance.shell_theory == ShellTheory.FSDT
    assert state.diagnostics.get("shell_theory") == str(ShellTheory.FSDT)
    assert state.diagnostics.get("kernel")

    runner = SectionLinearBucklingRunner.from_panel_inputs([inp])
    result = runner.run_cascade(station_id="eigen-smoke", z_station=0.0, n_modes=2)
    assert result.primary_results is not None
    assert result.primary_results.status == "ok"
    assert result.secondary_results is not None
    a = result.secondary_results.panel_assessments[0]
    assert a.buckling_index is not None
    assert abs(float(a.buckling_index) - 1.0 / float(state.lambdas[0])) < 1e-9


def test_eigen_product_skips_tension() -> None:
    abd = _isotropic_abd()
    inp = ShellPanelBucklingInput(
        panel_label="tens",
        panel_index=0,
        station_index=0,
        length_m=1.0,
        width_m=0.4,
        thickness_m=0.005,
        abd=abd,
        membrane_resultants=np.array([1.0e5, 0.0, 0.0], dtype=float),
        bending_resultants=np.zeros(3, dtype=float),
        provenance=ShellStationProvenance(),
    )
    state = solve_panel_linear_buckling_eigen(inp)
    assert state.status == "skipped"
    assert state.diagnostics.get("reason") == "not_compressive"
    assert state.n_modes_found == 0


def test_solve_eigen_skips_missing_abd() -> None:
    inp = ShellPanelBucklingInput(
        panel_label="no-abd",
        panel_index=0,
        station_index=0,
        length_m=1.0,
        width_m=0.4,
        thickness_m=0.005,
        abd=None,
        membrane_resultants=np.array([-1.0e5, 0.0, 0.0], dtype=float),
        bending_resultants=np.zeros(3, dtype=float),
        provenance=ShellStationProvenance(),
    )
    state = solve_panel_linear_buckling_eigen(inp)
    assert state.status == "skipped"
    assert state.diagnostics.get("reason") == "missing_abd"
    assert state.is_compressive
    assert state.n_modes_found == 0


def _static_cascade_my(panel_topology, *, My: float = 1.0e3):
    return LinearStaticCascadeResult.from_executed_runner(
        SectionLinearStaticRunner(
            panel_topology=panel_topology,
            My=My,
            section_strip_mesh=_strip_mesh(panel_topology),
            load_channel="equipollent",
        ).execute()
    )


def test_nested_resultants_from_static_cascade_panel_alignment() -> None:
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    cascade = _static_cascade_my(panel_topology)
    nested = nested_resultants_from_static_cascade(cascade)
    assert len(nested) == len(cascade.assembly.panel_maps)
    assert any(len(panel_rows or []) > 0 for panel_rows in nested)


def test_buckling_from_static_cascade_fold() -> None:
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    cascade = _static_cascade_my(panel_topology)
    buck = run_section_linear_buckling_from_static(
        cascade,
        station_id="buck-test",
        z_station=0.35,
    )
    assert buck.eigen_track == "section"
    assert buck.primary_results is not None
    assert buck.secondary_results is not None
    assert buck.station_bundle is not None
    assert buck.primary_results.eigen_track == "section"
    assert buck.primary_results.status in ("ok", "partial", "skipped") or buck.primary_results.status == "ok"
    assert len(buck.panel_inputs) > 0
    assert buck.station_bundle.station_id == "buck-test"
    assert buck.station_bundle.metadata.get("eigen_track") == "section"
    statuses = [s.status for s in buck.runner.eigen_states]
    assert any(s in ("ok", "skipped", "failed") for s in statuses)
    assert "not_computed" not in statuses
    assert buck.runner.Kg_global is not None


def test_buckling_panel_track_still_available() -> None:
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    cascade = _static_cascade_my(panel_topology)
    buck = run_section_linear_buckling_from_static(
        cascade,
        station_id="buck-panel",
        z_station=0.35,
        eigen_track="panel",
    )
    assert buck.eigen_track == "panel"
    assert buck.primary_results is not None
    assert buck.primary_results.eigen_track == "panel"


def test_enable_buckling_attaches_to_station_analysis() -> None:
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    analysis = build_fsdt_station_analysis(
        panel_topology.section,
        panel_topology.shell_inputs,
        z_station=0.35,
        My=1.0e3,
        section_strip_mesh=_strip_mesh(panel_topology),
        station_id="buck-attach",
        enable_buckling=True,
    )
    assert analysis.buckling is not None
    assert analysis.buckling.station_id == "buck-attach"
