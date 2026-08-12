"""Cascade M+P/S/T/Q mesh-density convergence (confidence Phase D / BL-003)."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
    DEFAULT_REFINEMENTS,
    drift,
    series_by_kind,
    sweep_cascade_convergence,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)

MAX_TIED_DRIFT = 2.0
MAX_CAP_DRIFT_VS_CONTROL = 3.0
MIN_UNTIED_CAP_DRIFT = 10.0


@pytest.fixture(scope="module")
def tied_rows():
    return sweep_cascade_convergence(
        DEFAULT_REFINEMENTS, cap_tie_mode="skin_offset", interface_constraint_mode="shared"
    )


@pytest.fixture(scope="module")
def untied_rows():
    return sweep_cascade_convergence(
        DEFAULT_REFINEMENTS, cap_tie_mode="none", interface_constraint_mode="shared"
    )


def test_runner_rejects_density_kwargs():
    class _Topo:
        panels: list = []

    class _Mesh:
        n_total_elements = 0

    runner = SectionLinearStaticRunner(
        panel_topology=_Topo(),  # type: ignore[arg-type]
        section_strip_mesh=_Mesh(),  # type: ignore[arg-type]
        target_element_length_m=0.01,
    )
    with pytest.raises(ValueError, match="Density kwargs"):
        runner._prepare_station_system()


def test_mesh_qa_passes_on_every_tied_rung(tied_rows):
    assert all(r.mesh_ok for r in tied_rows)
    assert all(r.cascade_trust_ok for r in tied_rows)


def test_tied_s_t_q_converge(tied_rows):
    for attr in ("peak_abs_n12", "peak_mid_sig", "peak_hashin"):
        series = series_by_kind(tied_rows, attr)
        for kind in ("cap", "skin", "web"):
            d = drift(series[kind])
            assert d <= MAX_TIED_DRIFT, f"tied {attr} {kind} drift {d:.3f}x values={series[kind]}"


def test_tied_extracts_full_secondary_tertiary_quaternary(tied_rows):
    """Ladder rungs expose the full P→S→T→Q kind-peak suite (finite on skin)."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
        KIND_PEAK_METRICS,
        QUATERNARY_KIND_METRICS,
        SECONDARY_KIND_METRICS,
        TERTIARY_KIND_METRICS,
    )

    row = tied_rows[-1]
    # Core secondary membrane + bending (M11/M22) + mid-σ + in-plane Hashin.
    # |M12| can be ~0 on some thin-wall recipes; Q/τ may be tiny.
    required_positive = (
        "peak_abs_n11",
        "peak_abs_n22",
        "peak_abs_n12",
        "peak_abs_m11",
        "peak_abs_m22",
        "peak_mid_sig",
        "peak_hashin",
    )
    for metric in required_positive:
        assert getattr(row, metric)["skin"] > 0.0, metric
    # Full registry present (may be zero when Q/τ not recovered).
    for metric in KIND_PEAK_METRICS:
        assert metric in SECONDARY_KIND_METRICS + TERTIARY_KIND_METRICS + QUATERNARY_KIND_METRICS
        mapping = getattr(row, metric)
        assert "skin" in mapping
        assert np.isfinite(float(mapping["skin"]))
        interior = getattr(row, f"{metric}_interior")
        assert "skin" in interior


def test_tied_interior_le_raw(tied_rows):
    for row in tied_rows:
        for kind in ("cap", "skin", "web"):
            for raw_attr, int_attr in (
                ("peak_abs_n12", "peak_abs_n12_interior"),
                ("peak_mid_sig", "peak_mid_sig_interior"),
                ("peak_hashin", "peak_hashin_interior"),
            ):
                raw = float(getattr(row, raw_attr).get(kind, float("nan")))
                interior = float(getattr(row, int_attr).get(kind, float("nan")))
                if np.isfinite(raw) and np.isfinite(interior) and raw > 0.0:
                    assert interior <= raw + 1e-12, (
                        f"{int_attr}[{kind}]={interior} > raw {raw}"
                    )


def test_tied_cap_tracks_control_on_hashin(tied_rows):
    series = series_by_kind(tied_rows, "peak_hashin")
    cap = drift(series["cap"])
    control = max(drift(series["skin"]), drift(series["web"]))
    assert cap <= MAX_CAP_DRIFT_VS_CONTROL * control


def test_untied_cap_s_and_q_diverge(untied_rows):
    """BL-003: untied null-space must still discriminate on raw cap peaks."""
    for attr in ("peak_abs_n12", "peak_hashin"):
        series = series_by_kind(untied_rows, attr)
        cap = drift(series["cap"])
        control = max(drift(series["skin"]), drift(series["web"]))
        assert cap >= MIN_UNTIED_CAP_DRIFT, (
            f"untied {attr} cap drift only {cap:.3f}x; fixture no longer discriminates"
        )
        assert cap > control


def test_connected_components_converge_untied(untied_rows):
    for attr in ("peak_abs_n12", "peak_hashin"):
        series = series_by_kind(untied_rows, attr)
        for kind in ("skin", "web"):
            assert drift(series[kind]) <= MAX_TIED_DRIFT
