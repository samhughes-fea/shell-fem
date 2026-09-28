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
# Measured 2026-09-25 after section_loads SL-001 (Channel C cells from the
# section's CellWallLoops): tied skin |N12| 1.60x, web 1.03x; tied Hashin skin
# 1.10x, web 1.19x. The previous 4.0 was set on the mis-celled load field
# (tied skin |N12| 3.03x), loose enough that the untied defect (2.9x) passed it.
# Tied Hashin shares the tolerance: the previous 6.0 (set when continuum-owned
# Hashin climbed on the mis-celled load field) would pass the untied defect
# (5.5x) as converged.
MAX_TIED_HASHIN_DRIFT = MAX_TIED_DRIFT
# The discriminator is relative, not a calibrated floor: the untied cap must
# fail the tolerance the tied assertions accept, and drift at least twice as
# much as the worst connected control (measured 2.84x on |N12|, 4.14x on Hashin).
MIN_UNTIED_OVER_CONTROL = 2.0


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


def test_kind_owner_counts_ignores_same_kind_strip_interfaces():
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_mesh_convergence import (
        _fi_sampling_excluded,
        _kind_owner_counts,
    )

    class _PM:
        def __init__(self, label, nodes, elements):
            self.panel_label = label
            self.global_nodes = nodes
            self.elements = elements

    # Two skin strips share node 5; skin–web junction at node 9.
    panels = [
        _PM("skin:A", [1, 5], [[0, 1]]),
        _PM("skin:B", [5, 6], [[0, 1]]),
        _PM("web:W", [9, 10], [[0, 1]]),
        _PM("skin:C", [9, 11], [[0, 1]]),
    ]
    counts = _kind_owner_counts(panels)
    assert counts[5] == 1  # same-kind strip interface
    assert counts[9] == 2  # kind-crossing junction
    assert not _fi_sampling_excluded(
        panel_maps=panels, pi=0, ei=0, kind_owner_count=counts, tie_slave_nodes=set()
    )
    assert _fi_sampling_excluded(
        panel_maps=panels, pi=3, ei=0, kind_owner_count=counts, tie_slave_nodes=set()
    )
    assert _fi_sampling_excluded(
        panel_maps=panels, pi=0, ei=0, kind_owner_count=counts, tie_slave_nodes={1}
    )


def test_runner_rejects_density_kwargs():
    class _Topo:
        panels: list = []
        zones: list = []

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
    """Membrane and mid-surface stress converge away from junctions.

    A kind-crossing junction (web foot, cap crossing a web) is a stress
    singularity of the shell model: its peak keeps rising under refinement for any
    load, so a convergence check that includes it is not a check at any
    threshold. The ``*_interior`` peaks exclude a fixed physical radius around
    every junction -- panel ends *and* interior T-junctions -- the same on every
    rung (``junction_masks``). Measured 2026-09-25 with exact Channel C loads:
    tied skin |N12| 65.6 / 64.3 / 66.0 kN/m, web 39.9 / 37.7 / 38.0 kN/m.
    """
    for attr in ("peak_abs_n12_interior", "peak_mid_sig_interior"):
        series = series_by_kind(tied_rows, attr)
        for kind in ("cap", "skin", "web"):
            d = drift(series[kind])
            assert d <= MAX_TIED_DRIFT, f"tied {attr} {kind} drift {d:.3f}x values={series[kind]}"
    # Hashin in-plane: continuum-owned peaks (skin/web). Cap slaves under
    # skin_offset are excluded from FI ownership — see test_tied_cap_hashin_slave_owned.
    # Bound growth on the 3-rung smoke ladder; full GCI uses a longer ladder.
    series_h = series_by_kind(tied_rows, "peak_hashin")
    for kind in ("skin", "web"):
        d = drift(series_h[kind])
        assert d <= MAX_TIED_HASHIN_DRIFT, (
            f"tied peak_hashin {kind} drift {d:.3f}x values={series_h[kind]}"
        )


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


def test_tied_cap_hashin_slave_owned(tied_rows):
    """Under skin_offset, cap Hashin is tie-slave owned → continuum peak is 0."""
    series = series_by_kind(tied_rows, "peak_hashin")
    assert all(float(v) == 0.0 for v in series["cap"])
    assert all(float(v) > 0.0 for v in series["skin"])
    assert all(float(v) > 0.0 for v in series["web"])


def test_untied_cap_s_and_q_diverge(untied_rows):
    """BL-003: untied null-space must still discriminate on raw cap peaks."""
    for attr in ("peak_abs_n12", "peak_hashin"):
        series = series_by_kind(untied_rows, attr)
        cap = drift(series["cap"])
        control = max(drift(series["skin"]), drift(series["web"]))
        assert cap > MAX_TIED_DRIFT, (
            f"untied {attr} cap drift {cap:.3f}x passes the tied tolerance "
            f"{MAX_TIED_DRIFT}x; fixture no longer discriminates"
        )
        assert cap >= MIN_UNTIED_OVER_CONTROL * control, (
            f"untied {attr} cap drift {cap:.3f}x is not clearly worse than the "
            f"connected control {control:.3f}x"
        )


def test_connected_components_converge_untied(untied_rows):
    for attr in ("peak_abs_n12", "peak_hashin"):
        series = series_by_kind(untied_rows, attr)
        for kind in ("skin", "web"):
            assert drift(series[kind]) <= MAX_TIED_DRIFT
