"""Unit + smoke tests for cap-peak sampling diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cap_peak_sampling import (
    ElementPeakTags,
    analyze_cap_peak_rung,
    endpoint_band_mask,
    percentile_stats,
    tag_element,
    verdict_notes,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.grid_independence_study import (
    load_cotide_station_recipe,
)


def test_endpoint_band_mask_basic():
    m = endpoint_band_mask(10, band=2)
    assert m.dtype == bool
    assert m.tolist() == [True, True, False, False, False, False, False, False, True, True]


def test_endpoint_band_mask_covers_short_strip():
    m = endpoint_band_mask(3, band=2)
    assert m.tolist() == [True, True, True]


def test_endpoint_band_mask_empty():
    assert endpoint_band_mask(0).tolist() == []


def test_percentile_stats_ordering():
    vals = [1.0, 2.0, 3.0, 4.0, 100.0]
    s = percentile_stats(vals)
    assert s["max"] == 100.0
    assert s["p50"] == 3.0
    assert s["p95"] < s["max"]
    assert s["p99"] <= s["max"]
    assert s["p95"] >= s["p50"]


def test_percentile_stats_empty():
    s = percentile_stats([])
    assert all(np.isnan(s[k]) for k in ("max", "p50", "p95", "p99"))


def test_tag_element_junction_tie_endpoint():
    owners = {1: 2, 2: 1, 3: 1}
    tags = tag_element(
        n_elem_panel=8,
        local_ei=0,
        element_gids=[1, 2],
        owner_count=owners,
        tie_slave_nodes={2},
        band=2,
    )
    assert tags.junction is True
    assert tags.tie_slave is True
    assert tags.endpoint_band is True
    assert tags.any_interface is True

    interior = tag_element(
        n_elem_panel=8,
        local_ei=3,
        element_gids=[3],
        owner_count=owners,
        tie_slave_nodes=set(),
        band=2,
    )
    assert interior == ElementPeakTags(junction=False, tie_slave=False, endpoint_band=False)


def test_verdict_sampling_artifact():
    note = verdict_notes(
        max_prod=3.0,
        max_lt=2.5,
        p95_prod=1.1,
        p95_lt=1.05,
        interior_prod=1.2,
        interior_lt=1.1,
        location_tags=[ElementPeakTags(junction=False, tie_slave=True, endpoint_band=True)],
    )
    assert note.startswith("sampling_artifact")


def test_verdict_sampling_artifact_when_prod_p95_drifts_but_last_two_stable():
    """Coarse-rung p95 collapses to max; last-two robust plateau is the signal."""
    note = verdict_notes(
        max_prod=6.4,
        max_lt=5.9,
        p95_prod=6.6,
        p95_lt=1.20,
        interior_prod=4.0,
        interior_lt=1.15,
        location_tags=[ElementPeakTags(tie_slave=True, endpoint_band=True)],
        interface_excluded_finite=False,
    )
    assert note == "sampling_artifact|all_cap_elems_tie_tagged"


def test_verdict_bulk_nonconvergence():
    note = verdict_notes(
        max_prod=3.0,
        max_lt=2.5,
        p95_prod=2.8,
        p95_lt=2.2,
        interior_prod=2.5,
        interior_lt=2.1,
        location_tags=[ElementPeakTags(tie_slave=True, endpoint_band=True)],
    )
    assert note == "bulk_nonconvergence"


def test_verdict_pass():
    note = verdict_notes(
        max_prod=1.2,
        max_lt=1.1,
        p95_prod=1.05,
        p95_lt=1.02,
        interior_prod=1.1,
        interior_lt=1.05,
        location_tags=[ElementPeakTags(endpoint_band=True)],
    )
    assert note == "pass"


@pytest.fixture(scope="module")
def cotide_recipe():
    return load_cotide_station_recipe()


def test_cotide_2b_cn_locator_smoke(cotide_recipe):
    """One coarse rung: locator fields populated; raw max >= p95 for cap Hashin."""
    row = analyze_cap_peak_rung(
        0.024,
        layout="2B-CN",
        **cotide_recipe.cascade_kwargs(),
    )
    assert row.mesh_ok and row.cascade_trust_ok
    cap_h = next(s for s in row.stats if s.kind == "cap" and s.metric == "peak_hashin")
    assert cap_h.n_elem > 0
    assert np.isfinite(cap_h.max) and cap_h.max > 0.0
    assert np.isfinite(cap_h.p95)
    assert cap_h.max >= cap_h.p95 - 1e-12
    assert cap_h.location is not None
    assert cap_h.location.panel_label.startswith("cap")
    assert cap_h.location.local_element_index >= 0
    # Cap-bearing layout with skin_offset should emit ties.
    assert row.n_cap_tie_links > 0
