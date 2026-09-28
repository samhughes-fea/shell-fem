"""FEA-EQ-001 / FEA-SC-002 exactness unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.equilibrium_checks import (
    _build_endpoint_clusters_raw,
    _panel_end_tangent,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.shear_centre_loading_axis import (
    shear_centre_about_yz_from_panel_topology,
)


def test_panel_end_tangent_refuses_degenerate():
    panel = SimpleNamespace(label="skin:a", nodes=np.array([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="FEA-EQ-001"):
        _panel_end_tangent(panel, "start")
    panel2 = SimpleNamespace(
        label="skin:a", nodes=np.array([[0.0, 0.0], [0.0, 0.0]])
    )
    with pytest.raises(ValueError, match="FEA-EQ-001"):
        _panel_end_tangent(panel2, "start")


def test_producer_clusters_bypass_proximity():
    """Distant endpoints still form one interface when producer says so."""
    zones = [
        SimpleNamespace(label="skin:a", nodes=np.array([[0.0, 0.0], [1.0, 0.0]])),
        SimpleNamespace(label="web:b", nodes=np.array([[10.0, 0.0], [10.0, 1.0]])),
    ]
    # Proximity alone would not join these (10 m apart).
    geo = _build_endpoint_clusters_raw(zones, tol=1e-6)
    assert all(len(c) == 1 for c in geo)
    producer = [
        [
            (0, "end", np.array([1.0, 0.0])),
            (1, "start", np.array([10.0, 0.0])),
        ],
        [(0, "start", np.array([0.0, 0.0]))],
        [(1, "end", np.array([10.0, 1.0]))],
    ]
    clusters = _build_endpoint_clusters_raw(
        zones, tol=1e-6, producer_clusters=producer
    )
    multi = [c for c in clusters if len(c) >= 2]
    assert len(multi) == 1
    members = {(m["pi"], m["end"]) for m in multi[0]}
    assert members == {(0, "end"), (1, "start")}


def test_sc_prefers_loops_path(monkeypatch):
    called = {"n": 0}

    class _SC:
        xs = 0.11
        ys = -0.22

    def _fake_sc(midlines, *, cell_wall_loops=None, **_kw):
        called["n"] += 1
        assert cell_wall_loops
        return _SC()

    monkeypatch.setattr(
        "blade_structure_utilities.geometry_utilities.section_properties.reference_axes.shear_centre.shear_centre_from_midlines",
        _fake_sc,
    )
    mid = SimpleNamespace(kind="skin", label="skin:a", thickness_m=0.01, points=None)
    shell = SimpleNamespace(midlines=(mid,), cell_wall_loops=(object(),))
    topo = SimpleNamespace(shell_inputs=shell)
    y, z = shear_centre_about_yz_from_panel_topology(topo)
    assert called["n"] == 1
    assert y == 0.11 and z == -0.22
