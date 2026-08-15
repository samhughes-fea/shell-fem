"""Study-only hybrid_kind_crossing cluster selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
    assign_global_nodes_hybrid_kind_crossing,
    cluster_is_kind_crossing,
)


def test_cluster_is_kind_crossing_detects_multi_kind():
    maps = [
        SimpleNamespace(panel_label="skin:outer_skin:SS:bay_0"),
        SimpleNamespace(panel_label="skin:outer_skin:SS:TE"),
        SimpleNamespace(panel_label="web:web_4"),
    ]
    same = [(0, "end", np.zeros(2)), (1, "start", np.zeros(2))]
    cross = [(0, "end", np.zeros(2)), (2, "start", np.zeros(2))]
    assert not cluster_is_kind_crossing(same, maps)  # type: ignore[arg-type]
    assert cluster_is_kind_crossing(cross, maps)  # type: ignore[arg-type]


def test_hybrid_kind_crossing_tb_only_on_noncollinear_kind_cross():
    """Same-kind non-collinear stays shared; kind-crossing non-collinear gets TB refs."""

    class _PM:
        def __init__(self, idx: int, label: str, n: int = 3):
            self.panel_index = idx
            self.panel_label = label
            self.s_nodes = np.linspace(0.0, 1.0, n)
            self.global_nodes = -np.ones(2 * n, dtype=int)

    maps = [
        _PM(0, "skin:A"),
        _PM(1, "skin:B"),
        _PM(2, "web:W"),
    ]
    # cid0: same-kind non-collinear → shared; cid1: kind-crossing non-collinear → TB
    clusters = [
        [(0, "end", np.array([0.0, 0.0])), (1, "start", np.array([0.0, 0.0]))],
        [(0, "start", np.array([1.0, 0.0])), (2, "end", np.array([1.0, 0.0]))],
    ]
    flags = {0: False, 1: False}
    out = assign_global_nodes_hybrid_kind_crossing(maps, clusters, flags)  # type: ignore[arg-type]
    assert (1, "bottom") in out["cluster_layer_node"]
    assert (0, "bottom") not in out["cluster_layer_node"]
