"""Study-only shared_cluster_frame: shared IDs + cluster-frame rotations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
    apply_shared_cluster_frame_to_cache,
    assign_global_nodes_shared,
    build_shared_cluster_frame_rotations,
    cluster_is_kind_crossing,
)


def test_shared_cluster_frame_keeps_shared_ids_and_builds_rotations():
    class _PM:
        def __init__(self, idx: int, label: str, n: int = 3):
            self.panel_index = idx
            self.panel_label = label
            self.s_nodes = np.linspace(0.0, 1.0, n)
            self.global_nodes = -np.ones(2 * n, dtype=int)
            self.elements = [(0, 1, n + 1, n)]

    class _Zone:
        def __init__(self, nodes):
            self.nodes = np.asarray(nodes, dtype=float)

    maps = [
        _PM(0, "skin:A"),
        _PM(1, "web:W"),
        _PM(2, "skin:B"),
    ]
    # Full endpoint cover (shared assign requires every end in a cluster).
    # cid0: kind-crossing non-collinear; cid1: same-kind; others: singletons.
    clusters = [
        [(0, "end", np.array([0.0, 0.0])), (1, "start", np.array([0.0, 0.0]))],
        [(0, "start", np.array([1.0, 0.0])), (2, "end", np.array([1.0, 0.0]))],
        [(1, "end", np.array([0.0, 1.0]))],
        [(2, "start", np.array([2.0, 0.1]))],
    ]
    flags = {0: False, 1: False, 2: True, 3: True}
    zones = [
        _Zone([[0.0, 0.0], [1.0, 0.0]]),  # skin A horizontal
        _Zone([[0.0, 0.0], [0.0, 1.0]]),  # web vertical
        _Zone([[1.0, 0.0], [2.0, 0.1]]),  # skin B nearly collinear-ish
    ]
    assert cluster_is_kind_crossing(clusters[0], maps)  # type: ignore[arg-type]
    assert not cluster_is_kind_crossing(clusters[1], maps)  # type: ignore[arg-type]

    next_n = assign_global_nodes_shared(maps, clusters)  # type: ignore[arg-type]
    assert next_n > 0
    # Shared IDs equal across panels in each cluster
    assert maps[0].global_nodes[2] == maps[1].global_nodes[0]  # end/start bottom cid0
    assert maps[0].global_nodes[0] == maps[2].global_nodes[2]  # start/end bottom cid1

    frame = build_shared_cluster_frame_rotations(maps, zones, clusters, flags)  # type: ignore[arg-type]
    assert 0 in frame["cids"]
    assert 1 not in frame["cids"]
    assert (0, "end") in frame["endpoint_frame_R"]
    assert (1, "start") in frame["endpoint_frame_R"]
    assert (0, "start") not in frame["endpoint_frame_R"]
    # No TB virtual refs in this mode
    assert "cluster_layer_node" not in frame


def test_apply_shared_cluster_frame_touches_endpoint_elements():
    class _PM:
        def __init__(self):
            self.panel_index = 0
            self.panel_label = "skin:A"
            self.s_nodes = np.linspace(0.0, 1.0, 3)
            self.global_nodes = np.arange(6, dtype=int)
            self.elements = [(0, 1, 4, 3)]  # start-end bottom/top of first elem

    class _Stiff:
        def __init__(self):
            self.panel_index = 0
            self.local_element_index = 0
            self.K_e = np.eye(24, dtype=float)
            self.global_dof_indices = np.arange(24, dtype=np.int64)

    class _Force:
        def __init__(self):
            self.F_e = np.ones(24, dtype=float)
            self.global_dof_indices = np.arange(24, dtype=np.int64)

    cache = SimpleNamespace(
        n_dof_node=6,
        stiffness_objects=[_Stiff()],
        force_objects=[_Force()],
    )
    # Non-diagonal Ke so congruence is observable under an orthogonal R.
    cache.stiffness_objects[0].K_e[1, 1] = 4.0
    cache.stiffness_objects[0].K_e[2, 2] = 1.0
    R = {(0, "start"): (0.0, 1.0, -1.0, 0.0)}  # 90 deg
    n = apply_shared_cluster_frame_to_cache(cache, [_PM()], R)  # type: ignore[arg-type]
    assert n == 1
    ke = cache.stiffness_objects[0].K_e
    # After 90° map, U_S/W diagonal entries swap
    assert abs(float(ke[1, 1]) - 1.0) < 1e-9
    assert abs(float(ke[2, 2]) - 4.0) < 1e-9
