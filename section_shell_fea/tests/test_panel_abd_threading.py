"""Panel ABD is computed once per panel per assembly cache miss."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.tests._panel_topology_factory import build_shell_panel_topology
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.global_shell_assembly import assemble_global_shell
from blade_structure_utilities.mesh_utilities.section_mesh import build_section_strip_mesh


def test_assembly_calls_panel_abd_once_per_panel() -> None:
    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    zones = list(panel_topology.zones)
    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
    mesh = build_section_strip_mesh(panel_topology.shell_inputs, n_elements_per_panel=4)
    call_count = 0
    original = __import__(
        "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.constitutive.panel_abd",
        fromlist=["panel_abd_tuple_from_lam"],
    ).panel_abd_tuple_from_lam

    def _counting(lam: object) -> tuple:
        nonlocal call_count
        call_count += 1
        return original(lam)

    with patch(
        "blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.formulation_factory.panel_abd_tuple_from_lam",
        side_effect=_counting,
    ):
        assemble_global_shell(
        zones, zeros, zeros, section_strip_mesh=mesh)

    n_valid = sum(1 for p in zones if len(np.asarray(p.s, dtype=float)) >= 2)
    assert call_count == n_valid, f"expected one ABD build per valid panel, got {call_count} for {n_valid} panels"

