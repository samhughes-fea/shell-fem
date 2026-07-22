"""Panel ABD is computed once per panel per assembly cache miss."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.engine.global_shell_assembly import assemble_global_shell


def test_assembly_calls_panel_abd_once_per_panel() -> None:
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    panels = pb.panels
    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    call_count = 0
    original = __import__(
        "blade_precompute.section_shell_model.pre_processing.constitutive.panel_abd",
        fromlist=["panel_abd_tuple_from_lam"],
    ).panel_abd_tuple_from_lam

    def _counting(lam: object) -> tuple:
        nonlocal call_count
        call_count += 1
        return original(lam)

    with patch(
        "blade_precompute.section_shell_model.pre_processing.element_library.formulation_factory.panel_abd_tuple_from_lam",
        side_effect=_counting,
    ):
        assemble_global_shell(panels, zeros, zeros, n_elements_per_panel=4)

    n_valid = sum(1 for p in panels if len(np.asarray(p.s, dtype=float)) >= 2)
    assert call_count == n_valid, f"expected one ABD build per valid panel, got {call_count} for {n_valid} panels"
