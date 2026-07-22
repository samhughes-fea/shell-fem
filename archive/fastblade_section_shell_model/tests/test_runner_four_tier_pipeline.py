"""End-to-end: SectionLinearStaticRunner drives all four result tiers.

Covers PROCESSING_ARCHITECTURE_PROPOSAL.md Phase 4 acceptance: tiers 2-4 are
opt-in runner methods (``compute_secondary_results`` / ``compute_tertiary_results``
/ ``compute_quaternary_results``), each requiring its upstream result set + the
retained ``cache`` (with ``panel_laminates`` threaded through from Phase 1),
and each raising ``RuntimeError`` when its prerequisite is missing. ``solve()``
itself stays tier-1-only (no change to default production behaviour).
"""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_precompute.section_shell_model.engine.global_shell_assembly import (
    _NDOF_NODE,
    _PanelGlobalMap,
    _dof,
)
from blade_precompute.section_shell_model.pre_processing.mesh.strip_discretise import (
    discretise_strip,
)
from blade_precompute.section_shell_model.pre_processing.element_library.formulation_factory import (
    build_formulation_cache,
)
from blade_precompute.section_shell_model.simulation_runner.section_linear_static import (
    SectionLinearStaticRunner,
)


def _cache_and_bc(*, return_panel_maps: bool = False):
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    panels = pb.panels
    nx = [np.linspace(0.0, 1000.0, len(np.asarray(p.s, dtype=float))) for p in panels]
    nxy = [np.zeros_like(n) for n in nx]

    panel_maps = []
    next_g = 0
    for pi, p in enumerate(panels):
        s_panel = np.asarray(p.s, dtype=float)
        if len(s_panel) < 2:
            panel_maps.append(_PanelGlobalMap(np.array([]), [], [], f"p{pi}", pi))
            continue
        s_nodes, elems = discretise_strip(s_panel, 4, panel_label="", nodes_yz=np.asarray(p.nodes))
        gn = [-1] * (2 * len(s_nodes))
        for i in range(2 * len(s_nodes)):
            gn[i] = next_g
            next_g += 1
        panel_maps.append(_PanelGlobalMap(s_nodes, elems, gn, f"p{pi}", pi))
    n_gdof = next_g * _NDOF_NODE

    cache = build_formulation_cache(panel_maps, panels, nx, nxy, shell_theory=ShellTheory.FSDT, L_x=1.0)

    # Clamp every DOF of the first global node so K_rr is nonsingular.
    gn0 = panel_maps[0].global_nodes[0]
    fixed = {_dof(gn0, k) for k in range(_NDOF_NODE)}
    if return_panel_maps:
        return cache, n_gdof, fixed, panel_maps
    return cache, n_gdof, fixed


def test_runner_default_solve_is_tier1_only():
    cache, n_gdof, fixed = _cache_and_bc()
    runner = SectionLinearStaticRunner.from_cache(cache, n_gdof, {}, fixed)
    result = runner.solve()
    assert result.primary_results is not None
    assert result.secondary_results is None
    assert result.tertiary_results is None
    assert result.quaternary_results is None


def test_runner_opt_in_tiers_require_upstream_prerequisites():
    cache, n_gdof, fixed = _cache_and_bc()
    runner = SectionLinearStaticRunner.from_cache(cache, n_gdof, {}, fixed)
    with pytest.raises(RuntimeError):
        runner.compute_secondary_results()
    with pytest.raises(RuntimeError):
        runner.compute_tertiary_results()
    with pytest.raises(RuntimeError):
        runner.compute_quaternary_results()


def test_runner_drives_all_four_tiers_end_to_end():
    cache, n_gdof, fixed = _cache_and_bc()
    runner = SectionLinearStaticRunner.from_cache(cache, n_gdof, {}, fixed)
    runner.solve()

    secondary = runner.compute_secondary_results(theory="fsdt")
    assert secondary.elemental_results is not None
    assert len(secondary.elemental_results.resultant_rows) == len(cache.stiffness_objects)
    assert secondary.gaussian_results is None
    assert secondary.nodal_results is None

    tertiary = runner.compute_tertiary_results()
    assert tertiary.ply_results is not None
    assert tertiary.panel_index is not None
    assert tertiary.ply_results.sig_ply_mat.shape[0] == len(secondary.elemental_results.resultant_rows)

    quaternary = runner.compute_quaternary_results()
    assert quaternary.hashin_fi is not None
    assert quaternary.hashin_fi.in_plane.shape[0] == tertiary.ply_results.sig_ply_mat.shape[0]
    assert quaternary.needs_review is not None
    assert quaternary.governing.get("panel_index", -1) >= -1

    # Runner state + SectionLinearStaticResult both carry the retained tiers.
    assert runner.secondary_results is secondary
    assert runner.tertiary_results is tertiary
    assert runner.quaternary_results is quaternary


def test_runner_secondary_gaussian_nodal_opt_in():
    """R10: opt-in Gaussian / nodal secondary resolutions stay off the hot path."""
    cache, n_gdof, fixed, panel_maps = _cache_and_bc(return_panel_maps=True)
    runner = SectionLinearStaticRunner.from_cache(cache, n_gdof, {}, fixed)
    runner.solve()

    secondary = runner.compute_secondary_results(
        theory="fsdt",
        panel_maps=panel_maps,
        compute_gaussian=True,
        compute_nodal=True,
    )
    assert secondary.gaussian_results is not None
    assert secondary.nodal_results is not None
    nx_gp = np.asarray(secondary.gaussian_results.fields["Nx"], dtype=float)
    assert nx_gp.shape == (len(cache.stiffness_objects), 4)
    assert np.isfinite(nx_gp).any()

    # Averaged elemental resultants should be consistent with centroid-level rows
    # on a coarse single-panel-ish mesh (relative finite, same order).
    elem_nx = np.asarray(secondary.nodal_results.fields["element_Nx"], dtype=float)
    assert elem_nx.shape == (len(cache.stiffness_objects), 4)
    rows = secondary.elemental_results.resultant_rows
    for i, row in enumerate(rows):
        # Mean of corner nodal Nx ≈ centroidal Nx for smooth fields
        assert np.isfinite(elem_nx[i]).all()
        assert abs(float(np.nanmean(elem_nx[i])) - float(row.Nx)) < max(1.0, 5.0 * abs(float(row.Nx)))
