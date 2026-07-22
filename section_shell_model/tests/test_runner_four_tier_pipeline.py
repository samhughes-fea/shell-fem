"""End-to-end: SectionLinearStaticRunner.solve() always runs all four result tiers."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.section_shell_model.pre_processing.mesh.api.pipeline_b import build_pipeline_b_section
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.section_shell_model.processing.static.operations.global_shell_assembly import (
    _NDOF_NODE,
    _PanelGlobalMap,
    _dof,
)
from blade_structure_utilities.section_shell_model.pre_processing.mesh.strip_discretise import (
    discretise_strip,
)
from blade_structure_utilities.section_shell_model.pre_processing.element_library.formulation_factory import (
    build_formulation_cache,
)
from blade_structure_utilities.section_shell_model.simulation_runner.section_linear_static_runner import (
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


def test_runner_solve_always_populates_all_four_tiers():
    cache, n_gdof, fixed = _cache_and_bc()
    runner = SectionLinearStaticRunner.from_cache(cache, n_gdof, {}, fixed)
    result = runner.solve(theory="fsdt")
    assert result.primary_results is not None
    assert result.secondary_results is not None
    assert result.tertiary_results is not None
    assert result.quaternary_results is not None
    assert len(result.secondary_results.elemental_results.resultant_rows) == len(
        cache.stiffness_objects
    )
    assert result.quaternary_results.hashin_fi is not None
    assert result.quaternary_results.needs_review is not None


def test_runner_recompute_tiers_require_upstream_prerequisites():
    cache, n_gdof, fixed = _cache_and_bc()
    runner = SectionLinearStaticRunner.from_cache(cache, n_gdof, {}, fixed)
    with pytest.raises(RuntimeError):
        runner.compute_secondary_results()
    with pytest.raises(RuntimeError):
        runner.compute_tertiary_results()
    with pytest.raises(RuntimeError):
        runner.compute_quaternary_results()


def test_solve_linear_system_only_skips_tiers():
    cache, n_gdof, fixed = _cache_and_bc()
    runner = SectionLinearStaticRunner.from_cache(cache, n_gdof, {}, fixed)
    d = runner.solve_linear_system_only()
    assert d.shape == (n_gdof,)
    assert runner.primary_results is None
    assert runner.secondary_results is None
    assert runner.quaternary_results is None


def test_runner_secondary_gaussian_nodal_via_solve_kwargs():
    """Gaussian / nodal secondary resolutions via solve kwargs (default off)."""
    cache, n_gdof, fixed, panel_maps = _cache_and_bc(return_panel_maps=True)
    runner = SectionLinearStaticRunner.from_cache(cache, n_gdof, {}, fixed)
    result = runner.solve(
        theory="fsdt",
        panel_maps=panel_maps,
        compute_gaussian=True,
        compute_nodal=True,
    )
    secondary = result.secondary_results
    assert secondary is not None
    assert secondary.gaussian_results is not None
    assert secondary.nodal_results is not None
    nx_gp = np.asarray(secondary.gaussian_results.fields["Nx"], dtype=float)
    assert nx_gp.shape == (len(cache.stiffness_objects), 4)
    assert np.isfinite(nx_gp).any()

    elem_nx = np.asarray(secondary.nodal_results.fields["element_Nx"], dtype=float)
    assert elem_nx.shape == (len(cache.stiffness_objects), 4)
    rows = secondary.elemental_results.resultant_rows
    for i, row in enumerate(rows):
        assert np.isfinite(elem_nx[i]).all()
        assert abs(float(np.nanmean(elem_nx[i])) - float(row.Nx)) < max(1.0, 5.0 * abs(float(row.Nx)))
