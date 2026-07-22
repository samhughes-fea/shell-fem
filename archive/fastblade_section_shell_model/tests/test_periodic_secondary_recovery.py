"""Unit tests for production periodic → secondary-orchestrator recovery bridge."""

from __future__ import annotations

import numpy as np

from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.engine.periodic_cache import _cached_periodic_k6
from blade_precompute.section_shell_model.engine.periodic_secondary_recovery import (
    nest_resultant_rows_by_panel,
    recover_resultants_from_displacement,
    recover_resultants_from_modes,
    scatter_element_displacements,
)
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
    ShellTheory,
)


def _periodic_geometry():
    pb = build_pipeline_b_section("2412", [0.35], chord_m=1.0)
    (
        _K,
        _f,
        _node_meta,
        panel_maps,
        K6,
        u_modes,
        _y_e,
        _z_e,
        _abd_p,
        _t_p,
        _g_p,
        _lab_p,
        cache,
    ) = _cached_periodic_k6(
        pb.panels,
        n_elements_per_panel=4,
        target_element_length_m=None,
        min_elements_per_panel=2,
        bc_mode="contour_support",
        interface_constraint_mode="shared",
    )
    assert cache is not None
    return pb, panel_maps, K6, u_modes, cache


def test_scatter_element_displacements_length_matches_stiffness_objects():
    _pb, _panel_maps, _K6, u_modes, cache = _periodic_geometry()
    d = np.asarray(u_modes[:, 0], dtype=float)
    U_e = scatter_element_displacements(cache, d)
    assert len(U_e) == len(cache.stiffness_objects)
    for stiff, ue in zip(cache.stiffness_objects, U_e):
        assert ue.shape == (int(stiff.global_dof_indices.size),)


def test_nest_resultant_rows_by_panel_groups_and_sorts():
    rows = [
        ShellPanelResultants(
            Nx=0.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
            panel_index=1, station_index=2, panel_label="b",
        ),
        ShellPanelResultants(
            Nx=0.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
            panel_index=0, station_index=1, panel_label="a",
        ),
        ShellPanelResultants(
            Nx=0.0, Ny=0.0, Nxy=0.0, Mx=0.0, My=0.0, Mxy=0.0,
            panel_index=1, station_index=0, panel_label="b",
        ),
    ]
    nested = nest_resultant_rows_by_panel(rows, n_panels=2)
    assert [r.station_index for r in nested[0]] == [1]
    assert [r.station_index for r in nested[1]] == [0, 2]


def test_recover_resultants_from_modes_returns_per_panel_rows():
    pb, panel_maps, K6, u_modes, cache = _periodic_geometry()
    q = np.zeros(6, dtype=float)
    q[0] = 1e-4  # small axial mode amplitude
    nested = recover_resultants_from_modes(
        u_modes,
        q,
        cache,
        panel_maps,
        pb.panels,
        shell_theory=ShellTheory.FSDT,
    )
    assert len(nested) == len(pb.panels)
    n_elems = sum(len(rows) for rows in nested)
    assert n_elems == len(cache.stiffness_objects)
    # Axial unit mode should produce finite membrane Nx on at least one element.
    nx_vals = [float(r.Nx) for rows in nested for r in rows]
    assert any(np.isfinite(v) for v in nx_vals)


def test_recover_resultants_from_displacement_matches_modes_path():
    pb, panel_maps, _K6, u_modes, cache = _periodic_geometry()
    q = np.array([1e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    via_modes = recover_resultants_from_modes(
        u_modes, q, cache, panel_maps, pb.panels, shell_theory=ShellTheory.FSDT,
    )
    d = u_modes @ q
    via_disp = recover_resultants_from_displacement(
        d, cache, panel_maps, pb.panels, shell_theory=ShellTheory.FSDT,
    )
    for rows_m, rows_d in zip(via_modes, via_disp):
        assert len(rows_m) == len(rows_d)
        for rm, rd in zip(rows_m, rows_d):
            assert float(rm.Nx) == float(rd.Nx)
            assert float(rm.Nxy) == float(rd.Nxy)
