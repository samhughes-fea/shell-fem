"""Tests for single periodic BVP beam + face-traction (no contour overlay)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _pb():
    from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section

    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def _nx(results):
    return np.array([float(r.Nx) for panel in results for r in panel], dtype=float)


def _traction_fixture(pb):
    from blade_precompute.section_shell_model.engine.global_shell_assembly import (
        assemble_global_shell,
    )
    from blade_precompute.section_shell_model.interface.traction_face_load import _skin_node_pairs
    from blade_precompute.section_shell_model.pre_processing.mesh.shell_mesh_map import normal_2d

    panels = pb.panels
    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    _K, _f, node_meta, _pm, *_rest = assemble_global_shell(
        panels, zeros, zeros, n_elements_per_panel=6,
    )
    locs, tangents, _ = _skin_node_pairs(node_meta)
    forces = np.array([5.0e3 * normal_2d(t) for t in tangents], dtype=float)

    class _TF:
        total_forces = forces

    return _TF(), locs


def test_fsdt_zero_traction_matches_beam_only() -> None:
    """FSDT periodic solve with traction=None equals beam-only path."""
    from blade_precompute.section_shell_model.api.periodic import run_section_periodic_bc_mitc4

    pb = _pb()
    kw = dict(N=1.0e5, My=3.0e4, Vz=2.0e4, n_elements_per_panel=6)
    b_beam = run_section_periodic_bc_mitc4(pb, **kw)
    b_none = run_section_periodic_bc_mitc4(
        pb, traction_field=None, traction_points_yz=None, **kw,
    )
    np.testing.assert_allclose(
        _nx(b_beam.all_panel_results), _nx(b_none.all_panel_results), rtol=1e-8, atol=1e-6,
    )


def test_fsdt_traction_perturbs_recovered_field() -> None:
    """Adding face traction on the periodic free set perturbs the recovered field."""
    from blade_precompute.section_shell_model.api.periodic import run_section_periodic_bc_mitc4
    from blade_precompute.section_shell_model.engine.periodic_traction_superposition import (
        PERIODIC_FACE_TRACTION_PROVENANCE,
    )

    pb = _pb()
    tf, locs = _traction_fixture(pb)
    kw = dict(N=1.0e5, My=3.0e4, n_elements_per_panel=6)
    b_no = run_section_periodic_bc_mitc4(pb, **kw)
    b_tr = run_section_periodic_bc_mitc4(
        pb, traction_field=tf, traction_points_yz=locs, **kw,
    )
    nx_no = _nx(b_no.all_panel_results)
    nx_tr = _nx(b_tr.all_panel_results)
    assert nx_no.size == nx_tr.size
    assert float(np.linalg.norm(nx_tr - nx_no)) > 0.0
    row = b_tr.all_panel_results[0][0]
    notes = {p.note for p in row.provenance.values()}
    assert any(PERIODIC_FACE_TRACTION_PROVENANCE in n for n in notes), notes


def test_single_bvp_identity_beam_plus_periodic_traction() -> None:
    """Combined solve equals sum of beam-only and periodic traction-only fields."""
    from blade_precompute.section_shell_model.api.periodic import run_section_periodic_bc_mitc4
    from blade_precompute.section_shell_model.engine.periodic_cache import (
        _cached_periodic_k6,
        get_cached_periodic_reduced_bvp,
    )
    from blade_precompute.section_shell_model.engine.periodic_traction_superposition import (
        traction_displacement,
    )
    from blade_precompute.section_shell_model.pre_processing.element_library.kernel import (
        MITC4_KERNEL,
    )

    pb = _pb()
    tf, locs = _traction_fixture(pb)
    n_el = 6
    (
        K_global,
        _f,
        node_meta,
        panel_maps,
        K6,
        u_modes,
        y_e,
        z_e,
        *_rest,
    ) = _cached_periodic_k6(
        pb.panels,
        n_elements_per_panel=n_el,
        target_element_length_m=None,
        min_elements_per_panel=2,
        bc_mode="contour_support",
        interface_constraint_mode="shared",
    )
    bvp = get_cached_periodic_reduced_bvp(
        pb.panels,
        n_elements_per_panel=n_el,
        target_element_length_m=None,
        min_elements_per_panel=2,
        bc_mode="contour_support",
        interface_constraint_mode="shared",
    )

    f_beam = np.array([1.0e5, 3.0e4, 0.0, 0.0, 0.0, 0.0], dtype=float)  # N, My
    q = np.linalg.solve(K6, f_beam)
    d_beam = u_modes @ q
    d_tr = traction_displacement(
        K_global,
        node_meta,
        panel_maps,
        traction_field=tf,
        traction_points_yz=locs,
        ndof=MITC4_KERNEL.ndof,
        periodic_bvp=bvp,
        y_e=y_e,
        z_e=z_e,
    )
    d_sum = d_beam + d_tr

    from blade_precompute.section_shell_model.engine.periodic_secondary_recovery import (
        recover_resultants_from_displacement,
    )

    bundle = run_section_periodic_bc_mitc4(
        pb, N=1.0e5, My=3.0e4, n_elements_per_panel=n_el,
        traction_field=tf, traction_points_yz=locs,
    )
    # Reconstruct combined displacement via the same recovery path's assembly cache.
    # Compare residual fields through recovered Nx from explicit d_sum vs API.
    (
        _K2,
        _f2,
        _nm2,
        pm2,
        _K62,
        _um2,
        _ye2,
        _ze2,
        _abd,
        _tp,
        _gp,
        _lab,
        cache,
    ) = _cached_periodic_k6(
        pb.panels,
        n_elements_per_panel=n_el,
        target_element_length_m=None,
        min_elements_per_panel=2,
        bc_mode="contour_support",
        interface_constraint_mode="shared",
    )
    from_sum = recover_resultants_from_displacement(
        d_sum, cache, pm2, pb.panels, provenance_tag="test", shell_theory=None,
    )
    np.testing.assert_allclose(_nx(from_sum), _nx(bundle.all_panel_results), rtol=1e-8, atol=1e-5)


def test_traction_path_does_not_invoke_contour_clamps() -> None:
    """Face traction must not call build_clamp_supports."""
    from blade_precompute.section_shell_model.api.periodic import run_section_periodic_bc_mitc4

    pb = _pb()
    tf, locs = _traction_fixture(pb)
    with patch(
        "blade_precompute.section_shell_model.engine.connected_assembly_core.build_clamp_supports",
        side_effect=AssertionError("contour clamps must not be used for face traction"),
    ):
        # Also patch the import site used by the retired overlay, if any remain.
        with patch(
            "blade_precompute.section_shell_model.engine.periodic_traction_superposition.build_clamp_supports",
            create=True,
            side_effect=AssertionError("contour clamps must not be used"),
        ):
            run_section_periodic_bc_mitc4(
                pb,
                N=1.0e5,
                traction_field=tf,
                traction_points_yz=locs,
                n_elements_per_panel=6,
            )


def test_periodic_bvp_factor_reused_for_repeated_traction_rhs() -> None:
    """Memoised PeriodicReducedBVP reuses the same SuperLU factor object."""
    from blade_precompute.section_shell_model.engine.periodic_cache import (
        clear_periodic_caches,
        get_cached_periodic_reduced_bvp,
        _cached_periodic_k6,
    )
    from blade_precompute.section_shell_model.engine.periodic_traction_superposition import (
        traction_displacement_periodic,
        traction_face_force_vector,
    )
    from blade_precompute.section_shell_model.pre_processing.element_library.kernel import (
        MITC4_KERNEL,
    )

    clear_periodic_caches()
    pb = _pb()
    tf, locs = _traction_fixture(pb)
    kwargs = dict(
        n_elements_per_panel=6,
        target_element_length_m=None,
        min_elements_per_panel=2,
        bc_mode="contour_support",
        interface_constraint_mode="shared",
    )
    _cached_periodic_k6(pb.panels, **kwargs)
    bvp1 = get_cached_periodic_reduced_bvp(pb.panels, **kwargs)
    bvp2 = get_cached_periodic_reduced_bvp(pb.panels, **kwargs)
    assert bvp1 is bvp2
    assert bvp1.K_rr_factor is bvp2.K_rr_factor

    (
        _K,
        _f,
        node_meta,
        _pm,
        *_rest,
    ) = _cached_periodic_k6(pb.panels, **kwargs)
    f_tr = traction_face_force_vector(
        node_meta,
        traction_field=tf,
        traction_points_yz=locs,
        ndof=MITC4_KERNEL.ndof,
    )
    d1 = traction_displacement_periodic(bvp1, f_tr)
    d2 = traction_displacement_periodic(bvp2, 0.5 * f_tr)
    assert d1.shape == d2.shape
    assert float(np.linalg.norm(d1)) > 0.0


def test_load_channel_rejects_nonlinear_regime() -> None:
    from blade_precompute.section_shell_model.pre_processing.containers.load_channel_types import (
        SectionExternalLoadRequest,
    )
    from blade_precompute.section_shell_model.pre_processing.containers.types import (
        ShellAnalysisRegime,
    )

    with pytest.raises(ValueError, match="LINEAR"):
        SectionExternalLoadRequest.from_beam_and_traction(
            N=1.0,
            analysis_regime=ShellAnalysisRegime.NONLINEAR,
        )


def test_periodic_reduced_bvp_builds_k6() -> None:
    from blade_precompute.section_shell_model.engine.homogenisation import (
        compute_elastic_centroid_from_panels,
        compute_k6_periodic_bc,
    )
    from blade_precompute.section_shell_model.engine.periodic_reduced_bvp import (
        build_periodic_reduced_bvp,
    )
    from blade_precompute.section_shell_model.engine.section_assembly import (
        assemble_section_from_panels,
    )
    from blade_precompute.section_shell_model.pre_processing.containers.types import ShellTheory

    pb = _pb()
    panels = pb.panels
    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels]
    asm = assemble_section_from_panels(
        panels, zeros, zeros, n_elements_per_panel=6, shell_theory=ShellTheory.FSDT,
    )
    ye, ze = compute_elastic_centroid_from_panels(panels)
    bvp = build_periodic_reduced_bvp(
        asm.K_global, asm.node_meta, asm.panel_maps, y_e=ye, z_e=ze,
    )
    K6_a, u_a = bvp.compute_k6_modes(asm.K_global)
    K6_b, u_b = compute_k6_periodic_bc(
        asm.K_global, asm.node_meta, asm.panel_maps, panels, ye, ze,
    )
    np.testing.assert_allclose(K6_a, K6_b, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(u_a, u_b, rtol=1e-12, atol=1e-12)
