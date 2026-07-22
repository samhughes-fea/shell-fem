"""End-to-end beam resultant handoff: transport + shell K6 RHS."""

from __future__ import annotations

import numpy as np
import pytest

from blade_precompute.contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_precompute.section_shell_model.api.periodic import run_section_periodic_bc
from blade_precompute.section_shell_model.api.pipeline_b import build_pipeline_b_section
from blade_precompute.section_shell_model.pre_processing.loading.beam_resultant_handoff import (
    resolve_beam_resultants_for_shell_k6,
)
from blade_precompute.section_shell_model.processing.diagnostics.k6_parity import (
    compare_strip_shell_k6,
)


def _pb():
    return build_pipeline_b_section("2412", [0.35], chord_m=1.0)


def test_resolve_centroid_shim_matches_legacy_f_beam() -> None:
    """Scalar kwargs (no loading_axis) → f_beam == [N, My, Mz, T, Vz, Vy]."""
    y_e, z_e = 0.12, -0.03
    N, Vy, Vz, My, Mz, T = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    _vec, f = resolve_beam_resultants_for_shell_k6(
        y_e=y_e, z_e=z_e, N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
    )
    np.testing.assert_allclose(f, [N, My, Mz, T, Vz, Vy])


def test_resolve_transport_matches_rxF_formula() -> None:
    y_e, z_e = 0.2, -0.1
    y_L, z_L = 0.0, 0.0
    N, Vy, Vz, My, Mz, T = 100.0, 10.0, -20.0, 30.0, -40.0, 50.0
    vec = BeamResultantVector.from_named(
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        about_yz=(y_L, z_L),
    )
    at_ec, f = resolve_beam_resultants_for_shell_k6(
        y_e=y_e, z_e=z_e, resultants=vec,
    )
    dy, dz = y_L - y_e, z_L - z_e
    assert at_ec.about_yz == (y_e, z_e)
    np.testing.assert_allclose(at_ec.T, T + dy * Vz - dz * Vy)
    np.testing.assert_allclose(at_ec.My, My + dz * N)
    np.testing.assert_allclose(at_ec.Mz, Mz - dy * N)
    np.testing.assert_allclose(f, [at_ec.N, at_ec.My, at_ec.Mz, at_ec.T, at_ec.Vz, at_ec.Vy])


def test_loading_axis_yz_emits_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="loading_axis_yz"):
        resolve_beam_resultants_for_shell_k6(
            y_e=0.0, z_e=0.0, T=1.0, Vy=1.0, Vz=0.0,
            loading_axis_yz=(0.5, 0.0),
        )


def test_centroid_referenced_vector_matches_scalar_kwargs_solve() -> None:
    """Vector referenced at (y_e, z_e) is bit-identical to the scalar shim path."""
    pb = _pb()
    from blade_precompute.section_shell_model.engine.homogenisation import (
        compute_elastic_centroid_from_panels,
    )

    y_e, z_e = compute_elastic_centroid_from_panels(pb.panels)
    loads = dict(N=0.0, Vy=1e3, Vz=2e3, My=5e3, Mz=-3e3, T=1e3)
    b_scalar = run_section_periodic_bc(
        pb, n_elements_per_panel=3, min_elements_per_panel=2, **loads,
    )
    vec = BeamResultantVector.from_named(
        **loads, frame=FRAME_SECTION_YZ, about_yz=(float(y_e), float(z_e)),
    )
    b_vec = run_section_periodic_bc(
        pb, resultants=vec, n_elements_per_panel=3, min_elements_per_panel=2,
    )
    np.testing.assert_allclose(
        b_scalar.beam_mode_amplitudes, b_vec.beam_mode_amplitudes, rtol=0, atol=0,
    )
    assert b_vec.y_e == pytest.approx(float(y_e))
    assert b_vec.z_e == pytest.approx(float(z_e))
    # Deprecated aliases still dual-written
    assert b_vec.y_sc == pytest.approx(float(y_e))
    assert b_vec.z_sc == pytest.approx(float(z_e))


def test_offset_axis_changes_torsion_mode_amplitude() -> None:
    """Offset about_yz changes T_ec → torsion mode amplitude vs centroid-ref."""
    pb = _pb()
    from blade_precompute.section_shell_model.engine.homogenisation import (
        compute_elastic_centroid_from_panels,
    )

    y_e, z_e = compute_elastic_centroid_from_panels(pb.panels)
    # Pure shear about beam axis — transport creates torque about EC.
    base = dict(N=0.0, Vy=5e3, Vz=-2e3, My=0.0, Mz=0.0, T=0.0)
    at_ec = BeamResultantVector.from_named(
        **base, about_yz=(float(y_e), float(z_e)),
    )
    at_origin = BeamResultantVector.from_named(
        **base, about_yz=BEAM_LOAD_AXIS_YZ,
    )
    b0 = run_section_periodic_bc(
        pb, resultants=at_ec, n_elements_per_panel=3, min_elements_per_panel=2,
    )
    b1 = run_section_periodic_bc(
        pb, resultants=at_origin, n_elements_per_panel=3, min_elements_per_panel=2,
    )
    # Mode 3 = torsion conjugate; transport of shears about offset axis is nonzero
    # whenever (y_e, z_e) != (0, 0).
    if abs(y_e) + abs(z_e) > 1e-9:
        assert not np.isclose(
            float(b0.beam_mode_amplitudes[3]),
            float(b1.beam_mode_amplitudes[3]),
        )


def test_k6_parity_named_diagonals_ignore_shear_slot_swap() -> None:
    strip = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # strip order N,My,Mz,T,Vy,Vz
    # Shell order N,My,Mz,T,Vz,Vy — swap last two diagonals vs strip numbering
    shell = np.diag([1.0, 2.0, 3.0, 4.0, 6.0, 5.0])
    report = compare_strip_shell_k6(strip, shell, rel_tol=0.01)
    assert report.ok
    assert report.rel_err_named["N"] == 0.0
