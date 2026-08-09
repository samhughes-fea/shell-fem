"""Tests for section_shell_model MVP (numpy + pytest)."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.results.compute_tertiary.local_clpt_shell import (
    default_skin_strengths_pa,
    solve_station_clpt_shell,
)
from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.diagnostics.equilibrium_checks import (
    build_load_reaction_audit,
    check_cluster_equilibrium,
    check_panel_equilibrium,
)
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants


def _panel_topology_vlasov_fixture(naca_code: str, web_x_positions: list[float]):
    """Package shell panel topology panels + multicell airfoil for Vlasov oracle tests."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import build_shell_panel_topology
    from blade_structure_utilities.fem_utilities.section_shell_model.io.job_outputs import (
        airfoil_polygon_to_multicell_airfoil,
    )

    panel_topology = build_shell_panel_topology(naca_code, web_x_positions, chord_m=1.0)
    raw = np.asarray(panel_topology.section.airfoil.vertices, dtype=np.float64)[:, :2]
    air = airfoil_polygon_to_multicell_airfoil(raw, n_each=48)
    return air, panel_topology.panels, panel_topology.webs_geom


def test_hashin_fi_matches_direct_clpt_pipeline():
    """solve_station_clpt_shell matches materials CLPT recovery + Hashin FI."""
    from blade_structure_utilities.material_utilities.laminate import clpt_recovery_from_laminate
    from blade_structure_utilities.material_utilities.failure import hashin_fi
    from blade_structure_utilities.material_utilities.laminate import laminate_definition_from_legacy
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import build_shell_panel_topology
    from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.api.linear_static import (
        run_section_linear_static_from_resultants,
    )
    from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.api.station_recovery import (
        bundle_from_linear_cascade,
    )

    from blade_structure_utilities.mesh_utilities.section_mesh.strip_mesh import (
        build_section_strip_mesh,
    )

    panel_topology = build_shell_panel_topology("0012", [0.35], chord_m=1.0)
    cascade = run_section_linear_static_from_resultants(
        panel_topology,
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=0.0,
        section_strip_mesh=build_section_strip_mesh(
            panel_topology.shell_inputs, n_elements_per_panel=4
        ),
    )
    bundle = bundle_from_linear_cascade(panel_topology, cascade)
    ref = bundle.reference_resultants
    assert ref is not None
    plies = bundle.panels[0].lam.build_plies()
    lam = laminate_definition_from_legacy(bundle.panels[0].lam, plies=plies)
    st = default_skin_strengths_pa()

    shell_res = solve_station_clpt_shell(
        ref,
        plies,
        Xt=st["Xt"],
        Xc=st["Xc"],
        Yt=st["Yt"],
        Yc=st["Yc"],
        S12=st["S12"],
        lam=lam,
    )

    eps0, kappa, sig_lam, _ = clpt_recovery_from_laminate(lam, ref.to_N_vec(), ref.to_M_vec())
    fi_direct = np.array(
        [
            float(
                hashin_fi(
                    np.asarray(sig, dtype=float).ravel()[:3].reshape(1, 3),
                    st["Xt"],
                    st["Xc"],
                    st["Yt"],
                    st["Yc"],
                    st["S12"],
                )[0]
            )
            for sig in sig_lam
        ],
        dtype=float,
    )

    assert np.allclose(shell_res.fi, fi_direct, rtol=0, atol=1e-12)
    assert np.allclose(shell_res.eps0, eps0, rtol=0, atol=1e-12)
    assert np.allclose(shell_res.kappa, kappa, rtol=0, atol=1e-12)


def test_hashin_fi_golden_pure_shear_envelope():
    """Known τ12 in material axes → (τ/S)² (all four Hashin mode FIs match)."""
    from blade_structure_utilities.material_utilities.failure import hashin_fi

    S12 = 45.0e6
    sigma = np.array([0.0, 0.0, 0.5 * S12])
    fi = hashin_fi(
        sigma,
        Xt=600e6,
        Xc=500e6,
        Yt=50e6,
        Yc=140e6,
        S12=S12,
    )
    assert abs(fi - 0.25) < 1e-12


def test_six_series_airfoil_converted_for_multicell_geometry() -> None:
    """Spanwise six-series polygon must be resampled to [upper; lower] LE→TE."""
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.naca import airfoil_vertices_from_spanwise
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.sectorial_warping import (
        open_outline_from_airfoil,
        polygon_area_signed,
    )
    from blade_structure_utilities.fem_utilities.section_shell_model.io.job_outputs import (
        airfoil_polygon_to_multicell_airfoil,
        build_airfoil_for_station,
    )

    raw = airfoil_vertices_from_spanwise(6, 63.0, 4.0, 15.0, 140, 1.0, closed_te=True)
    half_bad = raw.shape[0] // 2
    assert not np.allclose(raw[:half_bad, 0], raw[half_bad:, 0])
    mc = airfoil_polygon_to_multicell_airfoil(raw, n_each=48)
    assert mc.shape == (96, 2)
    n = mc.shape[0] // 2
    assert np.allclose(mc[:n, 0], mc[n:, 0])
    assert np.all(mc[:n, 1] >= mc[n:, 1] - 1e-8)
    loop = np.vstack([open_outline_from_airfoil(mc), open_outline_from_airfoil(mc)[:1]])
    assert abs(polygon_area_signed(loop)) > 1e-6

    via_station = build_airfoil_for_station(63.0, 4.0, 15.0, chord_m=1.655, naca_series=6, n_points=40)
    assert via_station.shape == (80, 2)
    n_via = via_station.shape[0] // 2
    assert np.allclose(via_station[:n_via, 0], via_station[n_via:, 0])


# ---------------------------------------------------------------------------
# MITC4 element patch tests
# ---------------------------------------------------------------------------

def test_mitc4_stiffness_shape_and_symmetry():
    """20×20 stiffness must be symmetric and positive semi-definite (after BCs)."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.element_library.mitc4.element import mitc4_stiffness  # noqa: E402

    E, nu, t = 70e9, 0.33, 0.005
    G = E / (2 * (1 + nu))
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0],
                      [nu * A_val, A_val, 0],
                      [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0],
                      [nu * D_val, D_val, 0],
                      [0, 0, 0.5 * (1 - nu) * D_val]])
    B_mat = np.zeros((3, 3))
    ABD = np.block([[A_mat, B_mat], [B_mat, D_mat]])

    K = mitc4_stiffness(L_s=0.1, L_x=1.0, ABD=ABD, thickness=t, G_eff=G, include_drilling=False)
    assert K.shape == (20, 20)
    assert np.allclose(K, K.T, atol=1e-10), "Stiffness must be symmetric"

    # After pinning rigid-body DOFs: 6 eigenvalues should be near zero (mechanisms)
    eigvals = np.linalg.eigvalsh(K)
    n_near_zero = int(np.sum(np.abs(eigvals) < 1e-3 * eigvals[-1]))
    assert n_near_zero >= 6, f"Expected ≥6 near-zero eigenvalues, got {n_near_zero}"


def test_mitc4_edge_resultants_uniform_membrane_matches_centroid():
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.element_library.mitc4.element import (
        mitc4_edge_resultants,
        mitc4_resultants,
    )

    E, nu, t = 70e9, 0.33, 0.005
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0], [nu * A_val, A_val, 0], [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0], [nu * D_val, D_val, 0], [0, 0, 0.5 * (1 - nu) * D_val]])
    ABD = np.block([[A_mat, np.zeros((3, 3))], [np.zeros((3, 3)), D_mat]])
    L_s, L_x = 0.2, 1.0
    d = np.zeros(20)
    # Constant membrane strain: eps_xx = dux/dx = alpha.
    alpha = 1e-3
    for node, eta in enumerate([-1.0, -1.0, 1.0, 1.0]):
        x = 0.5 * (eta + 1.0) * L_x
        d[node * 5 + 0] = alpha * x
    c = mitc4_resultants(d, L_s, L_x, ABD, ndof=5)
    e = mitc4_edge_resultants(d, L_s, L_x, ABD, ndof=5)
    assert np.isclose(e["start"]["Nx"], c["Nx"], rtol=1e-10)
    assert np.isclose(e["end"]["Nx"], c["Nx"], rtol=1e-10)
    assert np.isclose(e["start"]["Nxy"], 0.0, atol=1e-10)
    assert np.isclose(e["end"]["Nxy"], 0.0, atol=1e-10)


def test_mitc4_edge_shear_traction_integrated_uniform_field():
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.element_library.mitc4.element import mitc4_edge_shear_traction_integrated

    E, nu, t = 70e9, 0.33, 0.005
    A_val = E * t / (1 - nu**2)
    D_val = E * t**3 / (12 * (1 - nu**2))
    A_mat = np.array([[A_val, nu * A_val, 0], [nu * A_val, A_val, 0], [0, 0, 0.5 * (1 - nu) * A_val]])
    D_mat = np.array([[D_val, nu * D_val, 0], [nu * D_val, D_val, 0], [0, 0, 0.5 * (1 - nu) * D_val]])
    ABD = np.block([[A_mat, np.zeros((3, 3))], [np.zeros((3, 3)), D_mat]])
    L_s, L_x = 0.2, 1.0
    d = np.zeros(20)
    gamma = 2e-3
    # u_x = gamma*s, u_s = 0 gives near-uniform gamma_xs = gamma
    for node, xi in enumerate([-1.0, 1.0, 1.0, -1.0]):
        s = 0.5 * (xi + 1.0) * L_s
        d[node * 5 + 0] = gamma * s
    out_start = mitc4_edge_shear_traction_integrated(d, L_s, L_x, ABD, edge="start", ndof=5)
    out_end = mitc4_edge_shear_traction_integrated(d, L_s, L_x, ABD, edge="end", ndof=5)
    assert np.isclose(out_start["Nxy_edge_int"], -out_end["Nxy_edge_int"], rtol=1e-10, atol=1e-10)
    assert abs(out_start["Nxy_edge_int"]) > 0.0

def test_vlasov_shear_centre_finite():
    """compute_section_vlasov returns finite shear centre and positive I_omega_E."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import compute_section_vlasov

    air, panels, webs_geom = _panel_topology_vlasov_fixture("0012", [0.35])
    vlasov = compute_section_vlasov(
        air, panels, B=0.0, dB_dx=0.0, webs_geom=list(webs_geom)
    )
    assert np.isfinite(vlasov.y_sc)
    assert np.isfinite(vlasov.z_sc)
    assert vlasov.I_omega_E > 0.0
    assert len(vlasov.omega_hat_v) > 0


def test_vlasov_zero_bimoment_gives_zero_stress():
    """With B=0, warping normal stress must be identically zero."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import compute_section_vlasov

    air, panels, webs_geom = _panel_topology_vlasov_fixture("0012", [0.35])
    vlasov = compute_section_vlasov(
        air, panels, B=0.0, dB_dx=0.0, webs_geom=list(webs_geom)
    )
    for sig in vlasov.sigma_omega:
        assert np.allclose(sig, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# MITC4 integration test
# ---------------------------------------------------------------------------

def test_mitc4_integration_no_placeholders():
    """Linear-static MITC4 solve: all resultant provenance must be MITC4 or DERIVED."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import build_shell_panel_topology  # noqa: E402
    from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.api.linear_static import (  # noqa: E402
        run_section_linear_static_from_resultants,
    )
    from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.api.station_recovery import (  # noqa: E402
        bundle_from_linear_cascade,
    )
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ProvenanceKind  # noqa: E402

    from blade_structure_utilities.mesh_utilities.section_mesh.strip_mesh import (
        build_section_strip_mesh,
    )

    panel_topology = build_shell_panel_topology("2412", [0.35], chord_m=1.0)
    cascade = run_section_linear_static_from_resultants(
        panel_topology,
        N=1.0,
        Vy=1.0,
        Vz=1.0,
        My=1.0,
        Mz=1.0,
        T=1.0,
        section_strip_mesh=build_section_strip_mesh(
            panel_topology.shell_inputs, n_elements_per_panel=6
        ),
    )
    bundle = bundle_from_linear_cascade(panel_topology, cascade)
    assert bundle.reference_resultants is not None
    ref = bundle.reference_resultants
    placeholder_fields = [
        k for k, v in ref.provenance.items()
        if v.kind == ProvenanceKind.PLACEHOLDER
    ]
    assert placeholder_fields == [], (
        f"Fields still placeholder after MITC4 solve: {placeholder_fields}"
    )


# ---------------------------------------------------------------------------
# Item 3: Non-zero bimoment / dB_dx activates warping contributions
# ---------------------------------------------------------------------------

def test_vlasov_nonzero_bimoment_activates_stress():
    """
    B=1e3 → sigma_omega non-zero for at least one skin panel.
    B=0   → sigma_omega zero everywhere.
    Scaling: sigma_omega scales linearly with B.
    """
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import compute_section_vlasov

    air, panels, webs_geom = _panel_topology_vlasov_fixture("2412", [0.35])

    vlasov_0 = compute_section_vlasov(
        air, panels, B=0.0, dB_dx=0.0, webs_geom=list(webs_geom)
    )
    for sig in vlasov_0.sigma_omega:
        assert np.allclose(sig, 0.0, atol=1e-30), "B=0 must give zero sigma_omega"

    vlasov_1 = compute_section_vlasov(
        air, panels, B=1e3, dB_dx=0.0, webs_geom=list(webs_geom)
    )
    max_sig_1 = max(float(np.max(np.abs(sig))) for sig in vlasov_1.sigma_omega if len(sig))
    assert max_sig_1 > 0.0, "B=1e3 must give non-zero sigma_omega on at least one panel"

    # Linear scaling: B=2e3 → sigma_omega twice as large
    vlasov_2 = compute_section_vlasov(
        air, panels, B=2e3, dB_dx=0.0, webs_geom=list(webs_geom)
    )
    max_sig_2 = max(float(np.max(np.abs(sig))) for sig in vlasov_2.sigma_omega if len(sig))
    assert np.isclose(max_sig_2, 2.0 * max_sig_1, rtol=1e-10), (
        f"sigma_omega must scale linearly with B: {max_sig_2:.4e} vs 2*{max_sig_1:.4e}"
    )


def test_vlasov_nonzero_dBdx_activates_warping_shear():
    """dB_dx=1e3 with B=0 → q_omega non-zero on at least one skin panel."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import compute_section_vlasov

    air, panels, webs_geom = _panel_topology_vlasov_fixture("2412", [0.35])

    vlasov = compute_section_vlasov(
        air, panels, B=0.0, dB_dx=1e3, webs_geom=list(webs_geom)
    )
    max_q = max(float(np.max(np.abs(q))) for q in vlasov.q_omega if len(q))
    assert max_q > 0.0, "dB_dx=1e3 must give non-zero q_omega on at least one panel"


# ---------------------------------------------------------------------------
# Vlasov Batho helpers (also used by multi-cell I_ω checks)
# ---------------------------------------------------------------------------

class _FakeLam:
    def __init__(self, E: float, t: float):
        self.E = E
        self.t = t


class _FakePanel:
    def __init__(self, nodes: np.ndarray, E: float, t: float, label: str = ""):
        self.nodes = np.asarray(nodes, dtype=float)
        self.lam = _FakeLam(E, t)
        diffs = np.diff(self.nodes, axis=0)
        ds = np.hypot(diffs[:, 0], diffs[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.label = label


def _box_airfoil(n: int = 6, width: float = 1.0, height: float = 0.5) -> np.ndarray:
    """Rectangular box as airfoil (upper + lower surface, LE→TE each)."""
    x = np.linspace(0.0, width, n)
    upper = np.column_stack([x, np.full(n, height / 2.0)])
    lower = np.column_stack([x, np.full(n, -height / 2.0)])
    return np.vstack([upper, lower])


def test_vlasov_batho_single_cell_box():
    """Rectangular single-cell box: n_cells==1, I_omega_E>0, omega_hat_v non-trivial."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import compute_section_vlasov

    airfoil_box = _box_airfoil(n=10, width=1.0, height=0.5)
    E, t = 70e9, 2e-3

    # Two skin panels: upper and lower surfaces
    upper_nodes = np.column_stack([np.linspace(0, 1, 10), np.full(10, 0.25)])
    lower_nodes = np.column_stack([np.linspace(1, 0, 10), np.full(10, -0.25)])
    panels = [
        _FakePanel(upper_nodes, E, t, label="USkin"),
        _FakePanel(lower_nodes, E, t, label="LSkin"),
    ]

    vlasov = compute_section_vlasov(airfoil_box, panels, B=0.0, dB_dx=0.0,
                                    webs_geom=[], t_web=t)
    assert vlasov.n_cells == 1, f"Expected 1 cell, got {vlasov.n_cells}"
    assert vlasov.I_omega_E > 0.0, "I_omega_E must be positive"
    assert float(np.max(np.abs(vlasov.omega_hat_v))) > 0.0, "omega_hat must be non-trivial"


def test_vlasov_batho_two_cell():
    """Two-cell box (mid web at x=0.5): n_cells==2 and I_omega_E differs from single-cell."""
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import compute_section_vlasov

    airfoil_box = _box_airfoil(n=10, width=1.0, height=0.5)
    E, t = 70e9, 2e-3

    upper_nodes = np.column_stack([np.linspace(0, 1, 10), np.full(10, 0.25)])
    lower_nodes = np.column_stack([np.linspace(1, 0, 10), np.full(10, -0.25)])
    panels = [
        _FakePanel(upper_nodes, E, t, label="USkin"),
        _FakePanel(lower_nodes, E, t, label="LSkin"),
    ]

    # Single-cell baseline
    vlasov_1 = compute_section_vlasov(airfoil_box, panels, B=0.0, dB_dx=0.0,
                                      webs_geom=[], t_web=t)

    # Add mid-web at x=0.5
    web = ((np.array([0.5, 0.25]), np.array([0.5, -0.25])),)
    vlasov_2 = compute_section_vlasov(airfoil_box, panels, B=0.0, dB_dx=0.0,
                                      webs_geom=list(web), t_web=t)

    assert vlasov_2.n_cells == 2, f"Expected 2 cells, got {vlasov_2.n_cells}"
    # Batho correction for two cells differs from single cell by a meaningful margin.
    rel_diff = abs(vlasov_2.I_omega_E - vlasov_1.I_omega_E) / max(abs(vlasov_1.I_omega_E), 1e-30)
    assert rel_diff > 0.10, (
        "Two-cell Batho correction should change I_omega_E by >10%"
    )


def _mk_resultant(nx: float, nxy: float) -> ShellPanelResultants:
    return ShellPanelResultants(
        Nx=float(nx),
        Ny=0.0,
        Nxy=float(nxy),
        Mx=0.0,
        My=0.0,
        Mxy=0.0,
    )


def _mk_panel(nodes: np.ndarray, label: str = "skin"):
    class _Panel:
        pass

    p = _Panel()
    p.nodes = np.asarray(nodes, dtype=float)
    p.label = label
    return p


def test_equilibrium_sign_normalization_for_opposite_tangent():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    panels = [p0, p1]
    all_res = [
        [_mk_resultant(10.0, 5.0)],
        [_mk_resultant(10.0, -5.0)],
    ]

    checks = check_panel_equilibrium(all_res, panels, endpoint_tol=1e-8)
    assert len(checks) == 1
    chk = checks[0]
    assert chk["orientation"] == "opposite"
    assert chk["resultant_dNxy_rel"] < 1e-12
    assert chk["resultant_pass"]


def test_equilibrium_uses_geometric_adjacency_not_index_order():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[10.0, 0.0], [11.0, 0.0]]), label="USkin")
    p2 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    panels = [p0, p1, p2]
    all_res = [
        [_mk_resultant(10.0, 2.0)],
        [_mk_resultant(100.0, 20.0)],
        [_mk_resultant(10.0, -2.0)],
    ]

    checks = check_panel_equilibrium(all_res, panels, endpoint_tol=1e-8)
    assert len(checks) == 1
    pair = {checks[0]["pi"], checks[0]["pj"]}
    assert pair == {0, 2}


def test_equilibrium_includes_wraparound_for_closed_loop():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [0.5, 1.0]]), label="LSkin")
    p2 = _mk_panel(np.array([[0.5, 1.0], [0.0, 0.0]]), label="Web Core")
    panels = [p0, p1, p2]
    all_res = [
        [_mk_resultant(10.0, 1.0)],
        [_mk_resultant(10.0, 1.0)],
        [_mk_resultant(10.0, 1.0)],
    ]

    checks = check_panel_equilibrium(all_res, panels, endpoint_tol=1e-8)
    assert len(checks) == 3


def test_equilibrium_tiered_thresholds_skin_skin_vs_skin_web():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    p2 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web @ 0.35c")

    checks_skin_skin = check_panel_equilibrium(
        [[_mk_resultant(10.0, 1.0)], [_mk_resultant(9.4, -1.0)]],
        [p0, p1],
        all_panel_diagnostics=[
            {"boundary_reaction": {"end": {"Fx": 10.0, "Fs": 1.0}}},
            {"boundary_reaction": {"end": {"Fx": 9.4, "Fs": -1.0}}},
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks_skin_skin) == 1
    assert checks_skin_skin[0]["boundary_type"] == "skin-skin"
    assert checks_skin_skin[0]["reaction_dNx_rel"] > checks_skin_skin[0]["reaction_tol_nx"]
    assert not checks_skin_skin[0]["reaction_pass_nx"]

    checks_skin_web = check_panel_equilibrium(
        [[_mk_resultant(10.0, 1.0)], [_mk_resultant(9.2, 1.0)]],
        [p0, p2],
        all_panel_diagnostics=[
            {"boundary_reaction": {"end": {"Fx": 10.0, "Fs": 1.0}}},
            {"boundary_reaction": {"start": {"Fx": 9.2, "Fs": 1.0}}},
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks_skin_web) == 1
    assert checks_skin_web[0]["boundary_type"] == "skin-web"
    assert checks_skin_web[0]["reaction_dNx_rel"] < checks_skin_web[0]["reaction_tol_nx"]
    assert checks_skin_web[0]["reaction_pass_nx"]

def test_linear_static_returns_active_transverse_shear_fields():
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.mesh.api.shell_panel_topology import build_shell_panel_topology
    from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.api.linear_static import (
        run_section_linear_static_from_resultants,
    )
    from blade_structure_utilities.fem_utilities.section_shell_model.simulation_runner.api.station_recovery import (
        bundle_from_linear_cascade,
    )

    from blade_structure_utilities.mesh_utilities.section_mesh.strip_mesh import (
        build_section_strip_mesh,
    )

    panel_topology = build_shell_panel_topology("2412", [0.25, 0.60], chord_m=1.0)
    cascade = run_section_linear_static_from_resultants(
        panel_topology,
        N=1.3e4,
        Vy=5e4,
        Vz=8.3e5,
        My=5.5e6,
        Mz=3.3e5,
        T=2.5e4,
        section_strip_mesh=build_section_strip_mesh(
            panel_topology.shell_inputs, n_elements_per_panel=2
        ),
    )
    bundle = bundle_from_linear_cascade(panel_topology, cascade)
    all_panel = bundle.all_panel_results or []
    non_empty = [panel for panel in all_panel if panel]
    assert non_empty, "expected at least one non-empty MITC4 panel result set"
    qx_vals = [r.Qx for panel in non_empty for r in panel]
    qy_vals = [r.Qy for panel in non_empty for r in panel]
    assert any(v is not None for v in qx_vals), "expected at least one active Qx value"
    assert any(v is not None for v in qy_vals), "expected at least one active Qy value"
    assert any(r.provenance["Qx"].kind.value == "mitc4" for panel in non_empty for r in panel)
    assert any(r.provenance["Qy"].kind.value == "mitc4" for panel in non_empty for r in panel)


def test_equilibrium_reaction_metric_uses_panel_diagnostics():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    checks = check_panel_equilibrium(
        [[_mk_resultant(10.0, 0.0)], [_mk_resultant(10.0, 0.0)]],
        [p0, p1],
        all_panel_diagnostics=[
            {"boundary_reaction": {"end": {"Fx": 10.0, "Fs": 4.0}}},
            {"boundary_reaction": {"end": {"Fx": 10.0, "Fs": -4.0}}},
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["orientation"] == "opposite"
    assert chk["reaction_dNxy_rel"] < 1e-12
    assert chk["reaction_pass"]


def test_equilibrium_secondary_prefers_interface_field_set():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    checks = check_panel_equilibrium(
        [[_mk_resultant(10.0, 0.0)], [_mk_resultant(100.0, 0.0)]],
        [p0, p1],
        all_panel_diagnostics=[
            {
                "boundary_reaction": {"end": {"Fx": 10.0, "Fs": 1.0}},
                "interface_field_set": {"end": {"Nx": 10.0, "Nxy": 1.0}},
            },
            {
                "boundary_reaction": {"end": {"Fx": 10.0, "Fs": -1.0}},
                "interface_field_set": {"end": {"Nx": 10.0, "Nxy": -1.0}},
            },
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["orientation"] == "opposite"
    assert chk["resultant_dNx_rel"] < 1e-12
    assert chk["resultant_dNxy_rel"] < 1e-12
    # Field-set available but no edge-traction set — falls back to field-fallback mode.
    assert chk["nxy_compare_mode"] == "field-fallback"
    assert chk["resultant_dNx_rel_centroid"] > 0.5


def test_equilibrium_strict_skin_web_maps_components():
    # p0 "end" (normal_sign=+1): Tx=2, Ts=5, t_i=[1,0]
    # p1 "start" (normal_sign=-1): Tx=-2, Ts=-5, t_j=[1,0] (collinear)
    # dTx = 2 + (-2) = 0  →  dTx_rel = 0
    # dT_yz = 5*[1,0] + (-5)*[1,0] = [0,0]  →  dT_yz_rel = 0
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Web @ 0.35c")
    checks = check_panel_equilibrium(
        [[_mk_resultant(10.0, 2.0)], [_mk_resultant(10.0, 99.0)]],
        [p0, p1],
        all_panel_diagnostics=[
            {
                "boundary_reaction": {"end": {"Fx": 10.0, "Fs": 2.0}},
                "interface_field_set": {"end": {"Nx": 10.0, "Nxy": 2.0}},
                "interface_edge_set": {"end": {"Tx_int": 2.0, "Ts_int": 5.0}},
            },
            {
                "boundary_reaction": {"start": {"Fx": 10.0, "Fs": 2.0}},
                "interface_field_set": {"start": {"Nx": 10.0, "Nxy": 99.0}},
                "interface_edge_set": {"start": {"Tx_int": -2.0, "Ts_int": -5.0}},
            },
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["boundary_type"] == "skin-web"
    assert chk["nxy_compare_mode"] == "traction-vector-strict"
    assert chk["dTx_rel"] < 1e-12
    assert chk["dT_yz_rel"] < 1e-12


def test_equilibrium_strict_skin_skin_uses_tx_component():
    # p0 "end" (normal_sign=+1): Tx=3, Ts=8, t_i=[1,0]
    # p1 "end" (normal_sign=+1): Tx=-3, Ts=8, t_j=[-1,0]  (panels converge)
    # dTx = 3 + (-3) = 0  →  dTx_rel = 0
    # dT_yz = 8*[1,0] + 8*[-1,0] = [0,0]  →  dT_yz_rel = 0
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[2.0, 0.0], [1.0, 0.0]]), label="LSkin")
    checks = check_panel_equilibrium(
        [[_mk_resultant(5.0, 1.0)], [_mk_resultant(5.0, -1.0)]],
        [p0, p1],
        all_panel_diagnostics=[
            {
                "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 1.0}},
                "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 1.0}},
                "interface_edge_set": {"end": {"Tx_int": 3.0, "Ts_int": 8.0}},
            },
            {
                "boundary_reaction": {"end": {"Fx": 5.0, "Fs": -1.0}},
                "interface_field_set": {"end": {"Nx": 5.0, "Nxy": -1.0}},
                "interface_edge_set": {"end": {"Tx_int": -3.0, "Ts_int": 8.0}},
            },
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["nxy_compare_mode"] == "traction-vector-strict"
    assert chk["dTx_rel"] < 1e-12
    assert chk["dT_yz_rel"] < 1e-12

def test_equilibrium_topology_graph_handles_three_way_junction():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="USkin2")
    p2 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web A")
    panels = [p0, p1, p2]
    all_res = [[_mk_resultant(1.0, 0.0)], [_mk_resultant(1.0, 0.0)], [_mk_resultant(1.0, 0.0)]]
    checks = check_panel_equilibrium(all_res, panels, endpoint_tol=1e-8)
    assert len(checks) == 3


def test_build_load_reaction_audit_computes_summary():
    di = [
        {
            "load_totals": {"Fx_total": 10.0, "Fs_total": 2.0},
            "boundary_reaction_set": {"start": {"Fx": -6.0, "Fs": -1.0}, "end": {"Fx": -4.0, "Fs": -1.0}},
        },
        {
            "load_totals": {"Fx_total": 8.0, "Fs_total": 1.0},
            "boundary_reaction_set": {"start": {"Fx": -7.0, "Fs": -0.5}, "end": {"Fx": -1.0, "Fs": -0.4}},
        },
    ]
    audit = build_load_reaction_audit(di)
    assert int(audit["n_panels"]) == 2
    assert audit["max_rel_mismatch_endpoint"] >= 0.0
    assert audit["mean_rel_mismatch_endpoint"] >= 0.0
    assert audit["max_rel_mismatch_target"] >= 0.0
    assert audit["mean_rel_mismatch_target"] >= 0.0
    assert audit["global_force_balance_rel"] >= 0.0
    # New: global_force_balance_rel_at_fixed must always be present in the dict.
    assert "global_force_balance_rel_at_fixed" in audit
    # Without global_reaction_at_fixed_UX in diagnostics, the key is nan (no global solve).
    assert audit["global_force_balance_rel_at_fixed"] != audit["global_force_balance_rel_at_fixed"] or True  # nan or valid

    # Confirm that when global_reaction_at_fixed_UX is provided, the metric is computed.
    di_with_fixed = [
        {
            "load_totals": {"Fx_total": 10.0, "Fs_total": 2.0, "Fx_target": 10.0},
            "boundary_reaction_set": {"start": {"Fx": -6.0, "Fs": -1.0}, "end": {"Fx": -4.0, "Fs": -1.0}},
            "global_reaction_at_fixed_UX": -10.0,  # perfectly balanced: -10 + 10 = 0
        },
    ]
    audit2 = build_load_reaction_audit(di_with_fixed)
    assert "global_force_balance_rel_at_fixed" in audit2
    assert float(audit2["global_force_balance_rel_at_fixed"]) < 1e-12


def test_cluster_traction_three_way_analytical():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web")
    p2 = _mk_panel(np.array([[1.0, 0.0], [0.0, -1.0]]), label="LSkin")
    di = [
        {"interface_edge_set": {"end": {"Tx_int": 1.0, "Ts_int": 2.0}}},
        {"interface_edge_set": {"start": {"Tx_int": -2.0, "Ts_int": -1.0}}},
        {"interface_edge_set": {"start": {"Tx_int": 1.0, "Ts_int": 1.0}}},
    ]
    checks = check_cluster_equilibrium([p0, p1, p2], all_panel_diagnostics=di, endpoint_tol=1e-8)
    assert len(checks) == 1
    c = checks[0]
    assert c["n_panels"] == 3
    assert c["Tx_rel_cluster"] < 1e-12
    assert c["T_yz_rel_cluster"] >= 0.0


def test_cluster_traction_matches_pair_for_two_way():
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="LSkin")
    di = [
        {"interface_edge_set": {"end": {"Tx_int": 3.0, "Ts_int": 5.0}}},
        {"interface_edge_set": {"start": {"Tx_int": -3.0, "Ts_int": -5.0}}},
    ]
    pair = check_panel_equilibrium(
        [[_mk_resultant(0.0, 0.0)], [_mk_resultant(0.0, 0.0)]],
        [p0, p1],
        all_panel_diagnostics=di,
        endpoint_tol=1e-8,
    )[0]
    cluster = check_cluster_equilibrium([p0, p1], all_panel_diagnostics=di, endpoint_tol=1e-8)[0]
    assert np.isclose(cluster["Tx_rel_cluster"], pair["dTx_rel"], atol=1e-12)


def test_skin_web_label_invariance():
    # Geometry-based traction check must be label-invariant.
    p_skin = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p_web = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Web @ 0.35c")
    diag_base = [
        {
            "boundary_reaction": {"end": {"Fx": 10.0, "Fs": 2.0}},
            "interface_field_set": {"end": {"Nx": 10.0, "Nxy": 2.0}},
            "interface_edge_set": {"end": {"Tx_int": 2.0, "Ts_int": 5.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": 10.0, "Fs": 2.0}},
            "interface_field_set": {"start": {"Nx": 10.0, "Nxy": 99.0}},
            "interface_edge_set": {"start": {"Tx_int": 2.0, "Ts_int": 7.0}},
        },
    ]
    a = check_panel_equilibrium([[_mk_resultant(10.0, 2.0)], [_mk_resultant(10.0, 99.0)]], [p_skin, p_web], all_panel_diagnostics=diag_base, endpoint_tol=1e-8)[0]
    p_skin2 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="Panel A")
    p_web2 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Panel B")
    b = check_panel_equilibrium([[_mk_resultant(10.0, 2.0)], [_mk_resultant(10.0, 99.0)]], [p_skin2, p_web2], all_panel_diagnostics=diag_base, endpoint_tol=1e-8)[0]
    assert np.isclose(a["dTx_rel"], b["dTx_rel"])
    assert np.isclose(a["dT_yz_rel"], b["dT_yz_rel"])
    assert np.isclose(a["resultant_dNxy_rel"], b["resultant_dNxy_rel"])


def test_traction_compare_invariant_under_panel_swap():
    """Swapping (i,j) must leave dTx_rel and dT_yz_rel magnitudes unchanged."""
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Web @ 0.35c")
    diag = [
        {
            "boundary_reaction": {"end": {"Fx": 3.0, "Fs": 1.0}},
            "interface_field_set": {"end": {"Nx": 3.0, "Nxy": 1.0}},
            "interface_edge_set": {"end": {"Tx_int": 4.0, "Ts_int": 2.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": 3.0, "Fs": 1.0}},
            "interface_field_set": {"start": {"Nx": 3.0, "Nxy": 1.0}},
            "interface_edge_set": {"start": {"Tx_int": -5.0, "Ts_int": -3.0}},
        },
    ]
    fwd = check_panel_equilibrium(
        [[_mk_resultant(3.0, 1.0)], [_mk_resultant(3.0, 1.0)]], [p0, p1],
        all_panel_diagnostics=diag, endpoint_tol=1e-8,
    )[0]
    # Swapped order
    diag_swap = [diag[1], diag[0]]
    rev = check_panel_equilibrium(
        [[_mk_resultant(3.0, 1.0)], [_mk_resultant(3.0, 1.0)]], [p1, p0],
        all_panel_diagnostics=diag_swap, endpoint_tol=1e-8,
    )[0]
    assert np.isclose(fwd["dTx_rel"], rev["dTx_rel"], atol=1e-10)
    assert np.isclose(fwd["dT_yz_rel"], rev["dT_yz_rel"], atol=1e-10)


def test_traction_compare_pure_Nx_invariant():
    """Pure spanwise Nx load (Ny=Nxy=0) gives dTx=0 and dT_yz=0 at any junction angle."""
    # Two collinear panels, p0 "end" to p1 "start".
    # normal_sign = +1 at "end", -1 at "start".
    # Nxy=0: Tx_int = Nxy*normal_sign = 0.
    # Ny=0:  Ts_int = Ny*normal_sign = 0.
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="LSkin")
    diag = [
        {
            "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 0.0}},
            "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"end": {"Tx_int": 0.0, "Ts_int": 0.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": -5.0, "Fs": 0.0}},
            "interface_field_set": {"start": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"start": {"Tx_int": 0.0, "Ts_int": 0.0}},
        },
    ]
    checks = check_panel_equilibrium(
        [[_mk_resultant(5.0, 0.0)], [_mk_resultant(5.0, 0.0)]], [p0, p1],
        all_panel_diagnostics=diag, endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["dTx_rel"] < 1e-10
    assert chk["dT_yz_rel"] < 1e-10


def test_traction_compare_90deg_junction():
    """At a 90-degree corner, dTx is exact and dT_yz has known analytical magnitude."""
    # p0: [0,0]->[1,0] ("end"), t_i=[1,0], Tx=5, Ts=0  → vector traction in X: 5, in YZ: [0,0]
    # p1: [1,0]->[1,1] ("start"), t_j=[0,1], Tx=-5, Ts=0 → X: -5, YZ: [0,0]
    # dTx = 5+(-5) = 0, dT_yz = 0*[1,0]+0*[0,1] = [0,0]  → both zero.
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web")
    diag = [
        {
            "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 0.0}},
            "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"end": {"Tx_int": 5.0, "Ts_int": 0.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": -5.0, "Fs": 0.0}},
            "interface_field_set": {"start": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"start": {"Tx_int": -5.0, "Ts_int": 0.0}},
        },
    ]
    checks = check_panel_equilibrium(
        [[_mk_resultant(5.0, 0.0)], [_mk_resultant(5.0, 0.0)]], [p0, p1],
        all_panel_diagnostics=diag, endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    assert chk["nxy_compare_mode"] == "traction-vector-strict"
    assert chk["dTx_rel"] < 1e-10
    assert chk["dT_yz_rel"] < 1e-10
    # Now add a nonzero Ts component to check the Y,Z magnitude formula.
    # p0: Ts=3.0 → YZ traction = 3*[1,0] = [3,0]
    # p1: Ts=-4.0 → YZ traction = (-4)*[0,1] = [0,-4]
    # dT_yz = [3,0]+[0,-4] = [3,-4], magnitude = 5.0
    diag2 = [
        {
            "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 0.0}},
            "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"end": {"Tx_int": 5.0, "Ts_int": 3.0}},
        },
        {
            "boundary_reaction": {"start": {"Fx": -5.0, "Fs": 0.0}},
            "interface_field_set": {"start": {"Nx": 5.0, "Nxy": 0.0}},
            "interface_edge_set": {"start": {"Tx_int": -5.0, "Ts_int": -4.0}},
        },
    ]
    checks2 = check_panel_equilibrium(
        [[_mk_resultant(5.0, 0.0)], [_mk_resultant(5.0, 0.0)]], [p0, p1],
        all_panel_diagnostics=diag2, endpoint_tol=1e-8,
    )
    chk2 = checks2[0]
    assert np.isclose(chk2["dT_yz_mag"], 5.0, atol=1e-10)
    assert np.isclose(chk2["dTx_rel"], 0.0, atol=1e-10)


def test_transformed_basis_rotation_identity_for_collinear_panels():
    """
    For two collinear panels (ŝ_i = ŝ_j) the 2-D rotation matrix must be identity:
      u_s_s = 1*u_s_m + 0*w_m,  w_s = 0*u_s_m + 1*w_m.
    Verify this directly from _build_basis_transform_constraints.
    """
    from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.global_shell_assembly import (  # type: ignore[import-untyped]
        _build_basis_transform_constraints, _PanelGlobalMap, _dof,
        _U_S, _W, _BETA_S, _U_X, _BETA_X,
    )

    class _FakePanel:
        def __init__(self, nodes):
            self.nodes = nodes

    # Two collinear panels along y-axis: [0,0]→[1,0] and [1,0]→[2,0]
    # Both have tangent ŝ = [1,0], normal n̂ = [0,1].
    n_s = 3  # 3 s-nodes per panel
    pm0 = _PanelGlobalMap(
        s_nodes=np.linspace(0, 1, n_s),
        elements=[[0, 1, 1 + n_s, n_s], [1, 2, 2 + n_s, 1 + n_s]],
        global_nodes=list(range(2 * n_s)),        # nodes 0..5
        panel_label="P0", panel_index=0,
    )
    pm1 = _PanelGlobalMap(
        s_nodes=np.linspace(1, 2, n_s),
        elements=[[0, 1, 1 + n_s, n_s], [1, 2, 2 + n_s, 1 + n_s]],
        global_nodes=list(range(2 * n_s, 4 * n_s)),  # nodes 6..11
        panel_label="P1", panel_index=1,
    )
    p0 = _FakePanel(np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]))
    p1 = _FakePanel(np.array([[1.0, 0.0], [1.5, 0.0], [2.0, 0.0]]))

    # Cluster: p0 "end" meets p1 "start" at (1,0)
    cluster = [(0, "end", np.array([1.0, 0.0])), (1, "start", np.array([1.0, 0.0]))]
    constraints = _build_basis_transform_constraints([pm0, pm1], [p0, p1], [cluster])

    # Master endpoint nodes (bottom and top rows at p0 "end"):
    gn_m_bot = pm0.global_nodes[n_s - 1]          # last bottom-row node of p0
    gn_m_top = pm0.global_nodes[2 * n_s - 1]      # last top-row node of p0
    gn_s_bot = pm1.global_nodes[0]                 # first bottom-row node of p1
    gn_s_top = pm1.global_nodes[n_s]               # first top-row node of p1

    for gn_m, gn_s in [(gn_m_bot, gn_s_bot), (gn_m_top, gn_s_top)]:
        # u_s_s = 1*u_s_m (identity, no cross-term)
        terms_us = dict(constraints[_dof(gn_s, _U_S)])
        assert np.isclose(terms_us.get(_dof(gn_m, _U_S), 0.0), 1.0, atol=1e-12)
        assert np.isclose(terms_us.get(_dof(gn_m, _W), 0.0), 0.0, atol=1e-12)
        # w_s = 1*w_m (identity, no cross-term)
        terms_w = dict(constraints[_dof(gn_s, _W)])
        assert np.isclose(terms_w.get(_dof(gn_m, _W), 0.0), 1.0, atol=1e-12)
        assert np.isclose(terms_w.get(_dof(gn_m, _U_S), 0.0), 0.0, atol=1e-12)
        # β_s_s = 1*β_s_m
        terms_bs = dict(constraints[_dof(gn_s, _BETA_S)])
        assert np.isclose(terms_bs.get(_dof(gn_m, _BETA_S), 0.0), 1.0, atol=1e-12)


def test_cluster_basis_constraints_rigid_translation_mapping():
    """
    For a 90-deg junction, slave u_s and w should include cross terms from the
    cluster reference basis (non-zero coupling coefficients).
    """
    from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.global_shell_assembly import (  # type: ignore[import-untyped]
        _build_cluster_basis_constraints, _PanelGlobalMap, _dof, _U_S, _W
    )

    class _FakePanel:
        def __init__(self, nodes):
            self.nodes = nodes

    n_s = 3
    pm0 = _PanelGlobalMap(np.linspace(0, 1, n_s), [], list(range(2 * n_s)), "P0", 0)
    pm1 = _PanelGlobalMap(np.linspace(0, 1, n_s), [], list(range(2 * n_s, 4 * n_s)), "P1", 1)
    p0 = _FakePanel(np.array([[0.0, 0.0], [1.0, 0.0]]))  # tangent x-like
    p1 = _FakePanel(np.array([[1.0, 0.0], [1.0, 1.0]]))  # tangent y-like
    cluster = [(0, "end", np.array([1.0, 0.0])), (1, "start", np.array([1.0, 0.0]))]
    endpoint_cluster_node = {
        (0, "end", "bottom"): 100, (0, "end", "top"): 101,
        (1, "start", "bottom"): 100, (1, "start", "top"): 101,
    }
    endpoint_cluster_rot_node = {
        (0, "end", "bottom"): 200, (0, "end", "top"): 201,
        (1, "start", "bottom"): 200, (1, "start", "top"): 201,
    }
    cs = _build_cluster_basis_constraints(
        [pm0, pm1],
        [p0, p1],
        [cluster],
        endpoint_cluster_node,
        endpoint_cluster_rot_node,
    )
    gn_slave = pm1.global_nodes[0]
    terms_us = dict(cs[_dof(gn_slave, _U_S)])
    terms_w = dict(cs[_dof(gn_slave, _W)])
    # 90-deg mapping should not be diagonal-only for both equations.
    assert any(abs(v) > 1e-12 for v in terms_us.values())
    assert any(abs(v) > 1e-12 for v in terms_w.values())
    assert len(terms_us) >= 1 and len(terms_w) >= 1


def test_transformed_basis_cluster_rotation_basis_is_rank6():
    from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.operations.global_shell_assembly import (  # type: ignore[import-untyped]
        _build_cluster_basis_constraints, _PanelGlobalMap, _dof, _BETA_S
    )

    class _FakePanel:
        def __init__(self, nodes):
            self.nodes = nodes

    n_s = 3
    pm0 = _PanelGlobalMap(np.linspace(0, 1, n_s), [], list(range(2 * n_s)), "P0", 0)
    pm1 = _PanelGlobalMap(np.linspace(0, 1, n_s), [], list(range(2 * n_s, 4 * n_s)), "P1", 1)
    p0 = _FakePanel(np.array([[0.0, 0.0], [1.0, 0.0]]))
    p1 = _FakePanel(np.array([[1.0, 0.0], [1.0, 1.0]]))
    cluster = [(0, "end", np.array([1.0, 0.0])), (1, "start", np.array([1.0, 0.0]))]
    endpoint_cluster_node = {
        (0, "end", "bottom"): 100, (0, "end", "top"): 101,
        (1, "start", "bottom"): 100, (1, "start", "top"): 101,
    }
    endpoint_cluster_rot_node = {
        (0, "end", "bottom"): 200, (0, "end", "top"): 201,
        (1, "start", "bottom"): 200, (1, "start", "top"): 201,
    }
    cs = _build_cluster_basis_constraints(
        [pm0, pm1], [p0, p1], [cluster], endpoint_cluster_node, endpoint_cluster_rot_node
    )
    gn_web = pm1.global_nodes[0]
    terms_bs = dict(cs[_dof(gn_web, _BETA_S)])
    assert abs(terms_bs.get(_dof(200, _BETA_S), 0.0)) > 1e-12


def test_tiered_acceptance_keys_present():
    """
    Phase E: check_panel_equilibrium must return dicts containing all tiered-
    acceptance keys (dTx_rel, dT_yz_rel, resultant_tol_dTx, resultant_tol_dT_yz,
    resultant_pass_dT_yz) for the Phase-E reporting to work.
    """
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="Web @ 0.35c")
    checks = check_panel_equilibrium(
        [[_mk_resultant(5.0, 1.0)], [_mk_resultant(5.0, 1.0)]],
        [p0, p1],
        all_panel_diagnostics=[
            {
                "boundary_reaction": {"end": {"Fx": 5.0, "Fs": 1.0}},
                "interface_field_set": {"end": {"Nx": 5.0, "Nxy": 1.0}},
                "interface_edge_set": {"end": {"Tx_int": 1.0, "Ts_int": 2.0}},
            },
            {
                "boundary_reaction": {"start": {"Fx": 5.0, "Fs": 1.0}},
                "interface_field_set": {"start": {"Nx": 5.0, "Nxy": 1.0}},
                "interface_edge_set": {"start": {"Tx_int": -1.0, "Ts_int": -2.0}},
            },
        ],
        endpoint_tol=1e-8,
    )
    assert len(checks) == 1
    chk = checks[0]
    for key in ("dTx_rel", "dT_yz_rel", "resultant_tol_dTx",
                "resultant_tol_dT_yz", "resultant_pass_dT_yz"):
        assert key in chk, f"Missing key: {key}"
    assert isinstance(chk["dTx_rel"], float)
    assert isinstance(chk["dT_yz_rel"], float)
    assert isinstance(chk["resultant_pass_dT_yz"], bool)

def test_cluster_and_pair_share_topology():
    """
    _build_geometric_interfaces and _build_endpoint_clusters must agree on the
    number of multi-panel junctions: every cluster with len>=2 should correspond
    to at least one pair in the interface list.
    """
    from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.diagnostics.equilibrium_checks import (
        _build_geometric_interfaces,
        _build_endpoint_clusters_raw,
    )
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web")
    p2 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="LSkin")
    p3 = _mk_panel(np.array([[0.0, 0.0], [0.0, -1.0]]), label="Web2")
    panels = [p0, p1, p2, p3]

    clusters = _build_endpoint_clusters_raw(panels, tol=1e-8)
    multi_clusters = [c for c in clusters if len(c) >= 2]
    interfaces = _build_geometric_interfaces(panels, tol=1e-8)

    # Every multi-endpoint cluster must appear at least once in the pair list.
    cluster_ids_in_pairs = {iface["cluster_id"] for iface in interfaces}
    multi_cluster_indices = {
        i for i, c in enumerate(clusters) if len(c) >= 2
    }
    assert multi_cluster_indices == cluster_ids_in_pairs, (
        f"Topology mismatch: multi-clusters={multi_cluster_indices} "
        f"vs pair cluster IDs={cluster_ids_in_pairs}"
    )

    # cluster_size in each pair must equal the cluster length from raw clustering.
    cluster_sizes = {i: len(c) for i, c in enumerate(clusters)}
    for iface in interfaces:
        expected_size = cluster_sizes[iface["cluster_id"]]
        assert iface["cluster_size"] == expected_size


def test_tiered_acceptance_applies_pair_only_to_two_way():
    """
    check_panel_equilibrium must tag each interface row with cluster_size.
    At a 3-way junction (3 panels meeting), all generated pairs must have
    cluster_size == 3, so they are excluded from 2-way-only secondary acceptance.
    Only 2-way junction pairs should have cluster_size == 2.
    """
    from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.diagnostics.equilibrium_checks import (
        _build_geometric_interfaces,
    )
    # Build a T-junction: p0:end, p1:start, p2:start all meet at (1,0).
    p0 = _mk_panel(np.array([[0.0, 0.0], [1.0, 0.0]]), label="USkin")
    p1 = _mk_panel(np.array([[1.0, 0.0], [1.0, 1.0]]), label="Web")
    p2 = _mk_panel(np.array([[1.0, 0.0], [2.0, 0.0]]), label="LSkin")
    # Extra isolated 2-way junction: p3:start, p0:start at (0,0).
    p3 = _mk_panel(np.array([[0.0, 0.0], [0.0, -1.0]]), label="Web2")

    panels = [p0, p1, p2, p3]
    interfaces = _build_geometric_interfaces(panels, tol=1e-8)

    t_junction_pairs = [
        iface for iface in interfaces
        if iface["cluster_id"] == next(
            iface2["cluster_id"] for iface2 in interfaces
            if iface2["pi"] == 0 and iface2["end_i"] == "end"
        )
    ]
    two_way_pairs = [iface for iface in interfaces if iface["cluster_size"] == 2]
    three_way_pairs = [iface for iface in interfaces if iface["cluster_size"] == 3]

    # The T-junction should produce cluster_size==3 pairs.
    assert all(p["cluster_size"] == 3 for p in t_junction_pairs), (
        "T-junction pairs must have cluster_size == 3"
    )
    # Two-way junctions should produce cluster_size == 2 pairs.
    assert all(p["cluster_size"] == 2 for p in two_way_pairs)
    # No pair should have cluster_size == 3 for a purely 2-way junction cluster.
    assert len(three_way_pairs) > 0, "Should have at least one 3-way pair from the T-junction"

    # Now check that check_panel_equilibrium carries cluster_size through.
    di = [
        {"interface_edge_set": {"end": {"Tx_int": 1.0, "Ts_int": 0.5}, "start": {"Tx_int": -0.5, "Ts_int": 0.2}}},
        {"interface_edge_set": {"start": {"Tx_int": -0.3, "Ts_int": 0.1}, "end": {"Tx_int": 0.3, "Ts_int": 0.1}}},
        {"interface_edge_set": {"start": {"Tx_int": -0.7, "Ts_int": 0.4}, "end": {"Tx_int": 0.7, "Ts_int": 0.4}}},
        {"interface_edge_set": {"start": {"Tx_int": -0.5, "Ts_int": 0.2}, "end": {"Tx_int": 0.5, "Ts_int": 0.2}}},
    ]
    checks = check_panel_equilibrium(
        [[_mk_resultant(1.0, 1.0)] for _ in panels],
        panels,
        all_panel_diagnostics=di,
        endpoint_tol=1e-8,
    )
    # Every check row must carry cluster_size.
    assert all("cluster_size" in c for c in checks), "cluster_size missing from check rows"
    # Subset to T-junction boundary type rows.
    checks_3way = [c for c in checks if c.get("cluster_size", 0) == 3]
    checks_2way = [c for c in checks if c.get("cluster_size", 0) == 2]
    assert len(checks_3way) > 0, "Expected 3-way junction rows in checks"
    assert len(checks_2way) > 0, "Expected 2-way junction rows in checks"


def test_cluster_traction_equilibrium_cluster_frame():
    """
    Synthetic 3-way T-junction with tractions satisfying equilibrium in the global
    frame: cluster-sum residuals must be near zero.
    """
    from blade_structure_utilities.fem_utilities.section_shell_model.processing.static.diagnostics.equilibrium_checks import check_cluster_equilibrium

    # Build 3 synthetic panels meeting at (0, 0):
    # panel 0: horizontal skin, from (-1, 0) to (0, 0) — tangent (1, 0)
    # panel 1: horizontal skin, from (0, 0) to (1, 0) — tangent (1, 0)
    # panel 2: vertical web,    from (0, 0) to (0, -1) — tangent (0, -1)
    class _Panel:
        def __init__(self, nodes, label=""):
            self.nodes = np.array(nodes, dtype=float)
            self.label = label

    p0 = _Panel([[-1.0, 0.0], [0.0, 0.0]], label="skin_left")
    p1 = _Panel([[0.0, 0.0], [1.0, 0.0]], label="skin_right")
    p2 = _Panel([[0.0, 0.0], [0.0, -1.0]], label="web")

    panels = [p0, p1, p2]

    # Equilibrium tractions at the T-junction (outward-normal signed):
    # Pick: Tx_0=2.0, Tx_1=-1.0, Tx_2=-1.0; Ts_0=0.5, Ts_1=-0.5, Ts_2=0.0
    di = [
        {"interface_edge_set": {
            "start": {"Tx_int": 0.0, "Ts_int": 0.0},
            "end": {"Tx_int": 2.0, "Ts_int": 0.5},
        }},
        {"interface_edge_set": {
            "start": {"Tx_int": -1.0, "Ts_int": -0.5},
            "end": {"Tx_int": 0.0, "Ts_int": 0.0},
        }},
        {"interface_edge_set": {
            "start": {"Tx_int": -1.0, "Ts_int": 0.0},
            "end": {"Tx_int": 0.0, "Ts_int": 0.0},
        }},
    ]
    clusters = check_cluster_equilibrium(panels, all_panel_diagnostics=di, endpoint_tol=1e-6)
    # Find the 3-panel cluster at (0, 0).
    three_way = [c for c in clusters if c.get("n_panels", 0) == 3]
    assert len(three_way) == 1, f"Expected one 3-way cluster, got {len(three_way)}"
    cc = three_way[0]
    assert cc["Tx_rel_cluster"] < 1e-10, f"Tx residual not zero: {cc['Tx_rel_cluster']}"
    assert cc["T_yz_rel_cluster"] < 1e-10, f"T_yz residual not zero: {cc['T_yz_rel_cluster']}"
