"""Cap-to-skin rigid-link MPCs (BL-003 strategy S2).

The load-bearing property of a rigid link is that it reproduces rigid-body
motion *exactly*: if the whole section translates and rotates as a solid body,
every tie must be satisfied identically, otherwise the link injects spurious
strain energy and the cap is welded to the skin with a stiffness that has no
physical origin. That is what these tests check, over both a zero offset (where
the link degenerates to shared-node compatibility) and the real, nonzero cap
offsets produced by the meshers.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def assembled_5df():
    """Assemble a cap-bearing section; returns the SectionAssemblyResult."""
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import (
        build_section_view,
        resolve_system_type,
    )
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import AirfoilSDF
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh import (
        build_section_strip_mesh,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
        shell_panel_topology_from_shell_inputs,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.section_assembly import (
        assemble_section_from_zones,
    )

    twist = float(np.deg2rad(34.1))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=0.241, closed_te=True
    )
    section = build_section_view(
        airfoil,
        resolve_system_type("5D-F"),
        twist_angle_rad=twist,
        skin_thickness=0.001,
        web_thickness=0.0015,
        cap_height=0.004,
    )
    shell = shell_mesh_inputs_from_section(
        section, twist_rad=twist, layout_key="5D-F", n_web_samples=20, n_cap_samples=80
    )
    mesh = build_section_strip_mesh(shell, target_element_length_frac=0.012, endpoint_tol=1e-6)
    zones = list(shell_panel_topology_from_shell_inputs(shell, allow_demo_bind=True).zones)
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
    nxy = [np.zeros_like(v) for v in nx]
    return assemble_section_from_zones(
        zones,
        nx,
        nxy,
        L_x=1.0,
        interface_constraint_mode="shared",
        section_strip_mesh=mesh,
    )


def _rigid_body_at(point_yz, tangent_yz, ndof, *, a, omega):
    """Rigid-body DOF values at one point, expressed in the frame ``tangent_yz``.

    ``a`` is the translation ``(a_x, a_y, a_z)`` and ``omega`` the rotation
    vector ``(w_x, w_y, w_z)`` about the section origin. Local DOFs are
    ``[u_x, u_s, w, beta_s, beta_x, beta_z]`` with ``u_s`` along the tangent
    ``t`` and ``w`` along its normal ``n``.
    """
    ax, ay, az = a
    wx, wy, wz = omega
    y, z = float(point_yz[0]), float(point_yz[1])
    tv = np.array([float(tangent_yz[0]), float(tangent_yz[1])])
    nv = np.array([-tv[1], tv[0]])
    # u = a + omega x r, with r = (0, y, z)
    u_ip = np.array([ay - wx * z, az + wx * y])
    out = np.zeros(ndof)
    out[0] = ax + (wy * z - wz * y)
    out[1] = float(np.dot(u_ip, tv))
    out[2] = float(np.dot(u_ip, nv))
    out[3] = float(wy * tv[0] + wz * tv[1])  # omega . t
    out[4] = wx
    if ndof >= 6:
        out[5] = float(wy * nv[0] + wz * nv[1])  # omega . n
    return out


def _rigid_body_field(node_meta, ndof, *, a, omega):
    """Exact rigid-body DOF vector over every global node, in each node's frame."""
    q = np.zeros(node_meta.n_nodes * ndof)
    for gn in range(node_meta.n_nodes):
        p = node_meta.yz[gn]
        t = node_meta.tangent_yz[gn]
        if not (np.all(np.isfinite(p)) and np.all(np.isfinite(t))):
            continue
        q[gn * ndof : (gn + 1) * ndof] = _rigid_body_at(p, t, ndof, a=a, omega=omega)
    return q


def _residuals(constraints, q):
    """Max |q_slave - sum(coeff * q_master)| over all constraint rows."""
    worst = 0.0
    for slave, terms in constraints.items():
        lhs = q[slave]
        rhs = sum(c * q[m] for m, c in terms)
        worst = max(worst, abs(lhs - rhs))
    return worst


def test_ties_are_found_for_every_cap(assembled_5df):
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
        build_cap_skin_ties,
    )

    asm = assembled_5df
    report = build_cap_skin_ties(asm.panel_maps, asm.node_meta)

    assert report.caps_seen > 0, "fixture has no cap panels"
    assert report.caps_tied == report.caps_seen, (
        f"only {report.caps_tied}/{report.caps_seen} caps tied; {report.notes}"
    )
    assert report.n_links > 0
    # Both layers must be tied, or the cap is only attached on one face.
    layers = {lk.layer for lk in report.links}
    assert layers == {"bottom", "top"}, f"tied layers {layers}"


def test_offsets_are_nonzero_and_bounded(assembled_5df):
    """Links must carry the real cap standoff, not collapse onto the skin."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
        build_cap_skin_ties,
    )

    asm = assembled_5df
    report = build_cap_skin_ties(asm.panel_maps, asm.node_meta)
    lo, med, hi = report.offset_stats_mm()

    assert med > 0.1, f"median cap offset {med:.4f} mm — links have collapsed onto the skin"
    # Nothing should be tied across the section; a cap standoff is millimetres.
    assert hi < 50.0, f"max cap offset {hi:.3f} mm is implausibly large"


def test_rigid_body_motion_passes_exactly(assembled_5df):
    """The defining property: a solid-body field satisfies every tie identically."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
        build_cap_tie_constraints,
    )

    asm = assembled_5df
    ndof = int(asm.cache.n_dof_node)
    mpcs, report = build_cap_tie_constraints(
        asm.panel_maps, asm.node_meta, ndof=ndof, mode="skin_offset"
    )
    assert mpcs, "no cap-tie constraints were produced"

    scale = float(np.nanmax(np.abs(asm.node_meta.yz)))
    for a, omega in (
        ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),      # pure spanwise translation
        ((0.0, 1.0, -0.5), (0.0, 0.0, 0.0)),     # pure in-plane translation
        ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0)),      # twist about the beam axis
        ((0.0, 0.0, 0.0), (0.0, 0.2, -0.4)),     # bending rotations
        ((0.7, -0.2, 0.4), (0.15, -0.3, 0.25)),  # general rigid-body motion
    ):
        q = _rigid_body_field(asm.node_meta, ndof, a=a, omega=omega)
        res = _residuals(mpcs, q)
        tol = 1e-9 * max(1.0, scale * float(np.linalg.norm(omega)) + float(np.linalg.norm(a)))
        assert res < tol, (
            f"rigid-body motion a={a}, omega={omega} leaves residual {res:.3e} "
            f"(tol {tol:.3e}) — the link algebra is wrong"
        )


def test_zero_offset_link_reduces_to_frame_rotation(assembled_5df):
    """With r=0 the link must become plain shared-node compatibility."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
        CapTieLink,
        CapTieReport,
        cap_tie_constraints,
    )

    asm = assembled_5df
    ndof = int(asm.cache.n_dof_node)

    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
        build_cap_skin_ties,
    )

    found = build_cap_skin_ties(asm.panel_maps, asm.node_meta)
    src = found.links[0]
    zeroed = CapTieReport(
        mode="skin_offset",
        links=[
            CapTieLink(
                cap_label=src.cap_label,
                slave_node=src.slave_node,
                skin_label=src.skin_label,
                master_nodes=src.master_nodes,
                weights=src.weights,
                offset_yz=(0.0, 0.0),
                layer=src.layer,
            )
        ],
        caps_seen=1,
        caps_tied=1,
    )
    mpcs = cap_tie_constraints(zeroed, asm.node_meta, ndof=ndof)

    # r=0 asserts the cap node moves exactly as the skin point does, so the
    # rigid-body field must be sampled at the *master* point (in the cap's
    # frame) rather than at the cap node's real, offset position.
    a, omega = (0.4, -0.3, 0.2), (0.1, 0.2, -0.15)
    q = _rigid_body_field(asm.node_meta, ndof, a=a, omega=omega)
    p_master = sum(
        w * asm.node_meta.yz[gm] for gm, w in zip(src.master_nodes, src.weights)
    )
    q[src.slave_node * ndof : (src.slave_node + 1) * ndof] = _rigid_body_at(
        p_master, asm.node_meta.tangent_yz[src.slave_node], ndof, a=a, omega=omega
    )
    assert _residuals(mpcs, q) < 1e-9

    # And with the moment arm gone, u_x must lose its rotation coupling.

    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
        U_X,
        gdof,
    )

    ux_terms = mpcs[gdof(src.slave_node, U_X, ndof)]
    assert all(
        abs(c) < 1e-12 or m % ndof == U_X for m, c in ux_terms
    ), "zero-offset link still couples u_x to rotations"


def test_expands_master_that_is_already_a_slave(assembled_5df):
    """Skin masters that are junction slaves expand through existing MPCs."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
        build_cap_skin_ties,
        cap_tie_constraints,
    )
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.connected_assembly_core import (
        U_X,
        gdof,
    )

    asm = assembled_5df
    ndof = int(asm.cache.n_dof_node)
    report = build_cap_skin_ties(asm.panel_maps, asm.node_meta)
    victim = report.links[0].master_nodes[0] * ndof  # u_x of a skin master
    free_master = victim + 1000  # synthetic independent DOF
    mpcs = cap_tie_constraints(
        report,
        asm.node_meta,
        ndof=ndof,
        existing={victim: [(free_master, 1.0)]},
    )
    slave_ux = gdof(report.links[0].slave_node, U_X, ndof)
    masters = {m for m, _c in mpcs[slave_ux]}
    assert victim not in masters
    assert free_master in masters


def test_skips_cap_slave_already_owned_by_junction(assembled_5df):
    """Cap nodes already constrained by junction MPCs are skipped, not raised."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.cap_tie_constraints import (
        build_cap_skin_ties,
        cap_tie_constraints,
    )

    asm = assembled_5df
    ndof = int(asm.cache.n_dof_node)
    report = build_cap_skin_ties(asm.panel_maps, asm.node_meta)
    cap_ux = report.links[0].slave_node * ndof
    mpcs = cap_tie_constraints(
        report, asm.node_meta, ndof=ndof, existing={cap_ux: [(0, 1.0)]}
    )
    assert cap_ux not in mpcs
    assert any("skipped" in n for n in report.notes)
