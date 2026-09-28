"""WP4: bind placement identity, x3 face map, loads about midsurface, FE face x3."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import (
    ShellMidlineStrip,
)
from blade_structure_utilities.geometry_utilities.section_geometry.placement import (
    anchored_wall_geometry_from_policy,
)
from blade_structure_utilities.material_utilities.layup import (
    SKIN_FROM_OUTER_MOULD_LINE,
    SPAR_CAP_FROM_SKIN_BOND_LINE,
    WEB_FROM_MIDSURFACE,
)
from blade_structure_utilities.material_utilities.section_bind.bind.gauge_match import (
    assert_strip_matches_placement,
)
from blade_structure_utilities.mesh_utilities.section_mesh.handoff.panel_x3_orientation import (
    mid_segment_frame,
    normal_2d,
)


def test_x3_normal_is_ccw_of_tangent():
    t = np.array([1.0, 0.0])
    n = normal_2d(t)
    assert n[0] == pytest.approx(0.0)
    assert n[1] == pytest.approx(1.0)


def test_mid_segment_frame_on_horizontal_strip():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    m, n = mid_segment_frame(nodes)
    # Central open segment is nodes[1]–nodes[2] for N=3.
    assert m[0] == pytest.approx(1.5)
    assert n[1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "policy, expected_ref",
    [
        (SKIN_FROM_OUTER_MOULD_LINE, "outer_mould_line"),
        (SPAR_CAP_FROM_SKIN_BOND_LINE, "skin_bond_line"),
        (WEB_FROM_MIDSURFACE, "midsurface"),
    ],
)
def test_bind_placement_matches_default_wall_policy(policy, expected_ref):
    strip = ShellMidlineStrip(
        label="panel",
        kind="skin",
        midline_twisted_pitch_axis_blade_frame=np.array([[0.0, 0.0], [1.0, 0.0]]),
        thickness_m=0.004,
        placement_policy_id=policy.name,
        placement_digest=policy.digest,
        reference_face=policy.reference_face.value,
    )
    assert_strip_matches_placement(
        strip,
        expected_policy_id=policy.name,
        expected_digest=policy.digest,
        expected_reference_face=expected_ref,
    )


def test_loads_integrate_about_midline_samples():
    """Channel C SoT: contour force densities live on the shell midline."""
    from blade_structure_utilities.load_utilities.section_loads.types.dto import (
        ContourForceSample,
    )
    from blade_structure_utilities.load_utilities.section_loads.equipollence import (
        integrate_contour_forces,
    )

    samples = [
        ContourForceSample(
            label="a",
            kind="skin",
            y=0.0,
            z=0.0,
            t=0.004,
            f_x=1.0,
            f_s=0.0,
            f_w=0.0,
            ds=0.5,
        ),
        ContourForceSample(
            label="b",
            kind="skin",
            y=1.0,
            z=0.0,
            t=0.004,
            f_x=1.0,
            f_s=0.0,
            f_w=0.0,
            ds=0.5,
        ),
    ]
    result = integrate_contour_forces(samples, about_xy=(0.0, 0.0))
    assert float(result.N) == pytest.approx(1.0)
    assert "midline" in (integrate_contour_forces.__doc__ or "").lower()


def test_ply_face_x3_from_anchored_half_thickness():
    """ABD about laminate mid; physical faces at ±t/2 from wall mid (default skin)."""
    wall = anchored_wall_geometry_from_policy(SKIN_FROM_OUTER_MOULD_LINE, 0.004)
    # Skin: reference at OML (0), mid at t/2, cavity free face at t.
    # Laminate-local x3 faces relative to mid: reference face at -t/2, free at +t/2.
    half = 0.5 * wall.thickness_m
    x3_ref = wall.reference_offset_m - wall.midsurface_offset_m
    x3_free = wall.free_face_offset_m - wall.midsurface_offset_m
    assert x3_ref == pytest.approx(-half)
    assert x3_free == pytest.approx(+half)


def test_web_faces_symmetric_about_midsurface():
    wall = anchored_wall_geometry_from_policy(WEB_FROM_MIDSURFACE, 0.006)
    assert wall.midsurface_offset_m == 0.0
    assert wall.free_face_offset_m == pytest.approx(0.003)
    assert wall.second_free_face_offset_m == pytest.approx(-0.003)
