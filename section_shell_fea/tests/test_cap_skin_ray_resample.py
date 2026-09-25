"""
Tests for cap interior resampling via inward rays from the skin master ring.
"""

from __future__ import annotations

import numpy as np
import pytest

_MIN_SEG = 1e-12


@pytest.fixture(scope="module")
def airfoil():
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import AirfoilSDF

    return AirfoilSDF.from_naca("0012", n_points=100, chord=1.0)


def _cap_endpoints_after_split_only(airfoil, layout_key: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.geometry_utilities.section_geometry.midlines.shell_midline_export import build_shell_midline_strips
    from blade_structure_utilities.geometry_utilities.section_geometry.topology.shell_strip_topology import (
        _split_skin_at_junctions,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import (
        _validate_strips,
    )

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    strips = build_shell_midline_strips(
        section, twist_rad=0.0, n_web_samples=20, n_cap_samples=40
    )
    _validate_strips(strips)
    split = _split_skin_at_junctions(strips)
    _validate_strips(split)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m in split:
        if m.kind == "cap":
            arr = np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float)
            out[m.label] = (arr[0].copy(), arr[-1].copy())
    return out


def _cap_endpoints_before_split(airfoil, layout_key: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.geometry_utilities.section_geometry.midlines.shell_midline_export import build_shell_midline_strips

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    strips = build_shell_midline_strips(
        section, twist_rad=0.0, n_web_samples=20, n_cap_samples=40
    )
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for m in strips:
        if m.kind == "cap":
            arr = np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float)
            out[m.label] = (arr[0].copy(), arr[-1].copy())
    return out


@pytest.mark.parametrize("layout_key", ["2D-F", "2D-CN"])
def test_split_skin_moves_only_nonweb_cap_endpoints_to_skin(layout_key, airfoil):
    """Class A/B: non-web cap ends snap to skin; web-coupled ends may differ."""
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.geometry_utilities.section_geometry.midlines.shell_midline_export import build_shell_midline_strips
    from blade_structure_utilities.geometry_utilities.section_geometry.topology.shell_strip_topology import (
        _point_to_open_polyline_nearest,
        _split_skin_at_junctions,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import (
        _validate_strips,
    )

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    strips = build_shell_midline_strips(
        section, twist_rad=0.0, n_web_samples=20, n_cap_samples=40
    )
    _validate_strips(strips)
    split = _split_skin_at_junctions(strips)
    _validate_strips(split)

    web_polys = [
        np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float)
        for m in split
        if m.kind == "web" and np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float).shape[0] >= 2
    ]
    skin_pts = np.vstack([np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float) for m in split if m.kind == "skin"])
    before = {
        m.label: np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float)
        for m in strips
        if m.kind == "cap"
    }
    after = {
        m.label: np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float)
        for m in split
        if m.kind == "cap"
    }

    for lab, arr in after.items():
        b = before[lab]
        for row in (0, -1):
            p = np.asarray(b[row], dtype=float).reshape(2)
            d_web = min(_point_to_open_polyline_nearest(p, w)[0] for w in web_polys) if web_polys else float("inf")
            q = np.asarray(arr[row], dtype=float).reshape(2)
            if d_web > 5e-4:
                # Class A: non-web cap end should coincide with a skin node.
                d_skin = float(np.linalg.norm(skin_pts - q.reshape(1, 2), axis=1).min())
                assert d_skin <= 1e-9, (layout_key, lab, row, d_skin)
            else:
                # Class B: web-coupled end can be unchanged from exported cap.
                assert np.linalg.norm(q - p) <= 1e-3, (layout_key, lab, row)


@pytest.mark.parametrize("layout_key", ["2D-F", "2D-CN"])
def test_cap_endpoints_unchanged_after_skin_ray_resample(airfoil, layout_key):
    """Resampling must not move cap strip endpoints (skin–web junction snap)."""
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )

    ends_split = _cap_endpoints_after_split_only(airfoil, layout_key)

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    shell = shell_mesh_inputs_from_section(
        section,
        twist_rad=0.0,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=40,
    )

    for m in shell.midlines:
        if m.kind != "cap":
            continue
        arr = np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float)
        e0, e1 = ends_split[m.label]
        # x3 normalisation may reverse the strip; endpoint *set* must match.
        got = {tuple(np.round(arr[0], 12)), tuple(np.round(arr[-1], 12))}
        exp = {tuple(np.round(e0, 12)), tuple(np.round(e1, 12))}
        assert got == exp, (layout_key, m.label, got, exp)


@pytest.mark.parametrize("layout_key", ["2D-F"])
def test_cap_interior_segments_nondegenerate(airfoil, layout_key):
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    shell = shell_mesh_inputs_from_section(
        section,
        twist_rad=0.0,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=40,
    )
    for m in shell.midlines:
        if m.kind != "cap":
            continue
        pts = np.asarray(m.midline_twisted_pitch_axis_blade_frame, dtype=float)
        if pts.shape[0] < 2:
            continue
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        assert np.all(d >= _MIN_SEG), (layout_key, m.label, float(d.min()))


def test_cap_resample_disabled_preserves_midline_count(airfoil):
    """With resampling off, every cap keeps its split-only geometry.

    The cap-to-skin bond pass (section_geometry SG-017) may add correspondence
    vertices, but only *on* the existing polyline: the split-only vertices must
    all survive, in order, and every added vertex must lie on the original cap.
    Counting vertices would conflate the two.
    """
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.geometry_utilities.section_geometry.midlines.shell_midline_export import build_shell_midline_strips
    from blade_structure_utilities.geometry_utilities.section_geometry.topology.shell_strip_topology import (
        _split_skin_at_junctions,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import (
        _validate_strips,
    )
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )

    layout_key = "2D-F"
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    strips = build_shell_midline_strips(
        section, twist_rad=0.0, n_web_samples=20, n_cap_samples=40
    )
    _validate_strips(strips)
    split = _split_skin_at_junctions(strips)
    split_caps = {m.label: np.asarray(m.midline_twisted_pitch_axis_blade_frame) for m in split if m.kind == "cap"}

    shell = shell_mesh_inputs_from_section(
        section,
        twist_rad=0.0,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=40,
        cap_resample_from_skin_rays=False,
    )
    for m in shell.midlines:
        if m.kind != "cap":
            continue
        original = split_caps[m.label]
        final = np.asarray(m.midline_twisted_pitch_axis_blade_frame)
        if np.hypot(*(final[0] - original[0])) > np.hypot(*(final[-1] - original[0])):
            final = final[::-1]  # the mesh handoff may reverse a strip for +x3
        matched = [int(np.argmin(np.hypot(*(final - point).T))) for point in original]
        assert all(np.hypot(*(final[i] - point)) < 1e-12 for i, point in zip(matched, original))
        assert matched == sorted(matched), "split-only cap vertices must survive in order"
        segments = list(zip(original[:-1], original[1:]))
        for index, point in enumerate(final):
            if index in matched:
                continue
            distance = min(
                np.hypot(*(point - (a + np.clip(np.dot(point - a, b - a) / np.dot(b - a, b - a), 0, 1) * (b - a))))
                for a, b in segments
            )
            assert distance < 1e-12, "an added vertex must lie on the original cap polyline"
