"""MITC4 strip nodes include polyline knots (cap/web; sparse skin) — junction snap."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def airfoil():
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import AirfoilSDF

    return AirfoilSDF.from_naca("0012", n_points=100, chord=1.0)


def _target_element_length_demo_m(chord_m: float) -> float:
    return float(max(0.015, 0.012 * chord_m))


def _web_polylines_from_shell(shell: object) -> list[np.ndarray]:
    webs: list[np.ndarray] = []
    for m in shell.midlines:  # type: ignore[attr-defined]
        if m.kind != "web":
            continue
        arr = np.asarray(m.midline_b, dtype=float)
        if arr.shape[0] >= 2:
            webs.append(arr)
    return webs


def _min_dist_to_polylines(q: np.ndarray, polylines: list[np.ndarray]) -> np.ndarray:
    """Per-point minimum distance to any segment of any polyline."""
    best = np.full(q.shape[0], np.inf)
    for w in polylines:
        for j in range(w.shape[0] - 1):
            a = w[j]
            ab = w[j + 1] - a
            l2 = float(np.dot(ab, ab))
            if l2 <= 0.0:
                continue
            t = np.clip(((q - a) @ ab) / l2, 0.0, 1.0)
            proj = a + t[:, None] * ab
            best = np.minimum(best, np.linalg.norm(q - proj, axis=1))
    return best


@pytest.mark.parametrize("layout_key", ["2C-CN", "2D-CN", "3C-CN"])
def test_cap_mesh_nodes_intersect_web_midlines(layout_key: str):
    """Cap mesh carries web-centric shared nodes at cap-web intersections."""
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import AirfoilSDF
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import build_section_strip_mesh
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )

    chord_m = 1.655959896
    twist_rad = float(np.deg2rad(34.110405083))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=chord_m, closed_te=True
    )
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)
    shell = shell_mesh_inputs_from_section(
        section,
        twist_rad=twist_rad,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=80,
    )
    webs = _web_polylines_from_shell(shell)
    assert webs, layout_key

    mesh = build_section_strip_mesh(
        shell,
        target_element_length_m=_target_element_length_demo_m(chord_m),
        endpoint_tol=1e-6,
    )
    cap_yz = np.vstack(
        [np.asarray(pm.yz_nodes, dtype=float) for pm in mesh.panel_meshes if pm.kind == "cap"]
    )
    assert cap_yz.size > 0, layout_key
    best = float("inf")
    for w in webs:
        for j in range(w.shape[0] - 1):
            a = w[j]
            b = w[j + 1]
            ab = b - a
            l2 = float(np.dot(ab, ab)) + 1e-30
            rel = cap_yz - a.reshape(1, 2)
            t = np.clip((rel @ ab.reshape(2, 1)).ravel() / l2, 0.0, 1.0)
            proj = a.reshape(1, 2) + t.reshape(-1, 1) * ab.reshape(1, 2)
            d = float(np.linalg.norm(cap_yz - proj, axis=1).min())
            best = min(best, d)
    assert best < 1e-4, (layout_key, best)


@pytest.mark.parametrize("layout_key", ["2C-CN", "2D-CN", "5D-F"])
def test_mitc4_cap_nodes_pin_web_junctions_without_slivers(layout_key: str):
    """Class C: cap nodes pin cap-web junctions and honour the target length.

    Caps used to keep *every* midline knot. The cap midline is a dense sampling
    of the offset airfoil arc (~200 knots against a ~55-element target), so that
    both over-refined the cap and — wherever a cap-web junction insert landed
    microns from a sampling knot — left sliver elements whose spurious in-plane
    shear dominated the section's Hashin envelope and drove laminate sizing off
    a meshing artefact.

    What must actually hold: strip ends are exact, every cap-web junction keeps
    a shared node, and no element is a sliver. A junction closer to a strip end
    than the sliver floor is merged into that end — creating the sliver to
    preserve it exactly would defeat the point — so it is only required to be
    within the floor.
    """
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import AirfoilSDF
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import build_section_strip_mesh
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )
    from blade_structure_utilities.mesh_utilities.line_1d import LineMeshSpec

    chord_m = 1.655959896
    twist_rad = float(np.deg2rad(34.110405083))
    airfoil = AirfoilSDF.from_naca_series(
        6, 63.0, 4.0, 15.0, n_points=200, chord=chord_m, closed_te=True
    )
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)
    shell = shell_mesh_inputs_from_section(
        section,
        twist_rad=twist_rad,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=80,
    )
    h_target = _target_element_length_demo_m(chord_m)
    mesh = build_section_strip_mesh(
        shell,
        target_element_length_m=h_target,
        endpoint_tol=1e-6,
    )
    cap_mesh = {
        pm.panel_label.split(":", 1)[1]: np.asarray(pm.yz_nodes, dtype=float)
        for pm in mesh.panel_meshes
        if pm.kind == "cap"
    }
    assert cap_mesh
    webs = _web_polylines_from_shell(shell)
    sliver_floor = LineMeshSpec().min_segment_length_frac * h_target

    for m in shell.midlines:  # type: ignore[attr-defined]
        if m.kind != "cap":
            continue
        arr = np.asarray(m.midline_b, dtype=float)
        yz = cap_mesh[m.label]

        assert np.allclose(yz[0], arr[0], atol=1e-12, rtol=0.0), (layout_key, m.label)
        assert np.allclose(yz[-1], arr[-1], atol=1e-12, rtol=0.0), (layout_key, m.label)

        junctions = arr[_min_dist_to_polylines(arr, webs) <= 1e-6] if webs else arr[:0]
        for q in junctions:
            d = float(np.linalg.norm(yz - q.reshape(1, 2), axis=1).min())
            near_end = min(
                float(np.linalg.norm(q - arr[0])), float(np.linalg.norm(q - arr[-1]))
            )
            tol = sliver_floor if near_end < sliver_floor else 1e-9
            assert d <= tol, (layout_key, m.label, d, tol)

        el = np.linalg.norm(np.diff(yz, axis=0), axis=1)
        assert el.min() >= 0.2 * h_target, (layout_key, m.label, el.min(), h_target)
        assert el.max() <= 1.6 * h_target, (layout_key, m.label, el.max(), h_target)


def test_1a_cn_skin_mesh_endpoints_match_panel_polyline(airfoil):
    """Skin strip MITC4 endpoints match underlying panel polyline ends (no drift)."""
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import build_section_strip_mesh
    from blade_structure_utilities.mesh_utilities.section_mesh.handoff._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )
    from blade_structure_utilities.material_utilities.section_bind import bind_shell_material_zones

    layout_key = "1A-CN"
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    shell = shell_mesh_inputs_from_section(
        section, twist_rad=0.0, layout_key=layout_key, n_web_samples=20, n_cap_samples=40
    )
    zones, _, _, _ = bind_shell_material_zones(shell)
    mesh = build_section_strip_mesh(shell, n_elements_per_panel=5, endpoint_tol=1e-6)

    for pi, pm in enumerate(mesh.panel_meshes):
        if pm.kind != "skin" or pm.yz_nodes.size == 0:
            continue
        p = zones[pi]
        nodes = np.asarray(p.nodes, dtype=float)
        assert np.allclose(pm.yz_nodes[0], nodes[0], atol=1e-9, rtol=0.0)
        assert np.allclose(pm.yz_nodes[-1], nodes[-1], atol=1e-9, rtol=0.0)
