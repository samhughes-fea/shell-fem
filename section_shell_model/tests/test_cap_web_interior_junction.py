"""Regression for web-centric cap–web T-junctions after shell mesh inputs.

Cap/web shared nodes are driven by cap–web segment intersections on web
midlines. Cap endpoint coincidence with web feet is *not* required by this
contract.
"""

from __future__ import annotations

import numpy as np
import pytest


def _dist_point_to_segment(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> float:
    ab = b - a
    l2 = float(np.dot(ab, ab)) + 1e-30
    t = float(np.dot(p - a, ab)) / l2
    t = max(0.0, min(1.0, t))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def _min_dist_cap_interior_vertices_to_webs(shell: object) -> float:
    webs: list[np.ndarray] = []
    for m in shell.midlines:  # type: ignore[attr-defined]
        if m.kind == "web":
            webs.append(np.asarray(m.midline_b, dtype=float))
    best = float("inf")
    for m in shell.midlines:  # type: ignore[attr-defined]
        if m.kind != "cap":
            continue
        pts = np.asarray(m.midline_b, dtype=float)
        if pts.shape[0] < 3:
            continue
        for p in pts[1:-1]:
            for w in webs:
                if w.shape[0] < 2:
                    continue
                for i in range(w.shape[0] - 1):
                    d = _dist_point_to_segment(p, w[i], w[i + 1])
                    best = min(best, d)
    return best


@pytest.fixture(scope="module")
def airfoil():
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import AirfoilSDF

    return AirfoilSDF.from_naca("0012", n_points=100, chord=1.0)


@pytest.mark.parametrize("layout_key", ["1C-CN"])
def test_discrete_caps_retain_web_intersections_without_webfoot_forcing(layout_key: str):
    """Discrete caps (Y=C): cap strips still intersect webs without foot forcing."""
    from blade_structure_utilities.geometry_utilities.section_geometry.sdf.airfoil import AirfoilSDF
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.mesh_utilities.section_mesh._section_to_shell_inputs import (
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
    d = _min_dist_cap_interior_vertices_to_webs(shell)
    assert d < 1e-7, (layout_key, d)


@pytest.mark.parametrize("layout_key", ["2C-CN", "3D-F"])
def test_interior_cap_vertex_on_web_after_mesh_inputs(airfoil, layout_key: str):
    from blade_structure_utilities.geometry_utilities.section_geometry.api.factory import build_section_view, resolve_system_type
    from blade_structure_utilities.mesh_utilities.section_mesh._section_to_shell_inputs import (
        shell_mesh_inputs_from_section,
    )

    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=0.0)
    shell = shell_mesh_inputs_from_section(
        section,
        twist_rad=0.0,
        layout_key=layout_key,
        n_web_samples=20,
        n_cap_samples=80,
    )
    d = _min_dist_cap_interior_vertices_to_webs(shell)
    assert d < 1e-7, (layout_key, d)
