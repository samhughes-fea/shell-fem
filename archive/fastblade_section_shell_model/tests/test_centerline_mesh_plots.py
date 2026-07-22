"""Smoke tests for section centerline mesh figure helper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from blade_precompute.section_geometry.api.factory import build_section_view, resolve_system_type
from blade_utilities.sdf.airfoil import AirfoilSDF
from blade_precompute.section_shell_model.pre_processing.mesh.section_strip_mesh import build_section_strip_mesh
from blade_precompute.section_shell_model.pre_processing.mesh.shell_inputs_from_section import build_shell_mesh_inputs
from blade_precompute.section_shell_model.vis import save_section_centerline_mesh_figure


@pytest.fixture(scope="module")
def layout_case():
    layout_key = "2D-F"
    airfoil = AirfoilSDF.from_naca("2412", chord=1.0)
    twist_rad = float(np.deg2rad(12.0))
    layout = resolve_system_type(layout_key)
    section = build_section_view(airfoil, layout, twist_angle_rad=twist_rad)
    shell_inputs = build_shell_mesh_inputs(
        section=section,
        twist_rad=twist_rad,
        layout_key=layout_key,
        n_web_samples=12,
        n_cap_samples=24,
    )
    mesh = build_section_strip_mesh(
        shell_inputs,
        target_element_length_m=0.05,
        endpoint_tol=1e-2,
    )
    return {
        "layout_key": layout_key,
        "layout": layout,
        "airfoil": airfoil,
        "section": section,
        "shell_inputs": shell_inputs,
        "mesh": mesh,
        "twist_rad": twist_rad,
        "twist_deg": 12.0,
    }


def test_save_section_centerline_mesh_figure(tmp_path: Path, layout_case) -> None:
    out = tmp_path / "section_centerline_mesh.png"
    path = save_section_centerline_mesh_figure(
        out,
        mesh=layout_case["mesh"],
        section=layout_case["section"],
        shell_inputs=layout_case["shell_inputs"],
        layout=layout_case["layout"],
        layout_key=layout_case["layout_key"],
        airfoil=layout_case["airfoil"],
        twist_rad=layout_case["twist_rad"],
        twist_deg=layout_case["twist_deg"],
        target_element_length_m=0.05,
    )
    assert path.is_file()
    # 2x2 canvas is larger than the legacy 1x2 figure.
    assert path.stat().st_size > 20_000


def test_multicell_panel_count_exceeds_subcomponents(layout_case) -> None:
    """Junction-bounded skin splits → more strips/panels than SDF subcomponents."""
    section = layout_case["section"]
    shell_inputs = layout_case["shell_inputs"]
    mesh = layout_case["mesh"]
    n_sub = len(section.labels)
    n_strips = len(shell_inputs.midlines)
    n_panels = len(mesh.panel_meshes)
    assert n_strips == n_panels
    assert n_panels > n_sub
    n_skin_segs = sum(1 for m in shell_inputs.midlines if m.kind == "skin")
    assert n_skin_segs > 1
