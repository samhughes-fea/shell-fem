"""Shared fixtures for section_shell_fea tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def assembled_5df():
    """Assemble a cap-bearing 5D-F section; returns the SectionAssemblyResult.

    ``shell_midlines`` carry SG-017 bonds for FEA-CT-001 cap ties.
    """
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
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.bound_section_shell_inputs import (
        bind_section_shell_inputs,
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
    zones = list(bind_section_shell_inputs(shell, allow_demo_bind=True).zones)
    nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
    nxy = [np.zeros_like(v) for v in nx]
    return assemble_section_from_zones(
        zones,
        nx,
        nxy,
        L_x=1.0,
        interface_constraint_mode="shared",
        section_strip_mesh=mesh,
        shell_midlines=shell.midlines,
    )
