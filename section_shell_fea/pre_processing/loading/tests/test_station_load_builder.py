"""Smoke tests for section_loads traction reconstruction (synthetic aero)."""

from __future__ import annotations

import numpy as np
import pytest

from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (
    DistributedStationLoadSpec,
    SectionLoadsJobConfig,
    StationAeroContext,
    build_station_load_field,
)
from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import ShellMeshInputs
from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import ShellMidlineStrip


def _minimal_shell_inputs() -> ShellMeshInputs:
    ring = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.1],
            [0.0, 0.0],
            [0.5, -0.1],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    strip = ShellMidlineStrip(
        label="outer_skin",
        kind="skin",
        midline_twisted_pitch_axis_blade_frame=ring,
        thickness_m=0.01,
        closed=True,
    )
    return ShellMeshInputs(
        chord_m=1.0,
        twist_rad=0.0,
        layout_key="test",
        midlines=[strip],
    )


def test_build_station_load_field_matches_distributed_forces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADVANCED_BLADE_SECTION_LOADS_SYNTHETIC_AERO", "1")
    shell_inputs = _minimal_shell_inputs()
    job_config = SectionLoadsJobConfig(
        u_inf_m_s=10.0,
        tip_speed_ratio=5.0,
        use_direct_xfoil_scalars=True,
        direct_reynolds_number=1e6,
        direct_mach_number=0.1,
        direct_alpha_deg=5.0,
    )
    distributed = DistributedStationLoadSpec(
        z_station=0.5,
        q_y_Npm=1000.0,
        q_z_Npm=50.0,
        m_x_Nmpm=10.0,
    )
    ctx = StationAeroContext(
        station_index=0,
        z_station=0.5,
        radial_pos_m=40.0,
        chord_m=1.0,
        twist_deg=0.0,
        r_tip_m=60.0,
    )
    field = build_station_load_field(
        shell_inputs,
        distributed=distributed,
        job_config=job_config,
        station_context=ctx,
    )
    total = field.traction_field.get_total_force()
    np.testing.assert_allclose(total, [1000.0, 50.0], rtol=1e-6, atol=1e-3)
