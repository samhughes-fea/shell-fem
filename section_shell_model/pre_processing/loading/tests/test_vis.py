"""Tests for section_loads diagnostic PNG writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.containers.types import SectionLoadsJobConfig
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.containers.types import DistributedStationLoadSpec, StationAeroContext
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.builders.outline_frames import bframe_outline_to_sframe_chord
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.builders.station_load_builder import build_station_load_field
from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.vis import write_station_load_pngs
from blade_structure_utilities.mesh_utilities.section_mesh.shell_inputs import ShellMeshInputs
from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import ShellMidlineStrip
from blade_structure_utilities.load_utilities.beam_to_shell_loads.surface_traction.geometry import SectionOutlineGeometry


def _minimal_shell_inputs(*, twist_rad: float = 0.2) -> ShellMeshInputs:
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
        midline_b=ring,
        thickness_m=0.01,
        closed=True,
    )
    return ShellMeshInputs(
        chord_m=1.0,
        twist_rad=float(twist_rad),
        layout_key="test",
        midlines=[strip],
    )


def test_bframe_outline_to_sframe_chord_is_unit_chord() -> None:
    shell_inputs = _minimal_shell_inputs(twist_rad=0.3)
    ring = shell_inputs.midlines[0].midline_b
    bframe = SectionOutlineGeometry.from_closed_contour(ring, ensure_clockwise=True)
    sframe, _ = bframe_outline_to_sframe_chord(bframe, shell_inputs.twist_rad, te_first=False)
    coords = sframe.coordinates
    assert float(coords[:, 0].min()) >= -1e-9
    assert np.isclose(float(coords[:, 0].max()), 1.0, atol=1e-6)


def test_write_station_load_pngs_creates_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")

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
    load_field = build_station_load_field(
        shell_inputs,
        distributed=DistributedStationLoadSpec(
            z_station=1.0,
            q_y_Npm=100.0,
            q_z_Npm=50.0,
            m_x_Nmpm=0.0,
        ),
        job_config=job_config,
        station_context=StationAeroContext(
            station_index=0,
            z_station=1.0,
            radial_pos_m=5.0,
            chord_m=1.0,
            twist_deg=float(np.rad2deg(shell_inputs.twist_rad)),
            r_tip_m=10.0,
        ),
    )

    station_dir = tmp_path / "station"
    paths = write_station_load_pngs(load_field, station_dir, persist=True)
    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_write_station_load_pngs_skips_when_persist_false(tmp_path: Path) -> None:
    shell_inputs = _minimal_shell_inputs()
    job_config = SectionLoadsJobConfig(
        u_inf_m_s=10.0,
        tip_speed_ratio=5.0,
        use_direct_xfoil_scalars=True,
        direct_reynolds_number=1e6,
        direct_mach_number=0.1,
        direct_alpha_deg=5.0,
    )
    load_field = build_station_load_field(
        shell_inputs,
        distributed=DistributedStationLoadSpec(
            z_station=1.0,
            q_y_Npm=100.0,
            q_z_Npm=50.0,
            m_x_Nmpm=0.0,
        ),
        job_config=job_config,
        station_context=StationAeroContext(
            station_index=0,
            z_station=1.0,
            radial_pos_m=5.0,
            chord_m=1.0,
            twist_deg=float(np.rad2deg(shell_inputs.twist_rad)),
            r_tip_m=10.0,
        ),
    )
    assert write_station_load_pngs(load_field, tmp_path, persist=False) == []
