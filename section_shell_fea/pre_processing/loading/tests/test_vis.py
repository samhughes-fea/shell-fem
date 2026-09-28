"""Tests for section_loads diagnostic PNG writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (
    DistributedStationLoadSpec,
    SectionLoadsJobConfig,
    StationAeroContext,
)
from blade_structure_utilities.load_utilities.section_loads.surface_traction.outline_frames import blade_outline_to_unit_chord
from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (
    build_station_load_field,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.vis import write_station_load_pngs
from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import ShellStripMeshingInputs
from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import ShellMidlineStrip
from blade_structure_utilities.geometry_utilities.coordinate_frames import rotate_S_to_B
from blade_structure_utilities.load_utilities.section_loads import SectionOutlineGeometry


_PIVOT = 0.333  # matches coordinate_frames.PIVOT_FRACTION_DEFAULT


def _minimal_shell_inputs(*, twist_rad: float = 0.2) -> ShellStripMeshingInputs:
    # Canonical B: (X_B, Y_B) with X_B the thickness axis, +Y_B toward the TE, and
    # the pivot chordline at the origin -- so Y_LE < 0 < Y_TE. Unit chord.
    ring = np.array(
        [
            [0.0, 1.0 - _PIVOT],
            [0.1, 0.5 - _PIVOT],
            [0.0, 0.0 - _PIVOT],
            [-0.1, 0.5 - _PIVOT],
            [0.0, 1.0 - _PIVOT],
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
    return ShellStripMeshingInputs(
        chord_m=1.0,
        twist_rad=float(twist_rad),
        layout_key="test",
        midlines=[strip],
    )


def test_blade_outline_to_unit_chord_is_unit_chord() -> None:
    # The fixture ring is already the zero-twist outline, so pass twist 0 here;
    # the twisted case is test_blade_outline_to_unit_chord_untwists below.
    shell_inputs = _minimal_shell_inputs(twist_rad=0.0)
    ring = shell_inputs.midlines[0].midline_twisted_pitch_axis_blade_frame
    outline_B = SectionOutlineGeometry.from_closed_contour(ring, ensure_clockwise=True)
    chord_frame, _ = blade_outline_to_unit_chord(
        outline_B, shell_inputs.twist_rad, te_first=False
    )
    coords = chord_frame.coordinates
    assert float(coords[:, 0].min()) >= -1e-9
    assert np.isclose(float(coords[:, 0].max()), 1.0, atol=1e-6)
    # Column 1 is thickness, not a second chord axis. Reading the chord off the
    # canonical-B thickness column (the pre-canon bug) would put ~1.0 here.
    assert float(np.abs(coords[:, 1]).max()) < 0.3


def test_blade_outline_to_unit_chord_untwists() -> None:
    """The B→NACA hop must remove the twist, not double it."""
    twist = 0.3
    shell_inputs = _minimal_shell_inputs(twist_rad=twist)
    ring = np.asarray(shell_inputs.midlines[0].midline_twisted_pitch_axis_blade_frame, dtype=float)
    # Rotate the canonical-B fixture *into* a twisted attitude, then ask the
    # adapter to take it back out; the result must match the untwisted case.
    twisted = rotate_S_to_B(ring, twist)
    outline_B = SectionOutlineGeometry.from_closed_contour(twisted, ensure_clockwise=True)
    chord_frame, _ = blade_outline_to_unit_chord(outline_B, twist, te_first=False)
    # Untwisted, the fixture's max thickness sits at 0.5 chord with |y/c| = 0.1.
    coords = chord_frame.coordinates
    i_thick = int(np.argmax(np.abs(coords[:, 1])))
    assert float(coords[i_thick, 0]) == pytest.approx(0.5, abs=1e-6)
    assert float(np.abs(coords[i_thick, 1])) == pytest.approx(0.1, abs=1e-6)


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
