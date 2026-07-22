"""Round-trip tests for SectionStationLoadField JSON serialization."""

from __future__ import annotations

import numpy as np

from blade_precompute.section_shell_model.pre_processing.loading.containers.types import (
    DistributedStationLoadSpec,
    IncidentAeroSpec,
    LoadFieldProvenance,
    SectionStationLoadField,
)
from blade_utilities.point_to_surftraction.tractions import TractionField


def _example_load_field() -> SectionStationLoadField:
    outline = np.array([[0.0, 0.0], [1.0, 0.1], [0.0, 0.2]], dtype=np.float64)
    n_panels = outline.shape[0] - 1
    zeros = np.zeros((n_panels, 2), dtype=np.float64)
    traction = TractionField(
        pressure_forces=zeros + np.array([1.0, 0.0]),
        shear_forces=zeros + np.array([0.0, 0.5]),
        total_forces=zeros + np.array([1.0, 0.5]),
    )
    return SectionStationLoadField(
        station_id="i000_rz0.000",
        z_station=0.0,
        chord_m=2.0,
        twist_rad=0.05,
        outline_yz=outline,
        traction_field=traction,
        distributed_source=DistributedStationLoadSpec(
            z_station=0.0,
            q_y_Npm=100.0,
            q_z_Npm=20.0,
            m_x_Nmpm=3.0,
        ),
        incident=IncidentAeroSpec(
            u_rel_m_s=50.0,
            reynolds_number=1e6,
            mach_number=0.1,
            alpha_deg=4.0,
            omega_rad_s=1.2,
            u_tan_m_s=45.0,
        ),
        xfoil_artifacts={"cp": [0.1, 0.2]},
        provenance=LoadFieldProvenance(
            derivation="synthetic",
            xfoil_used=False,
            synthetic_aero=True,
            direct_scalar_override=False,
            load_matcher_rank=2,
            notes="test",
        ),
    )


def test_section_station_load_field_json_round_trip() -> None:
    original = _example_load_field()
    payload = original.to_json_dict()
    assert payload["schema_version"] == "section_load_field_v1"
    restored = SectionStationLoadField.from_json_dict(payload)
    assert restored.station_id == original.station_id
    assert restored.z_station == original.z_station
    assert restored.chord_m == original.chord_m
    assert restored.twist_rad == original.twist_rad
    np.testing.assert_allclose(restored.outline_yz, original.outline_yz)
    np.testing.assert_allclose(
        restored.traction_field.total_forces,
        original.traction_field.total_forces,
    )
    assert restored.distributed_source.q_y_Npm == original.distributed_source.q_y_Npm
    assert restored.incident.alpha_deg == original.incident.alpha_deg
    assert restored.provenance.derivation == original.provenance.derivation
    assert restored.xfoil_artifacts == original.xfoil_artifacts


def test_section_station_load_field_accepts_legacy_schema_key() -> None:
    original = _example_load_field()
    payload = original.to_json_dict()
    payload.pop("schema_version")
    payload["schema"] = "section_load_field_v1"
    restored = SectionStationLoadField.from_json_dict(payload)
    assert restored.station_id == original.station_id
