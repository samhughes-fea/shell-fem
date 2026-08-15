"""Compat re-export — prefer section_loads.surface_traction.station_load."""

from __future__ import annotations

import warnings
from pathlib import Path

from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (
    DistributedStationLoadSpec,
    SectionLoadsJobConfig,
    SectionStationLoadField,
    StationAeroContext,
    build_station_load_field as _build_station_load_field,
)
from blade_structure_utilities.mesh_utilities.section_mesh.handoff.shell_inputs import ShellMeshInputs

warnings.warn(
    "Import build_station_load_field from "
    "blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load.",
    DeprecationWarning,
    stacklevel=2,
)


def build_station_load_field(
    shell_inputs: ShellMeshInputs,
    *,
    distributed: DistributedStationLoadSpec,
    job_config: SectionLoadsJobConfig,
    station_context: StationAeroContext,
    cache_dir: Path | None = None,
    station_id: str | None = None,
) -> SectionStationLoadField:
    return _build_station_load_field(
        shell_inputs,
        distributed=distributed,
        job_config=job_config,
        station_context=station_context,
        cache_dir=cache_dir,
        station_id=station_id,
    )


__all__ = ["build_station_load_field"]
