"""Public API for per-station load field construction."""

from __future__ import annotations

from pathlib import Path

from ..containers.types import SectionLoadsJobConfig
from ..containers.types import (
    DistributedStationLoadSpec,
    SectionStationLoadField,
    StationAeroContext,
)
from ..engine.station_load_builder import (
    build_station_load_field as _build_station_load_field,
)
from blade_precompute.section_shell_model.pre_processing.mesh.shell_inputs_from_section import ShellMeshInputs


def build_station_load_field(
    shell_inputs: ShellMeshInputs,
    *,
    distributed: DistributedStationLoadSpec,
    job_config: SectionLoadsJobConfig,
    station_context: StationAeroContext,
    cache_dir: Path | None = None,
    station_id: str | None = None,
) -> SectionStationLoadField:
    """Construct a :class:`~blade_precompute.section_shell_model.pre_processing.loading.containers.types.SectionStationLoadField`."""
    return _build_station_load_field(
        shell_inputs,
        distributed=distributed,
        job_config=job_config,
        station_context=station_context,
        cache_dir=cache_dir,
        station_id=station_id,
    )


__all__ = ["build_station_load_field"]
