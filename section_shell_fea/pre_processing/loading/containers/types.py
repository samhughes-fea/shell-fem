"""Compat re-export — station load-field DTOs live on the loads worker.

Canonical home:
``blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load``.
"""

from __future__ import annotations

import warnings

from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (  # noqa: F401
    DistributedStationLoadSpec,
    IncidentAeroSpec,
    LoadFieldProvenance,
    SectionLoadsJobConfig,
    SectionStationLoadField,
    StationAeroContext,
)

warnings.warn(
    "Import station load-field DTOs from "
    "blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load "
    "(or the section_loads barrel). FE containers.types is a temporary re-export.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DistributedStationLoadSpec",
    "IncidentAeroSpec",
    "LoadFieldProvenance",
    "SectionLoadsJobConfig",
    "SectionStationLoadField",
    "StationAeroContext",
]
