"""Compat barrel — Channel B / station-load products live on ``section_loads``.

FE-owned SFV → nodal apply:
``processing.static.operations.assembly.load_application.section_force_vector_to_nodal_load``.
``beam_resultants_to_nodal_load`` is regression/reference only.

Accessing section_loads re-exports from this package emits ``DeprecationWarning``.
FE-owned submodules under this tree (e.g. ``traction_face_load``) remain importable
without that barrel warning.
"""

from __future__ import annotations

import warnings
from typing import Any

from .beam_resultants_to_nodal_load import (
    beam_resultants_to_nodal_load,
    integrate_nodal_load_resultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.assembly.load_application import (
    section_force_vector_to_nodal_load,
)

_SECTION_LOADS_EXPORTS = frozenset(
    {
        "DistributedStationLoadSpec",
        "IncidentAeroSpec",
        "LoadFieldProvenance",
        "SectionLoadsJobConfig",
        "SectionStationLoadField",
        "StationAeroContext",
        "build_station_load_field",
    }
)

__all__ = [
    "DistributedStationLoadSpec",
    "IncidentAeroSpec",
    "LoadFieldProvenance",
    "SectionLoadsJobConfig",
    "SectionStationLoadField",
    "StationAeroContext",
    "beam_resultants_to_nodal_load",
    "build_station_load_field",
    "integrate_nodal_load_resultants",
    "section_force_vector_to_nodal_load",
]


def __getattr__(name: str) -> Any:
    if name in _SECTION_LOADS_EXPORTS:
        warnings.warn(
            "Import Channel B / station-load products from "
            "blade_structure_utilities.load_utilities.section_loads. "
            "FE pre_processing.loading is a temporary re-export barrel.",
            DeprecationWarning,
            stacklevel=2,
        )
        from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (
            DistributedStationLoadSpec,
            IncidentAeroSpec,
            LoadFieldProvenance,
            SectionLoadsJobConfig,
            SectionStationLoadField,
            StationAeroContext,
            build_station_load_field,
        )

        mapping = {
            "DistributedStationLoadSpec": DistributedStationLoadSpec,
            "IncidentAeroSpec": IncidentAeroSpec,
            "LoadFieldProvenance": LoadFieldProvenance,
            "SectionLoadsJobConfig": SectionLoadsJobConfig,
            "SectionStationLoadField": SectionStationLoadField,
            "StationAeroContext": StationAeroContext,
            "build_station_load_field": build_station_load_field,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
