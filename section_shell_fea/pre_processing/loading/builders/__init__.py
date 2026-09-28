"""Compat re-export — load-field builders live on ``section_loads``."""

from __future__ import annotations

import warnings

warnings.warn(
    "Import load-field builders from "
    "blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load "
    "(or the section_loads barrel). FE loading.builders is a temporary re-export.",
    DeprecationWarning,
    stacklevel=2,
)

from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (
    build_station_load_field,
    build_xfoil_run_config,
    derive_incident_aero,
)

__all__ = [
    "build_station_load_field",
    "build_xfoil_run_config",
    "derive_incident_aero",
]
