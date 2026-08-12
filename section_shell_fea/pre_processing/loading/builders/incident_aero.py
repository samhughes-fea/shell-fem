"""Compat re-export — incident aero helpers live on section_loads."""

from __future__ import annotations

import warnings

from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (  # noqa: F401
    build_xfoil_run_config,
    derive_incident_aero,
)

warnings.warn(
    "Import derive_incident_aero / build_xfoil_run_config from "
    "blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load. "
    "FE builders.incident_aero is a temporary re-export.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["build_xfoil_run_config", "derive_incident_aero"]
