"""Compat re-export — Channel B station builder lives on section_loads."""

from __future__ import annotations

import warnings

from blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load import (  # noqa: F401
    build_station_load_field,
)

warnings.warn(
    "Import build_station_load_field from "
    "blade_structure_utilities.load_utilities.section_loads.surface_traction.station_load. "
    "FE builders.station_load_builder is a temporary re-export.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["build_station_load_field"]
