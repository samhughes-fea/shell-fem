"""JSON serialization for :class:`~blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.containers.types.SectionStationLoadField`."""

from __future__ import annotations

from typing import Any


def load_field_json_payload(load_field: Any) -> dict[str, Any]:
    """Return the ``section_load_field_v1`` JSON payload for a load field."""
    return load_field.to_json_dict()


__all__ = ["load_field_json_payload"]
