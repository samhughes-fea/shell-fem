"""Opt-in DOF provenance dump tooling (``SHELL_DOF_PROVENANCE_DUMP``).

Extracted from :class:`~blade_structure_utilities.section_shell_model.simulation_runner.section_linear_static_runner.SectionLinearStaticRunner`.
Serializes the assembly/constraint/reduction/reconstruction index maps to a
JSON file for offline inspection when ``SHELL_DOF_PROVENANCE_DUMP`` is set.
"""

from __future__ import annotations

import json
import os
from typing import Any

from ..results.containers.container_hopper import ShellIndexMapSet
from ..results.containers.map_results import ShellMapEntry


def _serialize_map_entries(entries: list[ShellMapEntry] | None) -> list[dict[str, Any]]:
    if not entries:
        return []
    return [
        {
            "global_dof": e.global_dof,
            "fixed": e.fixed,
            "independent": e.independent,
            "slave": e.slave,
            "masters": e.masters,
            "reduced_index": e.reduced_index,
            "reconstructed_value": e.reconstructed_value,
        }
        for e in entries
    ]


def dump_index_maps(index_maps: ShellIndexMapSet, *, path: str | None = None) -> None:
    """Write ``index_maps`` to ``SHELL_DOF_PROVENANCE_PATH`` (default ``shell_dof_provenance.json``)."""
    resolved_path = path or os.environ.get("SHELL_DOF_PROVENANCE_PATH", "shell_dof_provenance.json")
    payload = {
        "assembly_map": _serialize_map_entries(index_maps.assembly_map),
        "constraint_map": _serialize_map_entries(index_maps.constraint_map),
        "reduction_map": _serialize_map_entries(index_maps.reduction_map),
        "reconstruction_map": _serialize_map_entries(index_maps.reconstruction_map),
    }
    with open(resolved_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


__all__ = ["dump_index_maps"]
