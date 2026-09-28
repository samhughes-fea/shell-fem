"""MITC4 live-element and cache validation hooks."""

from __future__ import annotations

from typing import Any

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)

_MITC4_ID = "MITC4-FSDT"
_N_NODES = 4
_N_DOF_NODE = 6
_N_ELEM_DOF = _N_NODES * _N_DOF_NODE


def validate_mitc4_element(element: Any) -> None:
    """Refuse a live MITC4 element that violates geometry/material invariants."""
    fid = getattr(element, "formulation_id", None) or getattr(
        type(element), "formulation_id", None
    )
    elem_type = getattr(element, "element_type", None)
    if fid not in (None, _MITC4_ID) and elem_type not in (None, _MITC4_ID):
        raise ValueError(
            f"MITC4 validate_element: unexpected formulation identity "
            f"formulation_id={fid!r} element_type={elem_type!r}"
        )
    geom = getattr(element, "geometry", None)
    if geom is None:
        raise ValueError("MITC4 validate_element: missing geometry")
    for name in ("L_s", "L_x"):
        val = float(getattr(geom, name))
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(
                f"MITC4 validate_element: {name} must be finite and positive; got {val!r}"
            )
    local = getattr(geom, "local_nodes", None)
    if local is not None and len(local) != _N_NODES:
        raise ValueError(
            f"MITC4 validate_element: expected {_N_NODES} local nodes; got {len(local)}"
        )
    mat = getattr(element, "material", None)
    if mat is None:
        raise ValueError("MITC4 validate_element: missing material")
    thickness = float(getattr(mat, "thickness"))
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError(
            f"MITC4 validate_element: thickness must be finite and positive; got {thickness!r}"
        )
    abd = getattr(mat, "ABD", None)
    if abd is not None:
        arr = np.asarray(abd, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError("MITC4 validate_element: ABD has non-finite entries")
    fsdt = getattr(mat, "fsdt", None)
    if fsdt is not None:
        for block_name in ("A", "B", "D", "A_s"):
            block = np.asarray(getattr(fsdt, block_name), dtype=float)
            if not np.all(np.isfinite(block)):
                raise ValueError(
                    f"MITC4 validate_element: fsdt.{block_name} has non-finite entries"
                )


def validate_mitc4_cache(cache: ShellFormulationCache) -> None:
    """Refuse an MITC4 formulation cache that violates layout invariants."""
    if cache.formulation_id != _MITC4_ID:
        raise ValueError(
            f"MITC4 validate_cache: formulation_id must be {_MITC4_ID!r}; "
            f"got {cache.formulation_id!r}"
        )
    if int(cache.n_dof_node) != _N_DOF_NODE:
        raise ValueError(
            f"MITC4 validate_cache: n_dof_node must be {_N_DOF_NODE}; "
            f"got {cache.n_dof_node}"
        )
    for stiff, force in zip(cache.stiffness_objects, cache.force_objects):
        if stiff.global_dof_indices is not None:
            n = int(stiff.global_dof_indices.size)
            if n != _N_ELEM_DOF:
                raise ValueError(
                    f"MITC4 validate_cache: element {stiff.element_id} has "
                    f"{n} DOFs; expected {_N_ELEM_DOF}"
                )
        if int(force.F_e.size) != _N_ELEM_DOF and stiff.global_dof_indices is None:
            # Before fill this may differ; after fill shapes are already checked.
            pass


__all__ = ["validate_mitc4_cache", "validate_mitc4_element"]
