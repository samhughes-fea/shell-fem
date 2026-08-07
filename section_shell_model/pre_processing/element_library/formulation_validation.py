"""Strict-shape validation for ShellFormulationCache (mirror FormulationResultSet gate).

Owned by pre_processing so ``formulation_factory`` does not import ``processing``.
"""

from __future__ import annotations

import os

import numpy as np

from ..containers.formulation_objects import ShellFormulationCache


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def validate_formulation_cache_shapes(cache: ShellFormulationCache) -> None:
    """Raise ValueError when Ke/fe shapes are inconsistent (opt-in via env)."""
    if not _env_truthy("SHELL_FORMULATION_CACHE_STRICT_SHAPE"):
        return
    n = cache.n_dof_node
    n_elem_dof = 4 * n
    if len(cache.stiffness_objects) != len(cache.force_objects):
        raise ValueError(
            f"cache: {len(cache.stiffness_objects)} stiffness vs "
            f"{len(cache.force_objects)} force objects"
        )
    for stiff, force in zip(cache.stiffness_objects, cache.force_objects):
        if stiff.element_id != force.element_id:
            raise ValueError(
                f"element id mismatch: stiff {stiff.element_id} vs force {force.element_id}"
            )
        ke, fe = stiff.K_e, force.F_e
        if ke.shape != (n_elem_dof, n_elem_dof):
            raise ValueError(f"element {stiff.element_id}: Ke shape {ke.shape}")
        if fe.shape != (n_elem_dof,):
            raise ValueError(f"element {stiff.element_id}: fe shape {fe.shape}")
        if not np.all(np.isfinite(ke)) or not np.all(np.isfinite(fe)):
            raise ValueError(f"element {stiff.element_id}: non-finite Ke or fe")


__all__ = ["validate_formulation_cache_shapes"]
