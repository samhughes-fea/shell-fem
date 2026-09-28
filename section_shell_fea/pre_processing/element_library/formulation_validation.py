"""Always-on formulation-cache validation (shapes, pairing, finite values)."""

from __future__ import annotations

import numpy as np

from ..containers.formulation_objects import ShellFormulationCache


def validate_formulation_cache_shapes(cache: ShellFormulationCache) -> None:
    """Raise ValueError when Ke/fe shapes, pairing, or finiteness fail."""
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
        n_elem_dof = int(fe.size)
        if stiff.global_dof_indices is not None:
            n_elem_dof = int(stiff.global_dof_indices.size)
        if ke.shape != (n_elem_dof, n_elem_dof):
            raise ValueError(f"element {stiff.element_id}: Ke shape {ke.shape}")
        if fe.shape != (n_elem_dof,):
            raise ValueError(f"element {stiff.element_id}: fe shape {fe.shape}")
        if not np.all(np.isfinite(ke)) or not np.all(np.isfinite(fe)):
            raise ValueError(f"element {stiff.element_id}: non-finite Ke or fe")
        if ke.shape[0] != ke.shape[1]:
            raise ValueError(f"element {stiff.element_id}: Ke must be square")
        if not np.allclose(ke, ke.T, atol=1e-9, rtol=1e-9):
            raise ValueError(f"element {stiff.element_id}: Ke is not symmetric")


__all__ = ["validate_formulation_cache_shapes"]
