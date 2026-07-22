"""Shell-owned constitutive producer for structural-grid station runtime data."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .constitutive_adapter import build_shell_constitutive_input
from blade_precompute.section_shell_model.pre_processing.containers.types import BeamSectionConstitutiveInput, ShellStationProvenance


def build_structural_grid_shell_constitutive(
    *,
    station_z: Sequence[float],
    K6: np.ndarray,
    provenance: ShellStationProvenance | None = None,
) -> tuple[BeamSectionConstitutiveInput, ...]:
    """Temporary shell-owned constitutive producer backed by structural-grid K6 tables."""

    z = np.asarray(station_z, dtype=float).ravel()
    k6 = np.asarray(K6, dtype=float)
    if k6.shape != (z.size, 6, 6):
        raise ValueError(f"K6 must have shape ({z.size}, 6, 6); got {k6.shape!r}")
    prov = provenance or ShellStationProvenance()
    return tuple(
        build_shell_constitutive_input(
            z_station=float(z[i]),
            K6=k6[i],
            provenance=prov,
        )
        for i in range(z.size)
    )


__all__ = ["build_structural_grid_shell_constitutive"]
