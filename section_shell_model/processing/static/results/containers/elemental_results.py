"""Elemental-resolution results (U_e, element residuals, secondary resultants)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.containers.types import ShellPanelResultants


@dataclass
class ShellElementalResults:
    U_e: list[np.ndarray] = field(default_factory=list)
    element_residual_e: list[np.ndarray] = field(default_factory=list)
    element_residual_norms: list[float] = field(default_factory=list)
    resultant_rows: list[ShellPanelResultants] = field(default_factory=list)


__all__ = ["ShellElementalResults"]
