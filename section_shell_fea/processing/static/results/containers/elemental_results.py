"""Elemental-resolution results (U_e, element residuals, secondary resultants).

* ``U_e`` — per-element DOF vectors for the station shell mesh (geometry in **B**;
  strip/panel DOF basis used by the element kernels).
* ``resultant_rows`` — mid-surface shell resultants in **E** (element = laminate
  \\((x_1,x_2,x_3)\\)). Field names ``N11…M12`` / ``Q1,Q2`` are legacy labels for
  \\(N_{11},N_{22},N_{12},M_{11},M_{22},M_{12}\\) / \\(Q_1,Q_2\\).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    FRAME_E,
    FrameTag,
    require_frame_tag,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellPanelResultants,
)


@dataclass
class ShellElementalResults:
    """Per-element primary DOFs / residuals and secondary **E**-frame resultants."""

    U_e: list[np.ndarray] = field(default_factory=list)
    element_residual_e: list[np.ndarray] = field(default_factory=list)
    element_residual_norms: list[float] = field(default_factory=list)
    resultant_rows: list[ShellPanelResultants] = field(default_factory=list)
    resultant_frame: FrameTag = FRAME_E
    """Structural frame of ``resultant_rows`` (ARCH-MUST-016); always **E**."""

    def __post_init__(self) -> None:
        self.resultant_frame = require_frame_tag(self.resultant_frame)
        if self.resultant_frame != FRAME_E:
            raise ValueError(
                f"ShellElementalResults.resultant_frame must be {FRAME_E!r}; "
                f"got {self.resultant_frame!r}"
            )


__all__ = ["ShellElementalResults"]
