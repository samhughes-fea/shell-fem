"""Canonical analysis nodal DOF layout for station shell FE.

One authority for analysis-level DOF names, integer indices, and global
numbering. Formulation descriptors own only the native-to-analysis embedding;
they do not define a second layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    FRAME_C,
    FrameTag,
    require_frame_tag,
)

# Shared first-six analysis DOFs in curvilinear strip frame C (≡ E on each facet).
U_X, U_S, W, BETA_S, BETA_X, BETA_ZET = 0, 1, 2, 3, 4, 5

STANDARD_ANALYSIS_DOF_NAMES: tuple[str, ...] = (
    "u_x",
    "u_s",
    "w",
    "beta_s",
    "beta_x",
    "beta_z",
)


def gdof(node_idx: int, local: int, ndof: int) -> int:
    """Global DOF index for ``local`` DOF of node ``node_idx``."""
    return int(node_idx) * int(ndof) + int(local)


def _require_unique(names: Sequence[str], *, kind: str) -> tuple[str, ...]:
    out = tuple(str(n) for n in names)
    if len(set(out)) != len(out) or not out:
        raise ValueError(f"{kind} names must be unique and non-empty")
    return out


@dataclass(frozen=True, slots=True)
class AnalysisDofLayout:
    """Explicit analysis nodal DOF layout plus embedding of an element layout."""

    layout_id: str
    names: tuple[str, ...]
    frame: FrameTag
    formulation_dof_to_analysis: tuple[int, ...]

    def __post_init__(self) -> None:
        names = _require_unique(self.names, kind="analysis DOF")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "frame", require_frame_tag(self.frame))
        mapping = tuple(int(i) for i in self.formulation_dof_to_analysis)
        if any(i < 0 or i >= len(names) for i in mapping):
            raise ValueError("formulation_dof_to_analysis indexes the analysis layout")
        object.__setattr__(self, "formulation_dof_to_analysis", mapping)

    @property
    def n_dof_node(self) -> int:
        return len(self.names)

    def embed_nodal(self, native: Sequence[float]) -> tuple[float, ...]:
        """Embed one node's native DOFs into the analysis nodal vector with zeros."""
        out = [0.0] * self.n_dof_node
        if len(native) != len(self.formulation_dof_to_analysis):
            raise ValueError("native DOF count does not match formulation map")
        for src, dst in enumerate(self.formulation_dof_to_analysis):
            out[dst] = float(native[src])
        return tuple(out)

    @classmethod
    def station_c6(cls) -> AnalysisDofLayout:
        """Production six-DOF station layout (identity embedding)."""
        return cls(
            layout_id="station_c6",
            names=STANDARD_ANALYSIS_DOF_NAMES,
            frame=FRAME_C,
            formulation_dof_to_analysis=tuple(range(6)),
        )


__all__ = [
    "AnalysisDofLayout",
    "BETA_S",
    "BETA_X",
    "BETA_ZET",
    "STANDARD_ANALYSIS_DOF_NAMES",
    "U_S",
    "U_X",
    "W",
    "gdof",
]
