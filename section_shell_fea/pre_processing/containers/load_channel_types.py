"""Typed external-load channels for the section periodic BVP.

Beam resultants and surface traction are named channels of one constrained
problem. Linear static is the only production regime; mode/operator recovery
requires ``ShellAnalysisRegime.LINEAR``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    FRAME_B,
    FrameTag,
    require_frame_tag,
)
from .shell_theory_capabilities import ShellAnalysisRegime


class LoadChannelProvenance(str, Enum):
    """How an external load entered the section BVP."""

    BEAM_RESULTANTS = "beam_resultants"
    SURFACE_TRACTION = "surface_traction"
    RESIDUAL_TRACTION = "residual_traction"
    LINEAR_SUPERPOSED = "linear_superposed"


@dataclass(frozen=True)
class SectionExternalLoadRequest:
    """External loads for a section solve on a shared free-DOF set.

    ``f_beam_mode`` is the six-vector conjugate to the periodic K6 modes
    ``[N, My, Mz, T, Vz, Vy]`` (mode order, not API keyword order).
    ``f_face_global`` is an assembled nodal face-load vector (Channel B), or
    ``None`` when traction is off.
    """

    f_beam_mode: NDArray[np.float64]
    f_face_global: NDArray[np.float64] | None = None
    analysis_regime: ShellAnalysisRegime = ShellAnalysisRegime.LINEAR
    provenance: tuple[LoadChannelProvenance, ...] = field(
        default_factory=lambda: (LoadChannelProvenance.BEAM_RESULTANTS,),
    )
    traction_field: Any | None = None
    traction_points_yz: NDArray[np.float64] | None = None
    traction_pierce_frame: FrameTag = FRAME_B

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "traction_pierce_frame", require_frame_tag(self.traction_pierce_frame)
        )
        if self.traction_pierce_frame != FRAME_B:
            raise ValueError(
                f"SectionExternalLoadRequest.traction_pierce_frame must be {FRAME_B!r}; "
                f"got {self.traction_pierce_frame!r} (FEA-R-002)"
            )
        f6 = np.asarray(self.f_beam_mode, dtype=np.float64).ravel()
        if f6.size != 6:
            raise ValueError(f"f_beam_mode must have length 6, got {f6.size}")
        object.__setattr__(self, "f_beam_mode", f6)
        if self.f_face_global is not None:
            object.__setattr__(
                self,
                "f_face_global",
                np.asarray(self.f_face_global, dtype=np.float64).ravel(),
            )
        if self.analysis_regime is not ShellAnalysisRegime.LINEAR:
            raise ValueError(
                "SectionExternalLoadRequest supports only ShellAnalysisRegime.LINEAR; "
                "ShellAnalysisRegime.NONLINEAR is out of scope for section_shell_fea."
            )

    @property
    def has_face_traction(self) -> bool:
        return self.f_face_global is not None

    def require_linear_operator_recovery(self) -> None:
        """Raise if the request is unsuitable for linear mode / operator recovery."""
        if self.analysis_regime is not ShellAnalysisRegime.LINEAR:
            raise ValueError(
                "Linear operator recovery (K6) requires ShellAnalysisRegime.LINEAR."
            )

    @classmethod
    def from_beam_and_traction(
        cls,
        *,
        N: float = 0.0,
        Vy: float = 0.0,
        Vz: float = 0.0,
        My: float = 0.0,
        Mz: float = 0.0,
        T_ec: float = 0.0,
        f_face_global: NDArray[np.float64] | None = None,
        traction_field: Any | None = None,
        traction_points_yz: NDArray[np.float64] | None = None,
        analysis_regime: ShellAnalysisRegime = ShellAnalysisRegime.LINEAR,
    ) -> SectionExternalLoadRequest:
        """Build from API-order beam loads (shears enter mode slots as Vz, Vy)."""
        f_beam = np.array([N, My, Mz, T_ec, Vz, Vy], dtype=np.float64)
        has_tr = (
            f_face_global is not None
            or (traction_field is not None and traction_points_yz is not None)
        )
        if has_tr:
            provenance = (
                LoadChannelProvenance.BEAM_RESULTANTS,
                LoadChannelProvenance.SURFACE_TRACTION,
                LoadChannelProvenance.LINEAR_SUPERPOSED,
            )
        else:
            provenance = (LoadChannelProvenance.BEAM_RESULTANTS,)
        return cls(
            f_beam_mode=f_beam,
            f_face_global=f_face_global,
            analysis_regime=analysis_regime,
            provenance=tuple(provenance),
            traction_field=traction_field,
            traction_points_yz=traction_points_yz,
        )


__all__ = [
    "LoadChannelProvenance",
    "SectionExternalLoadRequest",
]
