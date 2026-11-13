"""Progressive damage processing: per-ply damage states and stiffness degradation.

``SmearedCrackDegradation`` reduces the ply stiffness direction by direction as
failure modes are reached; ``DamageState`` records which modes each ply has seen.
The incremental analysis loop that drives them is
``shell_fem.simulation_runner.progressive_damage.ProgressiveDamageAnalysis``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from laminate_constitutive_model.laminate_failure_criteria import FailureMode, FailureResult
from laminate_constitutive_model.laminate_materials_3d import ElasticProperties3D


@dataclass(frozen=True)
class DamageState:
    """Monotonic damage-mode flags for one ply of one element."""

    fiber_tension: bool = False
    fiber_compression: bool = False
    matrix_tension: bool = False
    matrix_compression: bool = False
    shear: bool = False
    delamination: bool = False

    @property
    def fiber(self) -> bool:
        return self.fiber_tension or self.fiber_compression

    @property
    def matrix(self) -> bool:
        return self.matrix_tension or self.matrix_compression

    @property
    def any(self) -> bool:
        return self.fiber or self.matrix or self.shear or self.delamination

    def merge(self, other: "DamageState") -> "DamageState":
        return DamageState(*(a or b for a, b in zip(self._flags(), other._flags())))

    def _flags(self) -> Tuple[bool, ...]:
        return (self.fiber_tension, self.fiber_compression, self.matrix_tension,
                self.matrix_compression, self.shear, self.delamination)

    def labels(self) -> List[str]:
        names = ("fiber_tension", "fiber_compression", "matrix_tension", "matrix_compression", "shear", "delamination")
        return [n for n, f in zip(names, self._flags()) if f]

    @classmethod
    def from_failure(cls, result: FailureResult) -> "DamageState":
        """Map a failure result onto damage flags.

        Mode-separated criteria flag every mode whose own index reached 1. Interactive
        criteria (Tsai-Wu, Tsai-Hill) flag the dominant contribution among fibre /
        matrix / shear / interlaminar, using the stress sign stored by the caller.
        """
        if not result.failed:
            return cls()
        modes = result.active_modes() if result.mode is not FailureMode.INTERACTIVE else set()
        if result.mode is not FailureMode.INTERACTIVE and not modes:
            modes = {result.mode}
        return cls(
            fiber_tension=FailureMode.FIBER_TENSION in modes,
            fiber_compression=FailureMode.FIBER_COMPRESSION in modes,
            matrix_tension=FailureMode.MATRIX_TENSION in modes or FailureMode.TRANSVERSE_NORMAL in modes,
            matrix_compression=FailureMode.MATRIX_COMPRESSION in modes,
            shear=FailureMode.IN_PLANE_SHEAR in modes,
            delamination=FailureMode.DELAMINATION in modes,
        )

    @classmethod
    def from_interactive(cls, result: FailureResult, stress: np.ndarray) -> "DamageState":
        if not result.failed:
            return cls()
        dominant = max(result.mode_indices, key=lambda k: abs(result.mode_indices[k]))
        s1, s2 = float(stress[0]), float(stress[1])
        return cls(
            fiber_tension=dominant == "fiber" and s1 >= 0.0,
            fiber_compression=dominant == "fiber" and s1 < 0.0,
            matrix_tension=dominant == "matrix" and s2 >= 0.0,
            matrix_compression=dominant == "matrix" and s2 < 0.0,
            shear=dominant == "shear",
            delamination=dominant == "interlaminar",
        )


class DegradationStrategy(ABC):
    """Maps damage flags onto degraded elastic constants."""

    @abstractmethod
    def degrade(self, elastic: ElasticProperties3D, damage: DamageState) -> ElasticProperties3D:
        """Return the degraded elastic constants for the given damage state."""


class SmearedCrackDegradation(DegradationStrategy):
    """Direction-selective smeared-crack stiffness reduction.

    * Matrix cracking (tension or compression): E2, G12, G23 and the Poisson ratios
      v12, v23 are knocked down; E1 (fibre direction) is preserved.
    * Fibre failure: E1, G12, G13, v12, v13 are knocked down.
    * In-plane shear failure: G12 is knocked down.
    * Delamination: E3, G13, G23, v13, v23 are knocked down.

    A degraded modulus becomes ``max(residual_fraction * E_virgin, floor)`` where the
    floor is ``minimum_fraction`` of the virgin value, so a fraction of zero means
    "set to the residual floor" and keeps the stiffness matrix non-singular.
    Degraded Poisson ratios are multiplied by ``residual_fraction``.
    """

    def __init__(self, residual_fraction: float = 0.01, minimum_fraction: float = 1.0e-6, compressive_matrix_fraction: Optional[float] = None):
        if not 0.0 <= residual_fraction <= 1.0:
            raise ValueError("residual_fraction must lie in [0, 1]")
        if not 0.0 < minimum_fraction <= 1.0:
            raise ValueError("minimum_fraction must lie in (0, 1]")
        self.residual_fraction = float(residual_fraction)
        self.minimum_fraction = float(minimum_fraction)
        self.compressive_matrix_fraction = (
            self.residual_fraction if compressive_matrix_fraction is None else float(compressive_matrix_fraction)
        )

    def _knock(self, value: float, fraction: float) -> float:
        return max(fraction * value, self.minimum_fraction * value)

    def degrade(self, elastic: ElasticProperties3D, damage: DamageState) -> ElasticProperties3D:
        if not damage.any:
            return elastic
        values = elastic.as_dict()
        knocked: Dict[str, float] = {}

        def apply(names: Sequence[str], fraction: float) -> None:
            for name in names:
                knocked[name] = min(knocked.get(name, 1.0), fraction)

        r = self.residual_fraction
        if damage.matrix_tension:
            apply(("E2", "G12", "G23", "v12", "v23"), r)
        if damage.matrix_compression:
            apply(("E2", "G12", "G23", "v12", "v23"), self.compressive_matrix_fraction)
        if damage.fiber:
            apply(("E1", "G12", "G13", "v12", "v13"), r)
        if damage.shear:
            apply(("G12",), r)
        if damage.delamination:
            apply(("E3", "G13", "G23", "v13", "v23"), r)
        for name, fraction in knocked.items():
            if name.startswith("v"):
                values[name] = values[name] * fraction
            else:
                values[name] = self._knock(values[name], fraction)
        return ElasticProperties3D(**values)


# ---------------------------------------------------------------------------
# Incremental progressive damage analysis
# ---------------------------------------------------------------------------
