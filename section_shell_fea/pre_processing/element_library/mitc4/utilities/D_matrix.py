"""Typed ABDAs material-stiffness operator for Reissner--Mindlin MITC4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.material_utilities.laminate import (
    FSDTLaminateStiffness,
)


@dataclass(frozen=True, slots=True)
class MaterialStiffnessOperator:
    """Element-facing adapter over canonical :class:`FSDTLaminateStiffness`.

    Array/scalar arguments remain a compatibility surface.  They are converted
    once here; element assembly and recovery consume the same typed ``fsdt``.
    """

    ABD: NDArray[np.float64] | FSDTLaminateStiffness
    thickness: float
    ks: float = 5.0 / 6.0
    G_eff: float | None = None
    alpha_drill: float = 0.25
    A_s: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.thickness) or float(self.thickness) <= 0.0:
            raise ValueError("thickness must be finite and positive")
        if not np.isfinite(self.ks) or float(self.ks) <= 0.0:
            raise ValueError("ks must be finite and positive")
        if isinstance(self.ABD, FSDTLaminateStiffness):
            if self.A_s is not None or self.G_eff is not None:
                raise ValueError(
                    "A_s/G_eff must not be supplied with FSDTLaminateStiffness"
                )
            return
        abd = np.asarray(self.ABD, dtype=np.float64)
        if abd.shape != (6, 6) or not np.all(np.isfinite(abd)):
            raise ValueError("ABD must be a finite 6x6 matrix")
        if not np.allclose(abd, abd.T):
            raise ValueError("ABD must be symmetric")
        if self.A_s is not None:
            a_s = np.asarray(self.A_s, dtype=np.float64)
            provenance = "explicit"
        else:
            g_eff = self._legacy_g_eff(abd)
            a_s = np.eye(2, dtype=np.float64) * (
                float(self.ks) * g_eff * float(self.thickness)
            )
            provenance = "legacy_scalar"
        c_abdas = np.zeros((8, 8), dtype=np.float64)
        c_abdas[:6, :6] = abd
        c_abdas[6:, 6:] = a_s
        fsdt = FSDTLaminateStiffness(
            A=abd[:3, :3],
            B=abd[:3, 3:],
            D=abd[3:, 3:],
            A_s=a_s,
            ABD=abd,
            C_ABDAs=c_abdas,
            provenance=provenance,
        )
        object.__setattr__(self, "ABD", fsdt)

    def _legacy_g_eff(self, abd: NDArray[np.float64]) -> float:
        if self.G_eff is not None:
            value = float(self.G_eff)
        else:
            value = float(abd[2, 2]) / float(self.thickness)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("G_eff must be finite and positive")
        return value

    @property
    def fsdt(self) -> FSDTLaminateStiffness:
        assert isinstance(self.ABD, FSDTLaminateStiffness)
        return self.ABD

    @property
    def A_mat(self) -> NDArray[np.float64]:
        return self.fsdt.A

    @property
    def B_mat(self) -> NDArray[np.float64]:
        return self.fsdt.B

    @property
    def D_mat(self) -> NDArray[np.float64]:
        return self.fsdt.D

    @property
    def A_s_mat(self) -> NDArray[np.float64]:
        return self.fsdt.A_s

    @property
    def D_g(self) -> NDArray[np.float64]:
        """Canonical 8×8 constitutive matrix in ``epsilon/kappa/gamma13/gamma23`` order."""
        return self.fsdt.C_ABDAs

    @property
    def drilling_stiffness(self) -> float:
        """Separate Hughes--Brezzi penalty scale [N/m]."""
        uncorrected_mean = float(np.trace(self.A_s_mat)) / (2.0 * float(self.ks))
        return float(self.alpha_drill) * uncorrected_mean

    def assembly_form(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
    ]:
        """Return named ABDAs blocks and the separate drilling scale."""
        return (
            self.A_mat,
            self.B_mat,
            self.D_mat,
            self.A_s_mat,
            self.drilling_stiffness,
        )
