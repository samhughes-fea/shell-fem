"""MITC4 material stiffness operator (1d-fem utility template)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MaterialStiffnessOperator:
    """Wrap laminate ABD plus shear/drilling blocks for MITC4 assembly."""

    ABD: NDArray[np.float64]
    thickness: float
    ks: float = 5.0 / 6.0
    G_eff: float | None = None
    alpha_drill: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "ABD", np.asarray(self.ABD, dtype=np.float64))

    @property
    def A_mat(self) -> NDArray[np.float64]:
        return self.ABD[:3, :3]

    @property
    def B_mat(self) -> NDArray[np.float64]:
        return self.ABD[:3, 3:]

    @property
    def D_mat(self) -> NDArray[np.float64]:
        return self.ABD[3:, 3:]

    def _g_eff(self) -> float:
        if self.G_eff is not None:
            return float(self.G_eff)
        return float(self.A_mat[2, 2]) / max(float(self.thickness), 1e-30)

    def assembly_form(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float]:
        """Return membrane/bending blocks, shear ``H_s``, drilling scale."""
        g_eff = self._g_eff()
        h_s = self.ks * g_eff * float(self.thickness)
        c_drill = self.alpha_drill * g_eff * float(self.thickness)
        return self.A_mat, self.B_mat, self.D_mat, h_s, c_drill
