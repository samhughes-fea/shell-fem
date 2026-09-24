"""Theory-agnostic strip-element kernel protocol and adapters.

Element formulation specifics live here; meshing, assembly spine, and periodic
homogenisation consume kernels via :func:`resolve_kernel` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ProvenanceKind,
    ShellTheory,
    require_shell_theory_supports,
    resolve_shell_theory_capability,
)

if TYPE_CHECKING:
    from blade_structure_utilities.material_utilities.laminate import (
        FSDTLaminateStiffness,
    )


class StripElementKernel(Protocol):
    """Minimal interface between agnostic assembly/recovery and an element library."""

    name: str
    ndof: int
    theory: ShellTheory
    provenance_kind: ProvenanceKind

    def stiffness(
        self,
        L_s: float,
        L_x: float,
        ABD: NDArray[np.float64],
        *,
        thickness: float,
        G_eff: float,
    ) -> NDArray[np.float64]: ...

    def geometric_stiffness(
        self,
        L_s: float,
        L_x: float,
        N11: float,
        N22: float,
        N12: float,
        *,
        n_gauss: int = 2,
    ) -> NDArray[np.float64]: ...

    def recover_resultants(
        self,
        d_elem: NDArray[np.float64],
        L_s: float,
        L_x: float,
        ABD: NDArray[np.float64],
    ) -> dict[str, float]: ...

    def recover_shear(
        self,
        d_elem: NDArray[np.float64],
        L_s: float,
        L_x: float,
        *,
        G_eff: float,
        thickness: float,
    ) -> tuple[float | None, float | None]: ...


@dataclass(frozen=True)
class Mitc4StripKernel:
    """Production FSDT kernel (MITC4 quad)."""

    name: str = "mitc4_fsdt"
    ndof: int = 6
    theory: ShellTheory = ShellTheory.FSDT
    provenance_kind: ProvenanceKind = ProvenanceKind.MITC4

    def stiffness_from_constitutive(
        self,
        L_s: float,
        L_x: float,
        constitutive: "FSDTLaminateStiffness",
        *,
        thickness: float,
    ) -> NDArray[np.float64]:
        """Typed production path using the complete panel ABDAs object."""
        from .mitc4.element import assemble_mitc4_stiffness
        from .mitc4.utilities import MaterialStiffnessOperator, ShapeFunctionOperator

        return assemble_mitc4_stiffness(
            ShapeFunctionOperator(L_s, L_x),
            MaterialStiffnessOperator(constitutive, thickness),
            ndof=self.ndof,
        ).K_e

    def stiffness(
        self,
        L_s: float,
        L_x: float,
        ABD: NDArray[np.float64],
        *,
        thickness: float,
        G_eff: float,
    ) -> NDArray[np.float64]:
        from .mitc4.element import mitc4_stiffness

        return mitc4_stiffness(
            L_s, L_x, ABD, thickness, G_eff=G_eff, include_drilling=(self.ndof == 6)
        )

    def geometric_stiffness(
        self,
        L_s: float,
        L_x: float,
        N11: float,
        N22: float,
        N12: float,
        *,
        n_gauss: int = 2,
    ) -> NDArray[np.float64]:
        from .mitc4.element import mitc4_geometric_stiffness

        return mitc4_geometric_stiffness(
            L_s,
            L_x,
            float(N11),
            float(N22),
            float(N12),
            n_gauss=int(n_gauss),
            ndof=self.ndof,
        )

    def recover_resultants(
        self,
        d_elem: NDArray[np.float64],
        L_s: float,
        L_x: float,
        ABD: NDArray[np.float64],
    ) -> dict[str, float]:
        from .mitc4.element import mitc4_resultants

        return mitc4_resultants(d_elem, L_s, L_x, ABD, ndof=self.ndof)

    def recover_resultants_at(
        self,
        d_elem: NDArray[np.float64],
        L_s: float,
        L_x: float,
        ABD: NDArray[np.float64],
        *,
        xi: float,
        eta: float,
    ) -> dict[str, float]:
        """Recover shell resultants at parent-space ``(xi, eta)`` (R10 Gaussian tier)."""
        from .mitc4.element import mitc4_resultants_at

        return mitc4_resultants_at(
            d_elem, L_s, L_x, ABD, xi=xi, eta=eta, ndof=self.ndof,
        )

    def recover_shear(
        self,
        d_elem: NDArray[np.float64],
        L_s: float,
        L_x: float,
        *,
        G_eff: float,
        thickness: float,
    ) -> tuple[float | None, float | None]:
        if G_eff <= 0.0 or thickness <= 0.0:
            return None, None
        from .mitc4.element import mitc4_shear_resultants

        qx, qy = mitc4_shear_resultants(
            d_elem, L_s, L_x, G_eff, thickness, ndof=self.ndof,
        )
        return float(qx), float(qy)

    def recover_generalized(
        self,
        d_elem: NDArray[np.float64],
        L_s: float,
        L_x: float,
        constitutive: "FSDTLaminateStiffness",
        *,
        xi: float = 0.0,
        eta: float = 0.0,
    ) -> dict[str, float]:
        """Recover N/M/Q from the exact constitutive object used in assembly."""
        from .mitc4.element import mitc4_generalized_resultants_at

        return mitc4_generalized_resultants_at(
            d_elem,
            L_s,
            L_x,
            constitutive,
            xi=xi,
            eta=eta,
            ndof=self.ndof,
        )

    def recover_edge_shear_traction(
        self,
        d_elem: NDArray[np.float64],
        L_s: float,
        L_x: float,
        ABD: NDArray[np.float64],
        *,
        edge: str,
        gauss_n: int = 4,
    ) -> dict[str, object]:
        """Line-integrate membrane traction on an interface edge (Defect K harness)."""
        from .mitc4.element import mitc4_edge_shear_traction_integrated

        return mitc4_edge_shear_traction_integrated(
            d_elem, L_s, L_x, ABD, edge=edge, gauss_n=gauss_n, ndof=self.ndof,
        )


MITC4_KERNEL = Mitc4StripKernel()


def resolve_kernel(shell_theory: ShellTheory) -> StripElementKernel:
    """Return the strip kernel registered for ``shell_theory`` (FSDT-native path).

    Theories without a native strip kernel (e.g. CALCULIX_CCX) use dedicated
    recovery orchestrators; the ``has_native_strip_kernel`` capability flag in
    :data:`..containers.types.SHELL_THEORY_CAPABILITIES` is the single source of truth.
    Unknown or missing registrations fail through the formulation registry.
    """
    require_shell_theory_supports(shell_theory, feature="native_strip_kernel")
    from .registry import default_formulation_id, ensure_default_formulations, resolve_formulation

    ensure_default_formulations()
    return resolve_formulation(default_formulation_id(shell_theory)).kernel


def kernel_provenance_kind(shell_theory: ShellTheory) -> ProvenanceKind:
    """Provenance tag for resultants recovered via ``shell_theory``'s native kernel."""
    return resolve_shell_theory_capability(shell_theory).provenance_kind


__all__ = [
    "MITC4_KERNEL",
    "Mitc4StripKernel",
    "StripElementKernel",
    "kernel_provenance_kind",
    "resolve_kernel",
]
