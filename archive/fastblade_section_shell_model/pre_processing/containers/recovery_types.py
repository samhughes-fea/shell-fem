"""Recovery-point and panel-resultant DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .shell_theory_capabilities import (
    ProvenanceKind,
    provenance_is_placeholder,
    provenance_is_trusted,
)

@dataclass
class FieldProvenance:
    """Audit trail for one scalar resultant."""

    kind: ProvenanceKind
    note: str = ""

    @property
    def is_trusted(self) -> bool:
        return provenance_is_trusted(self.kind)

    @property
    def is_placeholder(self) -> bool:
        return provenance_is_placeholder(self.kind)


@dataclass(frozen=True)
class ShellRecoveryMaterialPoint:
    """Shell-native ply/material recovery point used to assemble recovery operators."""

    subcomponent_label: str
    ply_index: int
    ply_count: int
    z_coord: float
    q_bar: np.ndarray
    transform: np.ndarray
    strengths: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.subcomponent_label.strip():
            raise ValueError("subcomponent_label must be a non-empty string")
        if self.ply_index < 0:
            raise ValueError("ply_index must be non-negative")
        if self.ply_count <= 0:
            raise ValueError("ply_count must be positive")
        if self.ply_index >= self.ply_count:
            raise ValueError("ply_index must be less than ply_count")
        if not np.isfinite(self.z_coord):
            raise ValueError("z_coord must be finite")
        if self.q_bar.shape != (3, 3):
            raise ValueError(f"q_bar must have shape (3, 3); got {self.q_bar.shape!r}")
        if self.transform.shape != (3, 3):
            raise ValueError(f"transform must have shape (3, 3); got {self.transform.shape!r}")


@dataclass(frozen=True)
class ShellRecoveryIsotropicPoint:
    """Shell-native isotropic recovery point for non-composite subcomponents."""

    subcomponent_label: str
    thickness: float
    sigma_allow: float
    z_coord: float = 0.0

    def __post_init__(self) -> None:
        if not self.subcomponent_label.strip():
            raise ValueError("subcomponent_label must be a non-empty string")
        for name, value in (
            ("thickness", self.thickness),
            ("sigma_allow", self.sigma_allow),
            ("z_coord", self.z_coord),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")


@dataclass(frozen=True)
class ShellRecoverySectionPointSet:
    """Shell-native station recovery basis replacing runtime strip solve ownership."""

    composite_labels: tuple[str, ...]
    isotropic_labels: tuple[str, ...]
    composite_resultant_basis: np.ndarray
    isotropic_resultant_basis: np.ndarray
    abd_inv: np.ndarray
    material_points: tuple[ShellRecoveryMaterialPoint, ...]
    isotropic_points: tuple[ShellRecoveryIsotropicPoint, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        n_comp = len(self.composite_labels)
        n_iso = len(self.isotropic_labels)
        if self.composite_resultant_basis.shape[:2] != (n_comp, 6):
            raise ValueError(
                "composite_resultant_basis must have shape (n_comp, 6, 6) matching composite_labels"
            )
        if self.composite_resultant_basis.shape[-1] != 6:
            raise ValueError("composite_resultant_basis trailing dimension must be 6")
        if self.isotropic_resultant_basis.shape[:2] != (n_iso, 6):
            raise ValueError(
                "isotropic_resultant_basis must have shape (n_iso, 6, 3) matching isotropic_labels"
            )
        if self.isotropic_resultant_basis.shape[-1] != 3:
            raise ValueError("isotropic_resultant_basis trailing dimension must be 3")
        if self.abd_inv.shape != (n_comp, 6, 6):
            raise ValueError("abd_inv must have shape (n_comp, 6, 6)")
        grouped: dict[str, list[ShellRecoveryMaterialPoint]] = {
            label: [] for label in self.composite_labels
        }
        for point in self.material_points:
            if point.subcomponent_label not in grouped:
                raise ValueError(
                    f"material point label {point.subcomponent_label!r} is not present in composite_labels"
                )
            grouped[point.subcomponent_label].append(point)
        for label in self.composite_labels:
            points = sorted(grouped[label], key=lambda point: point.ply_index)
            if not points:
                continue
            expected = points[0].ply_count
            if len(points) != expected:
                raise ValueError("material_points must contain ply_count entries per composite label")
            if any(point.ply_count != expected for point in points):
                raise ValueError("all material_points for a label must share the same ply_count")
            if [point.ply_index for point in points] != list(range(expected)):
                raise ValueError("material_points for each label must provide contiguous ply_index values")
        iso_labels = {point.subcomponent_label for point in self.isotropic_points}
        if iso_labels != set(self.isotropic_labels):
            raise ValueError("isotropic_points labels must match isotropic_labels exactly")

@dataclass
class ShellPanelResultants:
    """
    Laminate mid-surface shell resultants [N/m] and moments [N·m/m] in strip axes.

    Convention (matches :func:`blade_precompute.materials.clpt_recovery.membrane_resultants_from_shell_stress`):
    index 0 = x (span / beam axis), 1 = y (contour / tangent in section plane), 2 = in-plane shear xy.

    Production paths populate the full membrane/bending set from shell kinematics
    (periodic MITC4 / CCX secondary recovery). Optional FSDT fields ``Qx``, ``Qy``
    carry transverse shear [N/m]. Each scalar may carry a :class:`FieldProvenance`
    record; consumers must gate failure indices on trusted provenance.
    """

    Nx: float
    Ny: float
    Nxy: float
    Mx: float
    My: float
    Mxy: float
    Qx: float | None = None  # optional transverse shear resultants [N/m] (FSDT)
    Qy: float | None = None
    # Optional HSDT-oriented local recovery summaries (prototype / future expansion)
    qx_top: float | None = None
    qx_mid: float | None = None
    qx_bot: float | None = None
    qy_top: float | None = None
    qy_mid: float | None = None
    qy_bot: float | None = None

    provenance: dict[str, FieldProvenance] = field(default_factory=dict)

    # Raw thin-wall diagnostics (SI)
    sigma_xx_pa: float = 0.0
    tau_xy_pa: float = 0.0
    q_n_per_m: float = 0.0
    thickness_m: float = 0.0

    panel_label: str = ""
    panel_index: int = 0
    station_index: int = 0

    def __post_init__(self) -> None:
        scalar_values = {
            "Nx": self.Nx,
            "Ny": self.Ny,
            "Nxy": self.Nxy,
            "Mx": self.Mx,
            "My": self.My,
            "Mxy": self.Mxy,
            "sigma_xx_pa": self.sigma_xx_pa,
            "tau_xy_pa": self.tau_xy_pa,
            "q_n_per_m": self.q_n_per_m,
            "thickness_m": self.thickness_m,
        }
        for name, value in scalar_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in {
            "Qx": self.Qx,
            "Qy": self.Qy,
            "qx_top": self.qx_top,
            "qx_mid": self.qx_mid,
            "qx_bot": self.qx_bot,
            "qy_top": self.qy_top,
            "qy_mid": self.qy_mid,
            "qy_bot": self.qy_bot,
        }.items():
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
        if self.panel_index < 0:
            raise ValueError("panel_index must be non-negative")
        if self.station_index < 0:
            raise ValueError("station_index must be non-negative")
        if self.thickness_m < 0.0:
            raise ValueError("thickness_m must be non-negative")

    def to_N_vec(self) -> np.ndarray:
        """[Nx, Ny, Nxy] for CLPT assembly."""
        return np.array([self.Nx, self.Ny, self.Nxy], dtype=float)

    def to_M_vec(self) -> np.ndarray:
        """[Mx, My, Mxy] for CLPT assembly."""
        return np.array([self.Mx, self.My, self.Mxy], dtype=float)


