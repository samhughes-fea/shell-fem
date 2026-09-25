"""Recovery-point and panel-resultant DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from blade_structure_utilities.geometry_utilities.coordinate_frames import (
    FRAME_E,
    FrameTag,
    require_frame_tag,
)

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
    sigma_y: float
    z_coord: float = 0.0

    def __post_init__(self) -> None:
        if not self.subcomponent_label.strip():
            raise ValueError("subcomponent_label must be a non-empty string")
        for name, value in (
            ("thickness", self.thickness),
            ("sigma_y", self.sigma_y),
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
    Mid-surface shell resultants in element / laminate frame **E**
    (:math:`x_1,x_2,x_3`; see ``geometry_utilities.coordinate_frames``).

    Components (E-frame naming):

    * ``N11, N22, N12`` — :math:`N_{11}, N_{22}, N_{12}` [N/m]
    * ``M11, M22, M12`` — :math:`M_{11}, M_{22}, M_{12}` [N·m/m]
    * ``Q1, Q2`` — :math:`Q_1, Q_2` [N/m] (optional FSDT transverse shear)

    with :math:`x_1` = strip / laminate longitudinal, :math:`x_2` = contour
    tangent, :math:`x_3` = through-thickness. Matches
    :func:`blade_structure_utilities.material_utilities.laminate.membrane_resultants_from_shell_stress`.

    Production paths populate the full membrane/bending set from shell kinematics
    (periodic MITC4 / CCX secondary recovery). Each scalar may carry a
    :class:`FieldProvenance` record; consumers must gate failure indices on
    trusted provenance.
    """

    N11: float
    N22: float
    N12: float
    M11: float
    M22: float
    M12: float
    Q1: float | None = None
    Q2: float | None = None
    # Optional HSDT-oriented recovery summaries (prototype / future expansion)
    q1_top: float | None = None
    q1_mid: float | None = None
    q1_bot: float | None = None
    q2_top: float | None = None
    q2_mid: float | None = None
    q2_bot: float | None = None

    provenance: dict[str, FieldProvenance] = field(default_factory=dict)

    # Thin-wall diagnostics in E (uniform-through-thickness estimates)
    sigma_11_pa: float = 0.0
    tau_12_pa: float = 0.0
    q_n_per_m: float = 0.0
    thickness_m: float = 0.0

    panel_label: str = ""
    panel_index: int = 0
    station_index: int = 0
    frame: FrameTag = FRAME_E
    """Structural frame of the mid-surface resultants (ARCH-MUST-016); always **E**."""

    def __post_init__(self) -> None:
        self.frame = require_frame_tag(self.frame)
        if self.frame != FRAME_E:
            raise ValueError(
                f"ShellPanelResultants.frame must be {FRAME_E!r} (element/laminate); "
                f"got {self.frame!r}"
            )
        scalar_values = {
            "N11": self.N11,
            "N22": self.N22,
            "N12": self.N12,
            "M11": self.M11,
            "M22": self.M22,
            "M12": self.M12,
            "sigma_11_pa": self.sigma_11_pa,
            "tau_12_pa": self.tau_12_pa,
            "q_n_per_m": self.q_n_per_m,
            "thickness_m": self.thickness_m,
        }
        for name, value in scalar_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in {
            "Q1": self.Q1,
            "Q2": self.Q2,
            "q1_top": self.q1_top,
            "q1_mid": self.q1_mid,
            "q1_bot": self.q1_bot,
            "q2_top": self.q2_top,
            "q2_mid": self.q2_mid,
            "q2_bot": self.q2_bot,
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
        """E-frame ``[N11, N22, N12]`` for CLPT assembly."""
        return np.array([self.N11, self.N22, self.N12], dtype=float)

    def to_M_vec(self) -> np.ndarray:
        """E-frame ``[M11, M22, M12]`` for CLPT assembly."""
        return np.array([self.M11, self.M22, self.M12], dtype=float)


