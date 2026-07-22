"""Station geometry / constitutive / provenance DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from blade_structure_utilities.section_contract.shell_midline_strip import ShellMidlineStrip

from .shell_theory_capabilities import (
    ShellAnalysisRegime,
    ShellTheory,
    validate_shell_configuration,
)

@dataclass(frozen=True)
class ShellStationProvenance:
    """Immutable provenance for shell-owned station geometry/analysis artifacts."""

    analysis_regime: ShellAnalysisRegime = ShellAnalysisRegime.LINEAR
    shell_theory: ShellTheory = ShellTheory.FSDT
    mesh_n_elements_per_panel: int | None = None
    mesh_element_length_frac: float | None = None
    solver_version: str = "section_shell_model"
    source: str = "section_shell_model"

    def __post_init__(self) -> None:
        validate_shell_configuration(self.analysis_regime, self.shell_theory)
        if not self.solver_version.strip():
            raise ValueError("solver_version must be a non-empty string")
        if not self.source.strip():
            raise ValueError("source must be a non-empty string")
        if self.mesh_n_elements_per_panel is not None and self.mesh_n_elements_per_panel <= 0:
            raise ValueError("mesh_n_elements_per_panel must be positive when provided")
        if self.mesh_element_length_frac is not None and self.mesh_element_length_frac <= 0.0:
            raise ValueError("mesh_element_length_frac must be positive when provided")


@dataclass(frozen=True)
class ShellSectionStationGeometry:
    """Canonical shell-owned station geometry contract for downstream analysis."""

    station_id: str
    z_station: float
    strips: tuple[ShellMidlineStrip, ...]
    strip_material_labels: dict[str, str] = field(default_factory=dict)
    boundary_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ShellStationProvenance = field(default_factory=ShellStationProvenance)

    def __post_init__(self) -> None:
        if not self.station_id.strip():
            raise ValueError("station_id must be a non-empty string")
        if not np.isfinite(self.z_station):
            raise ValueError("z_station must be finite")
        if not self.strips:
            raise ValueError("strips must contain at least one ShellMidlineStrip")


@dataclass(frozen=True)
class BeamSectionConstitutiveInput:
    """Beam-facing constitutive handoff for one station.

    ``K6`` is optional: Saint-Venant periodic homogenisation is deprecated.
    Main-track linear-static station analysis omits ``K6`` (``None``).
    """

    z_station: float
    K6: np.ndarray | None = None
    mass_per_length: float | None = None
    inertia_terms: dict[str, float] = field(default_factory=dict)
    centres: dict[str, float] = field(default_factory=dict)
    warping_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ShellStationProvenance = field(default_factory=ShellStationProvenance)

    def __post_init__(self) -> None:
        if not np.isfinite(self.z_station):
            raise ValueError("z_station must be finite")
        if self.K6 is not None:
            if self.K6.shape != (6, 6):
                raise ValueError(f"K6 must have shape (6, 6); got {self.K6.shape!r}")
            if not np.all(np.isfinite(self.K6)):
                raise ValueError("K6 must contain only finite values")
        if self.mass_per_length is not None and not np.isfinite(self.mass_per_length):
            raise ValueError("mass_per_length must be finite when provided")


