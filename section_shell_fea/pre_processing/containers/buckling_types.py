"""Panel / station buckling DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .recovery_types import ShellPanelResultants
from .shell_theory_capabilities import (
    ShellAnalysisRegime,
    ShellTheory,
    validate_shell_configuration,
)
from .station_types import ShellStationProvenance

@dataclass
class PanelBucklingResult:
    """Per-panel linear buckling result from MITC4 geometric stiffness solve."""

    lambdas: np.ndarray       # (n_modes,) critical load multipliers λ = N_crit / N_applied
    modeshapes: np.ndarray    # (n_modes, n_panel_dof) full free-DOF displacement fields
    n_modes_found: int
    panel_label: str
    panel_index: int
    is_compressive: bool      # True when dominant in-plane resultant is compressive (N11 < 0)


def _as_finite_array(
    values: np.ndarray | list[float] | tuple[float, ...] | None,
    *,
    name: str,
    expected_size: int | None = None,
) -> np.ndarray | None:
    """Normalise a numeric vector and reject non-finite or unexpected shapes."""

    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if expected_size is not None and arr.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} entries; got {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr.copy()


@dataclass(frozen=True)
class ShellPanelBucklingInput:
    """Shell-native panel buckling input contract for one station panel."""

    panel_label: str
    panel_index: int
    station_index: int
    length_m: float
    width_m: float
    thickness_m: float
    abd: np.ndarray | None
    membrane_resultants: np.ndarray
    bending_resultants: np.ndarray
    transverse_shear_resultants: np.ndarray | None = None
    boundary_condition: str = "simply_supported"
    material_label: str | None = None
    provenance: ShellStationProvenance = field(default_factory=ShellStationProvenance)
    source_resultant: ShellPanelResultants | None = None

    def __post_init__(self) -> None:
        if not self.panel_label.strip():
            raise ValueError("panel_label must be a non-empty string")
        if self.panel_index < 0:
            raise ValueError("panel_index must be non-negative")
        if self.station_index < 0:
            raise ValueError("station_index must be non-negative")
        for name, value in (("length_m", self.length_m), ("width_m", self.width_m), ("thickness_m", self.thickness_m)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not self.boundary_condition.strip():
            raise ValueError("boundary_condition must be a non-empty string")
        membrane = _as_finite_array(self.membrane_resultants, name="membrane_resultants", expected_size=3)
        bending = _as_finite_array(self.bending_resultants, name="bending_resultants", expected_size=3)
        shear = _as_finite_array(self.transverse_shear_resultants, name="transverse_shear_resultants", expected_size=2)
        object.__setattr__(self, "membrane_resultants", membrane if membrane is not None else np.zeros(3, dtype=float))
        object.__setattr__(self, "bending_resultants", bending if bending is not None else np.zeros(3, dtype=float))
        object.__setattr__(self, "transverse_shear_resultants", shear)
        if self.abd is not None:
            abd = np.asarray(self.abd, dtype=float)
            if abd.shape != (6, 6):
                raise ValueError(f"abd must have shape (6, 6); got {abd.shape!r}")
            if not np.all(np.isfinite(abd)):
                raise ValueError("abd must contain only finite values")
            object.__setattr__(self, "abd", abd.copy())
        if self.source_resultant is not None:
            if self.source_resultant.panel_label and self.source_resultant.panel_label != self.panel_label:
                raise ValueError("source_resultant panel_label must match panel_label")
            if self.source_resultant.panel_index != self.panel_index:
                raise ValueError("source_resultant panel_index must match panel_index")
            if self.source_resultant.station_index != self.station_index:
                raise ValueError("source_resultant station_index must match station_index")


@dataclass(frozen=True)
class ShellPanelBucklingAssessment:
    """Stable shell-owned panel buckling assessment consumed by runtime and optimisation."""

    panel_label: str
    panel_index: int
    station_index: int
    load_multiplier_min: float | None
    load_multipliers: np.ndarray
    n_modes_found: int
    is_compressive: bool
    buckling_index: float | None = None
    governing_mode: int | None = None
    governing_resultant: str | None = None
    status: str = "ok"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.panel_label.strip():
            raise ValueError("panel_label must be a non-empty string")
        if self.panel_index < 0:
            raise ValueError("panel_index must be non-negative")
        if self.station_index < 0:
            raise ValueError("station_index must be non-negative")
        multipliers = _as_finite_array(self.load_multipliers, name="load_multipliers")
        object.__setattr__(self, "load_multipliers", multipliers if multipliers is not None else np.zeros(0, dtype=float))
        if self.n_modes_found < 0:
            raise ValueError("n_modes_found must be non-negative")
        if self.load_multiplier_min is not None and not np.isfinite(self.load_multiplier_min):
            raise ValueError("load_multiplier_min must be finite when provided")
        if self.buckling_index is not None and not np.isfinite(self.buckling_index):
            raise ValueError("buckling_index must be finite when provided")
        if self.governing_mode is not None and self.governing_mode < 0:
            raise ValueError("governing_mode must be non-negative when provided")
        if not self.status.strip():
            raise ValueError("status must be a non-empty string")


@dataclass(frozen=True)
class ShellStationBucklingBundle:
    """Station-level shell buckling contract containing per-panel inputs and assessments."""

    station_id: str
    station_index: int | None
    z_station: float
    panel_inputs: tuple[ShellPanelBucklingInput, ...] = field(default_factory=tuple)
    panel_results: tuple[ShellPanelBucklingAssessment, ...] = field(default_factory=tuple)
    governing_panel_label: str | None = None
    governing_panel_index: int | None = None
    governing_buckling_index: float | None = None
    analysis_regime: ShellAnalysisRegime = ShellAnalysisRegime.LINEAR
    shell_theory: ShellTheory = ShellTheory.FSDT
    provenance: ShellStationProvenance = field(default_factory=ShellStationProvenance)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.station_id.strip():
            raise ValueError("station_id must be a non-empty string")
        if not np.isfinite(self.z_station):
            raise ValueError("z_station must be finite")
        if self.station_index is not None and self.station_index < 0:
            raise ValueError("station_index must be non-negative when provided")
        validate_shell_configuration(self.analysis_regime, self.shell_theory)
        if self.provenance.analysis_regime != self.analysis_regime:
            raise ValueError("provenance analysis_regime must match bundle analysis_regime")
        if self.provenance.shell_theory != self.shell_theory:
            raise ValueError("provenance shell_theory must match bundle shell_theory")
        panel_inputs = tuple(self.panel_inputs)
        panel_results = tuple(self.panel_results)
        object.__setattr__(self, "panel_inputs", panel_inputs)
        object.__setattr__(self, "panel_results", panel_results)
        input_by_index = {panel.panel_index: panel for panel in panel_inputs}
        result_by_index = {panel.panel_index: panel for panel in panel_results}
        for panel in panel_inputs:
            if panel.provenance != self.provenance:
                raise ValueError("panel_inputs provenance must match station provenance")
            if self.station_index is not None and panel.station_index != self.station_index:
                raise ValueError("panel_inputs station_index must match bundle station_index")
        for panel in panel_results:
            if self.station_index is not None and panel.station_index != self.station_index:
                raise ValueError("panel_results station_index must match bundle station_index")
            if panel.panel_index in input_by_index and input_by_index[panel.panel_index].panel_label != panel.panel_label:
                raise ValueError("panel_results labels must match panel_inputs labels")
        if self.governing_panel_index is not None and self.governing_panel_index not in input_by_index and self.governing_panel_index not in result_by_index:
            raise ValueError("governing_panel_index must reference a panel present in inputs or results")
        if self.governing_buckling_index is not None and not np.isfinite(self.governing_buckling_index):
            raise ValueError("governing_buckling_index must be finite when provided")

    def governing_panel_result(self) -> ShellPanelBucklingAssessment | None:
        """Return the governing panel assessment, preferring the explicit governing index when available."""

        if not self.panel_results:
            return None
        if self.governing_panel_index is not None:
            for panel in self.panel_results:
                if panel.panel_index == self.governing_panel_index:
                    return panel
        candidates = [
            panel for panel in self.panel_results if panel.buckling_index is not None and np.isfinite(panel.buckling_index)
        ]
        if candidates:
            return max(candidates, key=lambda panel: float(panel.buckling_index or -np.inf))
        return self.panel_results[0]

    def max_buckling_index(self) -> float | None:
        """Return the governing buckling index for station-level optimisation and reporting."""

        governing = self.governing_panel_result()
        if governing is None:
            return None
        return governing.buckling_index


