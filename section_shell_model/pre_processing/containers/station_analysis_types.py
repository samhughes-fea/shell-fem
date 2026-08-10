"""Station analysis / recovery-bundle / runtime artifact DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import ShellMidlineStrip

from .buckling_types import ShellStationBucklingBundle
from .recovery_types import ShellPanelResultants, ShellRecoverySectionPointSet
from .station_types import (
    BeamSectionConstitutiveInput,
    ShellSectionStationGeometry,
    ShellStationProvenance,
)

if TYPE_CHECKING:
    from blade_structure_utilities.fem_utilities.section_shell_model.pre_processing.loading.thin_wall.section_vlasov import SectionVlasovResult

@dataclass(frozen=True)
class ThinWallResult:
    """Bredt thin-wall section recovery outputs (frozen sub-DTO of SectionShellRecoveryBundle)."""

    panels: Any
    booms: Any
    webs_geom: Any
    q_tot: Any
    sig_p: Any
    sig_b: Any
    q0: Any
    props: Any
    y_sc: float  # deprecated alias of elastic centroid y (see y_e)
    z_sc: float  # deprecated alias of elastic centroid z (see z_e)
    areas: list[float]
    I_omega: float
    gamma_y: float
    gamma_z: float
    GA_y: float
    GA_z: float
    q_primary: Any
    q_warp: Any
    reference_resultants: ShellPanelResultants | None = None
    y_e: float | None = None
    z_e: float | None = None


@dataclass(frozen=True)
class ShellPanelSolveResult:
    """Shell solve outputs (frozen sub-DTO of :class:`SectionShellRecoveryBundle`)."""

    all_panel_results: list | None = None
    all_panel_diagnostics: list | None = None
    vlasov_result: SectionVlasovResult | None = None
    panel_abd: Any | None = None
    panel_thickness_m: Any | None = None
    panel_G_eff: Any | None = None
    panel_labels: list[str] | None = None
    station_buckling: ShellStationBucklingBundle | None = None
    shell_midline_strips: tuple[ShellMidlineStrip, ...] = ()
    reference_clpt_result: Any | None = None
    all_panel_clpt_results: list | None = None
    beam_mode_amplitudes: Any | None = None


@dataclass
class SectionShellRecoveryBundle:
    """Section recovery outputs plus shell-mapped stations (adapter output).

    ``y_e`` / ``z_e`` are the A11-weighted elastic centroid when known.
    Historical fields ``y_sc`` / ``z_sc`` store the **same** values
    (they are not the shear centre) — prefer ``y_e`` / ``z_e``.

    On the linear-static cascade path, thin-wall slots (``q_tot``, ``sig_p``, …)
    may be empty lists / zeros; consumers must use ``all_panel_results`` for N/M.
    """

    panels: Any
    booms: Any
    webs_geom: Any
    q_tot: Any
    sig_p: Any
    sig_b: Any
    q0: Any
    props: Any
    y_sc: float  # deprecated alias of y_e (elastic centroid, not shear centre)
    z_sc: float  # deprecated alias of z_e
    areas: list[float]
    I_omega: float
    gamma_y: float
    gamma_z: float
    GA_y: float
    GA_z: float
    q_primary: Any
    q_warp: Any
    # Shell handoff for one reference station (optional; filled by adapter helper)
    reference_resultants: ShellPanelResultants | None = None
    # Per-panel element resultants (list[list[ShellPanelResultants]])
    all_panel_results: list | None = None
    # Per-panel solve diagnostics (residual/reaction/load balance details)
    all_panel_diagnostics: list | None = None
    # Vlasov warping result (optional; thin-wall / warping diagnostics)
    vlasov_result: SectionVlasovResult | None = None
    # Panel-wise 6×6 ABD used for shell assembly (one row per panel with len(s)>=2)
    panel_abd: Any | None = None  # np.ndarray (n_panel, 6, 6)
    panel_thickness_m: Any | None = None  # (n_panel,)
    panel_G_eff: Any | None = None  # (n_panel,)
    panel_labels: list[str] | None = None
    station_buckling: ShellStationBucklingBundle | None = None
    shell_midline_strips: tuple[ShellMidlineStrip, ...] = ()
    # Native shell CLPT/derived outputs built from shell resultants (not thin-wall reference plots)
    reference_clpt_result: Any | None = None
    all_panel_clpt_results: list | None = None
    # Beam-mode amplitudes from kinematic-BC K6 solve (6,) float64, physical units.
    # Index ordering matches _build_unit_mode_matrix: [ε₀, κ_y, κ_z, γ_t, γ_s_y, γ_s_z].
    # Set only by run_section_linear_static_from_resultants_mitc4; None for all Bredt-based paths.
    beam_mode_amplitudes: Any | None = None
    # Canonical elastic-centroid coordinates (prefer over y_sc/z_sc).
    y_e: float | None = None
    z_e: float | None = None

    def __post_init__(self) -> None:
        if self.y_e is None:
            self.y_e = float(self.y_sc)
        if self.z_e is None:
            self.z_e = float(self.z_sc)

    @property
    def thin_wall(self) -> ThinWallResult:
        """View of the Bredt thin-wall fields as a frozen sub-DTO."""
        return ThinWallResult(
            panels=self.panels,
            booms=self.booms,
            webs_geom=self.webs_geom,
            q_tot=self.q_tot,
            sig_p=self.sig_p,
            sig_b=self.sig_b,
            q0=self.q0,
            props=self.props,
            y_sc=self.y_sc,
            z_sc=self.z_sc,
            areas=self.areas,
            I_omega=self.I_omega,
            gamma_y=self.gamma_y,
            gamma_z=self.gamma_z,
            GA_y=self.GA_y,
            GA_z=self.GA_z,
            q_primary=self.q_primary,
            q_warp=self.q_warp,
            reference_resultants=self.reference_resultants,
            y_e=self.y_e,
            z_e=self.z_e,
        )

    @property
    def shell_solve(self) -> ShellPanelSolveResult:
        """View of the shell solve fields as a frozen sub-DTO."""
        return ShellPanelSolveResult(
            all_panel_results=self.all_panel_results,
            all_panel_diagnostics=self.all_panel_diagnostics,
            vlasov_result=self.vlasov_result,
            panel_abd=self.panel_abd,
            panel_thickness_m=self.panel_thickness_m,
            panel_G_eff=self.panel_G_eff,
            panel_labels=self.panel_labels,
            station_buckling=self.station_buckling,
            shell_midline_strips=self.shell_midline_strips,
            reference_clpt_result=self.reference_clpt_result,
            all_panel_clpt_results=self.all_panel_clpt_results,
            beam_mode_amplitudes=self.beam_mode_amplitudes,
        )


@dataclass(frozen=True)
class SectionShellStationRuntimeArtifacts:
    """In-memory shell stage payload for one structural station."""

    station_index: int
    r_z_m: float
    airfoil: NDArray[np.float64]
    spars_m: tuple[float, ...]
    load_resultants: tuple[float, float, float, float, float, float]
    station_tag: str
    thin_wall_bundle: SectionShellRecoveryBundle
    mitc4_bundle: SectionShellRecoveryBundle
    station_json_path: Path
    station_analysis: ShellSectionStationAnalysis | None = None
    station_buckling: ShellStationBucklingBundle | None = None
    png_paths: tuple[Path, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ShellSectionStationAnalysis:
    """Canonical shell-owned station analysis artifact."""

    geometry: ShellSectionStationGeometry
    constitutive: BeamSectionConstitutiveInput
    shell_resultants: tuple[ShellPanelResultants, ...] = field(default_factory=tuple)
    recovery_point_set: ShellRecoverySectionPointSet | None = None
    buckling: ShellStationBucklingBundle | None = None
    recovery_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: ShellStationProvenance = field(default_factory=ShellStationProvenance)

    def __post_init__(self) -> None:
        if self.geometry.provenance != self.provenance:
            raise ValueError("geometry provenance must match station analysis provenance")
        if self.constitutive.provenance != self.provenance:
            raise ValueError("constitutive provenance must match station analysis provenance")
        if not np.isclose(self.geometry.z_station, self.constitutive.z_station):
            raise ValueError("geometry and constitutive z_station must match")
        if self.buckling is not None:
            if self.buckling.provenance != self.provenance:
                raise ValueError("buckling provenance must match station analysis provenance")
            if self.buckling.station_id != self.geometry.station_id:
                raise ValueError("buckling station_id must match geometry station_id")
            if not np.isclose(self.buckling.z_station, self.geometry.z_station):
                raise ValueError("buckling z_station must match geometry z_station")

