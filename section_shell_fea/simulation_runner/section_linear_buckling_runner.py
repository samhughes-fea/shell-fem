"""Section linear buckling simulation runner (panel classical + full-section K/Kg)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import scipy.sparse as sp

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.buckling_types import (
    ShellPanelBucklingInput,
    ShellStationBucklingBundle,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    ShellFormulationCache,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellPanelResultants,
    ShellStationProvenance,
    ShellTheory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.topology.shell_mesh_map import (
    PanelGlobalMap,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.operations.assemble_geometric import (
    assemble_section_geometric_stiffness,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.operations.eigen_states import (
    BucklingEigenSolveState,
    PanelPrestressState,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.operations.solve_eigen import (
    prestress_map_from_nested_resultants,
    solve_panel_linear_buckling_eigen,
    solve_section_linear_buckling_eigen,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.compute_primary import (
    ShellBucklingPrimaryResultsOrchestrator,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.compute_secondary import (
    ShellBucklingSecondaryResultsOrchestrator,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.containers import (
    ShellBucklingPrimaryResultSet,
    ShellBucklingSecondaryResultSet,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.fold_station_bundle import (
    fold_station_buckling_bundle,
)

EigenTrack = Literal["section", "panel"]


def prestress_from_resultant(row: ShellPanelResultants) -> tuple[float, float, float, bool]:
    """Return ``(N11, N22, N12, is_compressive)`` from a static secondary row."""
    N11 = float(getattr(row, "N11", 0.0) or 0.0)
    N22 = float(getattr(row, "N22", 0.0) or 0.0)
    N12 = float(getattr(row, "N12", 0.0) or 0.0)
    return N11, N22, N12, bool(N11 < 0.0)


def panel_inputs_from_nested_resultants(
    nested_rows: Sequence[Sequence[ShellPanelResultants]],
    *,
    station_index: int = 0,
    panel_labels: Sequence[str] | None = None,
    panel_abd: Sequence[np.ndarray] | None = None,
    panel_thickness_m: Sequence[float] | None = None,
    panel_constitutive_specs: Sequence[Any] | None = None,
    panel_widths_m: Sequence[float] | None = None,
    length_m: float = 1.0,
    provenance: ShellStationProvenance | None = None,
    boundary_condition: str = "simply_supported",
) -> list[ShellPanelBucklingInput]:
    """Build buckling panel inputs from nested static-secondary resultants.

    ``station_index`` is the blade/station index shared by all panels (not the
    per-element index stored on secondary resultant rows).
    """
    prov = provenance or ShellStationProvenance()
    si = int(station_index)
    inputs: list[ShellPanelBucklingInput] = []
    for pi, panel_rows in enumerate(nested_rows):
        rows = list(panel_rows or [])
        if not rows:
            continue
        source = rows[len(rows) // 2]
        label = (
            str(panel_labels[pi])
            if panel_labels is not None and pi < len(panel_labels)
            else (str(source.panel_label) if source.panel_label else f"panel_{pi}")
        )
        thickness = (
            float(panel_thickness_m[pi])
            if panel_thickness_m is not None and pi < len(panel_thickness_m)
            else float(getattr(source, "thickness_m", 0.0) or 0.0)
        )
        if not np.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                f"panel {label!r}: thickness_m missing or ≤0; refusing silent "
                f"t=1e-3 default (FEA-BK-001)"
            )
        if panel_widths_m is None or pi >= len(panel_widths_m):
            raise ValueError(
                f"panel {label!r}: panel_widths_m required; refusing silent "
                f"width=1.0 m default (FEA-BK-001)"
            )
        width = float(panel_widths_m[pi])
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError(
                f"panel {label!r}: width_m must be positive; got {width!r} (FEA-BK-001)"
            )
        abd = None
        if panel_abd is not None and pi < len(panel_abd) and panel_abd[pi] is not None:
            abd = np.asarray(panel_abd[pi], dtype=float)
        inputs.append(
            ShellPanelBucklingInput(
                panel_label=label,
                panel_index=pi,
                station_index=si,
                length_m=float(length_m),
                width_m=float(width),
                thickness_m=float(thickness),
                abd=abd,
                membrane_resultants=source.to_N_vec(),
                bending_resultants=source.to_M_vec(),
                transverse_shear_resultants=np.array(
                    [float(source.Q1 or 0.0), float(source.Q2 or 0.0)],
                    dtype=float,
                ),
                boundary_condition=boundary_condition,
                material_label=label,
                provenance=prov,
                # Omit source_resultant: secondary rows use element index in
                # station_index, which must not collide with the blade station id.
                source_resultant=None,
                fsdt=(
                    panel_constitutive_specs[pi].fsdt
                    if panel_constitutive_specs is not None
                    and pi < len(panel_constitutive_specs)
                    and panel_constitutive_specs[pi] is not None
                    else None
                ),
            )
        )
    return inputs


@dataclass
class SectionLinearBucklingResult:
    """Typed output of one buckling cascade (P→S + station fold)."""

    primary_results: ShellBucklingPrimaryResultSet | None = None
    secondary_results: ShellBucklingSecondaryResultSet | None = None
    station_bundle: ShellStationBucklingBundle | None = None
    panel_inputs: tuple[ShellPanelBucklingInput, ...] = ()
    eigen_states: list[BucklingEigenSolveState] = field(default_factory=list)
    prestress: list[PanelPrestressState] = field(default_factory=list)
    eigen_track: EigenTrack = "section"


class SectionLinearBucklingRunner:
    """Orchestrates linear buckling: ops → primary → secondary → fold.

    **Default (full-section):** connected-mesh ``K_r φ = λ (−Kg_r) φ`` using the
    same ``T`` reduction as the static prestress solve (1d-fem spectral analogue).

    **Optional (panel classical):** per-panel strip eigenproducts via
    :meth:`from_panel_inputs` with ``eigen_track=\"panel\"``.
    """

    def __init__(self) -> None:
        self.panel_inputs: list[ShellPanelBucklingInput] = []
        self.prestress: list[PanelPrestressState] = []
        self.eigen_states: list[BucklingEigenSolveState] = []
        self.primary_results: ShellBucklingPrimaryResultSet | None = None
        self.secondary_results: ShellBucklingSecondaryResultSet | None = None
        self.station_bundle: ShellStationBucklingBundle | None = None
        self._provenance: ShellStationProvenance = ShellStationProvenance()
        self.eigen_track: EigenTrack = "section"
        # Full-section inputs (optional until from_static_solve)
        self.K_global: sp.spmatrix | None = None
        self.T: sp.spmatrix | None = None
        self.cache: ShellFormulationCache | None = None
        self.panel_maps: list[PanelGlobalMap] = []
        self.nested_resultants: list[list[ShellPanelResultants]] = []
        self.L_x: float = 1.0
        self.n_gdof: int = 0
        self.shell_theory: ShellTheory = ShellTheory.FSDT
        self.Kg_global: sp.spmatrix | None = None
        self.constraint_set: Any | None = None

    @classmethod
    def from_panel_inputs(
        cls,
        panel_inputs: Sequence[ShellPanelBucklingInput],
        *,
        provenance: ShellStationProvenance | None = None,
        eigen_track: EigenTrack = "panel",
    ) -> "SectionLinearBucklingRunner":
        """Panel-classical path (legacy / optional)."""
        runner = cls()
        runner.eigen_track = eigen_track
        runner.panel_inputs = list(panel_inputs)
        runner._provenance = provenance or ShellStationProvenance()
        runner.prestress = []
        for inp in runner.panel_inputs:
            mem = np.asarray(inp.membrane_resultants, dtype=float)
            N11, N22, N12 = float(mem[0]), float(mem[1]), float(mem[2])
            runner.prestress.append(
                PanelPrestressState(
                    panel_index=int(inp.panel_index),
                    panel_label=str(inp.panel_label),
                    N11=N11,
                    N22=N22,
                    N12=N12,
                    is_compressive=bool(N11 < 0.0),
                )
            )
        return runner

    @classmethod
    def from_static_solve(
        cls,
        *,
        K_global: sp.spmatrix,
        T: sp.spmatrix,
        cache: ShellFormulationCache,
        panel_maps: Sequence[PanelGlobalMap],
        nested_resultants: Sequence[Sequence[ShellPanelResultants]],
        panel_inputs: Sequence[ShellPanelBucklingInput],
        n_gdof: int,
        L_x: float = 1.0,
        shell_theory: ShellTheory | None = None,
        provenance: ShellStationProvenance | None = None,
        constraint_set: Any | None = None,
    ) -> "SectionLinearBucklingRunner":
        """Full-section path: reuse static ``K`` / ``T`` and secondary prestress."""
        runner = cls()
        runner.eigen_track = "section"
        runner.K_global = K_global
        runner.T = T
        runner.cache = cache
        runner.panel_maps = list(panel_maps)
        runner.nested_resultants = [list(rows or []) for rows in nested_resultants]
        runner.panel_inputs = list(panel_inputs)
        runner.n_gdof = int(n_gdof)
        runner.L_x = float(L_x)
        runner.constraint_set = constraint_set
        runner._provenance = provenance or ShellStationProvenance()
        runner.shell_theory = shell_theory or getattr(
            runner._provenance, "shell_theory", None
        ) or ShellTheory.FSDT
        runner.prestress = []
        for inp in runner.panel_inputs:
            mem = np.asarray(inp.membrane_resultants, dtype=float)
            runner.prestress.append(
                PanelPrestressState(
                    panel_index=int(inp.panel_index),
                    panel_label=str(inp.panel_label),
                    N11=float(mem[0]),
                    N22=float(mem[1]),
                    N12=float(mem[2]),
                    is_compressive=bool(float(mem[0]) < 0.0),
                )
            )
        return runner

    def solve(self, *, n_modes: int = 3, n_elements: int = 4) -> list[BucklingEigenSolveState]:
        """Assemble K/Kg and solve the buckling eigenproduct for the active track."""
        if self.eigen_track == "section":
            return self._solve_section(n_modes=int(n_modes))
        return self._solve_panel(n_modes=int(n_modes), n_elements=int(n_elements))

    def _solve_panel(
        self, *, n_modes: int, n_elements: int
    ) -> list[BucklingEigenSolveState]:
        states: list[BucklingEigenSolveState] = []
        for inp in self.panel_inputs:
            states.append(
                solve_panel_linear_buckling_eigen(
                    inp,
                    n_modes=int(n_modes),
                    n_elements=int(n_elements),
                )
            )
        self.eigen_states = states
        self.prestress = [
            PanelPrestressState(
                panel_index=int(s.panel_index),
                panel_label=str(s.panel_label),
                N11=float(s.diagnostics.get("N11", 0.0) or 0.0),
                N22=float(s.diagnostics.get("N22", 0.0) or 0.0),
                N12=float(s.diagnostics.get("N12", 0.0) or 0.0),
                is_compressive=bool(s.is_compressive),
            )
            for s in states
        ]
        return states

    def _solve_section(self, *, n_modes: int) -> list[BucklingEigenSolveState]:
        if self.K_global is None or self.T is None or self.cache is None:
            raise RuntimeError(
                "section eigen_track requires from_static_solve() "
                "(K_global, T, and formulation cache)"
            )
        prestress_map = prestress_map_from_nested_resultants(self.nested_resultants)
        self.Kg_global = assemble_section_geometric_stiffness(
            self.cache,
            self.panel_maps,
            prestress_map,
            self.n_gdof,
            L_x=self.L_x,
            shell_theory=self.shell_theory,
        )
        section_state = solve_section_linear_buckling_eigen(
            K_global=self.K_global,
            Kg_global=self.Kg_global,
            T=self.T,
            n_modes=int(n_modes),
        extra_diagnostics={
            "shell_theory": str(self.shell_theory),
            "n_prestress_elements": len(prestress_map),
            "n_panels": len(self.panel_inputs),
            **(
                {
                    "interface_constraint_mode": self.constraint_set.report.interface_constraint_mode,
                    "cap_tie_mode": self.constraint_set.report.cap_tie_mode,
                    "n_constraint_equations": len(self.constraint_set.equations),
                    "n_fixed_dofs": len(self.constraint_set.fixed_dofs),
                }
                if getattr(self, "constraint_set", None) is not None
                else {}
            ),
        },
        )
        state = section_state.as_eigen_state()
        self.eigen_states = [state]
        return self.eigen_states

    def compute_primary_results(self) -> ShellBucklingPrimaryResultSet:
        if not self.eigen_states:
            self.solve()
        self.primary_results = ShellBucklingPrimaryResultsOrchestrator(self.eigen_states).compute()
        return self.primary_results

    def compute_secondary_results(self, *, station_index: int | None = None) -> ShellBucklingSecondaryResultSet:
        if self.primary_results is None:
            self.compute_primary_results()
        assert self.primary_results is not None
        si = (
            int(station_index)
            if station_index is not None
            else (int(self.panel_inputs[0].station_index) if self.panel_inputs else 0)
        )
        if self.eigen_track == "section" and self.panel_inputs and self.eigen_states:
            # Station-bundle contract requires assessment labels == panel_inputs.
            # Broadcast section λ onto one assessment per prestress panel.
            from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.buckling_types import (
                PanelBucklingResult,
            )
            from blade_structure_utilities.fem_utilities.section_shell_fea.processing.buckling.results.compute_secondary.secondary_results_orchestrator import (
                assessment_from_panel_buckling_result,
            )

            section = self.eigen_states[0]
            assessments = []
            for inp in self.panel_inputs:
                raw = PanelBucklingResult(
                    lambdas=section.lambdas,
                    modeshapes=section.modeshapes,
                    n_modes_found=int(section.n_modes_found),
                    panel_label=str(inp.panel_label),
                    panel_index=int(inp.panel_index),
                    is_compressive=bool(inp.membrane_resultants[0] < 0.0),
                )
                assessments.append(
                    assessment_from_panel_buckling_result(
                        raw,
                        station_index=si,
                        diagnostics={
                            **dict(section.diagnostics),
                            "status": section.status,
                            "eigen_track": "section",
                        },
                    )
                )
            self.secondary_results = ShellBucklingSecondaryResultSet(
                panel_assessments=assessments,
                status=section.status if section.status in ("ok", "skipped", "failed") else "ok",
                diagnostics={
                    "n_panels": len(assessments),
                    "eigen_track": "section",
                    "section_status": section.status,
                },
            )
            return self.secondary_results

        panel_diag = [dict(s.diagnostics, status=s.status) for s in self.eigen_states]
        self.secondary_results = ShellBucklingSecondaryResultsOrchestrator(
            self.primary_results,
            station_index=si,
            panel_diagnostics=panel_diag,
        ).compute()
        return self.secondary_results

    def fold_station_buckling_bundle(
        self,
        *,
        station_id: str,
        z_station: float,
        provenance: ShellStationProvenance | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ShellStationBucklingBundle:
        if self.secondary_results is None:
            self.compute_secondary_results()
        assert self.secondary_results is not None
        prov = provenance or self._provenance
        meta = dict(metadata or {})
        meta.setdefault("eigen_track", self.eigen_track)
        self.station_bundle = fold_station_buckling_bundle(
            station_id=station_id,
            z_station=float(z_station),
            panel_inputs=self.panel_inputs,
            secondary=self.secondary_results,
            provenance=prov,
            metadata=meta,
        )
        return self.station_bundle

    def run_cascade(
        self,
        *,
        station_id: str,
        z_station: float,
        n_modes: int = 3,
        provenance: ShellStationProvenance | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SectionLinearBucklingResult:
        """Ops → primary → secondary → station fold."""
        self.solve(n_modes=n_modes)
        self.compute_primary_results()
        self.compute_secondary_results()
        bundle = self.fold_station_buckling_bundle(
            station_id=station_id,
            z_station=z_station,
            provenance=provenance,
            metadata=metadata,
        )
        return SectionLinearBucklingResult(
            primary_results=self.primary_results,
            secondary_results=self.secondary_results,
            station_bundle=bundle,
            panel_inputs=tuple(self.panel_inputs),
            eigen_states=list(self.eigen_states),
            prestress=list(self.prestress),
            eigen_track=self.eigen_track,
        )


__all__ = [
    "SectionLinearBucklingResult",
    "SectionLinearBucklingRunner",
    "panel_inputs_from_nested_resultants",
    "prestress_from_resultant",
]
