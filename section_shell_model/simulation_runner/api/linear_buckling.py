"""Public API for the section linear buckling cascade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from blade_structure_utilities.section_shell_model.pre_processing.containers.buckling_types import (
    ShellPanelBucklingInput,
    ShellStationBucklingBundle,
)
from blade_structure_utilities.section_shell_model.pre_processing.containers.types import (
    ShellPanelResultants,
    ShellStationProvenance,
    ShellTheory,
)
from blade_structure_utilities.section_shell_model.processing.static.results.compute_secondary.panel_row_nesting import (
    nest_resultant_rows_by_panel,
)
from blade_structure_utilities.section_shell_model.simulation_runner.api.linear_static import (
    LinearStaticCascadeResult,
)
from blade_structure_utilities.section_shell_model.simulation_runner.section_linear_buckling_runner import (
    EigenTrack,
    SectionLinearBucklingResult,
    SectionLinearBucklingRunner,
    panel_inputs_from_nested_resultants,
)


@dataclass(frozen=True)
class LinearBucklingCascadeResult:
    """Primary → secondary buckling cascade plus station-bundle fold."""

    runner: SectionLinearBucklingRunner
    primary_results: Any
    secondary_results: Any
    station_bundle: ShellStationBucklingBundle
    panel_inputs: tuple[ShellPanelBucklingInput, ...]
    eigen_track: EigenTrack = "section"


def nested_resultants_from_static_cascade(
    cascade: LinearStaticCascadeResult,
) -> list[list[ShellPanelResultants]]:
    """Extract nested panel resultants from a static cascade secondary set."""
    secondary = cascade.runner.secondary_results
    if secondary is None or secondary.elemental_results is None:
        return [[] for _ in cascade.assembly.panel_maps]
    rows = list(secondary.elemental_results.resultant_rows or [])
    return nest_resultant_rows_by_panel(rows, len(cascade.assembly.panel_maps))


def run_section_linear_buckling_from_static(
    cascade: LinearStaticCascadeResult,
    *,
    station_id: str = "station-0",
    z_station: float = 0.0,
    provenance: ShellStationProvenance | None = None,
    n_modes: int = 3,
    length_m: float = 1.0,
    panel_widths_m: Sequence[float] | None = None,
    metadata: dict[str, Any] | None = None,
    eigen_track: Literal["section", "panel"] = "section",
) -> LinearBucklingCascadeResult:
    """Static secondary prestress → buckling runner P→S → station bundle.

    Default ``eigen_track=\"section\"`` solves the connected-section geometric-
    stiffness eigenproblem (reuse static ``K`` / ``T``). Pass
    ``eigen_track=\"panel\"`` for the legacy per-panel classical strip path.
    """
    nested = nested_resultants_from_static_cascade(cascade)
    asm = cascade.assembly
    prov = provenance or ShellStationProvenance()
    abd_seq = None
    if getattr(asm, "abd_p", None) is not None:
        abd_seq = [np.asarray(a, dtype=float) for a in asm.abd_p]
    t_seq = None
    if getattr(asm, "t_p", None) is not None:
        t_seq = [float(t) for t in np.asarray(asm.t_p, dtype=float)]
    widths = panel_widths_m
    if widths is None and getattr(asm, "panel_maps", None) is not None:
        derived: list[float] = []
        for pm in asm.panel_maps:
            s = np.asarray(getattr(pm, "s_nodes", []), dtype=float)
            derived.append(float(s[-1] - s[0]) if s.size >= 2 else 1.0)
        widths = derived
    panel_inputs = panel_inputs_from_nested_resultants(
        nested,
        station_index=0,
        panel_labels=list(asm.lab_p),
        panel_abd=abd_seq,
        panel_thickness_m=t_seq,
        panel_widths_m=widths,
        length_m=float(length_m),
        provenance=prov,
    )

    if eigen_track == "panel":
        runner = SectionLinearBucklingRunner.from_panel_inputs(
            panel_inputs, provenance=prov, eigen_track="panel"
        )
    else:
        solve_result = cascade.solve_result
        if solve_result.T is None:
            raise RuntimeError("static cascade missing reduction transform T")
        if asm.cache is None:
            raise RuntimeError("static cascade assembly missing formulation cache")
        K = solve_result.global_sys.K if solve_result.global_sys is not None else asm.K_global
        shell_theory = getattr(prov, "shell_theory", None) or ShellTheory.FSDT
        runner = SectionLinearBucklingRunner.from_static_solve(
            K_global=K,
            T=solve_result.T,
            cache=asm.cache,
            panel_maps=asm.panel_maps,
            nested_resultants=nested,
            panel_inputs=panel_inputs,
            n_gdof=int(asm.n_gdof),
            L_x=float(length_m),
            shell_theory=shell_theory,
            provenance=prov,
        )

    result: SectionLinearBucklingResult = runner.run_cascade(
        station_id=str(station_id),
        z_station=float(z_station),
        n_modes=int(n_modes),
        provenance=prov,
        metadata=metadata,
    )
    assert result.station_bundle is not None
    return LinearBucklingCascadeResult(
        runner=runner,
        primary_results=result.primary_results,
        secondary_results=result.secondary_results,
        station_bundle=result.station_bundle,
        panel_inputs=tuple(panel_inputs),
        eigen_track=runner.eigen_track,
    )


__all__ = [
    "LinearBucklingCascadeResult",
    "nested_resultants_from_static_cascade",
    "run_section_linear_buckling_from_static",
]
