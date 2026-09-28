"""Public API for the section linear buckling cascade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.buckling_types import (
    ShellPanelBucklingInput,
    ShellStationBucklingBundle,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellPanelResultants,
    ShellStationProvenance,
    ShellTheory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_secondary.panel_row_nesting import (
    nest_resultant_rows_by_panel,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.linear_static import (
    LinearStaticCascadeResult,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_buckling_runner import (
    EigenTrack,
    SectionLinearBucklingResult,
    SectionLinearBucklingRunner,
    panel_inputs_from_nested_resultants,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticResult,
    SectionLinearStaticRunner,
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


def _static_view(
    static: SectionLinearStaticRunner | LinearStaticCascadeResult,
) -> tuple[SectionLinearStaticRunner, Any, SectionLinearStaticResult]:
    """Normalize executed runner or transitional cascade view."""
    if isinstance(static, SectionLinearStaticRunner):
        if static.assembly is None:
            raise RuntimeError("static runner missing assembly; call execute() first")
        return static, static.assembly, static.get_results()
    return static.runner, static.assembly, static.solve_result


def nested_resultants_from_static_cascade(
    static: SectionLinearStaticRunner | LinearStaticCascadeResult,
) -> list[list[ShellPanelResultants]]:
    """Extract nested panel resultants from an executed static runner / cascade."""
    runner, asm, _ = _static_view(static)
    secondary = runner.secondary_results
    if secondary is None or secondary.elemental_results is None:
        return [[] for _ in asm.panel_maps]
    rows = list(secondary.elemental_results.resultant_rows or [])
    return nest_resultant_rows_by_panel(rows, len(asm.panel_maps))


def run_section_linear_buckling_from_static(
    static: SectionLinearStaticRunner | LinearStaticCascadeResult,
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

    ``static`` is an executed :class:`SectionLinearStaticRunner` (preferred) or a
    transitional :class:`LinearStaticCascadeResult`.

    Default ``eigen_track=\"section\"`` solves the connected-section geometric-
    stiffness eigenproblem (reuse static ``K`` / ``T``). Pass
    ``eigen_track=\"panel\"`` for the legacy per-panel classical strip path.
    """
    _runner, asm, solve_result = _static_view(static)
    nested = nested_resultants_from_static_cascade(static)
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
            if s.size < 2:
                raise ValueError(
                    f"panel {getattr(pm, 'panel_label', None)!r}: cannot derive "
                    "buckling width from s_nodes; pass panel_widths_m= "
                    "(FEA-BK-001)"
                )
            w = float(abs(s[-1] - s[0]))
            if w <= 0.0:
                raise ValueError(
                    f"panel {getattr(pm, 'panel_label', None)!r}: derived "
                    f"buckling width is {w!r} (FEA-BK-001)"
                )
            derived.append(w)
        widths = derived
    if widths is None:
        raise ValueError(
            "panel_widths_m required when assembly has no panel_maps "
            "(refusing silent width=1.0; FEA-BK-001)"
        )
    panel_inputs = panel_inputs_from_nested_resultants(
        nested,
        station_index=0,
        panel_labels=list(asm.lab_p),
        panel_abd=abd_seq,
        panel_thickness_m=t_seq,
        panel_constitutive_specs=getattr(asm.cache, "panel_constitutive_specs", None),
        panel_widths_m=widths,
        length_m=float(length_m),
        provenance=prov,
    )

    if eigen_track == "panel":
        runner = SectionLinearBucklingRunner.from_panel_inputs(
            panel_inputs, provenance=prov, eigen_track="panel"
        )
    else:
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
            constraint_set=getattr(_runner, "constraint_set", None),
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
