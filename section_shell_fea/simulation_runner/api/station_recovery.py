"""Orchestration-facing station solve + Hashin FI workflow (simulation_runner API).

Main track: :class:`SectionLinearStaticRunner` construct → ``execute()`` →
``get_results()`` (full P→S→T→Q container). Saint-Venant ``K6`` is omitted on
the linear-static path (deprecated).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.station_analysis_types import (
    SectionShellRecoveryBundle,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellSectionStationAnalysis,
    ShellStationProvenance,
    ShellTheory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.bound_section_shell_inputs import (
    BoundSectionShellInputs,
    bind_section_shell_inputs,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.section_linear_static_runner import (
    SectionLinearStaticRunner,
)

FiTrack = Literal["linear_static"]


def _require_section_strip_mesh(
    *,
    section_strip_mesh: Any | None,
    n_elements_per_panel: int | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int | None = None,
) -> Any:
    """Require a prebuilt mesh; reject density kwargs (owned outside FE)."""
    if section_strip_mesh is None:
        raise ValueError(
            "section_strip_mesh is required; build it with "
            "mesh_utilities.section_mesh.build_section_strip_mesh "
            "(density is owned outside FE)."
        )
    if (
        n_elements_per_panel is not None
        or target_element_length_m is not None
        or min_elements_per_panel is not None
    ):
        raise ValueError(
            "Density kwargs are not accepted on this shim; "
            "pass a prebuilt section_strip_mesh= instead."
        )
    return section_strip_mesh


def _bundle_from_linear_static(
    panel_topology: BoundSectionShellInputs,
    static: SectionLinearStaticRunner,
) -> SectionShellRecoveryBundle:
    """Recovery bundle from an executed station runner, with static-secondary panel rows.

    Thin-wall / Vlasov slots (``q_tot``, ``sig_p``, ``I_omega``, …) are left empty
    on purpose: the linear-static cascade does not populate Bredt thin-wall fields.
    Prefer ``all_panel_results``, ``panel_abd``, and ``y_e`` / ``z_e`` (zeros here
    unless a future helper fills elastic-centroid metadata).
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_secondary.panel_row_nesting import (
        nest_resultant_rows_by_panel,
    )

    zones = list(panel_topology.zones)
    asm = static.assembly
    if asm is None:
        raise RuntimeError("executed runner missing assembly")
    nested: list[list[Any]] = [[] for _ in zones]
    secondary = static.secondary_results
    if secondary is not None and secondary.elemental_results is not None:
        rows = list(secondary.elemental_results.resultant_rows or [])
        nested = nest_resultant_rows_by_panel(rows, len(zones))
    reference = None
    for panel_rows in nested:
        if panel_rows:
            reference = panel_rows[len(panel_rows) // 2]
            break
    return SectionShellRecoveryBundle(
        zones=zones,
        booms=[],  # thin-wall recovery slot; bind no longer emits booms
        webs_geom=getattr(panel_topology, "webs_geom", []),
        q_tot=[],
        sig_p=[],
        sig_b=[],
        q0=[],
        props=None,
        y_sc=0.0,
        z_sc=0.0,
        y_e=0.0,
        z_e=0.0,
        areas=[],
        I_omega=0.0,
        gamma_y=0.0,
        gamma_z=0.0,
        GA_y=0.0,
        GA_z=0.0,
        q_primary=[],
        q_warp=[],
        reference_resultants=reference,
        all_panel_results=nested,
        all_panel_diagnostics=[{} for _ in zones],
        panel_abd=np.asarray(asm.abd_p, dtype=float),
        panel_thickness_m=np.asarray(asm.t_p, dtype=float),
        panel_G_eff=np.asarray(asm.g_p, dtype=float),
        panel_constitutive_specs=list(asm.cache.panel_constitutive_specs),
        panel_labels=list(asm.lab_p),
        shell_midline_strips=tuple(getattr(getattr(panel_topology, "shell_inputs", None), "midlines", ()) or ()),
    )


def _bundle_from_linear_cascade(
    panel_topology: BoundSectionShellInputs,
    cascade: Any,
) -> SectionShellRecoveryBundle:
    """Backward-compatible alias: accept cascade view or executed runner."""
    if isinstance(cascade, SectionLinearStaticRunner):
        return _bundle_from_linear_static(panel_topology, cascade)
    return _bundle_from_linear_static(panel_topology, cascade.runner)


def build_fsdt_station_analysis(
    section_view: Any,
    shell_inputs: Any,
    *,
    z_station: float,
    section_force_vector: Any,
    N: float = 1.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    n_elements_per_panel: int | None = None,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int | None = None,
    section_strip_mesh: Any | None = None,
    panel_topology: BoundSectionShellInputs | None = None,
    station_id: str = "station-0",
    provenance: ShellStationProvenance | None = None,
    fi_track: FiTrack = "linear_static",
    enable_buckling: bool = False,
) -> ShellSectionStationAnalysis:
    """Main-track station analysis for recovery prepass (linear-static).

    Omits Saint-Venant ``K6`` on the constitutive handoff (deprecated).
    Requires ``section_strip_mesh=`` (density is owned outside FE) and a frozen
    ``section_force_vector=`` shaped by ``section_loads`` / L4 — this helper
    must not call ``shape_section_force_vector*``.

    Prefer ``panel_topology=`` with pre-bound zones from orchestration. When
    omitted, ``bind_section_shell_inputs(..., allow_demo_bind=True)``
    may demo-bind (named convenience leak, not the L4 seam).
    """
    from blade_structure_utilities.fem_utilities.section_shell_fea.interface.station_analysis_builder import (
        build_station_analysis_from_shell_bundle,
    )

    if fi_track != "linear_static":
        raise ValueError(
            f"unknown fi_track={fi_track!r}; expected 'linear_static' "
            "(Channel-A fi_track='periodic' has been removed)."
        )
    if section_force_vector is None:
        raise ValueError(
            "section_force_vector is required; shape SFV with "
            "section_loads.SectionLoadsRunner / shape_section_force_vector* "
            "outside FE (FE applies frozen SFV only)."
        )

    if panel_topology is None:
        panel_topology = bind_section_shell_inputs(
            shell_inputs, section=section_view, allow_demo_bind=True
        )
    strip_mesh = _require_section_strip_mesh(
        section_strip_mesh=section_strip_mesh,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
    )
    prov = provenance or ShellStationProvenance()
    resultants = BeamResultantVector.from_named(
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        frame=FRAME_SECTION_YZ,
        about_xy=BEAM_LOAD_AXIS_YZ,
    )
    static = SectionLinearStaticRunner(
        panel_topology=panel_topology,
        resultants=resultants,
        section_strip_mesh=strip_mesh,
        load_channel="thinwall",
        section_force_vector=section_force_vector,
    ).execute()
    bundle = _bundle_from_linear_static(panel_topology, static)
    if enable_buckling:
        from blade_structure_utilities.fem_utilities.section_shell_fea.simulation_runner.api.linear_buckling import (
            run_section_linear_buckling_from_static,
        )

        buck = run_section_linear_buckling_from_static(
            static,
            station_id=str(station_id),
            z_station=float(z_station),
            provenance=prov,
        )
        # Builder reads recovery_bundle.station_buckling when present.
        setattr(bundle, "station_buckling", buck.station_bundle)

    strips = tuple(
        getattr(shell_inputs, "midlines", ())
        or getattr(bundle, "shell_midline_strips", ())
        or ()
    )
    return build_station_analysis_from_shell_bundle(
        z_station=float(z_station),
        strips=strips,
        recovery_bundle=bundle,
        K6=None,
        station_id=str(station_id),
        provenance=prov,
    )


bundle_from_linear_cascade = _bundle_from_linear_cascade

__all__ = [
    "build_fsdt_station_analysis",
    "bundle_from_linear_cascade",
]
