"""Periodic unit-cell section solves (sole section shell FE path).

Theory-agnostic router :func:`run_section_periodic_bc` plus the FSDT prismatic
and tapered periodic solvers. Load-independent assembly/K6 caches live in
:mod:`..engine.periodic_cache` (re-exported here for existing probes).
CalculiX (ccx) is an opt-in validation backend — import
``calculix_ccx.periodic.run_section_periodic_bc_ccx`` directly (and call
``register_calculix_ccx_backend()`` before theory-dispatched routing).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from blade_precompute.contract.beam_resultant_vector import BeamResultantVector
from blade_precompute.section_shell_model.pre_processing.mesh.mesh_topology import NElements
from blade_precompute.section_shell_model.pre_processing.mesh.section_strip_mesh import MIN_ELEMENTS_PER_PANEL
from blade_precompute.section_shell_model.pre_processing.containers.types import (
    SectionShellRecoveryBundle,
    ShellTheory,
    require_shell_theory_supports,
)
from blade_precompute.section_shell_model.pre_processing.loading.beam_resultant_handoff import (
    resolve_beam_resultants_for_shell_k6,
)
from ..engine.periodic_cache import (
    _PERIODIC_ASSEMBLY_CACHE,
    _PERIODIC_K6_CACHE,
    _TAPERED_PERIODIC_ASSEMBLY_CACHE,
    _cached_periodic_k6,
    _cached_tapered_periodic_assembly,
    _unpack_assembly_result,
    clear_periodic_caches,
    get_periodic_geometry_state,
)
from .pipeline_b import PipelineBSection, TaperedPipelineBSection


def run_section_periodic_bc(
    pb: "PipelineBSection",
    *,
    shell_theory: ShellTheory = ShellTheory.FSDT,
    resultants: BeamResultantVector | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    n_elements_per_panel: NElements = 10,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = MIN_ELEMENTS_PER_PANEL,
    bc_mode: str = "contour_support",
    interface_constraint_mode: str = "shared",
    loading_axis_yz: tuple[float, float] | None = None,
    transport_to_elastic_centroid: bool = True,
    traction_field: Any | None = None,
    traction_points_yz: np.ndarray | None = None,
    external_backend_options: dict[str, Any] | None = None,
    emit_frd: bool = True,
) -> SectionShellRecoveryBundle:
    """Theory-agnostic periodic unit-cell section solve for beam loads.

    Prefer ``resultants`` (:class:`~blade_precompute.contract.BeamResultantVector`)
    carrying frame + ``about_yz``. Scalar ``N``…``T`` kwargs remain as a
    centroid-referenced shim. ``loading_axis_yz`` is deprecated.

    ``transport_to_elastic_centroid`` (default True) applies the full
    ``r×F`` map from ``resultants.about_yz`` to the elastic centroid. Set
    False to retag without transport (legacy ``T_ec = T`` Channel-A behaviour).

    Dispatches to the registered element kernel / external solver for
    ``shell_theory`` (default FSDT via MITC4). The set of supported theories and
    their per-theory traits (e.g. whether an external work dir is required) come
    from the single capability table in :data:`..core.types.SHELL_THEORY_CAPABILITIES`.

    ``external_backend_options`` may include ``work_dir`` and ``exe`` for backends
    that require an external work directory (e.g. CalculiX).
    """
    cap = require_shell_theory_supports(shell_theory, feature="periodic")
    common = dict(
        resultants=resultants,
        N=N,
        Vy=Vy,
        Vz=Vz,
        My=My,
        Mz=Mz,
        T=T,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
        loading_axis_yz=loading_axis_yz,
        transport_to_elastic_centroid=transport_to_elastic_centroid,
        traction_field=traction_field,
        traction_points_yz=traction_points_yz,
    )
    try:
        solver = _PERIODIC_SOLVERS[shell_theory]
    except KeyError:
        raise ValueError(
            f"No periodic-BC handler registered for {shell_theory!r}; "
            "opt-in backends (e.g. CalculiX ccx) require calling their "
            "register_*_backend() hook first."
        ) from None
    opts = dict(external_backend_options or {})
    if cap.requires_external_work_dir:
        work_dir = opts.get("work_dir")
        if work_dir is None:
            raise ValueError(
                f"shell_theory={shell_theory.value.upper()} requires "
                "external_backend_options with 'work_dir'=..."
            )
        return solver(
            pb,
            ccx_work_dir=work_dir,
            ccx_exe=str(opts.get("exe", "ccx")),
            emit_frd=emit_frd,
            **common,
        )
    return solver(pb, **common)


def run_section_periodic_bc_mitc4(
    pb: "PipelineBSection",
    *,
    resultants: BeamResultantVector | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    n_elements_per_panel: NElements = 10,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = MIN_ELEMENTS_PER_PANEL,
    bc_mode: str = "contour_support",
    interface_constraint_mode: str = "shared",
    loading_axis_yz: tuple[float, float] | None = None,
    transport_to_elastic_centroid: bool = True,
    traction_field: Any | None = None,
    traction_points_yz: np.ndarray | None = None,
) -> SectionShellRecoveryBundle:
    """MITC4 section solve for beam loads via the **periodic unit-cell BC**.

    The section is a unit-length (``L_x = 1 m``) spanwise slice. ``compute_k6_periodic_bc``
    imposes a prescribed displacement *jump* between the inboard (``x=0``) and
    outboard (``x=L_x``) cut faces encoding the six beam unit modes
    (ε₀, κ_y, κ_z, torsion, shear_y, shear_z); **neither cut face is clamped**, so
    there are no Saint-Venant end effects. ``K6`` comes from the energy bilinear
    form, the mode amplitudes solve ``K6 q = f_beam``, and per-element resultants
    (incl. Mx/My/Mxy) are recovered from ``u_modes @ q`` — enabling through-thickness
    CLT stress recovery.

    Parameters
    ----------
    pb
        Pipeline B section from :func:`build_pipeline_b_section`.
    resultants
        Preferred :class:`~blade_precompute.contract.BeamResultantVector` with
        frame + ``about_yz``. Transported to the elastic centroid internally
        when ``transport_to_elastic_centroid`` is True.
    N, Vy, Vz, My, Mz, T
        Legacy scalar beam-level section loads (centroid-referenced shim).
    """
    from ..engine.periodic_cache import get_cached_periodic_reduced_bvp
    from ..engine.periodic_secondary_recovery import (
        recover_resultants_from_displacement,
        recover_resultants_from_modes,
    )
    from ..engine.periodic_traction_superposition import (
        PERIODIC_FACE_TRACTION_PROVENANCE,
        superpose_traction_on_beam_displacement,
    )
    from ..pre_processing.containers.load_channel_types import SectionExternalLoadRequest
    from ..pre_processing.element_library.kernel import MITC4_KERNEL

    panels = pb.panels
    (
        K_global,
        _f_global,
        node_meta,
        panel_maps,
        K6,
        u_modes,
        y_e,
        z_e,
        abd_p,
        t_p,
        g_p,
        lab_p,
        formulation_cache,
    ) = _cached_periodic_k6(
        panels,
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    if formulation_cache is None:
        raise RuntimeError("periodic K6 cache missing ShellFormulationCache (R1/R2 regression)")

    at_ec, f_beam = resolve_beam_resultants_for_shell_k6(
        y_e=float(y_e),
        z_e=float(z_e),
        resultants=resultants,
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        loading_axis_yz=loading_axis_yz,
        transport_to_elastic_centroid=transport_to_elastic_centroid,
    )
    load_req = SectionExternalLoadRequest.from_beam_and_traction(
        N=at_ec.N,
        Vy=at_ec.Vy,
        Vz=at_ec.Vz,
        My=at_ec.My,
        Mz=at_ec.Mz,
        T_ec=at_ec.T,
        traction_field=traction_field,
        traction_points_yz=traction_points_yz,
    )
    load_req.require_linear_operator_recovery()
    # Prefer the named projection (guards against from_beam_and_traction drift).
    f_beam = np.asarray(f_beam, dtype=np.float64)
    try:
        q_beam_physical = np.linalg.solve(K6, f_beam)
    except np.linalg.LinAlgError:
        q_beam_physical = np.linalg.lstsq(K6, f_beam, rcond=None)[0]

    d_beam = u_modes @ q_beam_physical
    use_traction = traction_field is not None and traction_points_yz is not None
    if use_traction:
        periodic_bvp = get_cached_periodic_reduced_bvp(
            panels,
            n_elements_per_panel=n_elements_per_panel,
            target_element_length_m=target_element_length_m,
            min_elements_per_panel=min_elements_per_panel,
            bc_mode=bc_mode,
            interface_constraint_mode=interface_constraint_mode,
        )
        d_total = superpose_traction_on_beam_displacement(
            d_beam,
            K_global,
            node_meta,
            panel_maps,
            traction_field=traction_field,
            traction_points_yz=traction_points_yz,
            ndof=MITC4_KERNEL.ndof,
            bc_mode=bc_mode,
            periodic_bvp=periodic_bvp,
            y_e=y_e,
            z_e=z_e,
        )
        all_panel_results = recover_resultants_from_displacement(
            d_total,
            formulation_cache,
            panel_maps,
            panels,
            provenance_tag=PERIODIC_FACE_TRACTION_PROVENANCE,
            shell_theory=ShellTheory.FSDT,
        )
    else:
        all_panel_results = recover_resultants_from_modes(
            u_modes,
            q_beam_physical,
            formulation_cache,
            panel_maps,
            panels,
            provenance_tag="periodic BC",
            shell_theory=ShellTheory.FSDT,
        )

    n_panels = len(panels)
    ref: Any = None
    for panel_res in all_panel_results:
        if panel_res:
            ref = panel_res[len(panel_res) // 2]
            break

    bundle = SectionShellRecoveryBundle(
        panels=panels,
        booms=pb.booms,
        webs_geom=pb.webs_geom,
        q_tot=[],
        sig_p=[],
        sig_b=[],
        q0=[],
        props=None,
        y_sc=float(y_e),
        z_sc=float(z_e),
        y_e=float(y_e),
        z_e=float(z_e),
        areas=[],
        I_omega=0.0,
        gamma_y=0.0,
        gamma_z=0.0,
        GA_y=0.0,
        GA_z=0.0,
        q_primary=[],
        q_warp=[],
        reference_resultants=ref,
        all_panel_results=all_panel_results,
        all_panel_diagnostics=[{} for _ in range(n_panels)],
        panel_abd=abd_p,
        panel_thickness_m=t_p,
        panel_G_eff=g_p,
        panel_labels=lab_p,
        shell_midline_strips=tuple(getattr(pb.shell_inputs, "midlines", ())),
        beam_mode_amplitudes=q_beam_physical,
    )

    return bundle


def run_section_tapered_periodic_bc_mitc4(
    pb: "TaperedPipelineBSection",
    *,
    resultants: BeamResultantVector | None = None,
    N: float = 0.0,
    Vy: float = 0.0,
    Vz: float = 0.0,
    My: float = 0.0,
    Mz: float = 0.0,
    T: float = 0.0,
    n_elements_per_panel: NElements = 10,
    target_element_length_m: float | None = None,
    min_elements_per_panel: int = MIN_ELEMENTS_PER_PANEL,
    bc_mode: str = "contour_support",
    interface_constraint_mode: str = "shared",
    loading_axis_yz: tuple[float, float] | None = None,
    transport_to_elastic_centroid: bool = True,
) -> SectionShellRecoveryBundle:
    """Restricted tapered periodic MITC4 solve using matched inboard/outboard sections."""
    from ..engine.homogenisation import (
        compute_elastic_centroid_from_panels,
        compute_k6_periodic_bc,
    )
    from ..engine.periodic_secondary_recovery import recover_resultants_from_modes

    if interface_constraint_mode not in {"shared", "transformed_basis", "shared_rotated"}:
        raise ValueError(
            "run_section_tapered_periodic_bc_mitc4 currently supports only interface_constraint_mode in {'shared', 'transformed_basis', 'shared_rotated'}"
        )

    panels_in = pb.inboard.panels
    panels_out = pb.outboard.panels
    if len(panels_in) != len(panels_out):
        raise ValueError("Tapered periodic solve requires identical inboard/outboard panel counts")

    zeros = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in panels_in]
    assembly_result = _cached_tapered_periodic_assembly(
        panels_in,
        panels_out,
        zeros,
        zeros,
        L_x=float(pb.L_x),
        n_elements_per_panel=n_elements_per_panel,
        target_element_length_m=target_element_length_m,
        min_elements_per_panel=min_elements_per_panel,
        bc_mode=bc_mode,
        interface_constraint_mode=interface_constraint_mode,
    )
    (
        K_global,
        f_global,
        node_meta,
        panel_maps,
        abd_p,
        t_p,
        g_p,
        lab_p,
        formulation_cache,
    ) = _unpack_assembly_result(assembly_result, panels_in)

    y_e, z_e = compute_elastic_centroid_from_panels(panels_in)
    y_e_out, z_e_out = compute_elastic_centroid_from_panels(panels_out)
    _at_ec, f_beam = resolve_beam_resultants_for_shell_k6(
        y_e=float(y_e),
        z_e=float(z_e),
        resultants=resultants,
        N=N, Vy=Vy, Vz=Vz, My=My, Mz=Mz, T=T,
        loading_axis_yz=loading_axis_yz,
        transport_to_elastic_centroid=transport_to_elastic_centroid,
    )

    K6, u_modes = compute_k6_periodic_bc(
        K_global,
        node_meta,
        panel_maps,
        panels_in,
        y_e,
        z_e,
        L_x=float(pb.L_x),
        topological=True,
        y_e_out=y_e_out,
        z_e_out=z_e_out,
    )

    f_beam = np.asarray(f_beam, dtype=float)
    try:
        q_beam_physical = np.linalg.solve(K6, f_beam)
    except np.linalg.LinAlgError:
        q_beam_physical = np.linalg.lstsq(K6, f_beam, rcond=None)[0]

    if formulation_cache is None:
        raise RuntimeError(
            "tapered periodic assembly missing ShellFormulationCache (R1/R2/R9 regression)"
        )
    all_panel_results = recover_resultants_from_modes(
        u_modes,
        q_beam_physical,
        formulation_cache,
        panel_maps,
        panels_in,
        provenance_tag="periodic BC",
        shell_theory=ShellTheory.FSDT,
        L_x=float(pb.L_x),
    )

    n_panels = len(panels_in)
    ref: Any = None
    for panel_res in all_panel_results:
        if panel_res:
            ref = panel_res[len(panel_res) // 2]
            break

    return SectionShellRecoveryBundle(
        panels=panels_in,
        booms=pb.inboard.booms,
        webs_geom=pb.inboard.webs_geom,
        q_tot=[],
        sig_p=[],
        sig_b=[],
        q0=[],
        props=None,
        y_sc=float(y_e),
        z_sc=float(z_e),
        y_e=float(y_e),
        z_e=float(z_e),
        areas=[],
        I_omega=0.0,
        gamma_y=0.0,
        gamma_z=0.0,
        GA_y=0.0,
        GA_z=0.0,
        q_primary=[],
        q_warp=[],
        reference_resultants=ref,
        all_panel_results=all_panel_results,
        all_panel_diagnostics=[{} for _ in range(n_panels)],
        panel_abd=abd_p,
        panel_thickness_m=t_p,
        panel_G_eff=g_p,
        panel_labels=lab_p,
        shell_midline_strips=tuple(getattr(pb.inboard.shell_inputs, "midlines", ())),
        beam_mode_amplitudes=q_beam_physical,
    )


# Ownership of solver callables lives in :mod:`..engine.shell_theory_registry` (R4).
from ..engine.shell_theory_registry import (
    get_periodic_solver as _get_periodic_solver,
    periodic_solver_map as _periodic_solver_map,
    register_periodic_solver as _register_periodic_solver_unified,
)

_register_periodic_solver_unified(ShellTheory.FSDT, run_section_periodic_bc_mitc4)


class _PeriodicSolverMap(dict):
    """Dict-like view over the unified theory registry's periodic solvers."""

    def __getitem__(self, key: ShellTheory) -> Callable[..., SectionShellRecoveryBundle]:
        try:
            return _get_periodic_solver(key)
        except KeyError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key: object) -> bool:
        return key in _periodic_solver_map()

    def __setitem__(self, key: ShellTheory, value: Callable[..., SectionShellRecoveryBundle]) -> None:
        _register_periodic_solver_unified(key, value)

    def keys(self):
        return _periodic_solver_map().keys()

    def items(self):
        return _periodic_solver_map().items()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


_PERIODIC_SOLVERS: dict[ShellTheory, Callable[..., SectionShellRecoveryBundle]] = _PeriodicSolverMap()


def register_periodic_solver(
    theory: ShellTheory, solve_fn: Callable[..., SectionShellRecoveryBundle]
) -> None:
    """Register ``solve_fn`` as the periodic-BC handler for ``theory``."""
    _register_periodic_solver_unified(theory, solve_fn)


__all__ = [
    "run_section_periodic_bc",
    "run_section_periodic_bc_mitc4",
    "run_section_tapered_periodic_bc_mitc4",
    "register_periodic_solver",
    "get_periodic_geometry_state",
    "clear_periodic_caches",
]
