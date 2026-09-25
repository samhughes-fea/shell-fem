"""Section linear static simulation runner (mirror 1d-fem LinearStaticSimulationRunner)."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from blade_structure_utilities.geometry_utilities.section_contract.beam_resultant_vector import (
    BEAM_LOAD_AXIS_YZ,
    FRAME_SECTION_YZ,
    BeamResultantVector,
)
from blade_structure_utilities.geometry_utilities.section_contract.shell_midline_strip import (
    ShellMidlineStrip,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import ShellFormulationCache
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellTheory
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.mesh.api.shell_panel_topology import (
    ShellPanelTopology,
)
from blade_structure_utilities.load_utilities.section_loads import (
    MaterialMap,
    SectionForceVector,
)
from blade_structure_utilities.mesh_utilities.section_mesh.strip.strip_mesh import (
    SectionStripMesh,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.solve_states import (
    ConstrainedShellSystem,
    DisassembledShellResults,
    GlobalShellSystem,
    PrimaryShellResults,
    ReconstructedShellField,
    ReducedShellSystem,
    SolvedReducedShellSystem,
)
# diagnostics
from ..processing.static.diagnostics.dof_provenance import dump_index_maps
from ..processing.static.diagnostics.solve_phase_timing import (
    PHASE_ASSEMBLE_GLOBAL_SYSTEM,
    PHASE_CONSTRAIN_GLOBAL_SYSTEM,
    PHASE_PREPARE_LOCAL_SYSTEM,
    PHASE_PRIMARY_RESULTS,
    PHASE_QUATERNARY_RESULTS,
    PHASE_RECONSTRUCT_GLOBAL_FIELD,
    PHASE_REDUCE_CONSTRAINED_SYSTEM,
    PHASE_SECONDARY_RESULTS,
    PHASE_SOLVE_REDUCED_SYSTEM,
    PHASE_TERTIARY_RESULTS,
    SolvePhaseCollector,
)
from ..processing.static.results.containers.diagnostics_bin import DiagnosticsBin
# operations
from ..processing.static.operations.assemble_global import AssembleGlobalShellSystem
from ..processing.static.operations.constrain_global import ConstrainGlobalShellSystem
from ..processing.static.operations.prepare_local import PrepareLocalShellSystem
from ..processing.static.operations.reconstruct_field import ReconstructGlobalShellField
from ..processing.static.operations.reduce_constrained import KrrModifier, ReduceConstrainedShellSystem
from ..processing.static.operations.solve_constrained import solve_constrained_system
from ..processing.static.operations.solve_reduced import SolveReducedShellSystem
# results
from ..processing.static.results.compute_primary import ShellPrimaryResultsOrchestrator
from ..processing.static.results.compute_quaternary import ShellQuaternaryResultsOrchestrator
from ..processing.static.results.compute_secondary import ShellSecondaryResultsOrchestrator
from ..processing.static.results.compute_tertiary import ShellTertiaryResultsOrchestrator
from ..processing.static.results.containers.container_hopper import (
    ShellIndexMapSet,
    ShellPrimaryResultSet,
    ShellQuaternaryResultSet,
    ShellSecondaryResultSet,
    ShellTertiaryResultSet,
)
from ..processing.static.results.containers.map_results import ShellMapEntry


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def section_solve_legacy_dict(result: "SectionLinearStaticResult") -> dict[str, object]:
    """Return a dict mirroring the former ``reduce_propagate_solve`` key set.

    Kept for diagnostics / external callers; prefer typed
    :class:`SectionLinearStaticResult` fields.
    """
    return {
        "d_global": result.d_global,
        "r_full": result.r_full,
        "T": result.T,
        "independent": result.independent,
        "fixed_sorted": result.fixed_sorted,
        "constrained_slaves": result.constrained_slaves,
        "constrained_masters": result.constrained_masters,
        "K_rr_factor": result.K_rr_factor,
    }


@dataclass
class SectionLinearStaticResult:
    """P→S→T→Q cascade bin for one linear-static worker solve.

    Use-case agnostic: callers read whichever tiers they need (e.g. Q for FI
    sizing, T for rainflow). This type is the cascade bin; ``CascadeBin`` is an
    alias.
    """

    d_global: np.ndarray
    r_full: np.ndarray
    T: sp.csr_matrix
    independent: np.ndarray
    fixed_sorted: list[int]
    constrained_slaves: set[int]
    constrained_masters: set[int]
    K_rr_factor: spla.SuperLU | None
    index_maps: ShellIndexMapSet = field(default_factory=ShellIndexMapSet)
    primary_results: ShellPrimaryResultSet | None = None
    secondary_results: ShellSecondaryResultSet | None = None
    tertiary_results: ShellTertiaryResultSet | None = None
    quaternary_results: ShellQuaternaryResultSet | None = None
    diagnostics: DiagnosticsBin | None = None
    cache: ShellFormulationCache | None = None
    global_sys: GlobalShellSystem | None = None
    constrained: ConstrainedShellSystem | None = None
    reduced: ReducedShellSystem | None = None
    solved: SolvedReducedShellSystem | None = None
    field: ReconstructedShellField | None = None
    primary: PrimaryShellResults | None = None
    disassembled: DisassembledShellResults | None = None

    def legacy_dict(self) -> dict[str, object]:
        return section_solve_legacy_dict(self)


# Public contract name for the result cascade bin (P→S→T→Q).
CascadeBin = SectionLinearStaticResult


class SectionLinearStaticRunner:
    """Classical FEM facade: bound MaterialZones + external force vector → Ke/Fe → cascade.

    Use-case agnostic (no sizing/fatigue modes). Constitutive bind
    (``SectionBindRunner``) and external force-vector shaping
    (``SectionLoadsRunner`` / ``BeamResultantVector`` → ``SectionForceVector``)
    happen **outside** this worker. Thinwall path requires
    ``section_force_vector=``; the runner only applies external → global ``f``,
    forms ``Ke``/``Fe``, assembles, BCs, solves, and fills P→S→T→Q via
    :meth:`execute` / :meth:`get_results`::

        # L4: frozen zones on panel_topology + frozen SFV from SectionLoadsRunner
        static = SectionLinearStaticRunner(
            panel_topology=panel_topology,
            resultants=resultants,
            section_strip_mesh=section_strip_mesh,
            section_force_vector=sfv,
            load_channel="thinwall",
        ).execute()
        bin = static.get_results()  # CascadeBin

    Compatibility wrappers :meth:`from_cache` / :meth:`from_global` require a
    complete explicit constraint dict and never infer station policy. Prefer
    the internal kernel :func:`solve_constrained_system` for low-level tests.

    Prefer ``section_strip_mesh=`` from ``mesh_utilities.section_mesh`` so density
    stays outside FE. Remesh-via-density kwargs are rejected — callers (or FE
    shims) must build the strip mesh before construct/execute.
    """

    def __init__(
        self,
        *,
        panel_topology: ShellPanelTopology | None = None,
        resultants: BeamResultantVector | None = None,
        N: float = 0.0,
        Vy: float = 0.0,
        Vz: float = 0.0,
        My: float = 0.0,
        Mz: float = 0.0,
        T: float = 0.0,
        about_xy: tuple[float, float] | None = None,
        about_yz: tuple[float, float] | None = None,
        torsion_about_shear_centre: bool = False,
        shear_centre_E_mean: float = 70e9,
        traction_field: Any | None = None,
        traction_points_yz: np.ndarray | None = None,
        n_elements_per_panel: Any = None,
        target_element_length_m: float | None = None,
        min_elements_per_panel: int | None = None,
        section_strip_mesh: SectionStripMesh | None = None,
        L_x: float = 1.0,
        interface_constraint_mode: str = "shared",
        cap_tie_mode: str = "skin_offset",
        bc_mode: str = "contour_support",
        shell_theory: ShellTheory = ShellTheory.FSDT,
        load_channel: str = "thinwall",
        shell_midlines: list[ShellMidlineStrip] | tuple[ShellMidlineStrip, ...] | None = None,
        load_materials: MaterialMap | None = None,
        section_force_vector: SectionForceVector | None = None,
        compute_section_deformation_1a: bool = False,
        compute_gaussian: bool = False,
        compute_nodal: bool = False,
    ) -> None:
        self._phase_collector = SolvePhaseCollector()
        self._collector_session_open = False
        self.cache: ShellFormulationCache | None = None
        self.K: sp.spmatrix | None = None
        self.f: np.ndarray | None = None
        self.global_sys: GlobalShellSystem | None = None
        self.constrained: ConstrainedShellSystem | None = None
        self.reduced: ReducedShellSystem | None = None
        self.solved: SolvedReducedShellSystem | None = None
        self.field: ReconstructedShellField | None = None
        self.primary: PrimaryShellResults | None = None
        self.disassembled: DisassembledShellResults | None = None
        self.primary_results: ShellPrimaryResultSet | None = None
        self.secondary_results: ShellSecondaryResultSet | None = None
        self.tertiary_results: ShellTertiaryResultSet | None = None
        self.quaternary_results: ShellQuaternaryResultSet | None = None
        self.index_maps: ShellIndexMapSet = ShellIndexMapSet()
        self.T: sp.csr_matrix | None = None
        self.fixed: set[int] = set()
        self.constraints: dict[int, list[tuple[int, float]]] = {}
        self.n_gdof: int = 0
        self.factor: spla.SuperLU | None = None
        self._solver: SolveReducedShellSystem | None = None
        self._k_rr_modifier: KrrModifier | None = None
        self._last: SectionLinearStaticResult | None = None
        self._executed = False

        # Station-facade inputs (lazy; used by execute())
        self._panel_topology = panel_topology
        self._resultants_in = resultants
        self._N = float(N)
        self._Vy = float(Vy)
        self._Vz = float(Vz)
        self._My = float(My)
        self._Mz = float(Mz)
        self._T_load = float(T)
        if about_yz is not None:
            warnings.warn(
                "SectionLinearStaticRunner(about_yz=...) is deprecated; use about_xy=...",
                DeprecationWarning,
                stacklevel=2,
            )
            if about_xy is not None:
                raise TypeError("Pass only one of about_xy= or about_yz=")
            about_xy = about_yz
        self._about_xy = about_xy
        self._torsion_about_shear_centre = bool(torsion_about_shear_centre)
        self._shear_centre_E_mean = float(shear_centre_E_mean)
        self._traction_field = traction_field
        self._traction_points_yz = traction_points_yz
        self._n_elements_per_panel = n_elements_per_panel
        self._target_element_length_m = target_element_length_m
        self._min_elements_per_panel = min_elements_per_panel
        self._section_strip_mesh = section_strip_mesh
        self._L_x = float(L_x)
        self._interface_constraint_mode = str(interface_constraint_mode)
        self._cap_tie_mode = str(cap_tie_mode or "skin_offset")
        self._bc_mode = str(bc_mode)
        self.constraint_set: Any | None = None
        self.f_element: np.ndarray | None = None
        self.f_station: np.ndarray | None = None
        self._shell_theory = shell_theory
        self._load_channel = str(load_channel or "thinwall")
        self._shell_midlines = shell_midlines
        self._load_materials = load_materials
        self._section_force_vector_in = section_force_vector
        self._compute_section_deformation_1a = bool(compute_section_deformation_1a)
        self._compute_gaussian = bool(compute_gaussian)
        self._compute_nodal = bool(compute_nodal)

        # Populated by execute()
        self.assembly: Any | None = None
        self.resultants: BeamResultantVector | None = None
        self.f_nodal: np.ndarray | None = None
        self.used_traction: bool = False
        self.section_force_vector: Any | None = None
        self.load_channel_used: str = "thinwall"

    @property
    def cap_tie_report(self) -> Any | None:
        """Deprecated view of ``constraint_set.report.cap_tie``."""
        report = getattr(self.constraint_set, "report", None)
        return getattr(report, "cap_tie", None)

    @classmethod
    def from_global(
        cls,
        K_global: sp.spmatrix,
        f_global: np.ndarray,
        constraints: dict[int, list[tuple[int, float]]],
        fixed: set[int],
        n_gdof: int,
        *,
        K_rr_factor: spla.SuperLU | None = None,
    ) -> SectionLinearStaticRunner:
        runner = cls()
        runner.K = K_global
        runner.f = np.asarray(f_global, dtype=float)
        runner.global_sys = GlobalShellSystem(K=K_global, f=runner.f)
        runner.constraints = dict(constraints)
        runner.fixed = set(fixed)
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
            section_constraint_set_from_solver_inputs,
        )

        runner.constraint_set = section_constraint_set_from_solver_inputs(
            constraints, fixed
        )
        runner.n_gdof = n_gdof
        runner.factor = K_rr_factor
        return runner

    @classmethod
    def from_cache(
        cls,
        cache: ShellFormulationCache,
        n_gdof: int,
        constraints: dict[int, list[tuple[int, float]]],
        fixed: set[int],
    ) -> SectionLinearStaticRunner:
        runner = cls()
        runner.cache = cache
        runner.constraints = dict(constraints)
        runner.fixed = set(fixed)
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
            section_constraint_set_from_solver_inputs,
        )

        runner.constraint_set = section_constraint_set_from_solver_inputs(
            constraints, fixed
        )
        runner.n_gdof = n_gdof
        return runner

    @staticmethod
    def _resolve_traction_overlay(
        *,
        traction_field: Any | None,
        traction_points_yz: np.ndarray | None,
    ) -> tuple[Any | None, np.ndarray | None, bool]:
        """Validate optional traction kwargs; reject half-specified input."""
        has_field = traction_field is not None
        has_points = traction_points_yz is not None
        if has_field ^ has_points:
            raise ValueError(
                "traction_field and traction_points_yz must be supplied together "
                "(got one without the other)."
            )
        if not has_field:
            return None, None, False
        pts = np.asarray(traction_points_yz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(
                f"traction_points_yz must have shape (n, 2); got {pts.shape!r}"
            )
        return traction_field, pts, True

    def _prepare_station_system(self) -> tuple[np.ndarray, list[Any], str]:
        """Assemble section + nodal RHS from station facade inputs.

        Returns ``(f_nodal, zones, provenance_tag)``.
        """
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.compose import (
            build_section_constraint_set,
        )
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.constraints.types import (
            constraint_equations_to_solver_dict,
        )
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.operations.section_assembly import (
            assemble_section_from_zones,
        )

        panel_topology = self._panel_topology
        if panel_topology is None:
            raise RuntimeError("station prepare requires panel_topology")
        if self._shell_theory is not ShellTheory.FSDT:
            raise ValueError(
                "SectionLinearStaticRunner.execute() currently supports "
                f"ShellTheory.FSDT only; got {self._shell_theory!r}"
            )
        about_xy = self._about_xy
        if self._torsion_about_shear_centre and about_xy is not None:
            raise ValueError(
                "Pass either torsion_about_shear_centre=True or about_xy=..., not both"
            )
        if self._torsion_about_shear_centre:
            from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.shear_centre_loading_axis import (
                shear_centre_about_yz_from_panel_topology,
            )

            about_xy = shear_centre_about_yz_from_panel_topology(
                panel_topology, E_mean=float(self._shear_centre_E_mean)
            )

        tf, pts, used_traction = self._resolve_traction_overlay(
            traction_field=self._traction_field,
            traction_points_yz=self._traction_points_yz,
        )

        zones = list(panel_topology.zones)
        strip_mesh = self._section_strip_mesh
        if strip_mesh is None:
            raise ValueError(
                "section_strip_mesh is required; build it with "
                "mesh_utilities.section_mesh.build_section_strip_mesh "
                "(density is owned outside FE)."
            )
        if (
            self._n_elements_per_panel is not None
            or self._target_element_length_m is not None
            or self._min_elements_per_panel is not None
        ):
            raise ValueError(
                "Density kwargs are not accepted on SectionLinearStaticRunner; "
                "pass a prebuilt section_strip_mesh= instead."
            )
        nx = [np.zeros(len(np.asarray(p.s, dtype=float))) for p in zones]
        nxy = [np.zeros_like(a) for a in nx]
        asm = assemble_section_from_zones(
            zones,
            nx,
            nxy,
            L_x=self._L_x,
            interface_constraint_mode=self._interface_constraint_mode,
            shell_theory=self._shell_theory,
            phase_collector=self._phase_collector,
            section_strip_mesh=strip_mesh,
        )
        ndof = int(asm.cache.n_dof_node)
        constraint_set = build_section_constraint_set(
            asm,
            interface_constraint_mode=self._interface_constraint_mode,
            cap_tie_mode=self._cap_tie_mode,
            bc_mode=self._bc_mode,
        )
        constraints = constraint_equations_to_solver_dict(constraint_set.equations)
        fixed = set(constraint_set.fixed_dofs)
        self.constraint_set = constraint_set
        self.f_element = np.asarray(asm.f_element, dtype=float)

        resultants = self._resultants_in
        if resultants is None:
            origin = about_xy if about_xy is not None else BEAM_LOAD_AXIS_YZ
            resultants = BeamResultantVector.from_named(
                N=self._N,
                Vy=self._Vy,
                Vz=self._Vz,
                My=self._My,
                Mz=self._Mz,
                T=self._T_load,
                frame=FRAME_SECTION_YZ,
                about_xy=origin,
            )
        elif about_xy is not None:
            resultants = resultants.transported_to(float(about_xy[0]), float(about_xy[1]))

        channel = str(self._load_channel or "thinwall").strip()
        sfv = self._section_force_vector_in
        channel_l = channel.lower()
        thinwall_aliases = {"thinwall_shaped", "thinwall", "c1"}
        equipollent_aliases = {"equipollent", "a", "channel_a"}
        surface_aliases = {"surface_traction", "traction", "b", "channel_b"}

        if channel_l in thinwall_aliases:
            from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.section_force_vector_to_nodal_load import (
                section_force_vector_to_nodal_load,
            )

            if sfv is None:
                raise ValueError(
                    "load_channel='thinwall' requires a precomputed "
                    "section_force_vector (external force vector / SectionForceVector). "
                    "Shape BeamResultantVector → SectionForceVector outside the FE worker "
                    "via load_utilities.section_loads.distribute_beam_resultants; "
                    "this runner only applies external → global force vector f. "
                    "Use load_channel='equipollent' for mesh-native Channel A."
                )
            f = section_force_vector_to_nodal_load(
                asm.node_meta,
                sfv,
                panel_maps=asm.panel_maps,
                zones=zones,
                ndof=ndof,
                L_x=self._L_x,
            )
            channel_used = "thinwall"
            provenance_tag = "SectionLinearStaticRunner.execute + thinwall"
            # Do not stack matching face traction on thinwall-shaped cut resultants
            if used_traction:
                used_traction = False
        elif channel_l in surface_aliases:
            from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.section_force_vector_to_nodal_load import (
                section_force_vector_to_nodal_load,
            )

            if sfv is None:
                raise ValueError(
                    "load_channel='surface_traction' requires a precomputed "
                    "section_force_vector (external force vector / SectionForceVector). "
                    "Shape TractionField → SectionForceVector outside the FE worker "
                    "via load_utilities.section_loads.distribute_loads("
                    "channel='surface_traction', ...) or "
                    "traction_field_to_section_force_vector; this runner only applies "
                    "external → global force vector f."
                )
            f = section_force_vector_to_nodal_load(
                asm.node_meta,
                sfv,
                panel_maps=asm.panel_maps,
                zones=zones,
                ndof=ndof,
                L_x=self._L_x,
            )
            channel_used = "surface_traction"
            provenance_tag = "SectionLinearStaticRunner.execute + surface_traction"
            # Face traction already folded into SFV outside FE.
            if used_traction:
                used_traction = False
        elif channel_l in equipollent_aliases:
            from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.loading.section_force_vector_to_nodal_load import (
                section_force_vector_to_nodal_load,
            )

            if sfv is None:
                raise ValueError(
                    "load_channel='equipollent' requires a precomputed "
                    "section_force_vector (external force vector / SectionForceVector). "
                    "Shape BeamResultantVector → SectionForceVector outside the FE worker "
                    "via load_utilities.section_loads.equipollent_shape_on_midlines; "
                    "this runner only applies external → global force vector f. "
                    "beam_resultants_to_nodal_load remains a regression utility, not a "
                    "second production FE path."
                )
            f = section_force_vector_to_nodal_load(
                asm.node_meta,
                sfv,
                panel_maps=asm.panel_maps,
                zones=zones,
                ndof=ndof,
                L_x=self._L_x,
            )
            channel_used = "equipollent"
            provenance_tag = "SectionLinearStaticRunner.execute + equipollent"
            if used_traction:
                used_traction = False
        else:
            raise ValueError(
                f"Unknown load_channel={channel!r}; expected "
                "thinwall|equipollent|surface_traction"
            )

        self.assembly = asm
        self.resultants = resultants
        self.f_nodal = f
        self.f_station = np.asarray(f, dtype=float)
        self.used_traction = used_traction
        self.section_force_vector = sfv
        self.load_channel_used = channel_used
        self.cache = asm.cache
        self.constraints = constraints
        self.fixed = set(fixed)
        self.n_gdof = int(asm.n_gdof)
        f_element = (
            np.asarray(self.f_element, dtype=float)
            if self.f_element is not None
            else np.zeros(self.n_gdof, dtype=float)
        )
        f_total = f_element + np.asarray(f, dtype=float)
        self.global_sys = GlobalShellSystem(
            K=asm.K_global,
            f=f_total,
            diagnostics=asm.diagnostics,
        )
        return f, zones, provenance_tag

    def execute(self) -> SectionLinearStaticRunner:
        """Station assemble → nodal load → full P→S→T→Q cascade. Returns ``self``."""
        if self._executed:
            return self
        if self._panel_topology is None:
            raise RuntimeError(
                "execute() requires station construction with panel_topology=...; "
                "use from_cache(...).solve(...) for the low-level FE path"
            )
        self._phase_collector.reset()
        self._collector_session_open = True
        f, zones, provenance_tag = self._prepare_station_system()
        assert self.assembly is not None
        self.solve(
            L_x=self._L_x,
            theory="fsdt",
            panel_maps=self.assembly.panel_maps,
            zones=zones,
            provenance_tag=provenance_tag,
            compute_gaussian=self._compute_gaussian,
            compute_nodal=self._compute_nodal,
        )
        if self._compute_section_deformation_1a:
            assert self.resultants is not None
            self.compute_section_deformation_average_1a(
                node_meta=self.assembly.node_meta,
                about_xy=self.resultants.about_xy,
                zones=zones,
                panel_maps=self.assembly.panel_maps,
            )
        if self._last is None or self._last.quaternary_results is None:
            raise RuntimeError("execute() did not populate quaternary_results")
        self._executed = True
        return self

    def get_results(self) -> SectionLinearStaticResult:
        """Return the full P→S→T→Q cascade bin. Requires :meth:`execute`."""
        if not self._executed or self._last is None:
            raise RuntimeError(
                "SectionLinearStaticRunner.execute() must be called before get_results()"
            )
        return self._last

    def prepare_local_system(self) -> ShellFormulationCache:
        if self.cache is None:
            raise RuntimeError("prepare_local_system requires from_cache()")
        with self._phase_collector.phase(PHASE_PREPARE_LOCAL_SYSTEM):
            self.cache = PrepareLocalShellSystem().run(self.cache)
        return self.cache

    def assemble_global_system(self) -> GlobalShellSystem:
        with self._phase_collector.phase(PHASE_ASSEMBLE_GLOBAL_SYSTEM):
            if self.global_sys is not None:
                if self.cache is not None:
                    self.index_maps.assembly_map = AssembleGlobalShellSystem.build_assembly_map(
                        self.cache, self.n_gdof
                    )
                elif not self.index_maps.assembly_map:
                    self.index_maps.assembly_map = [
                        ShellMapEntry(global_dof=d, independent=True) for d in range(self.n_gdof)
                    ]
                return self.global_sys
            if self.cache is not None:
                self.global_sys, self.index_maps.assembly_map = AssembleGlobalShellSystem().run(
                    self.cache, self.n_gdof,
                )
            else:
                raise RuntimeError("assemble_global_system requires cache or pre-set global_sys")
        return self.global_sys

    def constrain_global_system(self) -> ConstrainedShellSystem:
        with self._phase_collector.phase(PHASE_CONSTRAIN_GLOBAL_SYSTEM):
            self.constrained, self.index_maps.constraint_map = ConstrainGlobalShellSystem().run(
                self.constraints, self.fixed, self.n_gdof,
            )
            self.fixed = set(self.constrained.fixed)
        return self.constrained

    def reduce_constrained_system(self) -> ReducedShellSystem:
        if self.global_sys is None or self.constrained is None:
            raise RuntimeError("reduce_constrained_system requires assembled + constrained system")
        with self._phase_collector.phase(PHASE_REDUCE_CONSTRAINED_SYSTEM):
            self.reduced, self.index_maps.reduction_map = ReduceConstrainedShellSystem().run(
                self.global_sys,
                self.constrained,
                self.n_gdof,
                k_rr_modifier=self._k_rr_modifier,
                constraint_map=self.index_maps.constraint_map,
            )
            self.T = self.reduced.T
        return self.reduced

    def solve_reduced_system(self, *, f_reduced: np.ndarray | None = None) -> SolvedReducedShellSystem:
        if self.reduced is None:
            raise RuntimeError("solve_reduced_system requires reduced system")
        with self._phase_collector.phase(PHASE_SOLVE_REDUCED_SYSTEM):
            self._solver = SolveReducedShellSystem(self.reduced, factor=self.factor)
            self.solved = self._solver.solve(f_reduced)
            self.factor = self.solved.factor
        return self.solved

    def reconstruct_global_field(self) -> ReconstructedShellField:
        if self.reduced is None or self.solved is None or self.constrained is None:
            raise RuntimeError("reconstruct_global_field requires reduced + solved + constrained")
        with self._phase_collector.phase(PHASE_RECONSTRUCT_GLOBAL_FIELD):
            self.field, self.index_maps.reconstruction_map = ReconstructGlobalShellField().run(
                self.reduced,
                self.solved,
                self.constrained.fixed,
                self.n_gdof,
                reduction_map=self.index_maps.reduction_map,
            )
        return self.field

    def compute_primary_results(self) -> ShellPrimaryResultSet:
        if self.global_sys is None or self.field is None or self.constrained is None:
            raise RuntimeError("compute_primary_results requires global_sys + field + constrained")
        with self._phase_collector.phase(PHASE_PRIMARY_RESULTS):
            orchestrator = ShellPrimaryResultsOrchestrator(
                K=self.global_sys.K,
                f=self.global_sys.f,
                d_global=self.field.d_global,
                fixed=self.constrained.fixed,
                cache=self.cache,
                global_sys=self.global_sys,
                field=self.field,
                T=self.reduced.T if self.reduced else None,
                K_rr_factor=self.factor,
            )
            self.primary_results = orchestrator.compute()
            self.primary = orchestrator.primary
            self.disassembled = orchestrator.disassembled
        return self.primary_results

    def compute_secondary_results(
        self,
        *,
        L_x: float = 1.0,
        theory: str | None = None,
        panel_meta: list[dict[str, Any]] | None = None,
        panel_maps: list[Any] | None = None,
        zones: list[Any] | None = None,
        provenance_tag: str = "processing.compute_secondary",
        compute_gaussian: bool = False,
        compute_nodal: bool = False,
    ) -> ShellSecondaryResultSet:
        """Tier 2: shell resultants per element from tier 1's disassembled U_e.

        Invoked automatically by :meth:`solve`. Public for explicit recompute.
        ``compute_gaussian`` / ``compute_nodal`` populate secondary Gaussian /
        nodal resolution containers; both default off.
        """
        if self.primary_results is None or self.cache is None:
            raise RuntimeError("compute_secondary_results requires primary_results + cache")
        with self._phase_collector.phase(PHASE_SECONDARY_RESULTS):
            self.secondary_results = ShellSecondaryResultsOrchestrator(
                primary=self.primary_results,
                cache=self.cache,
                L_x=L_x,
                theory=theory,
                panel_meta=panel_meta,
                panel_maps=panel_maps,
                zones=zones,
                provenance_tag=provenance_tag,
                compute_gaussian=compute_gaussian,
                compute_nodal=compute_nodal,
            ).compute()
        return self.secondary_results

    def compute_tertiary_results(self) -> ShellTertiaryResultSet:
        """Tier 3: CLT ply-stress recovery. Invoked by :meth:`solve`; public for recompute."""
        if self.secondary_results is None or self.cache is None:
            raise RuntimeError("compute_tertiary_results requires secondary_results + cache")
        with self._phase_collector.phase(PHASE_TERTIARY_RESULTS):
            self.tertiary_results = ShellTertiaryResultsOrchestrator(
                secondary=self.secondary_results, cache=self.cache,
            ).compute()
        return self.tertiary_results

    def compute_quaternary_results(self) -> ShellQuaternaryResultSet:
        """Tier 4: failure indices. Invoked by :meth:`solve`; public for recompute."""
        if self.tertiary_results is None or self.cache is None:
            raise RuntimeError("compute_quaternary_results requires tertiary_results + cache")
        with self._phase_collector.phase(PHASE_QUATERNARY_RESULTS):
            self.quaternary_results = ShellQuaternaryResultsOrchestrator(
                tertiary=self.tertiary_results, cache=self.cache,
            ).compute()
        return self.quaternary_results

    def compute_section_deformation_average_1a(
        self,
        *,
        node_meta: Any,
        about_xy: tuple[float, float] = (0.0, 0.0),
        zones: list[Any] | None = None,
        panel_maps: list[Any] | None = None,
        thickness: float = 1.0,
    ) -> Any:
        """Primary/Global opt-in 1a: thickness×arc weighted section deformation average."""
        if self.field is None or self.primary_results is None:
            raise RuntimeError(
                "compute_section_deformation_average_1a requires solve() / primary_results"
            )
        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.results.compute_primary.section_deformation_average import (
            compute_section_deformation_average_1a,
        )

        ndof = int(self.cache.n_dof_node) if self.cache is not None else 6
        avg = compute_section_deformation_average_1a(
            d_global=np.asarray(self.field.d_global, dtype=float),
            node_meta=node_meta,
            ndof=ndof,
            about_xy=about_xy,
            zones=zones,
            panel_maps=panel_maps,
            thickness=thickness,
        )
        self.primary_results.global_results.diagnostics["section_deformation_1a"] = avg
        return avg

    def solve_linear_system_only(
        self,
        *,
        f_override: np.ndarray | None = None,
        k_rr_modifier: KrrModifier | None = None,
    ) -> np.ndarray:
        """Kinematics-only: assemble → constrain → reduce → solve → reconstruct ``d``.

        Does not compute primary/secondary/tertiary/quaternary. Use :meth:`solve`
        for the full results cascade.
        """
        self._k_rr_modifier = k_rr_modifier
        production_rhs_frozen = self.assembly is not None and self.global_sys is not None
        if self.cache is not None:
            if self.global_sys is None:
                self.prepare_local_system()
                self.assemble_global_system()
            else:
                self.assemble_global_system()
            if f_override is not None and not production_rhs_frozen:
                self.global_sys = GlobalShellSystem(
                    K=self.global_sys.K,
                    f=np.asarray(f_override, dtype=float),
                    diagnostics=self.global_sys.diagnostics,
                )
        elif self.global_sys is None and self.K is not None and self.f is not None:
            f_vec = self.f if f_override is None else np.asarray(f_override, dtype=float)
            self.global_sys = GlobalShellSystem(K=self.K, f=f_vec)
        elif f_override is not None and self.global_sys is not None and not production_rhs_frozen:
            self.global_sys = GlobalShellSystem(
                K=self.global_sys.K,
                f=np.asarray(f_override, dtype=float),
                diagnostics=self.global_sys.diagnostics,
            )
        if self.constraint_set is not None and self.global_sys is not None:
            result = solve_constrained_system(
                self.global_sys,
                self.constraint_set,
                self.n_gdof,
                k_rr_modifier=k_rr_modifier,
                factor=self.factor,
            )
            self.constrained = result.constrained
            self.reduced = result.reduced
            self.solved = result.solved
            self.field = result.field
            self.index_maps.constraint_map = result.constraint_map
            self.index_maps.reduction_map = result.reduction_map
            self.index_maps.reconstruction_map = result.reconstruction_map
            self.T = result.T
            self.factor = result.factor
            self.fixed = set(self.constrained.fixed)
            self._solver = SolveReducedShellSystem(self.reduced, factor=self.factor)
            self._solver.last_residual_rel = result.residual_rel
            self._solver.last_pivot_ratio = result.pivot_ratio
            self._solver.last_quality_ok = result.quality_ok
            self._solver.last_quality_reason = result.quality_reason
            return self.field.d_global
        self.constrain_global_system()
        self.reduce_constrained_system()
        self.solve_reduced_system()
        self.reconstruct_global_field()
        assert self.field is not None
        return self.field.d_global

    def solve(
        self,
        *,
        f_override: np.ndarray | None = None,
        k_rr_modifier: KrrModifier | None = None,
        L_x: float = 1.0,
        theory: str | None = None,
        panel_meta: list[dict[str, Any]] | None = None,
        panel_maps: list[Any] | None = None,
        zones: list[Any] | None = None,
        provenance_tag: str = "processing.compute_secondary",
        compute_gaussian: bool = False,
        compute_nodal: bool = False,
    ) -> SectionLinearStaticResult:
        """Always run kinematics + primary → secondary → tertiary → quaternary.

        Requires :meth:`from_cache` (formulation cache) so secondary recovery can
        run. For kinematics-only / factor-once multi-RHS use
        :meth:`solve_linear_system_only` or :meth:`solve_many`.
        """
        if self.cache is None:
            raise RuntimeError(
                "solve() requires from_cache() so the P→S→T→Q cascade can run; "
                "use solve_linear_system_only() for kinematics without a formulation cache"
            )
        if not self._collector_session_open:
            self._phase_collector.reset()
            self._collector_session_open = True
        try:
            return self._solve_cascade(
                f_override=f_override,
                k_rr_modifier=k_rr_modifier,
                L_x=L_x,
                theory=theory,
                panel_meta=panel_meta,
                panel_maps=panel_maps,
                zones=zones,
                provenance_tag=provenance_tag,
                compute_gaussian=compute_gaussian,
                compute_nodal=compute_nodal,
            )
        finally:
            self._collector_session_open = False

    def _solve_cascade(
        self,
        *,
        f_override: np.ndarray | None = None,
        k_rr_modifier: KrrModifier | None = None,
        L_x: float = 1.0,
        theory: str | None = None,
        panel_meta: list[dict[str, Any]] | None = None,
        panel_maps: list[Any] | None = None,
        zones: list[Any] | None = None,
        provenance_tag: str = "processing.compute_secondary",
        compute_gaussian: bool = False,
        compute_nodal: bool = False,
    ) -> SectionLinearStaticResult:
        self.solve_linear_system_only(f_override=f_override, k_rr_modifier=k_rr_modifier)
        self.compute_primary_results()
        self.compute_secondary_results(
            L_x=L_x,
            theory=theory,
            panel_meta=panel_meta,
            panel_maps=panel_maps,
            zones=zones,
            provenance_tag=provenance_tag,
            compute_gaussian=compute_gaussian,
            compute_nodal=compute_nodal,
        )
        self.compute_tertiary_results()
        self.compute_quaternary_results()
        assert self.primary is not None
        assert self.reduced is not None
        assert self.field is not None
        assert self.quaternary_results is not None
        constrained_slaves = set(self.constraints.keys())
        constrained_masters = {m for terms in self.constraints.values() for m, _ in terms}
        if _env_truthy("SHELL_DOF_PROVENANCE_DUMP"):
            self._dump_index_maps()

        from blade_structure_utilities.fem_utilities.section_shell_fea.processing.static.diagnostics.cascade_trust import (
            fill_cascade_trust,
        )

        solve_residual_rel = None
        solve_pivot_ratio = None
        solve_quality_ok = True
        solve_quality_reason = ""
        if self._solver is not None:
            solve_residual_rel = self._solver.last_residual_rel
            solve_pivot_ratio = self._solver.last_pivot_ratio
            solve_quality_ok = bool(self._solver.last_quality_ok)
            solve_quality_reason = str(self._solver.last_quality_reason or "")

        if not solve_quality_ok and self.quaternary_results is not None:
            nr = self.quaternary_results.needs_review
            if nr is not None:
                self.quaternary_results.needs_review = np.ones_like(np.asarray(nr), dtype=bool)
            elif self.tertiary_results is not None and self.tertiary_results.panel_index is not None:
                n = int(np.asarray(self.tertiary_results.panel_index).shape[0])
                self.quaternary_results.needs_review = np.ones(n, dtype=bool)

        diagnostics = fill_cascade_trust(
            self._phase_collector.freeze(),
            field=self.field,
            primary_results=self.primary_results,
            secondary_results=self.secondary_results,
            tertiary_results=self.tertiary_results,
            quaternary_results=self.quaternary_results,
            solve_residual_rel=solve_residual_rel,
            solve_pivot_ratio=solve_pivot_ratio,
            solve_quality_ok=solve_quality_ok,
            solve_quality_reason=solve_quality_reason,
        )
        strip = self._section_strip_mesh
        if strip is not None:
            diagnostics.mesh_n_total_elements = int(strip.n_total_elements)
        if self.constraint_set is not None:
            report = self.constraint_set.report
            diagnostics.constraint_metadata = {
                "interface_constraint_mode": report.interface_constraint_mode,
                "cap_tie_mode": report.cap_tie_mode,
                "bc_mode": report.bc_mode,
                "n_junction_equations": int(report.n_junction_equations),
                "n_cap_tie_equations": int(report.n_cap_tie_equations),
                "n_fixed_dofs": int(report.n_fixed_dofs),
                "n_cap_links": int(report.n_cap_links),
                "n_tie_slave_nodes": len(report.tie_slave_nodes),
                "n_junction_slave_dofs": len(report.junction_slave_dofs),
            }

        result = SectionLinearStaticResult(
            d_global=np.asarray(self.field.d_global, dtype=float),
            r_full=np.asarray(self.primary.r_full, dtype=float),
            T=self.reduced.T,
            independent=self.reduced.independent,
            fixed_sorted=sorted(self.constrained.fixed if self.constrained else self.fixed),
            constrained_slaves=constrained_slaves,
            constrained_masters=constrained_masters,
            K_rr_factor=self.factor,
            index_maps=self.index_maps,
            primary_results=self.primary_results,
            secondary_results=self.secondary_results,
            tertiary_results=self.tertiary_results,
            quaternary_results=self.quaternary_results,
            diagnostics=diagnostics,
            cache=self.cache,
            global_sys=self.global_sys,
            constrained=self.constrained,
            reduced=self.reduced,
            solved=self.solved,
            field=self.field,
            primary=self.primary,
            disassembled=self.disassembled,
        )
        self._last = result
        self.fixed.clear()
        self.fixed.update(result.fixed_sorted)
        return result

    def solve_many(
        self,
        f_vectors: list[np.ndarray],
        *,
        k_rr_modifier: KrrModifier | None = None,
    ) -> list[np.ndarray]:
        """Factor once; solve multiple RHS for displacements only (no P→Q cascade)."""
        if not f_vectors:
            return []
        first_d = self.solve_linear_system_only(
            f_override=f_vectors[0], k_rr_modifier=k_rr_modifier,
        )
        out = [first_d]
        if self.factor is None or self.reduced is None or self._solver is None:
            return [np.zeros(self.n_gdof)] * len(f_vectors)
        for f_vec in f_vectors[1:]:
            if self.global_sys is None:
                break
            f_r = np.asarray(self.reduced.T.T @ f_vec, dtype=float)
            solved = self._solver.solve(f_r)
            d = np.asarray(self.reduced.T @ solved.q, dtype=float)
            out.append(d)
        return out

    def _dump_index_maps(self) -> None:
        """Opt-in DOF provenance dump; see :mod:`..processing.diagnostics.dof_provenance`."""
        dump_index_maps(self.index_maps)


__all__ = [
    "SectionLinearStaticResult",
    "SectionLinearStaticRunner",
    "section_solve_legacy_dict",
]
