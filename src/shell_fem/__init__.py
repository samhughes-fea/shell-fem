"""shell_fem - layered shell and solid finite elements for composite structures.

Public API overview
-------------------
Elements        ShellElement: Quad4ShellElement (alias CompositePlateElement), Tri3ShellElement,
                Quad8ShellElement, CurvedQuad8ShellElement, Tri6ShellElement; ElementRegistry;
                layered solids LayeredHex8Element, LayeredHex20Element (solid_block,
                couple_shell_to_solid)
Lay-up          Layup, PlyDefinition, PlyDrop, ReferenceDirection, AngleField, TabulatedDrape
Orientation     FixedDirection, CylindricalDirection, FieldDirection
Solution        FEAMesh, BoundaryConditions, LoadCase, LoadCombination; LinearStaticRunner
                (stages AssembleGlobalSystem, ConstrainGlobalSystem, ReduceConstrainedSystem,
                SolveReducedSystem, ReconstructGlobalField, ComputePrimaryResults);
                StructuralGlobalSolver (the earlier interface to the same stages)
Meshing         BladeBuilder (airfoil skins, spar caps, shear webs, from_geometry and
                from_shell_mesh; PcompAssignment and solve_kernel_pcomp for a kernel
                shell mesh), the implicit_geometry_kernel types Airfoil, Station,
                SkinRegion and ShearWeb (re-exported here), RectangularPlateBuilder,
                CylindricalPanelBuilder, GmshPlateBuilder, GmshMeshImporter (optional gmsh)
Connections     add_reference_node, rigid_link (RBE2), distributing_link (RBE3)
Beam properties beam_section_properties: beam stiffness extracted from a shell model
                (Timoshenko 6 x 6 with all couplings, classical 4 x 4, optional
                Timoshenko-Vlasov 7 x 7 with EΓ; shear/tension centres),
                transform_beam_stiffness
Results         ResultsExtractor (Gauss point, element, node, ply, solid and global levels);
                write_vtu, write_pvd, VTKExporter (ParaView output, per-ply cells)
Post-processing ExactStressRecoverer, EquilibriumStressRecoverer (sigma_z and interlaminar
                shear from 3D equilibrium), SmearedCrackDegradation, ProgressiveDamageAnalysis
Backends        NativePythonKernelAdapter, CalculiXKernelAdapter, NastranKernelAdapter
Verification    StressDistributionVerifier, ModelChecker (pre-solve checks)

The ply and laminate behaviour (materials, Ply, Laminate, ABD, the kinematic
theories CLPTModel / FSDTModel / RefinedZigZagModel / LayerwiseModel, hygrothermal
states, failure criteria, PaganoPlate) is ``laminate_constitutive_model``; its public
names are re-exported here, so a model can be built from this package alone.
Package layout: ``pre_processing`` prepares the model (mesh, elements, properties,
loads), ``processing`` computes (one class per solve stage, one folder per analysis
type), ``simulation_runner`` sequences the stages and owns the results, and
``post_processing`` exports them. The 0.1 module paths (``shell_fem.laminate_*``,
``shell_fem.kernels``) still import, with a DeprecationWarning, until 0.3.0.

Section theory (thin-wall sections, torsion, GBT) lives in
``anisotropic_section_model``; this package never imports it.
"""

from laminate_constitutive_model import *  # noqa: F401,F403  (materials, laminates, kinematics, failure)
from shell_fem.verification.benchmark_stress import DistributionErrorStatistics, StressDistributionVerifier, ThicknessProfile
from shell_fem.processing.static.results.equilibrium import EquilibriumStressRecoverer, InterfaceCheck
from shell_fem.pre_processing.model_check import ModelChecker, ModelIssue, ModelReport
from shell_fem.pre_processing.element_library.shell import (
    DEFAULT_ELEMENTS,
    CompositePlateElement,
    ElementRegistry,
    Quad4ShellElement,
    Quad8ShellElement,
    ShellElement,
    Tri3ShellElement,
    shape_functions,
)
from shell_fem.pre_processing.mesh_library.builders import CylindricalPanelBuilder, NamedMesh, RectangularPlateBuilder
from shell_fem.pre_processing.property_library.orientation import CylindricalDirection, FieldDirection, FixedDirection
from shell_fem.pre_processing.mesh_library.gmsh import (
    CircularHole,
    GmshMesh,
    GmshMeshImporter,
    GmshPlateBuilder,
    GmshSession,
    GmshUnavailableError,
    UnsupportedMeshError,
)
from shell_fem.simulation_runner.backends import (
    AnalysisModel,
    CalculiXDeckWriter,
    CalculiXKernelAdapter,
    CalculiXResultReader,
    FEAKernel,
    KernelRegistry,
    KernelResult,
    KernelStatus,
    NastranDeckWriter,
    NastranF06Reader,
    NastranFieldFormatter,
    NastranKernelAdapter,
    NativePythonKernelAdapter,
    ProcessRunner,
    UnsupportedFeatureError,
    create_kernel,
)
from shell_fem.processing.progressive_damage.degradation import DamageState, DegradationStrategy, SmearedCrackDegradation
from shell_fem.simulation_runner.progressive_damage.progressive_damage_simulation import (
    DamageStepRecord,
    ProgressiveDamageAnalysis,
    ProgressiveDamageResult,
)
from shell_fem.pre_processing.containers.mesh import DOF, EnrichedDOF, FEAMesh
from shell_fem.pre_processing.containers.boundary_conditions import BoundaryConditions, LinearConstraint
from shell_fem.pre_processing.load_library.load_case import BodyLoad, LineLoad, LoadCase, LoadCombination
from shell_fem.processing.static.operations import (
    AssembleGlobalSystem,
    AssembleLoadVector,
    ConstrainedSystem,
    ConstrainGlobalSystem,
    ConstraintEliminator,
    DenseLinearSolver,
    FactorisedSystem,
    GlobalAssembler,
    GlobalSystem,
    LinearSystemSolver,
    ReconstructGlobalField,
    ReduceConstrainedSystem,
    ReducedSystem,
    SingularStiffnessError,
    SolveReducedSystem,
    SparseLinearSolver,
)
from shell_fem.processing.static.results.compute_primary import ComputePrimaryResults, combine_solutions
from shell_fem.simulation_runner.static.linear_static_simulation import (
    LinearStaticResults,
    LinearStaticRunner,
    StaticDiagnostics,
    StructuralGlobalSolver,
)
from shell_fem.processing.static.results.containers import (
    ElementField,
    GaussPointField,
    GlobalTotals,
    NodalField,
    PlyPointField,
    SolidPointField,
    SolutionResult,
)
from shell_fem.processing.static.results.results_extractor import ResultsExtractor, extrapolation_matrix
from shell_fem.processing.static.results.stress_recovery import ExactStressRecoverer, PointStressState, StressRecoverer
from shell_fem.pre_processing.element_library.curved_shell import CurvedQuad8ShellElement, CurvedShellElement, Tri6ShellElement
from shell_fem.pre_processing.property_library.layup import AngleField, Layup, PlyDefinition, PlyDrop, ReferenceDirection, TabulatedDrape
from shell_fem.pre_processing.element_library.solid import (
    DEFAULT_SOLIDS,
    LayeredHex8Element,
    LayeredHex20Element,
    LayeredSolidElement,
    SolidPoint,
    SolidRegistry,
    couple_shell_to_solid,
    merge_coincident_nodes,
    solid_block,
)
from shell_fem.post_processing.vtk import VTKExporter, write_pvd, write_vtu
from shell_fem.pre_processing.mesh_library.blade import Airfoil, BladeBuilder, BladeMesh, ShearWeb, SkinRegion, Station
from shell_fem.post_processing.beam_section_properties import (
    BeamSectionProperties,
    beam_section_properties,
    transform_beam_stiffness,
)
from shell_fem.pre_processing.mesh_library.blade_kernel_mesh import PcompAssignment
from shell_fem.simulation_runner.static.kernel_pcomp import solve_kernel_pcomp
from shell_fem.pre_processing.element_library.connectors import add_reference_node, distributing_link, rigid_link

__version__ = "0.2.0"

from shell_fem import _legacy

_legacy.install()

import types as _types

__all__ = sorted(
    name for name, value in dict(globals()).items()
    if not name.startswith("_") and not isinstance(value, _types.ModuleType)
)
