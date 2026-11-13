"""shell_fem - layered shell and solid finite elements for composite structures.

Public API overview
-------------------
Elements        ShellElement: Quad4ShellElement (alias CompositePlateElement), Tri3ShellElement,
                Quad8ShellElement, CurvedQuad8ShellElement, Tri6ShellElement; ElementRegistry;
                layered solids LayeredHex8Element, LayeredHex20Element (solid_block,
                couple_shell_to_solid)
Lay-up          Layup, PlyDefinition, PlyDrop, ReferenceDirection, AngleField, TabulatedDrape
Orientation     FixedDirection, CylindricalDirection, FieldDirection
Solution        FEAMesh, BoundaryConditions, LoadCase, LoadCombination, StructuralGlobalSolver
Meshing         BladeBuilder (airfoil skins, spar caps, shear webs, from the
                implicit_geometry_kernel types Airfoil, Station, SkinRegion and ShearWeb, which
                are re-exported here), RectangularPlateBuilder, CylindricalPanelBuilder,
                GmshPlateBuilder, GmshMeshImporter (optional gmsh dependency)
Connections     add_reference_node, rigid_link (RBE2), distributing_link (RBE3)
Beam properties beam_section_properties: beam stiffness extracted from a shell model
                (Timoshenko 6 x 6 with all couplings, classical 4 x 4, optional
                Timoshenko-Vlasov 7 x 7 with EΓ; shear/tension centres),
                transform_beam_stiffness
Results         ResultsExtractor (Gauss point, element, node, ply, solid and global levels);
                write_vtu, write_pvd, VTKExporter (ParaView output, per-ply cells)
Post-processing ExactStressRecoverer, EquilibriumStressRecoverer (sigma_z and interlaminar
                shear from 3D equilibrium), SmearedCrackDegradation, ProgressiveDamageAnalysis
Kernels         NativePythonKernelAdapter, CalculiXKernelAdapter, NastranKernelAdapter
Verification    StressDistributionVerifier, ModelChecker (pre-solve checks)

The ply and laminate behaviour (materials, Ply, Laminate, ABD, the kinematic
theories CLPTModel / FSDTModel / RefinedZigZagModel / LayerwiseModel, hygrothermal
states, failure criteria, PaganoPlate) is ``laminate_constitutive_model``; its public
names are re-exported here, so a model can be built from this package alone.
Section theory (thin-wall sections, torsion, GBT) lives in
``anisotropic_section_model``; this package never imports it.
"""

from laminate_constitutive_model import *  # noqa: F401,F403  (materials, laminates, kinematics, failure)
from .laminate_benchmark_stress import DistributionErrorStatistics, StressDistributionVerifier, ThicknessProfile
from .laminate_equilibrium import EquilibriumStressRecoverer, InterfaceCheck
from .laminate_model_check import ModelChecker, ModelIssue, ModelReport
from .laminate_elements import (
    DEFAULT_ELEMENTS,
    CompositePlateElement,
    ElementRegistry,
    Quad4ShellElement,
    Quad8ShellElement,
    ShellElement,
    Tri3ShellElement,
    shape_functions,
)
from .laminate_mesh_builders import CylindricalPanelBuilder, NamedMesh, RectangularPlateBuilder
from .laminate_orientation import CylindricalDirection, FieldDirection, FixedDirection
from .laminate_gmsh import (
    CircularHole,
    GmshMesh,
    GmshMeshImporter,
    GmshPlateBuilder,
    GmshSession,
    GmshUnavailableError,
    UnsupportedMeshError,
)
from .kernels import (
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
from .laminate_progressive_damage import (
    DamageState,
    DegradationStrategy,
    ProgressiveDamageAnalysis,
    ProgressiveDamageResult,
    SmearedCrackDegradation,
)
from .laminate_solver_pipeline import (
    DOF,
    BodyLoad,
    BoundaryConditions,
    EnrichedDOF,
    LineLoad,
    LoadCombination,
    combine_solutions,
    ConstraintEliminator,
    DenseLinearSolver,
    FEAMesh,
    GlobalAssembler,
    LinearConstraint,
    LinearSystemSolver,
    LoadCase,
    SingularStiffnessError,
    SolutionResult,
    SparseLinearSolver,
    StructuralGlobalSolver,
)
from .laminate_results import (
    ElementField,
    GaussPointField,
    GlobalTotals,
    NodalField,
    PlyPointField,
    ResultsExtractor,
    SolidPointField,
    extrapolation_matrix,
)
from .laminate_stress_recovery import ExactStressRecoverer, PointStressState, StressRecoverer
from .laminate_curved_shell import CurvedQuad8ShellElement, CurvedShellElement, Tri6ShellElement
from .laminate_layup import AngleField, Layup, PlyDefinition, PlyDrop, ReferenceDirection, TabulatedDrape
from .laminate_solid import (
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
from .laminate_vtk import VTKExporter, write_pvd, write_vtu
from .laminate_blade import (
    Airfoil,
    BeamSectionProperties,
    BladeBuilder,
    BladeMesh,
    ShearWeb,
    SkinRegion,
    Station,
    beam_section_properties,
    transform_beam_stiffness,
)
from .laminate_blade_kernel_mesh import PcompAssignment, solve_kernel_pcomp
from .laminate_connectors import add_reference_node, distributing_link, rigid_link

__version__ = "0.2.0"

import types as _types

__all__ = sorted(
    name for name, value in dict(globals()).items()
    if not name.startswith("_") and not isinstance(value, _types.ModuleType)
)
