"""
FE kernel worker: section geometry + panel materials + loads → F=KU → cascade bin.

Use-case agnostic. Prefer::

    bin = SectionLinearStaticRunner(...).execute().get_results()  # CascadeBin / P→S→T→Q

Orchestration (sizing loops, rainflow, parallelism) lives in use-case packages.
See package README. Public strip meshing:
``build_shell_mesh_inputs`` → ``build_section_v2`` → ``build_section_strip_mesh``.
"""

from __future__ import annotations

from .pre_processing.mesh.section_strip_mesh import (
    SectionStripMesh,
    StripJunctionCluster,
    StripPanelMesh,
    build_section_strip_mesh,
)
from .pre_processing.mesh.shell_inputs_from_section import (
    ShellMeshInputs,
    build_shell_mesh_inputs,
)
from .pre_processing.mesh.topology import build_section_v2
from .pre_processing.containers.types import (
    BeamSectionConstitutiveInput,
    FieldProvenance,
    ProvenanceKind,
    ShellAnalysisRegime,
    ShellPanelBucklingAssessment,
    ShellPanelBucklingInput,
    ShellPanelResultants,
    ShellRecoveryIsotropicPoint,
    ShellRecoveryMaterialPoint,
    ShellRecoverySectionPointSet,
    ShellSectionStationAnalysis,
    ShellSectionStationGeometry,
    ShellStationBucklingBundle,
    ShellStationProvenance,
    ShellTheory,
    SectionShellRecoveryBundle,
    supported_shell_theories,
    validate_shell_configuration,
)
from .simulation_runner.api.linear_static import run_section_linear_static_from_resultants
from .simulation_runner.api.station_recovery import (
    run_station_hashin,
    run_station_linear_static_hashin,
)
from .processing.static.results.containers.diagnostics_bin import DiagnosticsBin
from .simulation_runner.section_linear_static_runner import (
    CascadeBin,
    SectionLinearStaticResult,
    SectionLinearStaticRunner,
)
from .interface.recovery_point_set_builder import (
    build_shell_station_analysis_with_recovery_point_set,
    extract_shell_laminate_recovery_data,
)
from .interface.station_analysis_builder import build_station_analysis_from_shell_bundle
from .interface.smoke_station_builder import build_smoke_shell_station_analysis
from .interface.structural_station_constitutive import build_structural_grid_shell_constitutive
from .interface.constitutive_adapter import (
    build_shell_constitutive_input,
    build_shell_constitutive_stack,
    build_shell_constitutive_stack_from_legacy_section_properties,
)
from .processing.static.results.compute_tertiary.local_clpt_shell import (
    solve_station_clpt_shell,
    StationCLPTShellResult,
)

__all__ = [
    "build_section_strip_mesh",
    "SectionStripMesh",
    "StripJunctionCluster",
    "StripPanelMesh",
    "ShellMeshInputs",
    "build_section_v2",
    "build_shell_mesh_inputs",
    "BeamSectionConstitutiveInput",
    "FieldProvenance",
    "ProvenanceKind",
    "ShellAnalysisRegime",
    "ShellPanelBucklingAssessment",
    "ShellPanelBucklingInput",
    "ShellPanelResultants",
    "ShellRecoveryIsotropicPoint",
    "ShellRecoveryMaterialPoint",
    "ShellRecoverySectionPointSet",
    "ShellSectionStationAnalysis",
    "ShellSectionStationGeometry",
    "ShellStationBucklingBundle",
    "ShellStationProvenance",
    "ShellTheory",
    "SectionShellRecoveryBundle",
    "supported_shell_theories",
    "validate_shell_configuration",
    "build_shell_constitutive_input",
    "build_shell_constitutive_stack",
    "build_shell_constitutive_stack_from_legacy_section_properties",
    "CascadeBin",
    "DiagnosticsBin",
    "SectionLinearStaticResult",
    "SectionLinearStaticRunner",
    "run_section_linear_static_from_resultants",
    "run_station_hashin",
    "run_station_linear_static_hashin",
    "build_shell_station_analysis_with_recovery_point_set",
    "extract_shell_laminate_recovery_data",
    "build_station_analysis_from_shell_bundle",
    "build_smoke_shell_station_analysis",
    "build_structural_grid_shell_constitutive",
    "solve_station_clpt_shell",
    "StationCLPTShellResult",
]
