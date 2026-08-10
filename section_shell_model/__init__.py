"""
FE kernel worker: element ``Ke``/``Fe`` → assemble / solve → cascade bin.

Use-case agnostic. Prefer::

    bin = SectionLinearStaticRunner(...).execute().get_results()  # CascadeBin / P→S→T→Q

**Target contract:** prebuilt mesh with constitutive data already assigned;
this worker owns formulation, assembly, BCs, solve, and P→S→T→Q recovery.
Strip meshing lives in ``mesh_utilities.section_mesh``. Constitutive bind
lives in ``material_utilities.adapters`` (``bind_shell_panels``). Orchestration
(sizing loops, rainflow, parallelism) lives in use-case packages.
"""

from __future__ import annotations

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
    "build_shell_station_analysis_with_recovery_point_set",
    "extract_shell_laminate_recovery_data",
    "build_station_analysis_from_shell_bundle",
    "build_smoke_shell_station_analysis",
    "build_structural_grid_shell_constitutive",
    "solve_station_clpt_shell",
    "StationCLPTShellResult",
]
