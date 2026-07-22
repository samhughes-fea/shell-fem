"""
Shell handoff DTOs: local panel resultants + per-field provenance for MVP auditing.

Compatibility facade — prefer importing from the leaf modules:
``station_types``, ``recovery_types``, ``buckling_types``, ``station_analysis_types``,
``shell_theory_capabilities``.
"""

from __future__ import annotations

from .buckling_types import (  # noqa: F401
    PanelBucklingResult,
    ShellPanelBucklingAssessment,
    ShellPanelBucklingInput,
    ShellStationBucklingBundle,
)
from .recovery_types import (  # noqa: F401
    FieldProvenance,
    ShellPanelResultants,
    ShellRecoveryIsotropicPoint,
    ShellRecoveryMaterialPoint,
    ShellRecoverySectionPointSet,
)
from .shell_theory_capabilities import (  # noqa: F401
    EXPERIMENTAL_SHELL_THEORIES_BY_REGIME,
    ProvenanceKind,
    SHELL_THEORY_CAPABILITIES,
    SUPPORTED_SHELL_THEORY_BY_REGIME,
    ShellAnalysisRegime,
    ShellTheory,
    ShellTheoryCapability,
    experimental_shell_theories,
    provenance_is_placeholder,
    provenance_is_trusted,
    require_shell_theory_supports,
    resolve_shell_theory_capability,
    shell_theories_supporting,
    supported_shell_theories,
    validate_shell_configuration,
)
from .station_analysis_types import (  # noqa: F401
    SectionShellRecoveryBundle,
    SectionShellStationRuntimeArtifacts,
    ShellPanelSolveResult,
    ShellSectionStationAnalysis,
    ThinWallResult,
)
from .station_types import (  # noqa: F401
    BeamSectionConstitutiveInput,
    ShellSectionStationGeometry,
    ShellStationProvenance,
)

__all__ = [
    "SectionShellStationRuntimeArtifacts",
    "BeamSectionConstitutiveInput",
    "FieldProvenance",
    "PanelBucklingResult",
    "ProvenanceKind",
    "SectionShellRecoveryBundle",
    "ShellPanelSolveResult",
    "ThinWallResult",
    "provenance_is_placeholder",
    "provenance_is_trusted",
    "SUPPORTED_SHELL_THEORY_BY_REGIME",
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
    "ShellTheoryCapability",
    "SHELL_THEORY_CAPABILITIES",
    "resolve_shell_theory_capability",
    "shell_theories_supporting",
    "require_shell_theory_supports",
    "supported_shell_theories",
    "experimental_shell_theories",
    "EXPERIMENTAL_SHELL_THEORIES_BY_REGIME",
    "validate_shell_configuration",
]
