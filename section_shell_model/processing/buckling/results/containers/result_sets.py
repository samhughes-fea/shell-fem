"""Primary / secondary result sets for the linear buckling track."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from blade_structure_utilities.section_shell_model.pre_processing.containers.buckling_types import (
    PanelBucklingResult,
    ShellPanelBucklingAssessment,
)


@dataclass
class ShellBucklingPrimaryResultSet:
    """Tier-1 buckling: eigenpairs (per-panel classical and/or full-section)."""

    panel_results: list[PanelBucklingResult] = field(default_factory=list)
    status: str = "ok"
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # Full-section track (optional): global modes on the connected mesh.
    section_lambdas: Any = None
    section_modeshapes: Any = None
    eigen_track: str = "panel"  # "panel" | "section"


@dataclass
class ShellBucklingSecondaryResultSet:
    """Tier-2 buckling: per-panel assessments including buckling_index."""

    panel_assessments: list[ShellPanelBucklingAssessment] = field(default_factory=list)
    status: str = "ok"
    diagnostics: dict[str, Any] = field(default_factory=dict)
