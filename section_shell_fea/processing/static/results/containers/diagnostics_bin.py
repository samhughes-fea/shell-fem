"""Per-solve FE diagnostics adjacent to the P→S→T→Q cascade bin."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DiagnosticsBin:
    """Use-case-agnostic diagnostics for one FE worker call.

    Filled by the solve-phase collector during assemble / F=KU / cascade fill.
    Callers may nest summaries into orchestration logs; this type never owns
    campaign file I/O or sizing trial traces.
    """

    phase_totals_s: dict[str, float] = field(default_factory=dict)
    phase_counts: dict[str, int] = field(default_factory=dict)
    wall_s: float = 0.0

    # Reduced-solve quality (Phase A).
    solve_residual_rel: float | None = None
    solve_pivot_ratio: float | None = None
    solve_quality_ok: bool = True
    solve_quality_reason: str = ""

    # Per-tier trust summary (poisoned upstream → downstream).
    primary_ok: bool = True
    secondary_ok: bool = True
    tertiary_ok: bool = True
    quaternary_ok: bool = True
    cascade_trust_ok: bool = True
    cascade_trust_reason: str = ""

    # Optional mesh fingerprint from the prebuilt SectionStripMesh.
    mesh_n_total_elements: int | None = None
    mesh_el_min_over_h: float | None = None

    # Constraint provenance (modes/counts/tags; never the equation graph).
    constraint_metadata: dict[str, object] = field(default_factory=dict)


__all__ = ["DiagnosticsBin"]
