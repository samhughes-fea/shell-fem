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


__all__ = ["DiagnosticsBin"]
