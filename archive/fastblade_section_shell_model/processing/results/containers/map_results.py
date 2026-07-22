"""DOF-space mapping entries between solver stages (mirror 1d-fem MapEntry)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ShellMapEntry:
    """Per-global-DOF provenance across assembly / constraint / reduction / reconstruction."""

    global_dof: int
    fixed: bool = False
    independent: bool = False
    slave: bool = False
    masters: list[tuple[int, float]] = field(default_factory=list)
    reduced_index: int | None = None
    reconstructed_value: float = 0.0

    @classmethod
    def from_provenance_entry(cls, entry: object) -> ShellMapEntry:
        """Build from legacy ``ShellDofProvenanceEntry`` during migration."""
        return cls(
            global_dof=int(entry.global_dof),
            fixed=bool(entry.fixed),
            independent=bool(entry.independent),
            slave=bool(entry.slave),
            masters=list(getattr(entry, "masters", []) or []),
            reconstructed_value=float(getattr(entry, "reconstructed_value", 0.0)),
        )


__all__ = ["ShellMapEntry"]
