"""Progressive damage: per-ply damage states and stiffness degradation."""

from shell_fem.processing.progressive_damage.degradation import DamageState, DegradationStrategy, SmearedCrackDegradation

__all__ = ["DamageState", "DegradationStrategy", "SmearedCrackDegradation"]
