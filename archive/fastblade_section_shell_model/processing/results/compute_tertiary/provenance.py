"""Tier 3 provenance gating: which secondary resultant rows may enter CLT recovery.

Moved from ``api/fi_recovery.py`` (see PROCESSING_ARCHITECTURE_PROPOSAL.md,
``api/fi_recovery.py`` decomposition table); re-exported there for compatibility.
"""

from __future__ import annotations

from blade_precompute.section_shell_model.pre_processing.containers.types import ShellPanelResultants

_CLPT_RESULTANT_FIELDS: tuple[str, ...] = ("Nx", "Ny", "Nxy", "Mx", "My", "Mxy")


def resultant_field_trusted(row: ShellPanelResultants, field: str) -> bool:
    """True when ``field`` has trusted provenance on ``row`` (default: trusted if absent)."""
    prov = row.provenance.get(field)
    if prov is None:
        return True
    return bool(prov.is_trusted)


def clpt_inputs_trusted(row: ShellPanelResultants) -> bool:
    """All membrane/bending resultants used for in-plane Hashin must be trusted."""
    return all(resultant_field_trusted(row, f) for f in _CLPT_RESULTANT_FIELDS)


def interlaminar_inputs_trusted(row: ShellPanelResultants) -> bool:
    """Transverse shear resultants for interlaminar FI must be trusted."""
    for f in ("Qx", "Qy"):
        prov = row.provenance.get(f)
        if prov is not None and not prov.is_trusted:
            return False
    return row.Qx is not None and row.Qy is not None


__all__ = [
    "clpt_inputs_trusted",
    "interlaminar_inputs_trusted",
    "resultant_field_trusted",
]
