"""Tier 3 provenance gating: which secondary resultant rows may enter CLT recovery.

Moved from ``api/fi_recovery.py`` (see PROCESSING_ARCHITECTURE_PROPOSAL.md,
``api/fi_recovery.py`` decomposition table); re-exported there for compatibility.
"""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import ShellPanelResultants

# Secondary ``ShellPanelResultants`` field names (legacy x/y labels). In the
# laminate / element frame these are :math:`N_{11},N_{22},N_{12}` and
# :math:`M_{11},M_{22},M_{12}` respectively.
_CLPT_RESULTANT_FIELDS: tuple[str, ...] = ("N11", "N22", "N12", "M11", "M22", "M12")


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
    """Transverse shear resultants (:math:`Q_1,Q_2` as ``Q1``/``Q2``) must be trusted."""
    for f in ("Q1", "Q2"):
        prov = row.provenance.get(f)
        if prov is not None and not prov.is_trusted:
            return False
    return row.Q1 is not None and row.Q2 is not None


__all__ = [
    "clpt_inputs_trusted",
    "interlaminar_inputs_trusted",
    "resultant_field_trusted",
]
