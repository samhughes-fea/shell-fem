"""Structured panel → subcomponent identity contract for shell recovery consumers."""

from __future__ import annotations

from typing import Any


def panel_subcomponent_name(panel: Any) -> str | None:
    """Return the owning subcomponent name when present on the panel object."""
    name = getattr(panel, "subcomponent_name", None)
    if name is not None and str(name).strip():
        return str(name)
    label = str(getattr(panel, "label", "") or getattr(panel, "name", "") or "")
    if ":" in label:
        return label.split(":", 1)[1].strip() or None
    return None


def panel_kind(panel: Any) -> str | None:
    kind = getattr(panel, "panel_kind", None)
    if kind is not None and str(kind).strip():
        return str(kind)
    label = str(getattr(panel, "label", "") or getattr(panel, "name", "") or "")
    if ":" in label:
        return label.split(":", 1)[0].strip() or None
    return None


def composite_subcomponent_index(
    panel: Any,
    comp_subs: list,
    *,
    legacy_name_mapper: Any | None = None,
) -> int | None:
    """Resolve composite subcomponent row index from structured panel identity."""
    sub_name = panel_subcomponent_name(panel)
    if sub_name:
        for i, s in enumerate(comp_subs):
            if s.name == sub_name:
                return i
    if legacy_name_mapper is not None:
        panel_name = str(getattr(panel, "name", None) or getattr(panel, "label", ""))
        return legacy_name_mapper(panel_name, comp_subs)
    return None
