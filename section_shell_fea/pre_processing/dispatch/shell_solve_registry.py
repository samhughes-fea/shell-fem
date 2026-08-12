"""Shell theory → global-solve backend registry.

Registration seam for opt-in global-coupled backends (e.g. CCX). Production
periodic does **not** call through this registry — it uses ``api/periodic.py`` /
``assemble_section_from_zones`` directly.

Owned by :mod:`shell_theory_registry` (R4); ``BACKENDS`` is a compatibility view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
    resolve_shell_theory_capability,
)
from .shell_theory_registry import (
    get_global_backend,
    register_global_backend,
)

SolveFn = Callable[..., tuple[list[list[Any]], list[dict]]]


@dataclass(frozen=True)
class ShellSolveBackend:
    theory: ShellTheory
    solve_global: SolveFn
    label: str


def _backend(theory: ShellTheory, solve_fn: SolveFn, label: str | None = None) -> ShellSolveBackend:
    resolved_label = label if label is not None else resolve_shell_theory_capability(theory).label
    return ShellSolveBackend(theory, solve_fn, resolved_label)


class _BackendsView(dict):
    def __getitem__(self, key: ShellTheory) -> ShellSolveBackend:
        try:
            solve_fn, label = get_global_backend(key)
        except KeyError as exc:
            raise KeyError(key) from exc
        return _backend(key, solve_fn, label)

    def __contains__(self, key: object) -> bool:
        try:
            get_global_backend(key)  # type: ignore[arg-type]
            return True
        except KeyError:
            return False

    def __setitem__(self, key: ShellTheory, value: ShellSolveBackend) -> None:
        register_global_backend(key, value.solve_global, label=value.label)

    def keys(self):
        from .shell_theory_registry import global_backend_theories

        return global_backend_theories()

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


BACKENDS: dict[ShellTheory, ShellSolveBackend] = _BackendsView()


def register_backend(theory: ShellTheory, solve_fn: SolveFn, *, label: str | None = None) -> None:
    """Register ``solve_fn`` as the global-solve backend for ``theory``."""
    register_global_backend(theory, solve_fn, label=label)


def resolve_backend(shell_theory: ShellTheory) -> ShellSolveBackend:
    """Look up a registered global-solve backend (introspection / opt-in checks)."""
    try:
        return BACKENDS[shell_theory]
    except KeyError:
        raise ValueError(f"No shell solve backend registered for {shell_theory!r}") from None


__all__ = [
    "ShellSolveBackend",
    "BACKENDS",
    "register_backend",
    "resolve_backend",
]
