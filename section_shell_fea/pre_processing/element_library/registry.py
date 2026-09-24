"""Unified discrete-element formulation registry.

Keyed by formulation ID. Continuum theory maps to a default formulation.
External backends (CalculiX) stay on a separate dispatch path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
)
from .contracts import ShellFormulationDescriptor, mitc4_fsdt_descriptor


class NativeToCanonicalShellProjector(Protocol):
    """Project native element products into canonical dense layouts."""

    formulation_id: str

    def project_resultants(self, native: dict[str, float]) -> tuple[tuple[float, ...], tuple[str, ...]]:
        """Return canonical N/M/Q values and per-component capabilities."""
        ...

    def project_material_stress(
        self,
        sigma11: float,
        sigma22: float,
        tau12: float,
        *,
        sigma33: float = 0.0,
        tau23: float = 0.0,
        tau13: float = 0.0,
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        """Return canonical six-component stress and capabilities."""
        ...


@dataclass(frozen=True, slots=True)
class ShellFormulationRegistration:
    """Bundle descriptor, factory, kernel, projector, and constitutive adapter."""

    descriptor: ShellFormulationDescriptor
    factory_cls: type
    kernel: Any
    projector: NativeToCanonicalShellProjector
    constitutive_adapter: Callable[..., Any] | None = None
    is_external_backend: bool = False


_FORMULATIONS: dict[str, ShellFormulationRegistration] = {}
_DEFAULT_BY_THEORY: dict[str, str] = {}


def register_formulation(registration: ShellFormulationRegistration) -> None:
    fid = registration.descriptor.formulation_id
    existing = _FORMULATIONS.get(fid)
    if existing is not None:
        if existing.descriptor.formulation_id == fid:
            return
        raise ValueError(f"formulation already registered: {fid}")
    _FORMULATIONS[fid] = registration
    theory = registration.descriptor.continuum_theory
    if theory not in _DEFAULT_BY_THEORY and not registration.is_external_backend:
        _DEFAULT_BY_THEORY[theory] = fid


def resolve_formulation(formulation_id: str) -> ShellFormulationRegistration:
    ensure_default_formulations()
    try:
        return _FORMULATIONS[formulation_id]
    except KeyError as exc:
        raise ValueError(f"No formulation registered for {formulation_id!r}") from exc


def default_formulation_id(theory: str | ShellTheory) -> str:
    if isinstance(theory, ShellTheory):
        key = theory.name
    else:
        key = str(theory)
    ensure_default_formulations()
    candidates = {key, key.upper(), key.lower()}
    for cand in candidates:
        if cand in _DEFAULT_BY_THEORY:
            return _DEFAULT_BY_THEORY[cand]
        for stored, fid in _DEFAULT_BY_THEORY.items():
            if stored.lower() == cand.lower():
                return fid
    raise ValueError(f"No default formulation for theory {key!r}")


def registered_formulation_ids() -> tuple[str, ...]:
    ensure_default_formulations()
    return tuple(_FORMULATIONS)


def ensure_default_formulations() -> None:
    """Lazily register production MITC4-FSDT without importing it at module import."""
    if "MITC4-FSDT" in _FORMULATIONS:
        return
    from .mitc4.registration import register_mitc4_fsdt

    register_mitc4_fsdt()


def clear_formulation_registry_for_tests() -> None:
    """Test helper: drop non-production registrations while restoring MITC4."""
    _FORMULATIONS.clear()
    _DEFAULT_BY_THEORY.clear()
    ensure_default_formulations()


__all__ = [
    "NativeToCanonicalShellProjector",
    "ShellFormulationRegistration",
    "clear_formulation_registry_for_tests",
    "default_formulation_id",
    "ensure_default_formulations",
    "register_formulation",
    "registered_formulation_ids",
    "resolve_formulation",
    "mitc4_fsdt_descriptor",
]
