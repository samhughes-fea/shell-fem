"""MITC4-FSDT self-registration (lazy; no generic-module eager import)."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
)
from ..contracts import mitc4_fsdt_descriptor
from ..registry import ShellFormulationRegistration, register_formulation


def register_mitc4_fsdt() -> None:
    from ..formulation_factory import register_element_factory
    from ..kernel import MITC4_KERNEL
    from .factory import Mitc4ElementFactory
    from .projector import Mitc4NativeToCanonicalProjector
    from .validation import validate_mitc4_cache, validate_mitc4_element

    register_formulation(
        ShellFormulationRegistration(
            descriptor=mitc4_fsdt_descriptor(),
            factory_cls=Mitc4ElementFactory,
            kernel=MITC4_KERNEL,
            projector=Mitc4NativeToCanonicalProjector(),
            constitutive_adapter=None,
            is_external_backend=False,
            validate_element=validate_mitc4_element,
            validate_cache=validate_mitc4_cache,
        )
    )
    register_element_factory(ShellTheory.FSDT, Mitc4ElementFactory)


__all__ = ["register_mitc4_fsdt"]
