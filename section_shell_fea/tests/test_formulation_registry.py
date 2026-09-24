"""Formulation registry, descriptors, and projector contracts."""

from __future__ import annotations

import numpy as np

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.contracts import (
    CANONICAL_STRESS_NAMES,
    mitc4_fsdt_descriptor,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.registry import (
    ShellFormulationRegistration,
    clear_formulation_registry_for_tests,
    default_formulation_id,
    register_formulation,
    registered_formulation_ids,
    resolve_formulation,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.kernel import (
    resolve_kernel,
)


def test_default_fsdt_is_mitc4() -> None:
    assert default_formulation_id(ShellTheory.FSDT) == "MITC4-FSDT"
    assert default_formulation_id("FSDT") == "MITC4-FSDT"
    assert "MITC4-FSDT" in registered_formulation_ids()
    assert resolve_kernel(ShellTheory.FSDT).name == "mitc4_fsdt"


def test_second_fsdt_formulation_does_not_replace_default() -> None:
    desc = mitc4_fsdt_descriptor()
    dummy = ShellFormulationRegistration(
        descriptor=desc.__class__(
            **{
                **{f.name: getattr(desc, f.name) for f in desc.__dataclass_fields__.values()},
                "formulation_id": "TEST-FSDT",
            }
        ),
        factory_cls=resolve_formulation("MITC4-FSDT").factory_cls,
        kernel=resolve_formulation("MITC4-FSDT").kernel,
        projector=resolve_formulation("MITC4-FSDT").projector,
    )
    register_formulation(dummy)
    try:
        assert default_formulation_id("FSDT") == "MITC4-FSDT"
        assert "TEST-FSDT" in registered_formulation_ids()
    finally:
        clear_formulation_registry_for_tests()
        assert default_formulation_id("FSDT") == "MITC4-FSDT"


def test_mitc4_descriptor_frames_and_sigma33_capability() -> None:
    desc = mitc4_fsdt_descriptor()
    assert desc.dof_layout.frame == "C"
    assert desc.generalized.frame == "E"
    assert desc.stress_frame == "M"
    assert desc.stress_capabilities["sigma33"] == "not_modelled_zero"
    assert CANONICAL_STRESS_NAMES[2] == "sigma33"


def test_mitc4_projector_forces_sigma33_zero() -> None:
    projector = resolve_formulation("MITC4-FSDT").projector
    values, caps = projector.project_material_stress(10.0, 2.0, 1.0, sigma33=99.0)
    assert values[2] == 0.0
    assert caps[2] == "not_modelled_zero"
    native = {"N11": 1.0, "N22": 2.0, "N12": 0.0, "M11": 0.0, "M22": 0.0, "M12": 0.0, "Q1": 3.0, "Q2": 4.0}
    r_values, r_caps = projector.project_resultants(native)
    assert r_values == (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0)
    assert r_caps[0] == "modelled"


def test_mitc4_register_is_idempotent() -> None:
    from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.registration import (
        register_mitc4_fsdt,
    )

    before = registered_formulation_ids()
    register_mitc4_fsdt()
    register_mitc4_fsdt()
    after = registered_formulation_ids()
    assert after.count("MITC4-FSDT") == 1
    assert after == before
    assert default_formulation_id("FSDT") == "MITC4-FSDT"
