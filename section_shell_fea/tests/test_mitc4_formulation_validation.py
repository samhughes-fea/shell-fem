"""MITC4 formulation validation hook unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    ShellForceObject,
    ShellFormulationCache,
    ShellStiffnessObject,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.mitc4.validation import (
    validate_mitc4_cache,
    validate_mitc4_element,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.element_library.registry import (
    clear_formulation_registry_for_tests,
    resolve_formulation,
)


def test_mitc4_registration_wires_validation_hooks() -> None:
    clear_formulation_registry_for_tests()
    reg = resolve_formulation("MITC4-FSDT")
    assert reg.validate_element is validate_mitc4_element
    assert reg.validate_cache is validate_mitc4_cache


def test_validate_mitc4_element_refuses_nonpositive_length() -> None:
    elem = SimpleNamespace(
        formulation_id="MITC4-FSDT",
        geometry=SimpleNamespace(L_s=0.0, L_x=1.0, local_nodes=(0, 1, 2, 3)),
        material=SimpleNamespace(thickness=0.01, ABD=np.eye(6), fsdt=None),
    )
    with pytest.raises(ValueError, match="L_s"):
        validate_mitc4_element(elem)


def test_validate_mitc4_cache_refuses_wrong_formulation_id() -> None:
    cache = ShellFormulationCache(
        stiffness_objects=[],
        force_objects=[],
        n_dof_node=6,
        formulation_id="OTHER",
    )
    with pytest.raises(ValueError, match="formulation_id"):
        validate_mitc4_cache(cache)


def test_validate_mitc4_cache_accepts_empty_mitc4_cache() -> None:
    cache = ShellFormulationCache(
        stiffness_objects=[],
        force_objects=[],
        n_dof_node=6,
        formulation_id="MITC4-FSDT",
    )
    validate_mitc4_cache(cache)


def test_validate_mitc4_cache_refuses_wrong_elem_dof_count() -> None:
    stiff = ShellStiffnessObject(
        element_id=0,
        element_type="MITC4-FSDT",
        K_e=np.eye(12),
        panel_index=0,
        local_element_index=0,
        L_s=0.1,
        L_s_donnell=0.1,
        L_x=1.0,
        formulation_id="MITC4-FSDT",
        global_dof_indices=np.arange(12, dtype=np.int64),
    )
    force = ShellForceObject(
        element_id=0,
        element_type="MITC4-FSDT",
        F_e=np.zeros(12),
        panel_index=0,
        local_element_index=0,
        L_s=0.1,
        L_s_donnell=0.1,
        L_x=1.0,
        formulation_id="MITC4-FSDT",
    )
    cache = ShellFormulationCache(
        stiffness_objects=[stiff],
        force_objects=[force],
        n_dof_node=6,
        formulation_id="MITC4-FSDT",
    )
    with pytest.raises(ValueError, match="24"):
        validate_mitc4_cache(cache)
