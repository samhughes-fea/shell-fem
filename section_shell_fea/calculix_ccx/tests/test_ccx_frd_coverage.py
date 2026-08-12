"""CCX FRD element-map coverage metrics."""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.results import FrdCcxNodeIndex


def frd_bending_element_coverage(
    frd_by_eid: dict[int, list],
    *,
    element_base: int,
    n_mesh_elements: int,
    frd_index: FrdCcxNodeIndex | None = None,
) -> float:
    """Fraction of FRD stress entities mapped to mesh element ids."""
    from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.results import (
        _remap_frd_eids_to_mesh,
    )

    remapped = _remap_frd_eids_to_mesh(
        frd_by_eid,
        element_base=element_base,
        n_mesh_elements=n_mesh_elements,
        frd_index=frd_index,
    )
    if not frd_by_eid:
        return 1.0
    return len(remapped) / max(len(frd_by_eid), 1)


def test_frd_element_map_coverage_with_3c_index():
    idx = FrdCcxNodeIndex(
        user_nodes={1: (101, 102, 103, 104)},
        node_to_users={101: {1}, 102: {1}, 103: {1}, 104: {1}},
    )
    cov = frd_bending_element_coverage(
        {101: [], 999: []},
        element_base=100,
        n_mesh_elements=5,
        frd_index=idx,
    )
    assert cov >= 0.5


def test_list_unmapped_frd_entities():
    from blade_structure_utilities.fem_utilities.section_shell_fea.calculix_ccx.results import list_unmapped_frd_entities

    unmapped = list_unmapped_frd_entities(
        {101: [], 99999: []},
        element_base=100,
        n_mesh_elements=5,
        frd_index=None,
    )
    assert 99999 in unmapped
    assert 101 not in unmapped
