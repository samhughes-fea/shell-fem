"""MITC4 Q4 element factory (formulation-owned; generic cache builder stays agnostic)."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.formulation_objects import (
    PanelConstitutiveSpec,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellTheory,
)
from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.constitutive.panel_abd import (
    panel_abd_tuple_from_lam,
)
from ...mesh.shell_mesh_map import PanelGlobalMap
from ..formulation_factory import ShellElementFactory, _element_L_s, _panel_kappa_nodes
from .mitc4_quad4_element import (
    Mitc4ElementGeometry,
    Mitc4ElementLoads,
    Mitc4ElementMaterial,
    Mitc4Quad4Element,
)


class Mitc4ElementFactory(ShellElementFactory):
    theory = ShellTheory.FSDT
    n_dof_node = 6
    formulation_id = "MITC4-FSDT"

    def iter_elements(
        self,
        panel_maps: list[PanelGlobalMap],
        zones: list[Any],
        Nx_panels: list[NDArray[np.float64]],
        Nxy_panels: list[NDArray[np.float64]],
        *,
        L_x: float = 1.0,
        zones_outboard: list[Any] | None = None,
        panel_abd_stacks: list[tuple] | None = None,
        panel_constitutive_specs: list[PanelConstitutiveSpec | None] | None = None,
    ) -> Iterator[Mitc4Quad4Element]:
        elem_id = 0
        for pi, pm in enumerate(panel_maps):
            if len(pm.s_nodes) == 0:
                continue
            p = zones[pi]
            nodes_yz = np.asarray(getattr(p, "nodes"), dtype=float)
            s_nodes = pm.s_nodes
            s_panel_full = np.asarray(p.s, dtype=float)
            Nx_nodes = np.interp(s_nodes, s_panel_full, np.asarray(Nx_panels[pi], dtype=float))
            Nxy_nodes = np.interp(s_nodes, s_panel_full, np.asarray(Nxy_panels[pi], dtype=float))
            kappa_nodes = _panel_kappa_nodes(nodes_yz, s_panel_full, s_nodes)
            panel_spec = (
                panel_constitutive_specs[pi]
                if panel_constitutive_specs is not None
                else None
            )
            if panel_spec is not None:
                ABD = panel_spec.fsdt.ABD
                thickness = panel_spec.thickness_m
                G_eff = float(np.trace(panel_spec.fsdt.A_s)) / (
                    2.0 * (5.0 / 6.0) * thickness
                )
            elif panel_abd_stacks is not None:
                ABD, thickness, _, _, G_eff = panel_abd_stacks[pi]
            else:
                ABD, thickness, _, _, G_eff = panel_abd_tuple_from_lam(p.lam)

            for ei, e in enumerate(pm.elements):
                i0, i1, i2, i3 = e
                L_s_plain = float(abs(s_nodes[i1] - s_nodes[i0]))
                L_s, _L_x_elem = _element_L_s(
                    pi, s_nodes, i0, i1, zones, zones_outboard, L_x
                )
                if L_s < 1e-30:
                    # Degenerate geometry is recorded, not silently dropped from
                    # downstream identity: zero-length elements cannot form K_e.
                    continue
                nx_avg = 0.5 * (Nx_nodes[i0] + Nx_nodes[i1])
                nxy_avg = 0.5 * (Nxy_nodes[i0] + Nxy_nodes[i1])
                kappa_avg = None
                if kappa_nodes is not None:
                    kappa_avg = float(0.5 * (kappa_nodes[i0] + kappa_nodes[i1]))
                yield Mitc4Quad4Element(
                    element_id=elem_id,
                    panel_index=pi,
                    local_element_index=ei,
                    geometry=Mitc4ElementGeometry(
                        L_s=L_s,
                        L_s_donnell=L_s_plain,
                        L_x=L_x,
                        local_nodes=(i0, i1, i2, i3),
                    ),
                    material=Mitc4ElementMaterial(
                        ABD=ABD,
                        thickness=thickness,
                        G_eff=G_eff,
                        fsdt=None if panel_spec is None else panel_spec.fsdt,
                    ),
                    loads=Mitc4ElementLoads(
                        nx_avg=nx_avg, nxy_avg=nxy_avg, kappa_avg=kappa_avg
                    ),
                    dof_per_node=self.n_dof_node,
                )
                elem_id += 1


__all__ = ["Mitc4ElementFactory"]
