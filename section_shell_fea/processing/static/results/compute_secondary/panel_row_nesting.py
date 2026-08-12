"""Theory-agnostic regrouping of flat secondary resultant rows by panel.

Moved out of the deprecated periodic secondary-recovery module so linear-static
and buckling callers do not depend on Channel-A / Saint-Venant packages.
"""

from __future__ import annotations

from blade_structure_utilities.fem_utilities.section_shell_fea.pre_processing.containers.types import (
    ShellPanelResultants,
)


def nest_resultant_rows_by_panel(
    rows: list[ShellPanelResultants],
    n_panels: int,
) -> list[list[ShellPanelResultants]]:
    """Regroup flat secondary rows into ``list[list[ShellPanelResultants]]`` by panel."""
    nested: list[list[ShellPanelResultants]] = [[] for _ in range(n_panels)]
    for row in rows:
        pi = int(row.panel_index)
        if 0 <= pi < n_panels:
            nested[pi].append(row)
    for panel_rows in nested:
        panel_rows.sort(key=lambda r: int(r.station_index))
    return nested


__all__ = ["nest_resultant_rows_by_panel"]
