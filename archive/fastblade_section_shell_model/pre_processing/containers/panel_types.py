"""Package-local panel geometry DTO used by section topology / assembly.

Mirrors the legacy stress-model Panel surface consumed in-package
(``nodes``, ``lam``, ``s``, ``label``, ``cell_id``, ``subcomponent_name``,
``panel_kind``). Arc-length ``s`` is derived in ``__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class Panel:
    """Thin-walled panel midline between two contour endpoints.

    Parameters
    ----------
    nodes
        ``(N, 2)`` array of ``[y, z]`` coordinates along the panel midline.
    lam
        Panel laminate (duck-typed: must expose ``E``, ``t``, and typically
        ``nu`` / ``build_plies()`` for CLPT recovery).
    cell_id
        Closed-cell index (diagnostics / legacy compatibility).
    end_boom
        Optional boom at the panel end (legacy thin-wall field; unused by
        production periodic assembly).
    label
        Stable panel identity string (e.g. ``"skin:OuterSkin"``).
    subcomponent_name
        Source subcomponent label from section geometry.
    panel_kind
        Subcomponent kind: ``"skin"``, ``"cap"``, or ``"web"``.
    """

    nodes: NDArray[np.floating]
    lam: Any
    cell_id: int = 0
    end_boom: object = None
    label: str = ""
    subcomponent_name: str | None = None
    panel_kind: str | None = None
    s: NDArray[np.floating] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        self.nodes = nodes
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(
                f"Panel.nodes must be (N, 2); got shape {getattr(nodes, 'shape', None)}"
            )
        if nodes.shape[0] < 2:
            self.s = np.asarray([0.0], dtype=float)
        else:
            ds = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
            self.s = np.concatenate([[0.0], np.cumsum(ds)])


__all__ = ["Panel"]
