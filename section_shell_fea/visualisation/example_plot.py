from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_demo_visualisation(output_path: str | Path) -> Path:
    """Render a deterministic demo figure for src.blade_structure_utilities.fem_utilities.section_shell_fea."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, 120)
    y_a = np.sin(2.0 * np.pi * x)
    y_b = 0.5 * np.cos(4.0 * np.pi * x + 0.3)

    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=140)
    ax.plot(x, y_a, label="primary")
    ax.plot(x, y_b, label="secondary")
    ax.set_title("src.blade_structure_utilities.fem_utilities.section_shell_fea demo visualisation")
    ax.set_xlabel("normalised span")
    ax.set_ylabel("response")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, format="png")
    plt.close(fig)
    return out
