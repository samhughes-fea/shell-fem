"""Matplotlib diagnostics for per-station surface traction load fields."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from blade_structure_utilities.section_shell_model.pre_processing.loading.containers.types import SectionStationLoadField
from blade_structure_utilities.section_shell_model.pre_processing.loading.builders.outline_frames import (
    bframe_outline_to_sframe_chord,
    rotate_traction_field,
)
from blade_structure_utilities.point_to_surftraction.geometry import SectionOutlineGeometry
from blade_structure_utilities.point_to_surftraction.plotting import (
    RunningLoadDistributions,
    draw_magnitude_normal_offset_ribbon,
    draw_signed_normal_offset_ribbon,
    ribbon_scale_from_values,
    running_load_distributions,
)


def _sframe_plot_context(
    load_field: SectionStationLoadField,
) -> tuple[SectionOutlineGeometry, object, float, float]:
    """Return S-frame geometry, S-frame traction, twist [deg], and alpha [deg]."""
    bframe = SectionOutlineGeometry.from_closed_contour(
        load_field.outline_yz,
        ensure_clockwise=True,
    )
    sframe, rot = bframe_outline_to_sframe_chord(bframe, float(load_field.twist_rad), te_first=False)
    traction_s = rotate_traction_field(load_field.traction_field, rot)
    twist_deg = float(np.rad2deg(load_field.twist_rad))
    alpha_deg = float(load_field.incident.alpha_deg)
    return sframe, traction_s, twist_deg, alpha_deg


def plot_force_vs_chord_png(
    load_field: SectionStationLoadField,
    out_png: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Write a 2×2 running-load figure to ``out_png``."""
    import matplotlib.pyplot as plt

    geometry, traction, twist_deg, alpha_deg = _sframe_plot_context(load_field)
    dist: RunningLoadDistributions = running_load_distributions(
        geometry,
        traction,
        twist_angle_deg=twist_deg,
    )
    x = dist.x_chord

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(
        f"Running load distributions  |  α = {alpha_deg:.1f}°  |  twist = {twist_deg:.1f}°",
        fontsize=12,
    )

    panels = [
        (axes[0, 0], dist.chord_normal, "S-frame chord-normal (lift)", "#3498db"),
        (axes[0, 1], dist.chord_parallel, "S-frame chord-parallel (drag)", "#e74c3c"),
        (axes[1, 0], dist.flapwise, "B-frame flapwise", "#2ecc71"),
        (axes[1, 1], dist.edgewise, "B-frame edgewise", "#e67e22"),
    ]

    for ax, values, title, color in panels:
        ax.plot(x, values, color=color, linewidth=1.8)
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.fill_between(x, values, 0.0, alpha=0.15, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x / c")
        ax.set_ylabel("dF / dx  [N / m]")
        ax.grid(True, linestyle="--", alpha=0.4)

    out_png = Path(out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_airfoil_area_diagrams_png(
    load_field: SectionStationLoadField,
    out_png: Path,
    *,
    dpi: int = 200,
) -> Path:
    """Write a 1×3 signed ribbon area-diagram figure to ``out_png``."""
    import matplotlib.pyplot as plt

    geometry, traction, _, _ = _sframe_plot_context(load_field)
    outline = geometry.coordinates
    nodes = np.asarray(geometry.panel_centres, dtype=float)

    pressure_scalar = np.sum(traction.pressure_forces * geometry.normals, axis=1)
    shear_scalar = np.sum(traction.shear_forces * geometry.tangents, axis=1)
    total_scalar = np.sum(traction.total_forces * geometry.normals, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    titles = ["Pressure area diagram", "Wall-shear area diagram", "Total traction area diagram"]
    values_list = [pressure_scalar, shear_scalar, total_scalar]
    colors = [("#3498db", "#e74c3c"), ("#2ecc71", None), ("#8e44ad", "#c0392b")]

    for ax, title, values, (pos_color, neg_color) in zip(axes, titles, values_list, colors):
        ax.plot(outline[:, 0], outline[:, 1], color="black", linewidth=1.2, zorder=10)
        scale = ribbon_scale_from_values(values, span_fraction=0.08)
        if neg_color is None:
            draw_magnitude_normal_offset_ribbon(ax, nodes, values, scale, pos_color, alpha=0.55)
        else:
            draw_signed_normal_offset_ribbon(
                ax,
                nodes,
                values,
                scale,
                pos_color,
                neg_color,
                alpha=0.55,
            )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle="--", alpha=0.4)

    out_png = Path(out_png).resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_station_vs_envelope_loads(
    z_stations: np.ndarray,
    *,
    station_q_y: np.ndarray | None = None,
    station_q_z: np.ndarray | None = None,
    envelope_z: np.ndarray | None = None,
    envelope_q_y: np.ndarray | None = None,
    envelope_q_z: np.ndarray | None = None,
    title: str = "Station distributed loads vs envelope",
):
    """Overlay per-station line loads against a spanwise envelope tabulation."""
    import matplotlib.pyplot as plt

    z = np.asarray(z_stations, dtype=np.float64).ravel()
    if z.size == 0:
        raise ValueError("z_stations must be non-empty when plotting envelope overlays.")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    panels = (
        (axes[0], station_q_y, envelope_q_y, "q_y [N/m]", "station q_y", "q_y (envelope)"),
        (axes[1], station_q_z, envelope_q_z, "q_z [N/m]", "station q_z", "q_z (envelope)"),
    )
    z_env = np.asarray(envelope_z, dtype=np.float64).ravel() if envelope_z is not None else None

    for ax_i, station_vals, envelope_vals, ylabel, station_label, envelope_label in panels:
        if z_env is not None and envelope_vals is not None:
            ax_i.plot(z_env, np.asarray(envelope_vals, dtype=np.float64).ravel(), "k--", alpha=0.6, label=envelope_label)
        if station_vals is not None:
            n = min(z.size, np.asarray(station_vals).size)
            ax_i.plot(z[:n], np.asarray(station_vals, dtype=np.float64).ravel()[:n], "C0.-", label=station_label)
        ax_i.set_ylabel(ylabel)
        ax_i.grid(True, alpha=0.3)
        ax_i.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("z [m]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def write_station_load_pngs(
    load_field: SectionStationLoadField,
    station_dir: Path,
    *,
    persist: bool = True,
    dpi: int = 200,
) -> list[Path]:
    """Write per-station traction diagnostic PNGs; returns written paths."""
    if not persist:
        return []

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return []

    station_dir = Path(station_dir).resolve()
    paths = [
        plot_force_vs_chord_png(load_field, station_dir / "force_vs_chord.png", dpi=dpi),
        plot_airfoil_area_diagrams_png(load_field, station_dir / "airfoil_area_diagrams.png", dpi=dpi),
    ]
    return paths
