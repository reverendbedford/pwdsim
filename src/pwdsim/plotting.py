"""Plotting routines."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pwdsim.track import Track
from pwdsim.units import DEGREE, LENGTH_UNITS


def plot_track(
    track: Track,
    unit: str = "m",
    n: int = 2000,
    equal_aspect: bool = False,
    axes: Sequence[Axes] | None = None,
) -> Figure:
    """Plot the geometry of a track.

    The top panel shows the track profile $y(x)$ with the start pin and finish line
    marked.  The bottom panel shows the track angle $\\theta(s)$ and curvature
    $\\kappa(s)$ along the track.

    Args:
        track: the track to plot.
        unit: length unit for the axes, one of the keys of
            [`LENGTH_UNITS`][pwdsim.units.LENGTH_UNITS].
        n: number of points to evaluate along the track.
        equal_aspect: plot the profile with equal axis scales.  Tracks are much
            longer than they are tall, so by default the height is exaggerated.
        axes: optional pair of matplotlib axes (profile, angle and curvature) to
            plot into.  If not provided, a new figure is created.

    Returns:
        The matplotlib figure.
    """
    if unit not in LENGTH_UNITS:
        raise ValueError(f"Unknown length unit {unit!r}")
    scale = LENGTH_UNITS[unit]

    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    ax_profile, ax_angle = axes
    fig = ax_profile.figure

    with torch.no_grad():
        s = torch.linspace(0, float(track.length), n, dtype=torch.float64)
        x, y = (v.numpy() for v in track.position(s))
        theta = track.angle(s).numpy()
        kappa = track.curvature(s).numpy()
        markers = torch.stack([track.s_start, track.s_finish])
        x_marker, y_marker = (v.numpy() for v in track.position(markers))
        s = s.numpy()

    ax_profile.plot(x / scale, y / scale, color="k")
    for xm, ym, label, color in zip(
        x_marker / scale,
        y_marker / scale,
        ("start", "finish"),
        ("tab:green", "tab:red"),
        strict=True,
    ):
        ax_profile.axvline(float(xm), color=color, ls=":", lw=1)
        ax_profile.plot(float(xm), float(ym), "o", color=color, label=label)
    ax_profile.set_xlabel(f"x ({unit})")
    ax_profile.set_ylabel(f"y ({unit})")
    ax_profile.legend()
    if equal_aspect:
        ax_profile.set_aspect("equal")

    ax_angle.plot(s / scale, theta / DEGREE, color="tab:blue")
    ax_angle.set_xlabel(f"s ({unit})")
    ax_angle.set_ylabel("angle (deg)", color="tab:blue")
    ax_curvature = ax_angle.twinx()
    ax_curvature.plot(s / scale, kappa * scale, color="tab:orange")
    ax_curvature.set_ylabel(f"curvature (1/{unit})", color="tab:orange")

    return fig
