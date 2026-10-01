"""Plotting routines."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pwdsim import geometry
from pwdsim.car import GeometricCar
from pwdsim.simulation import Run, Simulation
from pwdsim.track import Track
from pwdsim.units import DEGREE, LENGTH_UNITS


def _length_scale(unit):
    if unit not in LENGTH_UNITS:
        raise ValueError(f"Unknown length unit {unit!r}")
    return LENGTH_UNITS[unit]


def _batch_columns(x):
    """Reshape a time series with shape (n, *batch) to (n, nbatch)."""
    return x.reshape(x.shape[0], -1)


def _labels(labels, count):
    if labels is None:
        return [None] * count if count == 1 else [f"car {i}" for i in range(count)]
    if len(labels) != count:
        raise ValueError(f"Need {count} labels")
    return list(labels)


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
    scale = _length_scale(unit)

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


def plot_runs(
    runs: Sequence[Run],
    labels: Sequence[str] | None = None,
    unit: str = "m",
    axes: Sequence[Axes] | None = None,
) -> Figure:
    """Compare the performance of several runs.

    The top panel shows the speed of the center of gravity over time, with the
    finish marked.  The bottom panel shows the normal forces at the rear (solid) and
    front (dashed) axles along the track up to the finish.  A negative normal force
    means a wheel lifted off the track.

    Args:
        runs: the runs to plot.  Each car in each run's batch gets its own line.
        labels: optional labels, one for each car across all the runs.
        unit: length unit for the distance along the track.
        axes: optional pair of matplotlib axes to plot into.

    Returns:
        The matplotlib figure.
    """
    scale = _length_scale(unit)
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    ax_speed, ax_force = axes
    fig = ax_speed.figure

    columns = []
    with torch.no_grad():
        for run in runs:
            forces = run.normal_forces()
            before = _batch_columns(run._before_finish().expand(run.s.shape)).numpy()
            speed = _batch_columns(run.speed()).numpy()
            s = _batch_columns(run.s).numpy()
            rear = _batch_columns(forces[..., 0]).numpy()
            front = _batch_columns(forces[..., 1]).numpy()
            finish = run.finish_time.reshape(-1).numpy()
            times = run.times.numpy()
            for i in range(speed.shape[1]):
                columns.append(
                    (times, speed[:, i], s[:, i], rear[:, i], front[:, i])
                    + (before[:, i], finish[i])
                )

    for (times, speed, s, rear, front, mask, finish), label in zip(
        columns, _labels(labels, len(columns)), strict=True
    ):
        (line,) = ax_speed.plot(times, speed, label=label)
        color = line.get_color()
        ax_speed.axvline(finish, color=color, ls=":", lw=1)
        ax_force.plot(s[mask] / scale, rear[mask], color=color, label=label)
        ax_force.plot(s[mask] / scale, front[mask], color=color, ls="--")

    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.set_title("Speed of the center of gravity (dotted: finish)")
    ax_force.axhline(0, color="k", lw=0.8)
    ax_force.set_xlabel(f"rear wheel location s ({unit})")
    ax_force.set_ylabel("normal force (N)")
    ax_force.set_title("Normal forces: rear (solid) and front (dashed) axles")
    if labels is not None or len(columns) > 1:
        ax_speed.legend()
    return fig


def plot_run(
    run: Run,
    labels: Sequence[str] | None = None,
    unit: str = "m",
    axes: Sequence[Axes] | None = None,
) -> Figure:
    """Plot the performance of a run: see [`plot_runs`][pwdsim.plotting.plot_runs].

    Args:
        run: the run to plot.  Each car in a batch gets its own line.
        labels: optional labels for the cars in the batch.
        unit: length unit for the distance along the track.
        axes: optional pair of matplotlib axes to plot into.

    Returns:
        The matplotlib figure.
    """
    return plot_runs([run], labels=labels, unit=unit, axes=axes)


def plot_energy(run: Run, index: int = 0, ax: Axes | None = None) -> Figure:
    """Plot the energy budget over a run.

    Shows the kinetic and potential energy, the energy dissipated by drag and
    friction so far, and the total, which stays constant up to the error of the
    time integration.

    Args:
        run: the run to plot.
        index: which car in a batch to plot.
        ax: optional matplotlib axes to plot into.

    Returns:
        The matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    with torch.no_grad():
        energy = {k: _batch_columns(v)[:, index] for k, v in run.energy().items()}
    for name in ("kinetic", "potential", "dissipated", "total"):
        ax.plot(run.times.numpy(), energy[name].numpy(), label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("energy (J)")
    ax.legend()
    return ax.figure


def plot_car(
    simulation: Simulation,
    s: Sequence[float] | torch.Tensor,
    index: int = 0,
    unit: str = "in",
    margin: float = 0.1,
    ax: Axes | None = None,
) -> Figure:
    """Draw the car on the track at one or more locations.

    The car is drawn schematically, at true scale: its wheels, the line from the
    rear axle to the front of the car, and its center of gravity.

    Args:
        simulation: the simulation holding the track and car.
        s: rear wheel contact locations to draw the car at.
        index: which car in a batch to draw.
        unit: length unit for the axes.
        margin: length of track to show beyond the car, in meters.
        ax: optional matplotlib axes to plot into.

    Returns:
        The matplotlib figure.
    """
    scale = _length_scale(unit)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    car = simulation.car
    shape = car.batch_shape

    with torch.no_grad():
        s = torch.as_tensor(s, dtype=torch.float64).reshape(-1)
        s_batch = s.reshape((-1,) + (1,) * len(shape)).expand((len(s), *shape))
        config = simulation.kinematics.evaluate(s_batch)

        def pick(x, vector=True):
            x = x.reshape((len(s), -1, 2) if vector else (len(s), -1))
            return x[:, index].numpy()

        rear = pick(config.rear_axle.value)
        front_axle = pick(config.front_axle.value)
        front = pick(config.front.value)
        cg = pick(config.cg.value)
        sigma = pick(config.front_contact.value, vector=False)

        def radius(value):
            return value.expand(shape).reshape(-1)[index].item()

        r_rear, r_front = radius(car.rear_wheel_radius), radius(car.front_wheel_radius)
        track_s = torch.linspace(
            float(s.min()) - margin,
            float(sigma.max()) + margin,
            500,
            dtype=torch.float64,
        )
        x, y = simulation.track.position(track_s)

    ax.plot(x.numpy() / scale, y.numpy() / scale, color="k", lw=1)
    for i in range(len(s)):
        for center, r in ((rear[i], r_rear), (front_axle[i], r_front)):
            ax.add_patch(
                plt.Circle(center / scale, r / scale, fill=False, color="tab:blue")
            )
        ax.plot(
            [rear[i, 0] / scale, front[i, 0] / scale],
            [rear[i, 1] / scale, front[i, 1] / scale],
            color="tab:blue",
        )
        ax.plot(*(cg[i] / scale), "o", color="tab:red", label="CG" if i == 0 else None)
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.legend()
    return ax.figure


def _material_name(density):
    names = {
        geometry.TUNGSTEN: "tungsten",
        geometry.LEAD: "lead",
        geometry.VOID: "void",
        geometry.PINE: "pine",
        geometry.BASSWOOD: "basswood",
    }
    return names.get(density, f"{density:.0f} kg/m³")


def plot_geometry(
    car: GeometricCar, unit: str = "in", ax: Axes | None = None
) -> Figure:
    """Draw the side profile of a [`GeometricCar`][pwdsim.car.GeometricCar], at
    true scale.

    Shows the body, its inclusions filled by material, the wheels and axles, the
    track under the wheels, and the center of gravity.  Inclusions shallower than
    the body are drawn lighter.

    Args:
        car: the car to draw.
        unit: length unit for the axes.
        ax: optional matplotlib axes to plot into.

    Returns:
        The matplotlib figure.
    """
    scale = _length_scale(unit)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    profile = car.profile
    with torch.no_grad():
        x = torch.linspace(profile.x_rear, profile.x_front, 200, dtype=torch.float64)
        top = profile.top(x).numpy()
        x = x.numpy()
        bottom = profile.bottom
        outline_x = [x[0], *x, x[-1]]
        outline_y = [bottom, *top, bottom]
        ax.fill(
            [v / scale for v in outline_x],
            [v / scale for v in outline_y],
            color="burlywood",
            label=f"body ({_material_name(car.body_density.item())})",
        )
        labelled = set()
        for inclusion in car.inclusions:
            for rect in inclusion.rectangles(bottom):
                x0, x1 = rect.x0.item() / scale, rect.x1.item() / scale
                y0, y1 = rect.y0.item() / scale, rect.y1.item() / scale
                depth = car.thickness if rect.depth is None else rect.depth
                fraction = (depth / car.thickness).item()
                name = _material_name(rect.density)
                if rect.density == geometry.VOID:
                    style = dict(facecolor="white", edgecolor="0.4", linestyle="--")
                else:
                    style = dict(facecolor="0.25", edgecolor="0.1")
                ax.add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        x1 - x0,
                        y1 - y0,
                        alpha=0.3 + 0.7 * fraction,
                        label=None if name in labelled else name,
                        **style,
                    )
                )
                labelled.add(name)
        w = car.wheelbase.item()
        for center, radius in (
            (0.0, car.rear_wheel_radius.item()),
            (w, car.front_wheel_radius.item()),
        ):
            ax.add_patch(
                plt.Circle(
                    (center / scale, 0.0), radius / scale, fill=False, color="tab:blue"
                )
            )
            ax.plot(center / scale, 0.0, "+", color="tab:blue")
        track = -car.rear_wheel_radius.item()
        ax.axhline(track / scale, color="k", lw=0.8)
        cg = car.cg.detach().numpy() / scale
        ax.plot(*cg, "o", color="tab:red", label="CG")
    ax.set_aspect("equal")
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.legend(loc="upper right", fontsize="small")
    return ax.figure
