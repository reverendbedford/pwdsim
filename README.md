# pwdsim

A pytorch/python Pinewood Derby simulator and optimizer.

pwdsim simulates, visualizes, and optimizes pinewood derby cars. It has three parts:

1. **Simulation**: given the car and track, compute how the car runs down the track and
   how long it takes to reach the finish line.
2. **Optimization**: tune features of the car to improve performance.
3. **Visualization**: plot performance studies and car geometries.

The simulator is written in pytorch, so automatic differentiation gives parameter
sensitivities directly and torch optimizers can tune the car design. Most users will
work with pwdsim from Jupyter notebooks.

## Status

pwdsim is in early development.  What works now:

- **Tracks**: smooth spline tracks built from straights, circular arcs, and
  easements, including a model of the common BestTrack aluminum track.
- **Cars**: cars described directly by their essential properties, as trainable
  torch parameters, with batches of designs simulated together.
- **Forward simulation**: races a car down the track and reports its finish time,
  speed, energy budget, and the normal forces on its wheels, warning if a wheel
  lifts off the track.  Gradients of the results with respect to every car
  parameter come from the adjoint method.  The physics covers gravity, the
  translation and pitching of the car, the spin of its wheels, aerodynamic drag, and
  axle and rolling friction, and each piece can be switched on and off.
- **Visualization**: plots of the track, the car on the track, and its run.

Next up: optimization routines, and cars described by their shape.

## Quick start

```python
import pwdsim
from pwdsim.plotting import plot_run
from pwdsim.units import GRAM, INCH, OUNCE

# A car built from the standard BSA kit, in customary units
wheel_radius = 0.595 * INCH
wheel_inertia = 0.58 * (2.6 * GRAM) * wheel_radius**2
car = pwdsim.SimpleCar(
    cg=(1.0 * INCH, 0.4 * INCH),
    wheelbase=4.375 * INCH,
    front_offset=6.125 * INCH,
    mass=5 * OUNCE,
    body_inertia=3.9e-4,
    rear_wheel_inertia=wheel_inertia,
    front_wheel_inertia=wheel_inertia,
    rear_wheel_radius=wheel_radius,
    front_wheel_radius=wheel_radius,
    rear_axle_radius=0.0435 * INCH,
    front_axle_radius=0.0435 * INCH,
    rear_axle_friction=0.1,
    front_axle_friction=0.1,
    frontal_area=0.0014,
    drag_coefficient=0.4,
    rolling_friction=0.002,
)

# Race it down a 42 ft BestTrack
sim = pwdsim.Simulation(pwdsim.besttrack(42), car)
run = sim()
print(f"Finish time: {run.finish_time.item():.4f} s")

# Sensitivity of the finish time to the center of gravity, in ms per inch
run.finish_time.backward()
print(car.cg_.grad * INCH * 1000)

plot_run(run, unit="ft")
```

The [forward model example](docs/examples/forward_model.py) walks through a
complete simulation, and the [which physics matters?](docs/examples/physics_study.py)
example compares the effects of the different pieces of physics and ranks the car
parameters by their effect on the finish time.  The docs (`uv run mkdocs serve`)
describe the model in detail.

## Model overview

pwdsim uses a 2D model of a rigid car rolling along a track.

**Track.** A curve parameterized by arc length $s$, tracing the path of the wheel
contact points. A track provides its height $y(s)$, angle $\theta(s)$, curvature
$\kappa(s)$, and curvature derivatives, along with its total length $L$, the start
pin location $s_\mathrm{start}$, and the finish line location
$s_\mathrm{finish} \le L$.

**Car.** A rigid body connecting two axles, described by the position $s$ of the rear
wheel contact along the track and its speed $\dot{s}$. Car parameters include:

| Symbol | Description |
| --- | --- |
| $\mathbf{x}_g$ | Center of gravity relative to the rear axle (body frame) |
| $w$ | Wheelbase |
| $d$ | Offset from the rear axle to the front of the car |
| $M$ | Total mass, including wheels |
| $I_b$ | Body moment of inertia about the center of gravity (pitch) |
| $n_r$, $n_f$ | Number of rear / front wheels |
| $I_r$, $I_f$ | Per-wheel moment of inertia, rear / front |
| $r_r$, $r_f$ | Wheel radius, rear / front |
| $a_r$, $a_f$ | Axle radius, rear / front |
| $\mu_r$, $\mu_f$ | Axle friction coefficient, rear / front |
| $A$ | Cross-sectional area |
| $C_d$ | Drag coefficient |
| $c_r$ | Rolling friction coefficient |

Global parameters are gravitational acceleration $g$ and air density $\rho$.

**Dynamics.** The equation of motion comes from Lagrange's equations with a Rayleigh
dissipation function.  It is assembled from modular physics terms, which can be
switched on and off: potential energy (gravity), kinetic energy (the translation and
pitching of the car, and the spin of the wheels rolling without slip), and
dissipation (drag, axle friction, and rolling friction).  The normal forces at each
axle come out as the constraint forces holding the axles on the track, so they
automatically include every enabled term.  Friction is proportional to the normal
forces, so the acceleration and normal forces are solved for together.  The
simulation flags when the model's assumptions break down, such as a wheel lifting
off the track.  `pwdsim.full_physics()` (the default) includes every term, and
`pwdsim.simple_physics()` just gravity and translation.

**Time integration.** [pyzag](https://github.com/applied-material-modeling/pyzag)
integrates the equation of motion with the backward Euler method, and provides
gradients of the results with the adjoint method.

**Units.** Everything is SI internally, with helpers to work in US customary units
(inches, ounces, ...).

## Installation

pwdsim requires Python 3.12 or later.

```bash
uv pip install -e .
```

## Development

The project is managed with [uv](https://docs.astral.sh/uv/). Create the development
environment (a project-local `.venv` with the package, tests, linting, and docs tools)
and install the git hooks:

```bash
uv sync
uv run pre-commit install
```

Common tasks:

```bash
uv run pytest                          # run the tests
uv run pytest --cov --cov-report=html  # tests with coverage
uv run ruff check . && uv run ruff format .  # lint and format
uv run pre-commit run --all-files      # run all hooks
uv run mkdocs serve                    # preview the docs locally
uv run mkdocs build --strict           # build the docs
```

### Example notebooks

The example notebooks in `docs/examples/` are stored as
[jupytext](https://jupytext.readthedocs.io/) percent-format `.py` scripts, so they
diff cleanly and never carry outputs in version control. To work on one
interactively, run `uv run jupyter lab` and open the `.py` file as a notebook
(right-click → *Open With* → *Notebook*). Jupytext pairs it with a `.ipynb` file,
which git ignores; edits are saved back to the `.py` script.

The docs build executes the notebooks and renders their outputs, and
`tests/test_examples.py` runs each one as a smoke test.

## License

MIT; see [LICENSE](LICENSE).
