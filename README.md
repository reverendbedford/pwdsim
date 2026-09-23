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

> **Status:** early development. The package structure is in place; the simulator is
> not yet implemented.

## Model overview

pwdsim uses a 2D model of a rigid car rolling along a track.

**Track.** A curve parameterized by arc length $s$, tracing the path of the wheel
contact points. A track provides its height $y(s)$, angle $\theta(s)$, and curvature
$\kappa(s)$, along with its total length $L$, the start pin location
$s_\mathrm{start}$, and the finish line location $s_\mathrm{finish} \le L$.

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

**Dynamics.** The equations of motion come from a Lagrangian with non-conservative
generalized forces. The Lagrangian includes the car's potential energy, the
translational and pitching kinetic energy of the body, and the rotational kinetic
energy of the wheels (rolling without slip). Axle friction, rolling friction, and
aerodynamic drag enter as generalized forces. The normal forces at each axle are
computed from a Newton-Euler balance on the body. The simulation also flags when the
model's assumptions break down, such as a wheel lifting off the track.

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
