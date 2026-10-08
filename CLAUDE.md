# pwdsim: A pytorch/python Pinewood Derby simulator and optimizer

## Purpose

This package is a tool to simulate, visualize, and optimize pinewood derby cars.  It contains three key modules:
1. A forward simulation tool that takes key characteristics of the car and track and simulates the performance of the car (i.e. how long it takes to complete a run).
2. An optimizer tool that tweaks features of the car to improve performance.
3. Visualization tools to display results including performance studies and car geometries.

The first two modules are made possible by implementing the forward simulation tool in pytorch, so that we can use AD and torch optimizers to tune the car's performance.

Generally we expect users to use notebooks to interface with the package to actually tune car performance.

## Code principles

A simple modern python package. Python 3.12, uv for environment and dependency management, src layout (`src/pwdsim`).  ruff for linting, pytest for testing, coverage.py for coverage checking, mkdocs for docs.  Pre-commit hooks to enforce basic formatting and linting.

CI (`.github/workflows/`): `ci.yml` runs the hooks, tests, notebooks, and `package.yml` (build the sdist and wheel, test the installed wheel) on every PR; `docs.yml` publishes the docs to GitHub Pages, with PR previews; `release.yml` publishes a GitHub release to PyPI by trusted publishing.  The version comes from the git tag (hatch-vcs), so never set it in `pyproject.toml`.

Example notebooks live in `docs/examples/` as jupytext percent-format `.py` scripts (never commit `.ipynb` files).  mkdocs-jupyter executes them when building the docs, and `tests/test_examples.py` runs them as smoke tests, marked slow: they only run with `pytest --runslow`.

Use float64 throughout the simulation.  Close races differ by ~1 ms over ~2.5 s, so float32 is too noisy for finish times and their gradients.

## Units

All internal calculations use SI units.  Users should be able to work in US customary units (inches, ounces, etc.) easily, so provide simple conversion helpers and/or constructors that accept customary units and convert at the boundary.

## Physics

We will use a 2D model of the car moving along a curve.

### Track

A track is a curve in $\mathbb{R}^2$ parameterized by arc length $s$.  The curve is the path of the wheel contact points.  We need to be able to quickly calculate $y(s)$, $\theta(s)$, defined as the angle of the track from the horizontal ($\theta = \mathrm{atan2}(y^\prime, x^\prime)$), and the curvature of the track $\kappa(s) = \theta^\prime(s)$ at any point.

A track is also described by:
1. The total track length $L$.  All tracks end at $y(L) = 0$.
2. The start location $s_\mathrm{start}$, where the fronts of the cars rest against the start pin (so the front of the car, not an axle or the center of gravity, starts at this common location).
3. The finish line location $s_\mathrm{finish} \le L$.  The finish line is generally before the end of the track, which includes a braking section.

### Car

A car is a rigid body connecting two axles, parameterized by the location of the rear wheel contact point along the track, $s$, and its velocity along the track, $\dot{s}$.  The car cannot leave the track (but see the checks below).

The rear and front axle centers are offset from the track along the track normal by the respective wheel radii.  The front wheel contact location $s_f(s)$ is not simply $s + w$: on curved sections it is found implicitly as the location where the distance between the axle centers equals the wheelbase $w$.  This geometry, along with the wheel radii, defines the pitch angle of the car body $\phi(s)$.

Car quantities are expressed in a body frame with its origin at the rear axle center, its first axis pointing toward the front axle center, and its second axis perpendicular, pointing away from the track.

A car is parameterized by:
1. The position of the center of gravity relative to the rear axle, in the body frame, $\mathbf{x}_g$.
2. The wheelbase of the car, $w$, which defines the location of the front axle.
3. The offset from the rear axle to the front of the car, $d$ (along the body frame first axis), needed for positioning the car at the start and determining when the car crosses the finish line.
4. The total mass of the car, $M$, including the wheels.
5. The moment of inertia of the car body about its center of gravity, $I_b$, for pitching motion.
6. The number of rear wheels ($n_r$) and front wheels ($n_f$).  Typically $n_r = n_f = 2$, but a raised front wheel gives $n_f = 1$.
7. The per-wheel moment of inertia of the rear wheels ($I_r$) and the front wheels ($I_f$).
8. The radius of the rear wheels ($r_r$) and the front wheels ($r_f$).
9. The radius of the rear axle ($a_r$) and the front axle ($a_f$).
10. The coefficient of friction of the rear axle ($\mu_r$) and the front axle ($\mu_f$).
11. The cross-sectional area of the car ($A$).
12. The drag coefficient of the car ($C_d$).
13. The rolling friction coefficient ($c_r$).

We may want these quantities to in turn depend on others.  For example, eventually we might want to model the car shape with a spline and calculate the mass, center of gravity, moment of inertia, cross-sectional area, and drag coefficient from this description.  We may also expand this list of parameters later as we add complexity to the model.

### Global parameters

1. Gravitational acceleration, $g$.
2. Air density, $\rho$.

### Dynamics

We'll take a Lagrangian approach (with non-conservative forces) to model the system, with $s$ as the single generalized coordinate.  We can then integrate out the dynamics from the Euler-Lagrange equations.  The conservative contributions to the Lagrangian are:
1. The potential energy of the car.  The height of the center of gravity comes from the rigid contact of the car over the wheelbase with the track.
2. The translational kinetic energy of the car's center of gravity.
3. The rotational kinetic energy of the car body as it pitches, $\tfrac{1}{2} I_b \dot{\phi}^2$.
4. The rotational kinetic energy of the wheels, assuming rolling without slip.

The non-conservative contributions enter as generalized forces:
1. Axle friction, a torque of $\mu N a$ on each axle, resisting motion at the wheel rim with force $\mu N a / r$.
2. Drag, $\tfrac{1}{2} \rho C_d A \dot{s}^2$.
3. Rolling friction, $c_r N$.

Axle and rolling friction depend on the normal forces at the rear and front axles, $N_r$ and $N_f$.  These are constraint forces, which we get as Lagrange multipliers: the generalized coordinates are $q = (s, h_r, h_f)$, where $h_r$ and $h_f$ lift each axle off the track along the track normal and are always held at zero.  The Euler-Lagrange equations along the lifts give the normal forces, automatically consistent with whichever physics terms are enabled.  They depend on the centripetal loading through curved sections ($\propto \dot{s}^2 \kappa$) and on $\ddot{s}$ itself, so with friction depending on them the equation of motion is a small linear system for $(\ddot{s}, N_r, N_f)$, solved at every state (`EquationsOfMotion` in `src/pwdsim/simulation.py`).  The wheel spin angles come from the rolling constraint as holonomic functions of the contact locations, $\Psi(c) = -(c - r\theta(c))/r$, which keeps the lift multipliers equal to the pure normal forces.

The physics is modular (`src/pwdsim/physics.py`).  Terms are kinetic (`KineticTerm`), potential (`PotentialTerm`), or dissipative (`DissipativeTerm`, Rayleigh dissipation functions), can be enabled and disabled by name, and supply **analytic** derivatives with respect to $s$ and the lifts (every derivative is checked against finite differences in the tests).  Kinetic energies are sums of squared *rates* $\rho = J(q) \cdot \dot{q}$ (CG velocity, pitch rate, wheel spin), so a kinetic term only supplies $J_s$, its first two $s$-derivatives, and the lift components.  The car geometry and its derivatives (including the implicit front contact solve) live in `src/pwdsim/kinematics.py`.

Time integration uses pyzag (backward Euler by default, $dt = 0.1$ ms) with the adjoint method for gradients.  Car properties may carry a leading batch dimension to simulate several designs at once.

### Validity checks

The model assumes the car stays on the track.  Simulations must check for violations of this assumption, in particular a negative normal force at either axle (e.g. $N_f < 0$, the front wheels lifting through the transition when the center of gravity is too far back).  These checks should be exposed so they can feed into constraints and bounds for optimization.

Later we may need to model the effect of cars swerving in the tracks or, the counter design principle to that effect, rail riding.

## Design optimization

As everything is done in torch we can use AD (combined with the adjoint method) to calculate parameter sensitivities and torch optimizers to tune performance.

Optimization must respect constraints and bounds, both from race rules (e.g. maximum mass, length, width, clearance) and from physical validity (e.g. the wheel lift checks above); without them the optimizer will drive parameters to unphysical or illegal values.

`pwdsim.optimize` (`src/pwdsim/optimization.py`) wraps SciPy's `trust-constr` (BFGS Hessian approximations, since the adjoint gives gradients but not second derivatives).  Users pass a `Simulation`, the variables to tune (car parameter names with optional `(lower, upper)` bounds; equal bounds fix an element), and constraints (`src/pwdsim/constraints.py`: `cg_between_axles`, `max_mass`, `bsa_rules`, and custom `Constraint`s).  The `lift_off` constraint, a smooth (KS) minimum of the normal forces, is always included unless `lift_off_tolerance=None`.  Variables are scaled to $[0, 1]$ between their bounds.

## Code design and package development

Start with simple descriptions of tracks and cars (i.e. cars are just the direct parameter values above).  Build the simulation and basic visualization toolkit.  Provide some handy routines for optimization.

Then build out more complex car models where we provide a way to work with the car geometry directly (i.e. profile as a spline).

`GeometricCar` (`src/pwdsim/car.py`, with `src/pwdsim/geometry.py`) does the first part: a `Profile` (flat bottom, vertical ends, cubic B-spline top through control heights; bounds on the control heights keep it inside the block by the convex hull property) extruded through a thickness, with rectangular `Inclusion`s (`Region`, `WeightPocket`: void from the bottom to `height - length`, then the weight) of other densities and optional depths, and wheels as point masses.  Mass, CG, pitch inertia, and front offset are computed exactly (Gauss quadrature on spline spans, closed-form rectangles); any can be overridden with a number.  Geometry parameters are optimized by dotted name (`profile.heights`, `inclusions.0.position`), with constraints `regions_inside_body`, `regions_within_thickness`, `regions_dont_overlap`, `regions_avoid_axles`.

The frontal area and drag coefficient also come from the profile (`src/pwdsim/aerodynamics.py`, `DragModel`): a semi-empirical Hoerner-style component buildup of the drag area (laminar skin friction on the wetted area, forebody pressure drag from the front face and steep forward-facing slopes, Hoerner base drag from the rear face and steep rear-facing slopes, and an empirical drag area for the exposed wheels with the rear ones in the front ones' wake).  Panel methods were rejected: potential flow gives zero drag, and viscous-inviscid coupling fails at the blunt, separated base.  All coefficients are overridable, as are `frontal_area` and `drag_coefficient` themselves.
