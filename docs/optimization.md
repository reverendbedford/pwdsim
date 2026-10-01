# Optimization

pwdsim can tune a car's parameters to minimize its finish time, subject to bounds,
race rules, and physical validity.  This page describes how to set up and run an
optimization, the standard constraints, and how the optimizer works.  The
[optimization example](examples/optimization.py) tunes the center of gravity and
mass of a stock car.

## Running an optimization

An optimization needs the system, the parameters to tune with their bounds, and any
constraints:

```python
import pwdsim
from pwdsim.units import INCH, OUNCE

sim = pwdsim.Simulation(track, car, dt=2e-4)  # the car, track, and physics

result = pwdsim.optimize(
    sim,
    variables={
        "cg": ((0.0, 0.2 * INCH), (4.375 * INCH, 1.0 * INCH)),  # (along, up)
        "mass": (3 * OUNCE, 6 * OUNCE),
    },
    constraints=pwdsim.constraints.bsa_rules(),
)
print(result.summary({"cg_": (INCH, "in"), "mass_": (OUNCE, "oz")}))
```

[`optimize`][pwdsim.optimization.optimize] minimizes the finish time over the chosen
variables, holding every other car parameter fixed, and writes the optimum into the
car.  It returns an [`OptimizationResult`][pwdsim.optimization.OptimizationResult]
with:
- the variables before and after;
- the finish times;
- the constraint values at the optimum;
- the history of every iteration;
- a copy of the starting car.

!!! warning "The car changes in place"
    A run's results are computed from the car's current parameters, which is what
    makes them differentiable.  So after an optimization changes the car, runs made
    before it can't compute their results any more, and raise an error instead.
    Simulate `result.initial_car` to compare with the starting design.

## Variables

The `variables` argument switches parameters on for tuning.  It is either a list of
names or a dict of names and bounds:

```python
variables = ["mass"]  # default bounds
variables = {"mass": (3 * OUNCE, 6 * OUNCE)}  # explicit bounds
variables = {
    "cg": (
        (0.25 * INCH, 0.4 * INCH),  # tune only the CG along the car:
        (2.0 * INCH, 0.4 * INCH),
    )
}  # its height is fixed at 0.4 in
```

- **Names** are the car's parameter names.  For a
  [`SimpleCar`][pwdsim.car.SimpleCar], the property names work too (`"cg"` for its
  `cg_` parameter).
- **Bounds** are `(lower, upper)` in SI units: either scalars, or one value per
  element of the parameter, like the along and up components of the center of
  gravity.
- **An element with equal lower and upper bounds is fixed** at that value, so part
  of a parameter can be tuned on its own.
- **Default bounds.** Variables without bounds get bounds of half to twice their
  current value.  That doesn't work for a zero value, so those need explicit
  bounds.

Every variable needs finite bounds.  The optimizer works with each variable scaled
to $[0, 1]$ between its bounds, $p = l + x(u - l)$.  This puts parameters of very
different sizes on an equal footing, from wheel inertias around $10^{-7}$ kg m² to
lengths around 0.1 m.

## Constraints

A [`Constraint`][pwdsim.constraints.Constraint] bounds a differentiable function of
the car and its run: `lower <= fun(car, run) <= upper`.
[`pwdsim.constraints`][pwdsim.constraints] provides the standard ones:

| Constraint | Keeps |
| --- | --- |
| [`lift_off(tolerance)`][pwdsim.constraints.lift_off] | the normal forces on both axles at least `tolerance`, all the way to the finish |
| [`cg_between_axles(margin)`][pwdsim.constraints.cg_between_axles] | the center of gravity between the axles, at least `margin` from each |
| [`max_mass(limit)`][pwdsim.constraints.max_mass] | the total mass at most `limit` (5 oz by default) |
| [`bsa_rules()`][pwdsim.constraints.bsa_rules] | a list with `max_mass(5 oz)` and `cg_between_axles()`, the common Scout rules |

Custom constraints are easy to write.  For example, a maximum overall length for a
car with 7/8 in behind its rear axle:

```python
max_length = pwdsim.constraints.Constraint(
    fun=lambda car, run: car.front_offset + 0.875 * INCH,
    upper=7 * INCH,
    name="max_length",
    scale=INCH,
    linear=True,
)
```

The `scale` gives the optimizer a typical size for the constraint's values.
`linear=True` tells it the constraint is linear in the car parameters.  Constraints
that don't use the run are cheap, because their gradients don't need the adjoint
method.

### No lift-off

The model assumes the wheels stay on the track.  A design that would lift a wheel
isn't valid, however fast the simulation says it is.  For example, moving the
center of gravity back makes the car faster until the front wheels lift off as the
car comes out of the curve.

So `optimize` always includes the [`lift_off`][pwdsim.constraints.lift_off]
constraint, with its tolerance set by `lift_off_tolerance`:

- `lift_off_tolerance=0.0`, the default, just keeps the wheels on the track.
- A positive tolerance, in newtons, demands a margin for a more robust design, at
  some cost in speed.
- `lift_off_tolerance=None` removes the constraint.

The constraint uses a smooth minimum of the normal forces,
[`Run.smooth_min_normal_force`][pwdsim.simulation.Run.smooth_min_normal_force],
because the exact minimum isn't differentiable wherever the location of the minimum
jumps.  The smooth minimum is the Kreisselmeier-Steinhauser aggregate

$$
-\frac{1}{\rho} \log \sum_i e^{-\rho N_i},
$$

over both axles and every time step before the finish.  It is never larger than the
exact minimum, and at most $\ln(n)/\rho$ smaller for $n$ values, about 0.02 N with
the default $\rho = 500$ /N, so it errs on the safe side.

## The optimizer

`optimize` uses SciPy's `trust-constr` method: a trust region interior point method
for problems with bounds and nonlinear constraints.  Nothing like it exists natively
in pytorch.

- **Curvature.** It approximates the curvature of the finish time and of each
  nonlinear constraint with BFGS updates built from their gradients.  The
  gradients come from the adjoint method, which gives exact gradients but not
  second derivatives.
- **Cost.** Each iteration costs one simulation, plus one backward pass for the
  finish time and one for each constraint value that depends on the run.
  - At $dt = 0.2$ ms on the 42 ft BestTrack, that's about 2 s an iteration.
  - The example's optimizations converge in about 8 iterations.
- **Methods.** `method="SLSQP"` selects SciPy's sequential quadratic programming
  method instead.

### Settings

`options` passes settings to `scipy.optimize.minimize`.  The defaults
([`DEFAULT_OPTIONS`][pwdsim.optimization.DEFAULT_OPTIONS]) are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `maxiter` | 50 | maximum number of iterations |
| `gtol` | $10^{-3}$ | tolerance on the optimality conditions, in ms per unit of the scaled variables |
| `xtol` | $10^{-4}$ | tolerance on the step size, in the scaled variables |
| `initial_barrier_parameter` | $10^{-3}$ | a small barrier, so the interior point method converges onto the bounds |

The tolerances are modest because the gradients are only accurate to about 1% (see
[Forward simulation](simulation.md#time-integration)).  For the same reason, use a
time step of 0.2 ms or less: coarser steps give noisy gradients, which confuse the
optimizer.

### Stopping

With more than a few variables, the noise in the gradients can keep the optimizer
from meeting `gtol`, long after the finish time has stopped improving.  So
`optimize` also stops once the finish time of the feasible designs has improved by
less than `stall_tolerance` (0.01 ms by default) over the last `stall_iterations`
(5) iterations.  Pass `stall_tolerance=None` to leave the stopping to the
optimizer's own tolerances.

An interior point method can also spend many iterations creeping toward bounds that
many variables end up on, gaining a few hundredths of a millisecond each time.  If
that isn't worth the time, cap the iterations with `options={"maxiter": ...}`, as
the [designing a car body](examples/car_design.py) example does.
