# Forward simulation

The forward simulation runs a [car](cars.md) down a [track](tracks.md) and reports
how long it takes to reach the finish line, along with its speed, the forces on its
wheels, and the gradients of the results with respect to the car parameters.  This
page describes how the equation of motion is built from modular physics terms, how
it is integrated in time, and how to use the results.  The [forward model
example](examples/forward_model.py) runs a complete simulation, and the
[API reference](api.md#simulation) documents every class.

## Running a simulation

```python
import pwdsim

track = pwdsim.besttrack(42)
car = pwdsim.SimpleCar(...)  # see the Cars page

sim = pwdsim.Simulation(track, car)
run = sim()

run.finish_time  # time for the front of the car to cross the finish line
run.finish_time.backward()  # gradients with respect to every car parameter
car.cg_.grad
```

A [`Simulation`][pwdsim.simulation.Simulation] holds the track, the car, the
physics terms, and the global parameters in an
[`Environment`][pwdsim.physics.Environment] (gravitational acceleration and air
density).  Calling it places the car with its front against the start pin, at rest,
and integrates the equation of motion.  It returns a [`Run`][pwdsim.simulation.Run]
with the results.

## Coordinates

The car rides on the track with both axles in contact, so its motion has a single
degree of freedom: $s$, the location of the rear wheel contact along the track.  The
state of the car is $(s, v)$ with $v = \dot{s}$.

Everything else follows from the geometry.  The rear axle center sits one wheel
radius off the track.  The front wheel contact $\sigma(s)$ is wherever the front
axle center is exactly one wheelbase from the rear axle center, found with Newton's
method.  The line between the axle centers sets the body frame and the pitch angle
$\phi(s)$ of the car, which locate every point on the car, including its center of
gravity $\mathbf{x}_g(s)$.

To compute the normal forces, the model also includes two coordinates that the
simulation holds at zero: the lifts $h_r$ and $h_f$ of the rear and front axles off
the track, along the track normal.  The generalized coordinates are
$q = (s, h_r, h_f)$.

[`CarKinematics`][pwdsim.kinematics.CarKinematics] computes the positions of points
on the car analytically, together with their first three derivatives with respect to
$s$ and their first derivatives with respect to the lifts.

## Equation of motion

The equation of motion follows from Lagrange's equations with a Rayleigh
dissipation function $F$:

$$
\frac{d}{dt} \frac{\partial T}{\partial \dot{q}_i}
- \frac{\partial T}{\partial q_i}
+ \frac{\partial V}{\partial q_i}
+ \frac{\partial F}{\partial \dot{q}_i} = Q_i,
$$

where $T$ is the kinetic energy and $V$ the potential energy.  The constraint forces
$Q_i$ vanish along $s$ and are the normal forces along the lifts.

### Physics terms

The contributions to $T$, $V$, and $F$ are modular
[physics terms][pwdsim.physics], each of one of three kinds:

| Kind | Base class | Provides |
| --- | --- | --- |
| Kinetic energy | [`KineticTerm`][pwdsim.physics.KineticTerm] | rates making up $T$ |
| Potential energy | [`PotentialTerm`][pwdsim.physics.PotentialTerm] | $V$ and its derivatives |
| Dissipation | [`DissipativeTerm`][pwdsim.physics.DissipativeTerm] | derivatives of $F$ |

The implemented terms are:

| Name | Class | Contribution |
| --- | --- | --- |
| `gravity` | [`Gravity`][pwdsim.physics.Gravity] | $V = M g\, y_g$ |
| `translation` | [`Translation`][pwdsim.physics.Translation] | $T = \tfrac{1}{2} M \lvert\dot{\mathbf{x}}_g\rvert^2$ |

With only these two terms the car rolls without friction or drag, and its wheels
and body have no rotational inertia.  More terms will add the spin of the wheels,
the pitching of the body, aerodynamic drag, and axle and rolling friction.

Terms are switched on and off by name:

```python
sim.disable("gravity")
sim.enable("gravity")
```

### Rates

Every kinetic energy in the model is a sum of squares of *rates*: quantities that
are linear in the generalized velocities, $\rho = J(q) \cdot \dot{q}$.  Examples are
the components of the velocity of the center of gravity, the pitch rate of the body,
and the spin rate of a wheel.  A rate with weight $m$ (a mass or a moment of inertia)
contributes $T = \tfrac{1}{2} m \rho^2$.

For such a term the left side of Lagrange's equations reduces to
$m J_i \dot{\rho}$.  On the track, where only $s$ changes,
$\dot{\rho} = J_s a + J_s' v^2$ with $a = \ddot{s}$.  So a kinetic term only needs
to supply $J_s$ and its derivatives with respect to $s$, and the lift components
$J_{h_r}$ and $J_{h_f}$.

Summing the terms along $s$ gives

$$
M(s)\, a + C(s)\, v^2 + V_s + D_s = 0, \qquad
M = \sum m J_s^2, \quad C = \sum m J_s J_s',
$$

with $V_s = \partial V / \partial s$ and $D_s = \partial F / \partial \dot{s}$.
$M(s)$ is the effective mass of the car along the track.  This gives the
acceleration $a(s, v)$.  The time integration also needs the Jacobian of $a$ with
respect to $s$ and $v$, which uses the higher derivatives.  Every term provides its
derivatives analytically, and the test suite checks each one against finite
differences.

### Normal forces

Along the lifts, the constraint forces are the normal forces at each axle:

$$
N_{r,f} = \sum m J_{h_{r,f}} \left(J_s a + J_s' v^2\right)
+ \frac{\partial V}{\partial h_{r,f}} + \frac{\partial F}{\partial \dot{h}_{r,f}}.
$$

Because they come from the same terms as the equation of motion, the normal forces
automatically account for all the enabled physics.  They are linear in $a$, so
friction that depends on the normal forces will make the equation of motion a small
linear system for $a$, $N_r$, and $N_f$ at each state.

A positive normal force pushes the car away from the track.  A negative one means
the track would have to pull the wheels down to keep them on it.  In reality that
wheel lifts off, which the model doesn't allow.  If a normal force goes negative
before the finish, the simulation raises a
[`LiftOffWarning`][pwdsim.simulation.LiftOffWarning].
[`Run.min_normal_force`][pwdsim.simulation.Run.min_normal_force] reports the minimum
normal forces, which an optimizer can use to avoid these designs.

## Time integration

[pyzag](https://github.com/applied-material-modeling/pyzag) integrates the equation
of motion $\dot{x} = (v, a)$ on a fixed time grid, with step `dt` for a total time
`duration`.  The default is the backward Euler method with $dt = 0.1$ ms.  Forward
Euler is also available (`integrator="forward-euler"`).

Both methods are first order: the error in the finish time is proportional to $dt$.
The backward Euler method also loses a little energy, about 0.03% over a run at the
default step, again proportional to $dt$.

The first-order convergence relies on a smooth track.  A jump in the curvature or
its first two derivatives adds an error to the step where an axle crosses it, and
that error depends on where the crossing falls within the step.  This makes the
finish time a slightly jagged function of the car's design, and its gradients
noisy.  [`SplineTrack`][pwdsim.track.SplineTrack] keeps the curvature and its first
two derivatives continuous to avoid this (see [Track models](tracks.md)).

!!! note "Accuracy"
    On the 42 ft BestTrack, the finish time at the default step is within about
    0.25 ms of the converged value.  Gradients of the finish time are stable to
    about 1% for steps of 0.4 ms and below, including small sensitivities like the
    one to the wheelbase (about 0.02 ms/in).  Coarser steps can't resolve the 1 in
    easements on the track, and the gradients become noisy.

pyzag solves blocks of time steps together (`block_size`, default 1000) to
vectorize the calculation.

### Gradients

With `sim(adjoint=True)`, the default, gradients come from the adjoint method.  It
costs about as much as one more simulation, and memory doesn't grow with the number
of time steps.  Parameters with gradients turned off are skipped.  `sim(adjoint=False)`
instead backpropagates through every time step, which is exact but uses much more
memory.

### Batches of cars

A car whose properties have a batch dimension describes several designs (see
[Cars](cars.md)).  The simulation runs them all at once, and every result gets the
same batch dimension:

```python
cg = torch.tensor([[0.5, 0.4], [1.0, 0.4], [1.5, 0.4]]) * INCH
cars = pwdsim.SimpleCar(..., cg=cg)
pwdsim.Simulation(track, cars)().finish_time  # shape (3,)
```

## Results

A [`Run`][pwdsim.simulation.Run] holds the times and states and computes results
from them:

| Result | Description |
| --- | --- |
| `finish_time` | time for the front of the car to cross the finish line |
| `finish_speed` | speed of the center of gravity at the finish |
| `max_speed` | maximum speed of the center of gravity before the finish |
| `min_normal_force` | minimum normal forces at the rear and front axles before the finish |
| `speed()` | speed of the center of gravity at every step |
| `normal_forces()` | normal forces at every step |
| `energy()` | kinetic, potential, and total energy at every step |

The finish time is interpolated within the step where the front of the car crosses
the finish line, a vertical line through the track at $s_\mathrm{finish}$.  The
scalar results are differentiable.  Cars that don't finish within `duration` get a
finish time of NaN and a [`DidNotFinishWarning`][pwdsim.simulation.DidNotFinishWarning].

The [`plotting`][pwdsim.plotting] module draws the results:

- [`plot_run`][pwdsim.plotting.plot_run]: speed over time and normal forces along
  the track;
- [`plot_energy`][pwdsim.plotting.plot_energy]: kinetic, potential, and total
  energy;
- [`plot_car`][pwdsim.plotting.plot_car]: the car on the track at chosen locations,
  at true scale.
