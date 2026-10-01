# pwdsim

A pytorch/python Pinewood Derby simulator and optimizer.

pwdsim simulates, visualizes, and optimizes pinewood derby cars:

- **Simulation**: race a car down a track, and compute its finish time, its speed,
  and the forces on its wheels.
- **Optimization**: automatic differentiation through the pytorch simulator gives
  the sensitivity of the finish time to every car parameter, so torch optimizers
  can tune the car design within the constraints of the race rules.
- **Visualization**: plot the track, the car, and its run.

!!! note "Status"
    Early development.  The forward simulation works, with the full physics of the
    model: gravity, the translation and pitching of the car, the spin of its
    wheels, aerodynamic drag, and axle and rolling friction, with gradients from the
    adjoint method.  The optimizer tunes car parameters within bounds, race rules,
    and a no-lift-off constraint.  Cars can be described by their shape and
    weights, with the mass, center of gravity, and inertia computed from the
    geometry; computing the drag from the shape is next.

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

The [forward model example](examples/forward_model.py) walks through a complete
simulation: setting up the track and the car, racing it, computing sensitivities,
and comparing a batch of designs.  The
[which physics matters?](examples/physics_study.py) example switches the physics
terms on and off to show which aspects of a car's physics, and which of its
parameters, matter most for its speed.  The
[optimizing a car](examples/optimization.py) example tunes the center of gravity
and mass of a car, and the [designing a car body](examples/car_design.py) example
optimizes the shape of a car body and the size and position of its weight.

## The model in brief

The car is a 2D rigid body rolling along a track curve parameterized by arc length
$s$.  Its state is the location $s$ of the rear wheel contact and its rate
$\dot{s}$.  The equation of motion comes from Lagrange's equations with a Rayleigh
dissipation function $F$,

$$
\frac{d}{dt} \frac{\partial T}{\partial \dot{q}_i}
- \frac{\partial T}{\partial q_i}
+ \frac{\partial V}{\partial q_i}
+ \frac{\partial F}{\partial \dot{q}_i} = Q_i,
$$

assembled from modular physics terms that can be switched on and off:

- the gravitational potential energy, $V = M g\, y_g$;
- the kinetic energy of the car moving with its center of gravity and pitching
  through the curve, and of its wheels spinning as they roll;
- the Rayleigh dissipation of aerodynamic drag, axle friction, and rolling friction.

The constraint forces $Q_i$ holding the axles on the track are the normal forces on
the wheels.  Axle and rolling friction are proportional to them, so the simulation
solves for the acceleration and the normal forces together.  It warns if a normal
force goes negative: a wheel lifting off the track.

The model is documented piece by piece:

- [Track models](tracks.md): how pwdsim describes track geometry, and the standard
  tracks it includes.
- [Cars](cars.md): the essential properties that describe a car, and the car
  classes that provide them.
- [Forward simulation](simulation.md): how the equation of motion is assembled
  from modular physics terms and integrated in time, and the results of a run.
- [Car geometry](geometry.md): cars described by the shape of their body and the
  weights in it.
- [Optimization](optimization.md): tuning car parameters within bounds and
  constraints.

## Installation

pwdsim requires Python 3.12 or later.

```bash
uv pip install -e .
```
