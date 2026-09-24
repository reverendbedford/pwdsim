# pwdsim

A pytorch/python Pinewood Derby simulator and optimizer.

pwdsim simulates, visualizes, and optimizes pinewood derby cars:

- **Simulation**: given the car and track, compute how long the car takes to reach the
  finish line.
- **Optimization**: automatic differentiation through the pytorch simulator gives
  parameter sensitivities, and torch optimizers tune the car design within the
  constraints of the race rules.
- **Visualization**: plot performance studies and car geometries.

!!! note "Status"
    Early development.  The forward simulation works with gravity and the
    translational kinetic energy of the car; friction, drag, and rotational inertia
    are next.

## The model in brief

The car is a 2D rigid body rolling along a track curve parameterized by arc length
$s$. Its state is the rear wheel contact position $s$ and speed $\dot{s}$. The
equation of motion comes from the Lagrangian

$$
\mathcal{L} = T_\mathrm{body} + T_\mathrm{wheels} - M g\, y_g(s),
$$

where $T_\mathrm{body}$ includes the translation of the center of gravity and the
pitching of the body, and $T_\mathrm{wheels}$ is the spin of the wheels. Axle friction,
rolling friction, and aerodynamic drag enter as non-conservative generalized forces.

The model is documented piece by piece:

- [Track models](tracks.md): how pwdsim describes track geometry, and the standard
  tracks it includes.
- [Cars](cars.md): the essential properties that describe a car, and the car
  classes that provide them.
- [Forward simulation](simulation.md): how the equation of motion is assembled
  from modular physics terms and integrated in time, and the results of a run.

## Getting started

```bash
uv pip install -e .
```

```python
import pwdsim
```
