# Car geometry

A [`GeometricCar`][pwdsim.car.GeometricCar] computes its mass, center of gravity,
and pitching moment of inertia from a description of its body: the shape it's cut
to, and the weights and pockets in it.  This lets the optimizer work on the things
a builder actually controls: how the block is cut, and where the weight goes.  The
[designing a car body](examples/car_design.py) example optimizes the shape of a
car and the size and position of its weight.

Everything is in the car's body frame: the origin at the rear axle center, $x$
along the car toward the front axle, and $y$ up.

## The body

The body is a block of wood cut to a side profile and extruded through the block's
thickness.  A [`Profile`][pwdsim.geometry.Profile] describes the profile:

- a flat bottom at $y = y_b$ (usually a little below the axles);
- vertical rear and front faces at $x = x_r$ and $x = x_f$;
- a top at $y = y_b + h(x)$, where the height $h(x)$ is a cubic B-spline through a
  set of control heights $c_i$.

```python
from pwdsim.geometry import Profile
from pwdsim.units import INCH

# An uncut BSA block: 7 in long and 1.25 in tall, the rear axle 7/8 in from the back
profile = Profile.block(
    x_rear=-0.875 * INCH,
    x_front=6.125 * INCH,
    bottom=-0.15 * INCH,
    height=1.25 * INCH,
    n=9,
)
```

A B-spline always lies within the range of its control values (the convex hull
property).  So keeping every control height between 0 and the block height keeps
the whole body inside the block.  The shape can't fold over on itself either.

## Inclusions

Inside the body, rectangular *inclusions* of other materials replace the wood.  An
inclusion has a density, and optionally a depth across the car.  Where it's
shallower than the body, the rest of the thickness stays wood.  There are two
kinds:

- A [`Region`][pwdsim.geometry.Region] is a single rectangle of one material, such
  as a weight or a void.
- A [`WeightPocket`][pwdsim.geometry.WeightPocket] is a pocket drilled up from the
  bottom of the car, holding a weight at its top.  It is centered at `position`
  along the car and is `width` long, with total `height` and a weight `length`:
  - empty from the bottom up to `height - length`;
  - the weight from there up to `height`.

```python
from pwdsim.geometry import TUNGSTEN, WeightPocket

weight = WeightPocket(
    position=-0.5 * INCH,
    width=0.5 * INCH,
    height=0.6 * INCH,
    length=0.15 * INCH,
    material=TUNGSTEN,
    depth=0.5 * INCH,
)
```

The [`geometry`][pwdsim.geometry] module has approximate densities for common
materials: `PINE`, `BASSWOOD`, `TUNGSTEN`, `LEAD`, and `VOID`.

## Mass properties

The car's mass properties combine the body, the inclusions, and the wheels, which
are point masses at the axle centers.  For each quantity
$q \in \{1, x, y, x^2 + y^2\}$,

$$
\int q\, dm = \rho_b t \int_\text{body} q\, dA
+ \sum_k (\rho_k - \rho_b)\, d_k \int_{\text{rectangle } k} q\, dA
+ m_w \left(n_r\, q(0, 0) + n_f\, q(w, 0)\right),
$$

with:
- the body density $\rho_b$ and thickness $t$;
- each inclusion rectangle's density $\rho_k$ and depth $d_k$;
- the wheel mass $m_w$.

The results:
- the mass is $M = \int dm$;
- the center of gravity is $\mathbf{x}_g = (\int x\, dm, \int y\, dm) / M$;
- the pitching moment of inertia about the center of gravity is
  $I_b = \int (x^2 + y^2)\, dm - M |\mathbf{x}_g|^2$.

The integrals over the body reduce to integrals of polynomials in $h(x)$ along the
car, which Gauss-Legendre quadrature on each span of the spline computes exactly.
The rectangles have closed forms.  Everything is a differentiable function of the
control heights and the inclusion parameters.

The front offset, which places the car at the start and finish, is the front of
the profile, $x_f$.

## Building the car

[`GeometricCar`][pwdsim.car.GeometricCar] takes the profile, the inclusions (listed
from the rear of the car to the front), and the properties that don't come from
the geometry:

```python
import pwdsim
from pwdsim.geometry import PINE

car = pwdsim.GeometricCar(
    profile, [weight],
    thickness=1.75 * INCH, body_density=PINE, wheel_mass=2.6 * GRAM,
    wheelbase=4.375 * INCH, ...,  # wheels, axles, friction, and drag
)
car.mass, car.cg, car.body_inertia  # computed from the geometry
```

### Mixing in numbers

Any computed property (`mass`, `cg`, `body_inertia`, or `front_offset`) can be
given as a number instead, which replaces the geometric value:

```python
car = pwdsim.GeometricCar(profile, [weight], ..., mass=5 * OUNCE)
car.overrides  # ["mass"]
```

This is useful for checking the geometry against a measured car, or for holding a
property fixed.  The drag coefficient and frontal area are plain numbers for now.
A later version will compute them from the shape too, with the same option to
override them.

## Optimizing the geometry

The geometry's parameters are car parameters like any other, so
[`optimize`][pwdsim.optimization.optimize] can tune them by their dotted names:

| Variable | What it tunes |
| --- | --- |
| `profile.heights` | the control heights of the body's top |
| `inclusions.0.position` | where the first inclusion is along the car |
| `inclusions.0.height`, `inclusions.0.length` | how deep a pocket goes, and how much weight it holds |
| `inclusions.0.depth` | how thick the weight is across the car |
| `inclusions.0.x0`, `inclusions.0.y1`, ... | the edges of a `Region` |

### Bounds

[`Profile.bounds`][pwdsim.geometry.Profile.bounds] gives the bounds for the control
heights.  They run from a minimum height to the block height, which keeps the body
inside the block.  Its `fixed` argument holds the profile fixed over ranges of
$x$, for example to leave wood around the axle slots:

```python
heights = car.profile.bounds(
    fixed=[(-0.5 * INCH, 0.5 * INCH), (3.875 * INCH, 4.875 * INCH)],
    minimum=0.25 * INCH,
)
pwdsim.optimize(sim, {"profile.heights": heights, ...}, ...)
```

Control heights whose Greville abscissae fall in a fixed range get equal lower and
upper bounds, which fixes them.  The Greville abscissa is where each control height
has the most influence.

### Constraints

[`pwdsim.constraints`][pwdsim.constraints] has constraints for the inclusions:

| Constraint | Keeps |
| --- | --- |
| [`regions_inside_body(margin)`][pwdsim.constraints.regions_inside_body] | each inclusion inside the body, with at least `margin` of wood around it, and each pocket's weight inside its pocket |
| [`regions_within_thickness()`][pwdsim.constraints.regions_within_thickness] | each inclusion no deeper than the body is thick |
| [`regions_dont_overlap()`][pwdsim.constraints.regions_dont_overlap] | the inclusions from overlapping, in the order they're listed |
| [`regions_avoid_axles(car, clearance)`][pwdsim.constraints.regions_avoid_axles] | each inclusion clear of the axle slots, in the section of the car where it starts |

`regions_inside_body` checks the top of each inclusion against the profile with a
smooth minimum over sample points, so it is differentiable.  The other three are
linear in the parameters.  The general constraints (`bsa_rules()`, `max_mass`,
`cg_between_axles`, and the default lift-off constraint) work unchanged, since the
mass and center of gravity are now computed.

Because `regions_avoid_axles` keeps each inclusion in the section where it starts,
a weight can't move from one side of an axle to the other.  Start it in the
section you want it in.
