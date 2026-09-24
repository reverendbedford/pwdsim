# Cars

A car tells the simulator how it's built: how heavy it is and where that weight
sits, how its axles and wheels are arranged, and how much friction and drag it
has.  This page describes the car model, the essential properties every car
provides, and the car classes in pwdsim.  The [forward model
example](examples/forward_model.py) sets up a typical car, and the
[API reference](api.md#cars) documents every class.

## The car model

pwdsim models a car as a rigid body riding on two axles.  Its position on the track is
given by $s$, the location of the rear wheel contact point along the
[track](tracks.md).  The front wheel contact follows from the car's geometry.

Car dimensions are measured in a **body frame** attached to the car:

- the origin is at the center of the rear axle;
- the first axis points toward the center of the front axle;
- the second axis is perpendicular to the first, pointing up, away from the track.

So a center of gravity at $\mathbf{x}_g = (1\ \mathrm{in}, 0.4\ \mathrm{in})$ sits
1 in ahead of the rear axle and 0.4 in above the line through the axle centers.
The body frame pitches with the car as it moves through the curve, so
$\mathbf{x}_g$ is fixed however the car is oriented.

## Essential properties

Every car provides these properties, which are everything the simulation needs to
know about it:

| Property | Symbol | Description | SI unit |
| --- | --- | --- | --- |
| `cg` | $\mathbf{x}_g$ | Center of gravity relative to the rear axle, in the body frame | m |
| `wheelbase` | $w$ | Distance between the axle centers | m |
| `front_offset` | $d$ | Distance from the rear axle to the front of the car | m |
| `mass` | $M$ | Total mass, including the wheels | kg |
| `body_inertia` | $I_b$ | Pitching moment of inertia about the center of gravity | kg m² |
| `n_rear_wheels`, `n_front_wheels` | $n_r$, $n_f$ | Number of wheels on the track at each axle | - |
| `rear_wheel_inertia`, `front_wheel_inertia` | $I_r$, $I_f$ | Moment of inertia of each wheel about its axle | kg m² |
| `rear_wheel_radius`, `front_wheel_radius` | $r_r$, $r_f$ | Wheel radius | m |
| `rear_axle_radius`, `front_axle_radius` | $a_r$, $a_f$ | Axle radius | m |
| `rear_axle_friction`, `front_axle_friction` | $\mu_r$, $\mu_f$ | Wheel to axle friction coefficient | - |
| `frontal_area` | $A$ | Cross-sectional area facing the air | m² |
| `drag_coefficient` | $C_d$ | Aerodynamic drag coefficient | - |
| `rolling_friction` | $c_r$ | Rolling friction coefficient of the wheels on the track | - |

A few notes:

- **Front offset.**  The simulation positions a car at the start, and decides when
  it finishes, using the front of the car.  Only the distance from the rear axle to
  the front matters, not the total length of the car.
- **Wheel counts.**  Cars normally have two wheels on each axle.  A popular trick is
  to raise one front wheel so it never touches the track, giving
  $n_f = 1$.  The wheel inertias and friction apply per wheel, so the counts
  scale them.
- **Axle friction.**  Friction between a wheel and its axle acts at the axle
  radius, so it resists the car with a force of $\mu N a / r$ at the wheel rim, for
  a normal load $N$.  Thin axles and large wheels reduce it.

Values are always SI inside pwdsim.  To work in customary units, multiply by the
constants in [`pwdsim.units`][pwdsim.units] (`5 * OUNCE`, `4.375 * INCH`) and
divide to convert back.

### Typical values

For a legal car built from the standard BSA kit:

| Property | Typical value | Notes |
| --- | --- | --- |
| `mass` | 5 oz (0.142 kg) | the usual race limit |
| `wheelbase` | 4 3/8 in | kit axle slot spacing |
| `front_offset` | 6 1/8 in | 7 in block with the rear slot 7/8 in from the back |
| `cg` | (1.0 in, 0.4 in) | fast cars often put the center of gravity 3/4 to 1 in ahead of the rear axle |
| `body_inertia` | $3.9 \times 10^{-4}$ kg m² | uniform 7 in by 1.25 in block |
| wheel radius | 0.595 in | 95.0 mm circumference |
| wheel inertia | $3.4 \times 10^{-7}$ kg m² | 2.6 g wheel, $I \approx 0.58\, m r^2$ |
| axle radius | 0.0435 in | 0.087 in diameter axles |
| axle friction | 0.1 | lubricated; about 0.24 dry |
| `rolling_friction` | 0.002 | |
| `drag_coefficient` | 0.4 | |
| `frontal_area` | 0.0014 m² | uncut block |

Most of these come from the
[Wikibooks derby physics page](https://en.wikibooks.org/wiki/How_To_Build_a_Pinewood_Derby_Car/Physics).
They're starting points, not measurements of any particular car.

## Car classes

### `Car`

[`Car`][pwdsim.car.Car] is the abstract base class that defines the interface.
Each essential property is an abstract, read-only property.  A concrete car class
decides where the values come from, whether that's stored values, measurements, or
calculations from a detailed description of the car.  The simulation only ever uses
these properties, so it works with any car class.

`Car` is also a `torch.nn.Module`.  The parameters of a concrete car, whatever they
are, are available from `car.parameters()` to hand to a torch optimizer.  Because
the properties are computed with torch operations from those parameters, gradients
of race times flow back through them.

`Car` provides two helper methods:

- [`summary`][pwdsim.car.Car.summary] returns the current property values as plain
  numbers, for display.
- [`check`][pwdsim.car.Car.check] raises an error if the properties are not
  physically meaningful: for example a negative mass, axles larger than the wheels,
  or a front axle ahead of the front of the car.

### `SimpleCar`

[`SimpleCar`][pwdsim.car.SimpleCar] is the simplest car: you give it every
essential property directly, and it stores each one (except the wheel counts) as a
float64 `torch.nn.Parameter`.  The parameters are named after the properties with a
trailing underscore, so the `mass` property returns the `mass_` parameter.

```python
import pwdsim
from pwdsim.units import GRAM, INCH, OUNCE

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
    n_front_wheels=1,  # raise a front wheel
)
```

Every parameter is trainable by default.  Most optimization studies only vary a few
of them, so turn off the gradient for the ones that should stay fixed:

```python
for parameter in car.parameters():
    parameter.requires_grad_(False)
car.cg_.requires_grad_(True)  # only optimize the center of gravity
```

`SimpleCar` doesn't enforce any bounds while optimizing.  The optimization routines
will be responsible for keeping the parameters physical and within the race rules.

### Writing a new car class

To describe a car in another way, subclass `Car`, store the fundamental parameters,
and implement each essential property as a calculation from them.

For small variations it's often easier to extend `SimpleCar`.  For example, here's
a car whose wheels and axles are the same front and rear.  It shares one parameter
between each front and rear pair, so an optimizer can't make them differ:

```python
from pwdsim import SimpleCar


class MatchedWheelCar(SimpleCar):
    """A car with identical front and rear wheels and axles."""

    def __init__(
        self, wheel_radius, wheel_inertia, axle_radius, axle_friction, **kwargs
    ):
        super().__init__(
            rear_wheel_radius=wheel_radius,
            front_wheel_radius=wheel_radius,
            rear_wheel_inertia=wheel_inertia,
            front_wheel_inertia=wheel_inertia,
            rear_axle_radius=axle_radius,
            front_axle_radius=axle_radius,
            rear_axle_friction=axle_friction,
            front_axle_friction=axle_friction,
            **kwargs,
        )
        # Share one parameter between the front and rear wheels
        self.front_wheel_radius_ = self.rear_wheel_radius_
        self.front_wheel_inertia_ = self.rear_wheel_inertia_
        self.front_axle_radius_ = self.rear_axle_radius_
        self.front_axle_friction_ = self.rear_axle_friction_
```

A car built from its shape would instead derive its mass, center of gravity,
inertia, and frontal area from a spline profile of the body and the density of the
wood and weights.
