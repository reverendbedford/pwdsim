# Track models

A track tells the simulator where the car's wheels are.  This page covers how pwdsim
describes track geometry, the spline model used to build tracks, and the standard
tracks included with the package.  The [forward model
example](examples/forward_model.py) shows these tracks in use, and the
[API reference](api.md#tracks) documents every function.

## Track geometry

pwdsim models a track as a curve in the vertical plane, parameterized by the arc
length $s$ measured along the track from its top end.  The curve traces the path of
the **wheel contact points**, not the track surface as a whole or the axle centers.
The axle centers sit one wheel radius off this curve, along its normal.

Every track provides these functions of $s$:

| Quantity | Symbol | Method |
| --- | --- | --- |
| Angle from the horizontal | $\theta(s)$ | [`angle`][pwdsim.track.Track.angle] |
| Curvature | $\kappa(s) = \theta'(s)$ | [`curvature`][pwdsim.track.Track.curvature] |
| Curvature derivative | $\kappa'(s)$ | [`curvature_derivative`][pwdsim.track.Track.curvature_derivative] |
| Curvature second derivative | $\kappa''(s)$ | [`curvature_second_derivative`][pwdsim.track.Track.curvature_second_derivative] |
| Position | $(x(s), y(s))$ | [`position`][pwdsim.track.Track.position] |
| Height | $y(s)$ | [`height`][pwdsim.track.Track.height] |

Because $s$ is arc length, the position follows from the angle alone:

$$
x(s) = \int_0^s \cos\theta(\sigma)\, d\sigma, \qquad
y(s) = y_0 + \int_0^s \sin\theta(\sigma)\, d\sigma.
$$

The track starts at $x(0) = 0$, and the constant $y_0$ is chosen so the track ends at
$y(L) = 0$, where $L$ is the total track length.  Heights are therefore measured
from the end of the track, which is usually the flat run.

The angle is negative going downhill.  A curve that bends upward, like the transition
from the ramp to the flat run, has positive curvature.

### Start and finish

A track also marks two locations along its length:

- $s_\mathrm{start}$, the **start pin**.  The fronts of the cars rest against it at
  the start of a heat, so cars with different lengths and wheelbases start with
  their axles at different locations.
- $s_\mathrm{finish} \le L$, the **finish line**.  A car finishes when its front
  crosses this line.  The finish is usually before the end of the track, which has
  a braking section.

### Beyond the ends

Past either end of $[0, L]$, tracks continue as straight lines at their end angles,
with zero curvature.  The simulation never runs the car off the track, but it does
evaluate the geometry slightly outside it, for example while solving for the front
axle location when the rear wheels are near an end.

## Spline tracks

[`SplineTrack`][pwdsim.track.SplineTrack] describes the angle $\theta(s)$ as a
piecewise polynomial.  The track is divided into segments at a set of knots
$0 = s_0 < s_1 < \dots < s_n = L$.  At each knot you give the angle $\theta_i$ and the
curvature $\kappa_i$, and optionally the first two derivatives of the curvature,
$\kappa'_i$ and $\kappa''_i$ (zero by default).  On each segment the angle is the
unique degree 7 (septic) polynomial matching those four values at both ends: a
Hermite spline.  With $h_i = s_{i+1} - s_i$ and the local coordinate
$t = (s - s_i)/h_i \in [0, 1]$,

$$
\theta(s) = \sum_{j=0}^{3} \left[
H_{0j}(t)\, h_i^j\, \theta^{(j)}_i + H_{1j}(t)\, h_i^j\, \theta^{(j)}_{i+1}
\right],
$$

where $\theta^{(j)}$ is the $j$th derivative of the angle ($\theta$, $\kappa$,
$\kappa'$, $\kappa''$), and the basis polynomials $H_{0j}$ and $H_{1j}$ have
$j$th derivative 1 at $t = 0$ and $t = 1$ respectively, with all their other
derivatives up to the third zero at both ends.

The angle, the curvature, and the first two curvature derivatives are all
continuous along the track.

### Why so smooth?

The car's acceleration depends on the curvature and its derivative, and the Jacobian
the time integrator needs depends on $\kappa''$ too (see
[Forward simulation](simulation.md)).  A jump in any of them adds an error to the
time step where an axle crosses it.  That error depends on exactly where the
crossing falls within the step, which in turn depends on the car's design.  So the
finish time picks up a small sawtooth as a function of the design, and its gradients
pick up much larger noise.  With a cubic spline, which lets $\kappa'$ jump, the
gradients of the smaller sensitivities could be off by a factor of several.  With
the septic spline they converge smoothly.

### Why specify the curvature?

Real tracks are built from straight sections and curves.  Specifying the curvature at
each knot represents those pieces exactly:

- a **straight** section has the same angle at both ends and zero curvature, so the
  polynomial reduces to a constant;
- a **circular arc** of radius $R$ has curvature $1/R$ at both ends and an angle
  change of $h_i / R$, so the polynomial reduces to a straight line in $\theta$;
- an **easement**, where the curvature changes smoothly from one value to another,
  is reproduced exactly as long as the curvature follows a polynomial of degree 6
  or less.

A standard interpolating spline, which only uses the knot angles, can't do this.  It
also forces derivatives of the curvature to be continuous, which makes it overshoot
and wiggle next to the straight sections.

### Interpolating splines

When you only have angles, for example angles measured along an existing track,
[`SplineTrack.interpolate`][pwdsim.track.SplineTrack.interpolate] builds the classic
$C^2$ cubic spline.  It solves for the knot curvatures that make $\kappa'$
continuous at every interior knot, with the curvatures at the two ends of the track
fixed (zero by default).  The track then uses the cubic's curvature derivatives at
the knots.  A cubic spline's $\kappa''$ jumps at the knots, so the track takes the
average there, which makes it follow the cubic closely while staying smooth.

### Computing the position

There is no closed form for the integrals of $\cos\theta$ and $\sin\theta$ when
$\theta$ is a polynomial, so the position is integrated numerically.  Gauss-Legendre
quadrature is applied within each segment: 16 points per segment by default, which
is accurate to around $10^{-9}$ m for typical tracks.  The positions at the knots are
computed once, when the track is created.  Evaluating the position at any $s$ then
costs one quadrature over part of a single segment.

Everything is computed with float64 pytorch operations, so the geometry is
differentiable with respect to $s$ and to the knot values.

## Standard tracks

### Ramp tracks

Most derby tracks share the same layout: a straight ramp, a curved transition, and a
long flat run.  [`ramp_track`][pwdsim.track.ramp_track] builds this shape from its
ramp length and angle, the transition radius $R$, the total length, and the start and
finish locations.

The transition is a circular arc of radius $R$ that turns the track from the ramp
angle to horizontal.  Joining an arc directly to a straight section would make the
curvature jump from $0$ to $1/R$, but a `SplineTrack` keeps the curvature and its
first two derivatives continuous.  So short **easements** (1 inch long by default)
join the arc to the straights.  Over each one the curvature follows the smooth step

$$
\kappa = \frac{1}{R}\left(10u^3 - 15u^4 + 6u^5\right),
$$

where $u$ runs from 0 to 1 along the easement, so that $\kappa'$ and $\kappa''$ are
zero at both ends.  Railway transition curves use the same idea.  Physically, an
easement means the load on the car's wheels builds up over a short distance instead
of jumping instantly.

### BestTrack

[`besttrack`][pwdsim.track.besttrack] approximates the aluminum tracks sold by
[BestTrack](https://www.besttrack.com/), one of the most common commercial tracks,
in its 35, 42, and 49 ft lengths.  It uses the manufacturer's published
dimensions from their [FAQ](https://www.besttrack.com/faq.htm) and
[specifications](https://www.besttrack.com/track_specs.html):

| Feature | Published value |
| --- | --- |
| Curve section | 41 in long, 48 in radius |
| Straight sections after the curve | 84 in each: 3, 4, or 5 of them |
| Stop section | 40 in |
| Start pin to end of track (42 ft) | about 37 ft 9 in |
| Racing distance, start pin to timer | 358, 442, or 526 in |
| Starting gate height | about 4 ft from the floor |

BestTrack does not publish the ramp angle.  pwdsim infers it by treating the whole
curve section as a circular arc: $41/48$ rad, or about 49°.  With that angle, the
start pin comes out about 44 in above the flat run.  That is consistent with the
quoted 4 ft gate height if the flat sections stand a few inches off the floor.

## Example

```python
import pwdsim
from pwdsim.plotting import plot_track
from pwdsim.units import DEGREE, FOOT, INCH

# A standard 42 ft BestTrack
track = pwdsim.besttrack(42)
print(track.height(track.s_start) / INCH)  # height of the start pin, in inches

# A custom ramp track, specified in customary units
custom = pwdsim.ramp_track(
    length=32 * FOOT,
    ramp_length=8 * FOOT,
    ramp_angle=30 * DEGREE,
    radius=5 * FOOT,
    s_start=1 * FOOT,
    s_finish=28 * FOOT,
)
plot_track(custom, unit="ft")
```
