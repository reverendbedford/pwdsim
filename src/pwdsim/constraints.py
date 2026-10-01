"""Constraints for design optimization.

A [`Constraint`][pwdsim.constraints.Constraint] bounds a differentiable function of
the car and, optionally, of its run: `lower <= fun(car, run) <= upper`.  This module
provides the standard ones:

- [`lift_off`][pwdsim.constraints.lift_off]: keep the wheels on the track.
  [`optimize`][pwdsim.optimization.optimize] includes it by default.
- [`cg_between_axles`][pwdsim.constraints.cg_between_axles]: keep the center of
  gravity between the axles.
- [`max_mass`][pwdsim.constraints.max_mass]: a maximum mass.
- [`bsa_rules`][pwdsim.constraints.bsa_rules]: the common Scout race rules.

For a [`GeometricCar`][pwdsim.car.GeometricCar] there are also constraints on its
inclusions (weights and voids):

- [`regions_inside_body`][pwdsim.constraints.regions_inside_body]: inside the body.
- [`regions_within_thickness`][pwdsim.constraints.regions_within_thickness]: no
  deeper than the body is thick.
- [`regions_dont_overlap`][pwdsim.constraints.regions_dont_overlap]: not
  overlapping each other.
- [`regions_avoid_axles`][pwdsim.constraints.regions_avoid_axles]: clear of the
  axle slots.

The bounding box of the body, the uncut block, is handled by bounds on its control
heights from [`Profile.bounds`][pwdsim.geometry.Profile.bounds].
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from pwdsim.car import Car, GeometricCar
from pwdsim.geometry import Region, WeightPocket
from pwdsim.simulation import Run
from pwdsim.types import DTYPE
from pwdsim.units import INCH, OUNCE


@dataclass
class Constraint:
    """A constraint `lower <= fun(car, run) <= upper` on a car design.

    Constraints on the car alone, which don't use the run, are cheap: their
    gradients don't need the adjoint method.

    Attributes:
        fun: a differentiable function of the car and its run, returning a scalar or
            a tensor of values, each of which is constrained.
        lower: lower bound on the values, in the units of `fun`.
        upper: upper bound on the values, in the units of `fun`.
        name: a name for reports.
        scale: a typical size of the values, used to scale the constraint for the
            optimizer.
        linear: whether `fun` is linear in the car parameters, so the optimizer
            doesn't need to approximate its curvature.
    """

    fun: Callable[[Car, Run], torch.Tensor]
    lower: float = -math.inf
    upper: float = math.inf
    name: str = "constraint"
    scale: float = 1.0
    linear: bool = False

    def __call__(self, car: Car, run: Run) -> torch.Tensor:
        """The constrained values, flattened."""
        return torch.as_tensor(self.fun(car, run)).reshape(-1)


def lift_off(tolerance: float = 0.0, sharpness: float = 500.0) -> Constraint:
    """Keep the wheels on the track: the normal forces at both axles must stay at
    least `tolerance` all the way to the finish.

    Uses the smooth minimum
    [`Run.smooth_min_normal_force`][pwdsim.simulation.Run.smooth_min_normal_force],
    which is never larger than the exact minimum.

    Args:
        tolerance: the smallest allowed normal force, in newtons.  Zero just keeps
            the wheels on the track; a positive value leaves a margin for a more
            robust design.
        sharpness: sharpness of the smooth minimum, in 1/N.
    """
    return Constraint(
        fun=lambda car, run: run.smooth_min_normal_force(sharpness),
        lower=tolerance,
        name="lift_off",
        scale=0.1,
    )


def cg_between_axles(margin: float = 0.0) -> Constraint:
    """Keep the center of gravity between the axles, at least `margin` from each.

    Args:
        margin: the smallest allowed distance from the center of gravity to either
            axle, along the car, in meters.
    """
    return Constraint(
        fun=lambda car, run: torch.stack(
            [car.cg[0] - margin, car.wheelbase - margin - car.cg[0]]
        ),
        lower=0.0,
        name="cg_between_axles",
        scale=INCH,
        linear=True,
    )


def max_mass(limit: float = 5 * OUNCE) -> Constraint:
    """A maximum total mass for the car.

    Args:
        limit: the maximum mass, in kilograms.
    """
    return Constraint(
        fun=lambda car, run: car.mass,
        upper=limit,
        name="max_mass",
        scale=OUNCE,
        linear=True,
    )


def bsa_rules() -> list[Constraint]:
    """The common Scout race rules that apply to the car properties: a maximum mass
    of 5 oz, with the center of gravity between the axles.

    Returns a list, so it's easy to add more constraints.
    """
    return [max_mass(5 * OUNCE), cg_between_axles()]


def _inclusions(car: Car):
    if not isinstance(car, GeometricCar):
        raise TypeError("Region constraints need a GeometricCar")
    return list(car.inclusions)


def _extent(inclusion, bottom):
    """The x and y extent of an inclusion's rectangles."""
    rectangles = inclusion.rectangles(bottom)
    return (
        rectangles[0].x0,
        rectangles[0].x1,
        torch.min(torch.stack([r.y0 for r in rectangles])),
        torch.max(torch.stack([r.y1 for r in rectangles])),
    )


def regions_inside_body(
    margin: float = 0.0, samples: int = 16, sharpness: float = 1e4
) -> Constraint:
    """Keep each inclusion of a [`GeometricCar`][pwdsim.car.GeometricCar] inside the
    body, at least `margin` from its outline.

    The top of each inclusion must stay below the top of the profile across its
    width.  This uses a smooth minimum of the clearance at `samples` points, which
    is never larger than the exact minimum over those points.  The inclusions must
    also stay within the length of the block, regions above its bottom, and the
    weight of a [`WeightPocket`][pwdsim.geometry.WeightPocket] inside its pocket.

    Args:
        margin: the smallest allowed wall of body material around an inclusion, in
            meters.
        samples: number of points across each inclusion at which to check the top.
        sharpness: sharpness of the smooth minimum, in 1/m.
    """
    fractions = torch.linspace(0, 1, samples, dtype=torch.float64)

    def fun(car, run):
        profile = car.profile
        values = []
        for inclusion in _inclusions(car):
            x0, x1, y0, y1 = _extent(inclusion, profile.bottom)
            clearance = profile.top(x0 + (x1 - x0) * fractions) - y1 - margin
            values.append(-torch.logsumexp(-sharpness * clearance, 0) / sharpness)
            values.append(x0 - profile.x_rear - margin)
            values.append(profile.x_front - margin - x1)
            if isinstance(inclusion, Region):
                values.append(y0 - profile.bottom - margin)
            if isinstance(inclusion, WeightPocket):
                values.append(inclusion.height - inclusion.length)
        return torch.stack(values)

    return Constraint(fun=fun, lower=0.0, name="regions_inside_body", scale=INCH)


def regions_within_thickness() -> Constraint:
    """Keep each inclusion of a [`GeometricCar`][pwdsim.car.GeometricCar] with a
    given depth no deeper than the body is thick."""

    def fun(car, run):
        values = [
            car.thickness - inclusion.depth
            for inclusion in _inclusions(car)
            if inclusion.depth is not None
        ]
        return torch.stack(values) if values else torch.zeros(0, dtype=DTYPE)

    return Constraint(
        fun=fun, lower=0.0, name="regions_within_thickness", scale=INCH, linear=True
    )


def regions_dont_overlap() -> Constraint:
    """Keep the inclusions of a [`GeometricCar`][pwdsim.car.GeometricCar] from
    overlapping along the car.

    The inclusions must be listed from the rear of the car to the front, and keep
    that order: each one must end before the next one starts.
    """

    def fun(car, run):
        inclusions = _inclusions(car)
        extents = [_extent(inclusion, car.profile.bottom) for inclusion in inclusions]
        gaps = [
            after[0] - before[1]
            for before, after in zip(extents[:-1], extents[1:], strict=True)
        ]
        return torch.stack(gaps) if gaps else torch.zeros(0, dtype=DTYPE)

    return Constraint(
        fun=fun, lower=0.0, name="regions_dont_overlap", scale=INCH, linear=True
    )


def regions_avoid_axles(
    car: GeometricCar, clearance: float = 0.125 * INCH
) -> Constraint:
    """Keep the inclusions of a [`GeometricCar`][pwdsim.car.GeometricCar] clear of
    the axle slots.

    Each inclusion stays in the section of the car where it starts: behind the rear
    axle, between the axles, or ahead of the front axle, at least `clearance` from
    the axle centers.

    Args:
        car: the car, to find which section each inclusion starts in.
        clearance: the smallest allowed distance from an inclusion to an axle
            center, in meters.

    Raises:
        ValueError: if an inclusion starts over an axle slot.
    """
    w = car.wheelbase.item()
    bays = []
    with torch.no_grad():
        for inclusion in _inclusions(car):
            x0, x1, _, _ = (v.item() for v in _extent(inclusion, car.profile.bottom))
            if x1 <= -clearance:
                bays.append("rear")
            elif x0 >= clearance and x1 <= w - clearance:
                bays.append("middle")
            elif x0 >= w + clearance:
                bays.append("front")
            else:
                raise ValueError("An inclusion starts over an axle slot")

    def fun(car, run):
        values = []
        w = car.wheelbase
        for inclusion, bay in zip(_inclusions(car), bays, strict=True):
            x0, x1, _, _ = _extent(inclusion, car.profile.bottom)
            if bay == "rear":
                values.append(-clearance - x1)
            elif bay == "middle":
                values += [x0 - clearance, w - clearance - x1]
            else:
                values.append(x0 - w - clearance)
        return torch.stack(values) if values else torch.zeros(0, dtype=DTYPE)

    return Constraint(
        fun=fun, lower=0.0, name="regions_avoid_axles", scale=INCH, linear=True
    )
