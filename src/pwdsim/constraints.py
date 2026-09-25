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
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from pwdsim.car import Car
from pwdsim.simulation import Run
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
