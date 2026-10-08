"""Aerodynamic drag from the car's profile.

A pinewood derby car is a blunt body, a little wider than it is tall, running close
to the track at a Reynolds number of about $5 \\times 10^4$.  Potential-flow (panel)
methods predict no drag at all, and coupling them with a boundary layer calculation
breaks down behind the blunt rear of the car, where the flow separates.  So pwdsim
estimates the drag with a semi-empirical *component buildup*, in the style of
Hoerner's *Fluid-Dynamic Drag*: the drag area $C_d A$ is the sum of

- **skin friction** on the wetted surface, from flat-plate laws;
- **forebody pressure drag**, from the blunt front face and the forward-facing
  slopes of the profile;
- **base drag** behind the blunt rear face and any steep rear-facing slopes, from
  Hoerner's correlation with the forebody drag;
- **wheel drag**, for the exposed, rotating wheels.

[`DragModel`][pwdsim.aerodynamics.DragModel] holds the empirical coefficients and
computes the drag of a [`Profile`][pwdsim.geometry.Profile].  The model ignores the
weight pockets, the track floor and guide rail, and the variation of the Reynolds
number during a run.  It aims for the right trends and plausible magnitudes rather
than precision.
"""

import math
from dataclasses import dataclass

import torch

from pwdsim.geometry import Profile
from pwdsim.units import DEGREE, INCH


def _positive(z, width=1e-5):
    """A smooth version of max(z, 0)."""
    return (z + torch.sqrt(z**2 + width**2)) / 2


def _step(angle, start, end):
    """A smooth step from 0 below `start` to 1 above `end`."""
    t = torch.clamp((angle - start) / (end - start), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class DragModel:
    """A semi-empirical model of the drag of a car body from its side profile.

    For a profile with height $h(x)$, thickness $t$, and length $L$, with the height
    of the front face $h_f$ and the rear face $h_r$, the drag area is the sum of:

    1. **Skin friction**, $C_f S_\\text{wet}$.  The wetted area is both sides, the
       top, and the bottom,
       $S_\\text{wet} = 2 \\int h\\, dx + t \\int \\sqrt{1 + h'^2}\\, dx + t L$.
       The friction coefficient comes from the Reynolds number at a reference
       speed, $\\mathrm{Re} = V L / \\nu$: the laminar Blasius law
       $C_f = 1.328 / \\sqrt{\\mathrm{Re}}$, or above the transition Reynolds number
       the mixed law $C_f = 0.074\\, \\mathrm{Re}^{-1/5} - 1742 / \\mathrm{Re}$.
    2. **Forebody pressure drag**, $t\\, (C_\\text{face} H_\\text{blunt} +
       C_\\text{ramp} H_\\text{ramp})$.  Forward-facing slopes steeper than
       `separated_angle` count as blunt, like the front face, and those gentler than
       `attached_angle` as ramps, with a smooth blend in between:
       $H_\\text{blunt} = h_f + \\int s\\, (-h')^+ dx$ and
       $H_\\text{ramp} = \\int (1 - s)\\, (-h')^+ dx$, where $s$ is the blend.
    3. **Base drag**, behind the rear face and any steep rear-facing slopes, with
       base area $A_b = t\\, (h_r + \\int s\\, (h')^+ dx)$.  Hoerner's correlation
       gives the base drag coefficient from the forebody drag (friction and
       pressure) on the base area, $C_{D,b} = 0.029 / \\sqrt{C_{D,f}}$, so a cleaner
       forebody has a stronger base suction.  Gently tapering tails stay attached
       and add only friction.
    4. **Wheels**, each a $2r \\times b$ rectangle facing the flow with drag
       coefficient $C_\\text{wheel}$, as for exposed rotating wheels in ground
       contact.  The rear wheels run in the wake of the front wheels, so their drag
       is scaled by `rear_wheel_factor`, as for bodies in tandem.

    The frontal area is $A = t H_\\text{max}$, with a smooth maximum of the height,
    and the drag coefficient is $C_d = C_d A / A$.

    Attributes:
        face: drag coefficient of a blunt, sharp-edged face.
        ramp: pressure drag coefficient of gently sloping forward-facing surfaces.
        base_constant: the constant in Hoerner's base drag correlation.
        attached_angle: slopes gentler than this keep the flow attached, in radians.
        separated_angle: slopes steeper than this separate the flow, in radians.
        wheel: drag coefficient of each wheel, on its frontal area.
        wheel_width: width of the wheel treads.
        rear_wheel_factor: fraction of their drag the rear wheels keep, in the wake
            of the front wheels (an estimate).
        reference_speed: speed for the Reynolds number.
        kinematic_viscosity: kinematic viscosity of air.
        transition_reynolds: Reynolds number of transition to turbulence.
        sharpness: sharpness of the smooth maximum height, in 1/m.
    """

    face: float = 0.8
    ramp: float = 0.1
    base_constant: float = 0.029
    attached_angle: float = 10 * DEGREE
    separated_angle: float = 25 * DEGREE
    wheel: float = 0.6
    wheel_width: float = 0.3 * INCH
    rear_wheel_factor: float = 0.5
    reference_speed: float = 4.5
    kinematic_viscosity: float = 1.5e-5
    transition_reynolds: float = 5e5
    sharpness: float = 1e5

    def friction_coefficient(self, length: float) -> float:
        """The flat-plate skin friction coefficient for a body of this length."""
        reynolds = self.reference_speed * length / self.kinematic_viscosity
        if reynolds <= self.transition_reynolds:
            return 1.328 / math.sqrt(reynolds)
        return 0.074 * reynolds**-0.2 - 1742 / reynolds

    def breakdown(
        self,
        profile: Profile,
        thickness: torch.Tensor,
        wheel_radii: tuple[torch.Tensor, torch.Tensor] = (0.0, 0.0),
        wheels: tuple[int, int] = (0, 0),
    ) -> dict[str, torch.Tensor]:
        """The drag of a car body, broken down by component.

        Args:
            profile: the side profile of the body.
            thickness: the thickness of the body.
            wheel_radii: the radii of the rear and front wheels.
            wheels: the numbers of rear and front wheels.

        Returns:
            The drag areas (in m²) of the `friction`, `forebody`, `base`, and
            `wheels` components, and their `total`, along with the
            `frontal_area` and the `drag_coefficient`.
        """
        x, w, h, slope = profile.samples()
        angle = torch.atan(slope.abs())
        separated = _step(angle, self.attached_angle, self.separated_angle)
        falling, rising = _positive(-slope), _positive(slope)
        h_front = profile.height(profile.x_front)
        h_rear = profile.height(profile.x_rear)

        wetted = (
            2 * torch.sum(w * h)
            + thickness * torch.sum(w * torch.sqrt(1 + slope**2))
            + thickness * profile.length
        )
        friction = self.friction_coefficient(profile.length) * wetted

        blunt = h_front + torch.sum(w * separated * falling)
        gentle = torch.sum(w * (1 - separated) * falling)
        forebody = thickness * (self.face * blunt + self.ramp * gentle)

        base_area = thickness * (h_rear + torch.sum(w * separated * rising))
        forebody_on_base = (friction + forebody) / base_area
        base = self.base_constant / torch.sqrt(forebody_on_base) * base_area

        factors = (self.rear_wheel_factor, 1.0)
        wheel_drag = sum(
            factor * n * self.wheel * 2 * torch.as_tensor(r) * self.wheel_width
            for factor, n, r in zip(factors, wheels, wheel_radii, strict=True)
        )

        heights = profile.height(
            torch.linspace(profile.x_rear, profile.x_front, 200, dtype=x.dtype)
        )
        max_height = torch.logsumexp(self.sharpness * heights, 0) / self.sharpness
        frontal_area = thickness * max_height
        total = friction + forebody + base + wheel_drag
        return {
            "friction": friction,
            "forebody": forebody,
            "base": base,
            "wheels": wheel_drag + 0 * total,
            "total": total,
            "frontal_area": frontal_area,
            "drag_coefficient": total / frontal_area,
        }
