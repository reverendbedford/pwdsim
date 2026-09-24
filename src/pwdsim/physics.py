"""Physics terms for the equation of motion.

The equation of motion comes from the Lagrangian of the car with non-conservative
forces from a Rayleigh dissipation function.  Its contributions are split into
modular terms of three kinds:

- [`KineticTerm`][pwdsim.physics.KineticTerm]: kinetic energy,
- [`PotentialTerm`][pwdsim.physics.PotentialTerm]: potential energy,
- [`DissipativeTerm`][pwdsim.physics.DissipativeTerm]: Rayleigh dissipation.

Each term supplies its contribution along with the analytic derivatives the
simulation needs, with respect to the generalized coordinates $q = (s, h_r, h_f)$
(see [`pwdsim.kinematics`][pwdsim.kinematics]).  Terms can be switched on and off
with their `enabled` flag.

**Kinetic energy** is expressed through *rates*: quantities $\\rho = J(q) \\cdot
\\dot{q}$ that are linear in the generalized velocities, such as the components of
the velocity of the center of gravity or the spin rate of a wheel.  A rate with
weight $m$ contributes $T = \\tfrac{1}{2} m \\rho^2$, and its Euler-Lagrange
inertial force along coordinate $i$ is $m J_i \\dot{\\rho}$, with
$\\dot\\rho = J_s \\ddot{s} + J_s' \\dot{s}^2$ on the track.  So a kinetic term only
provides $J_s$ and its derivatives with respect to $s$, and the lift components
$J_{h_r}$ and $J_{h_f}$ and their derivatives with respect to $s$.

**Friction** that is proportional to the normal force at an axle has the Rayleigh
dissipation function $F = c N |\\rho|$, for a coefficient $c$ and a rate $\\rho$.
These terms are expressed as [`NormalFriction`][pwdsim.physics.NormalFriction]
contributions, since the normal forces are themselves unknowns solved for along
with the acceleration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from pwdsim.car import Car
from pwdsim.kinematics import FIELDS, Configuration, Derivatives


@dataclass
class Environment:
    """Global parameters of the simulation.

    Attributes:
        g: gravitational acceleration.
        rho: air density.
    """

    g: float = 9.80665
    rho: float = 1.2


@dataclass
class Rate:
    """A rate $\\rho = J \\cdot \\dot{q}$ contributing $T = \\tfrac{1}{2} m \\rho^2$.

    Rates may be vectors, with the components in a trailing dimension, which
    contribute $T = \\tfrac{1}{2} m |\\rho|^2$.  Each field has that trailing
    dimension, even for a scalar rate (size 1).

    Attributes:
        weight: the weight $m$, e.g. a mass or a moment of inertia.
        ds: $J_s$, the rate per unit $\\dot{s}$.
        dss: $\\partial J_s / \\partial s$.
        dsss: $\\partial^2 J_s / \\partial s^2$.
        dhr: $J_{h_r}$, the rate per unit $\\dot{h}_r$.
        dhf: $J_{h_f}$, the rate per unit $\\dot{h}_f$.
        dhr_ds: $\\partial J_{h_r} / \\partial s$.
        dhf_ds: $\\partial J_{h_f} / \\partial s$.
    """

    weight: torch.Tensor
    ds: torch.Tensor
    dss: torch.Tensor
    dsss: torch.Tensor
    dhr: torch.Tensor
    dhf: torch.Tensor
    dhr_ds: torch.Tensor
    dhf_ds: torch.Tensor

    @classmethod
    def of(cls, weight, quantity: Derivatives, scalar: bool = False) -> "Rate":
        """The rate of change of a quantity with known derivatives, e.g. a point or
        an angle.  Scalar quantities get a trailing dimension of size 1."""
        fields = ("ds", "dss", "dsss", "dhr", "dhf", "dhr_ds", "dhf_ds")
        values = [getattr(quantity, f) for f in fields]
        if scalar:
            values = [v[..., None] for v in values]
        return cls(weight, *values)


@dataclass
class Potential:
    """A potential energy $V(q)$ and its derivatives.

    Attributes:
        value: $V$.
        ds: $\\partial V / \\partial s$.
        dss: $\\partial^2 V / \\partial s^2$.
        dhr: $\\partial V / \\partial h_r$.
        dhf: $\\partial V / \\partial h_f$.
        dhr_ds: $\\partial^2 V / \\partial h_r \\partial s$.
        dhf_ds: $\\partial^2 V / \\partial h_f \\partial s$.
    """

    value: torch.Tensor
    ds: torch.Tensor
    dss: torch.Tensor
    dhr: torch.Tensor
    dhf: torch.Tensor
    dhr_ds: torch.Tensor
    dhf_ds: torch.Tensor


@dataclass
class Dissipation:
    """Derivatives of a Rayleigh dissipation function $F(q, \\dot{q})$ that doesn't
    depend on the normal forces, on the track, where $\\dot q = (v, 0, 0)$.

    The generalized dissipative force is $-\\partial F / \\partial \\dot{q}$.

    Attributes:
        s: $D_s = \\partial F / \\partial \\dot{s}$.
        s_ds: $\\partial D_s / \\partial s$.
        s_dv: $\\partial D_s / \\partial v$.
        hr: $D_{h_r} = \\partial F / \\partial \\dot{h}_r$.
        hr_ds: $\\partial D_{h_r} / \\partial s$.
        hr_dv: $\\partial D_{h_r} / \\partial v$.
        hf: $D_{h_f} = \\partial F / \\partial \\dot{h}_f$.
        hf_ds: $\\partial D_{h_f} / \\partial s$.
        hf_dv: $\\partial D_{h_f} / \\partial v$.
    """

    s: torch.Tensor
    s_ds: torch.Tensor
    s_dv: torch.Tensor
    hr: torch.Tensor
    hr_ds: torch.Tensor
    hr_dv: torch.Tensor
    hf: torch.Tensor
    hf_ds: torch.Tensor
    hf_dv: torch.Tensor


REAR = 0
"""Index of the rear axle."""

FRONT = 1
"""Index of the front axle."""


@dataclass
class NormalFriction:
    """Friction proportional to the normal force $N$ at an axle, with the Rayleigh
    dissipation function $F = c N |\\rho|$ for a rate $\\rho$.

    The sign of the rate is regularized as
    $\\mathrm{sgn}(\\rho) \\approx \\mathrm{sgn}(J_s) \\tanh(v / v_\\mathrm{reg})$
    on the track, which keeps the equation of motion smooth as the car starts from
    rest.

    Attributes:
        axle: which axle's normal force, [`REAR`][pwdsim.physics.REAR] or
            [`FRONT`][pwdsim.physics.FRONT].
        coefficient: the coefficient $c$.
        rate: the quantity whose rate of change is $\\rho$, with its derivatives.
        regularization: the velocity $v_\\mathrm{reg}$.
    """

    axle: int
    coefficient: torch.Tensor
    rate: Derivatives
    regularization: float


class PhysicsTerm(nn.Module, ABC):
    """Base class for the terms of the equation of motion.

    Args:
        enabled: whether the term contributes to the equation of motion.
    """

    name: str = "term"
    """Name of the term, used to switch it on and off in a simulation."""

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled


class KineticTerm(PhysicsTerm):
    """A kinetic energy contribution, made up of rates."""

    @abstractmethod
    def rates(self, config: Configuration, car: Car, env: Environment) -> list[Rate]:
        """The rates making up this kinetic energy."""


class PotentialTerm(PhysicsTerm):
    """A potential energy contribution."""

    @abstractmethod
    def potential(self, config: Configuration, car: Car, env: Environment) -> Potential:
        """The potential energy and its derivatives."""


class DissipativeTerm(PhysicsTerm):
    """A Rayleigh dissipation function contribution.

    A term provides the part of its dissipation that doesn't depend on the normal
    forces through [`dissipation`][pwdsim.physics.DissipativeTerm.dissipation], and
    friction proportional to the normal forces through
    [`frictions`][pwdsim.physics.DissipativeTerm.frictions].
    """

    def dissipation(
        self, config: Configuration, v: torch.Tensor, car: Car, env: Environment
    ) -> Dissipation | None:
        """Derivatives of the dissipation function that doesn't depend on the
        normal forces, at speed $v = \\dot{s}$."""
        return None

    def frictions(
        self, config: Configuration, car: Car, env: Environment
    ) -> list[NormalFriction]:
        """Friction proportional to the normal forces."""
        return []


class Gravity(PotentialTerm):
    """Gravitational potential energy, $V = M g\\, y_g$."""

    name = "gravity"

    def potential(self, config, car, env):
        cg = config.cg
        weight = car.mass * env.g
        return Potential(
            value=weight * cg.value[..., 1],
            ds=weight * cg.ds[..., 1],
            dss=weight * cg.dss[..., 1],
            dhr=weight * cg.dhr[..., 1],
            dhf=weight * cg.dhf[..., 1],
            dhr_ds=weight * cg.dhr_ds[..., 1],
            dhf_ds=weight * cg.dhf_ds[..., 1],
        )


class Translation(KineticTerm):
    """Kinetic energy of the car moving with its center of gravity,
    $T = \\tfrac{1}{2} M |\\dot{\\mathbf{x}}_g|^2$."""

    name = "translation"

    def rates(self, config, car, env):
        return [Rate.of(car.mass, config.cg)]


class BodyRotation(KineticTerm):
    """Kinetic energy of the car body pitching, $T = \\tfrac{1}{2} I_b \\dot\\phi^2$."""

    name = "body_rotation"

    def rates(self, config, car, env):
        return [Rate.of(car.body_inertia, config.pitch, scalar=True)]


def _zeros_like_derivatives(value) -> Derivatives:
    zero = torch.zeros_like(value)
    return Derivatives(value, zero, zero, zero, zero, zero, zero, zero)


def rear_contact(config: Configuration) -> Derivatives:
    """Location of the rear wheel contact, $s$, with its derivatives."""
    contact = _zeros_like_derivatives(config.rear_track.location)
    contact.ds = torch.ones_like(contact.value)
    return contact


def wheel_angles(config: Configuration, car: Car) -> tuple[Derivatives, Derivatives]:
    """Rotation angles of the rear and front wheels, with their derivatives.

    A wheel of radius $r$ rolling without slip on the track, with its contact at
    arc length $c$, turns through the angle
    $\\Psi(c) = -(c - r\\theta(c)) / r$: its center moves $1 - r\\kappa$ times as fast
    as the contact point, along the offset curve.  The rear contact is at $s$ and
    the front contact at $\\sigma(q)$, which gives the derivatives by the chain rule.
    The angles are absolute: they include the pitching of the car.
    """
    # Rear wheels: Psi(s)
    track, r = config.rear_track, car.rear_wheel_radius
    zero = torch.zeros_like(track.location)
    rear = Derivatives(
        value=-(track.location - r * track.angle) / r,
        ds=-(1 - r * track.curvature) / r,
        dss=track.dcurvature + zero,
        dsss=track.ddcurvature + zero,
        dhr=zero,
        dhf=zero,
        dhr_ds=zero,
        dhf_ds=zero,
    )

    # Front wheels: Psi(sigma(q)), with g = dPsi/dsigma
    track, r = config.front_track, car.front_wheel_radius
    sigma = config.front_contact
    g = -(1 - r * track.curvature) / r
    dg, ddg = track.dcurvature, track.ddcurvature
    front = Derivatives(
        value=-(track.location - r * track.angle) / r,
        ds=g * sigma.ds,
        dss=dg * sigma.ds**2 + g * sigma.dss,
        dsss=ddg * sigma.ds**3 + 3 * dg * sigma.ds * sigma.dss + g * sigma.dsss,
        dhr=g * sigma.dhr,
        dhf=g * sigma.dhf,
        dhr_ds=dg * sigma.ds * sigma.dhr + g * sigma.dhr_ds,
        dhf_ds=dg * sigma.ds * sigma.dhf + g * sigma.dhf_ds,
    )
    return rear, front


class WheelSpin(KineticTerm):
    """Kinetic energy of the spinning wheels,
    $T = \\tfrac{1}{2} n_r I_r \\dot\\Psi_r^2 + \\tfrac{1}{2} n_f I_f \\dot\\Psi_f^2$,
    for wheels rolling without slip (see
    [`wheel_angles`][pwdsim.physics.wheel_angles])."""

    name = "wheel_spin"

    def rates(self, config, car, env):
        rear, front = wheel_angles(config, car)
        return [
            Rate.of(car.n_rear_wheels * car.rear_wheel_inertia, rear, scalar=True),
            Rate.of(car.n_front_wheels * car.front_wheel_inertia, front, scalar=True),
        ]


class Drag(DissipativeTerm):
    """Aerodynamic drag at the center of gravity, a force of
    $\\tfrac{1}{2} \\rho C_d A |\\dot{\\mathbf{x}}_g|^2$ opposing its motion.

    The Rayleigh dissipation function is
    $F = \\tfrac{1}{6} \\rho C_d A |\\dot{\\mathbf{x}}_g|^3$.
    """

    name = "drag"

    def dissipation(self, config, v, car, env):
        k = 0.5 * env.rho * car.drag_coefficient * car.frontal_area
        cg = config.cg
        speed = torch.linalg.norm(cg.ds, dim=-1)  # |x_g'|, so |x_g dot| = speed |v|
        along = torch.sum(cg.ds * cg.dss, dim=-1)  # x_g' . x_g''
        vv = v * v.abs()

        def lift(dh, dh_ds):
            projection = torch.sum(cg.ds * dh, dim=-1)
            return (
                k * speed * vv * projection,
                k
                * vv
                * (
                    along / speed * projection
                    + speed
                    * (
                        torch.sum(cg.dss * dh, dim=-1)
                        + torch.sum(cg.ds * dh_ds, dim=-1)
                    )
                ),
                2 * k * speed * v.abs() * projection,
            )

        hr, hr_ds, hr_dv = lift(cg.dhr, cg.dhr_ds)
        hf, hf_ds, hf_dv = lift(cg.dhf, cg.dhf_ds)
        return Dissipation(
            s=k * speed**3 * vv,
            s_ds=3 * k * speed * along * vv,
            s_dv=2 * k * speed**3 * v.abs(),
            hr=hr,
            hr_ds=hr_ds,
            hr_dv=hr_dv,
            hf=hf,
            hf_ds=hf_ds,
            hf_dv=hf_dv,
        )


class AxleFriction(DissipativeTerm):
    """Friction between the wheels and their axles.

    Friction at an axle of radius $a$ with normal load $N$ exerts a torque
    $\\mu N a$ opposing the spin of the wheels relative to the car body, so
    $F = \\mu N a |\\dot\\Psi - \\dot\\phi|$ at each axle.  The axle load is taken to be
    the normal force at the wheel contact, neglecting the weight of the wheels.

    Args:
        regularization: velocity scale $v_\\mathrm{reg}$ of the regularized sign of
            the friction force.
        enabled: whether the term contributes to the equation of motion.
    """

    name = "axle_friction"

    def __init__(self, regularization: float = 1e-3, enabled: bool = True):
        super().__init__(enabled)
        self.regularization = regularization

    def frictions(self, config, car, env):
        rear, front = wheel_angles(config, car)
        pitch = config.pitch
        return [
            NormalFriction(
                axle,
                friction * radius,
                Derivatives(*(getattr(angle, f) - getattr(pitch, f) for f in FIELDS)),
                self.regularization,
            )
            for axle, angle, friction, radius in (
                (REAR, rear, car.rear_axle_friction, car.rear_axle_radius),
                (FRONT, front, car.front_axle_friction, car.front_axle_radius),
            )
        ]


class RollingFriction(DissipativeTerm):
    """Rolling resistance of the wheels on the track.

    A force $c_r N$ opposes the motion of each contact point along the track, so
    $F = c_r N |\\dot c|$ for the contact location $c$ ($s$ at the rear, $\\sigma$ at
    the front).

    Args:
        regularization: velocity scale $v_\\mathrm{reg}$ of the regularized sign of
            the friction force.
        enabled: whether the term contributes to the equation of motion.
    """

    name = "rolling_friction"

    def __init__(self, regularization: float = 1e-3, enabled: bool = True):
        super().__init__(enabled)
        self.regularization = regularization

    def frictions(self, config, car, env):
        return [
            NormalFriction(
                REAR, car.rolling_friction, rear_contact(config), self.regularization
            ),
            NormalFriction(
                FRONT, car.rolling_friction, config.front_contact, self.regularization
            ),
        ]


def simple_physics() -> list[PhysicsTerm]:
    """Gravity and the translational kinetic energy of the car only: a car rolling
    without friction or drag, with no rotational inertia."""
    return [Gravity(), Translation()]


def full_physics() -> list[PhysicsTerm]:
    """All of the physics terms, enabled."""
    return [
        Gravity(),
        Translation(),
        BodyRotation(),
        WheelSpin(),
        Drag(),
        AxleFriction(),
        RollingFriction(),
    ]


def default_physics() -> list[PhysicsTerm]:
    """The physics terms a simulation uses by default: all of them."""
    return full_physics()
