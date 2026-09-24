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
$J_{h_r}$ and $J_{h_f}$.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from pwdsim.car import Car
from pwdsim.kinematics import Configuration


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
    """

    weight: torch.Tensor
    ds: torch.Tensor
    dss: torch.Tensor
    dsss: torch.Tensor
    dhr: torch.Tensor
    dhf: torch.Tensor


@dataclass
class Potential:
    """A potential energy $V(q)$ and its derivatives.

    Attributes:
        value: $V$.
        ds: $\\partial V / \\partial s$.
        dss: $\\partial^2 V / \\partial s^2$.
        dhr: $\\partial V / \\partial h_r$.
        dhf: $\\partial V / \\partial h_f$.
    """

    value: torch.Tensor
    ds: torch.Tensor
    dss: torch.Tensor
    dhr: torch.Tensor
    dhf: torch.Tensor


@dataclass
class Dissipation:
    """Derivatives of a Rayleigh dissipation function $F(q, \\dot{q})$.

    The generalized dissipative force is $-\\partial F / \\partial \\dot{q}$.

    Attributes:
        s: $\\partial F / \\partial \\dot{s}$.
        s_ds: $\\partial^2 F / \\partial \\dot{s} \\partial s$.
        s_dv: $\\partial^2 F / \\partial \\dot{s}^2$.
        hr: $\\partial F / \\partial \\dot{h}_r$.
        hf: $\\partial F / \\partial \\dot{h}_f$.
    """

    s: torch.Tensor
    s_ds: torch.Tensor
    s_dv: torch.Tensor
    hr: torch.Tensor
    hf: torch.Tensor


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
    """A Rayleigh dissipation function contribution."""

    @abstractmethod
    def dissipation(
        self, config: Configuration, v: torch.Tensor, car: Car, env: Environment
    ) -> Dissipation:
        """Derivatives of the dissipation function at speed $v = \\dot{s}$."""


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
        )


class Translation(KineticTerm):
    """Kinetic energy of the car moving with its center of gravity,
    $T = \\tfrac{1}{2} M |\\dot{\\mathbf{x}}_g|^2$."""

    name = "translation"

    def rates(self, config, car, env):
        cg = config.cg
        return [Rate(car.mass, cg.ds, cg.dss, cg.dsss, cg.dhr, cg.dhf)]


def default_physics() -> list[PhysicsTerm]:
    """All the physics terms implemented so far, enabled."""
    return [Gravity(), Translation()]
