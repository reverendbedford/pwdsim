import pytest
import torch
from test_kinematics import make_car

from pwdsim.kinematics import CarKinematics
from pwdsim.physics import (
    Dissipation,
    DissipativeTerm,
    Environment,
    Gravity,
    Translation,
    default_physics,
)
from pwdsim.simulation import EquationsOfMotion
from pwdsim.track import besttrack

# Rear contact locations through the BestTrack curve, clear of the spline knots
S = torch.tensor([1.0, 2.25, 2.5, 2.8, 3.1, 5.0], dtype=torch.float64)
V = torch.tensor([0.5, 2.0, 3.0, 3.5, 4.0, 4.5], dtype=torch.float64)


@pytest.fixture
def kinematics():
    return CarKinematics(besttrack(), make_car())


def lift_kwargs(q):
    return {"h_r": q[1], "h_f": q[2]}


def positions(kinematics, q):
    """CG position at generalized coordinates q = (s, h_r, h_f)."""
    return kinematics.evaluate(q[0], **lift_kwargs(q)).cg.value


def central(f, h):
    """Fourth-order central difference of f(delta) at delta = 0."""
    return (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)


class LagrangianByDifferences:
    """Euler-Lagrange equations by finite differences, using only positions."""

    def __init__(self, kinematics, env):
        self.kinematics = kinematics
        self.env = env
        self.car = kinematics.car

    def velocity(self, q, qdot, eps=1e-4):
        return central(lambda d: positions(self.kinematics, q + d * qdot), eps)

    def unit(self, i):
        e = torch.zeros(3, 1, dtype=torch.float64)
        e[i] = 1.0
        return e

    def kinetic(self, q, qdot):
        return 0.5 * self.car.mass * torch.sum(self.velocity(q, qdot) ** 2, dim=-1)

    def potential(self, q):
        return self.car.mass * self.env.g * positions(self.kinematics, q)[..., 1]

    def dT_dqdot(self, q, qdot, i):
        # T is quadratic in qdot, so dT/dqdot_i = M xdot . dxdot/dqdot_i exactly
        return self.car.mass * torch.sum(
            self.velocity(q, qdot) * self.velocity(q, self.unit(i)), dim=-1
        )

    def generalized_force(self, q, qdot, qddot, i, delta=1e-3, h=1e-3):
        """d/dt dL/dqdot_i - dL/dq_i along a path through (q, qdot, qddot)."""
        ddt = central(
            lambda d: self.dT_dqdot(
                q + d * qdot + 0.5 * d**2 * qddot, qdot + d * qddot, i
            ),
            delta,
        )
        e = self.unit(i)
        dL_dq = central(
            lambda d: self.kinetic(q + d * e, qdot) - self.potential(q + d * e), h
        )
        return ddt - dL_dq


def state(a=None):
    zeros = torch.zeros_like(S)
    q = torch.stack([S, zeros, zeros])
    qdot = torch.stack([V, zeros, zeros])
    qddot = torch.stack([a if a is not None else zeros, zeros, zeros])
    return q, qdot, qddot


def test_equation_of_motion_matches_lagrangian(kinematics):
    env = Environment()
    eom = EquationsOfMotion(kinematics, [Gravity(), Translation()], env)
    reference = LagrangianByDifferences(kinematics, env)
    with torch.no_grad():
        a = eom.acceleration(S, V)
        q, qdot, qddot = state(a)
        # The s equation is satisfied by the computed acceleration
        residual = reference.generalized_force(q, qdot, qddot, 0)
        scale = kinematics.car.mass * env.g
        torch.testing.assert_close(
            residual / scale, torch.zeros_like(S), atol=1e-6, rtol=0
        )
        # The lift equations give the normal forces
        normal = eom.normal_forces(S, V)
        for i in (1, 2):
            expected = reference.generalized_force(q, qdot, qddot, i)
            torch.testing.assert_close(
                normal[..., i - 1], expected, rtol=1e-5, atol=1e-6
            )


def test_gravity_derivatives(kinematics):
    env = Environment()
    gravity = Gravity()
    h = 1e-5

    def potential(s, **lifts):
        return gravity.potential(kinematics.evaluate(s, **lifts), kinematics.car, env)

    with torch.no_grad():
        p = potential(S)
        plus, minus = potential(S + h), potential(S - h)
        torch.testing.assert_close(p.ds, (plus.value - minus.value) / (2 * h))
        torch.testing.assert_close(
            p.dss, (plus.ds - minus.ds) / (2 * h), rtol=1e-6, atol=1e-6
        )
        for lift, key in (("dhr", "h_r"), ("dhf", "h_f")):
            fd = (potential(S, **{key: h}).value - potential(S, **{key: -h}).value) / (
                2 * h
            )
            torch.testing.assert_close(getattr(p, lift), fd, rtol=1e-6, atol=1e-8)


def test_generalized_force_derivatives(kinematics):
    eom = EquationsOfMotion(kinematics, default_physics(), Environment())
    h = 1e-5
    with torch.no_grad():
        f = eom.generalized_forces(kinematics.evaluate(S), V)
        plus = eom.generalized_forces(kinematics.evaluate(S + h), V)
        minus = eom.generalized_forces(kinematics.evaluate(S - h), V)
        for value, derivative in (
            ("mass", "dmass"),
            ("coriolis", "dcoriolis"),
            ("potential", "dpotential"),
        ):
            fd = (getattr(plus, value) - getattr(minus, value)) / (2 * h)
            torch.testing.assert_close(
                getattr(f, derivative), fd, rtol=1e-5, atol=1e-7, msg=derivative
            )


class LinearDamping(DissipativeTerm):
    """F = c v^2 / 2 in the rear contact speed, to exercise the assembly."""

    name = "damping"

    def __init__(self, c):
        super().__init__()
        self.c = c

    def dissipation(self, config, v, car, env):
        zero = torch.zeros_like(v)
        return Dissipation(
            s=self.c * v, s_ds=zero, s_dv=self.c + zero, hr=zero, hf=zero
        )


@pytest.mark.parametrize("damping", [False, True])
def test_jacobian(kinematics, damping):
    terms = default_physics() + ([LinearDamping(0.05)] if damping else [])
    eom = EquationsOfMotion(kinematics, terms, Environment())
    h = 1e-6
    with torch.no_grad():
        a, da_ds, da_dv = eom.acceleration_and_jacobian(S, V)
        fd_s = (eom.acceleration(S + h, V) - eom.acceleration(S - h, V)) / (2 * h)
        fd_v = (eom.acceleration(S, V + h) - eom.acceleration(S, V - h)) / (2 * h)
    torch.testing.assert_close(da_ds, fd_s, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(da_dv, fd_v, rtol=1e-5, atol=1e-6)


def test_damping_slows_the_car(kinematics):
    env = Environment()
    free = EquationsOfMotion(kinematics, default_physics(), env)
    damped = EquationsOfMotion(
        kinematics, default_physics() + [LinearDamping(0.05)], env
    )
    with torch.no_grad():
        mass = free.generalized_forces(kinematics.evaluate(S), V).mass
        torch.testing.assert_close(
            damped.acceleration(S, V), free.acceleration(S, V) - 0.05 * V / mass
        )


def test_disabled_terms(kinematics):
    gravity = Gravity(enabled=False)
    eom = EquationsOfMotion(kinematics, [gravity, Translation()], Environment())
    with torch.no_grad():
        # No gravity: the car coasts without changing its energy
        kinetic, potential = eom.energy(S, V)
        torch.testing.assert_close(potential, torch.zeros_like(S))
        mass = eom.generalized_forces(kinematics.evaluate(S), V).mass
        torch.testing.assert_close(kinetic, 0.5 * mass * V**2)


def test_needs_kinetic_term(kinematics):
    eom = EquationsOfMotion(kinematics, [Gravity()], Environment())
    with pytest.raises(ValueError, match="kinetic"):
        eom.acceleration(S, V)


def test_names():
    assert [term.name for term in default_physics()] == ["gravity", "translation"]
