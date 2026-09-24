import pytest
import torch
from test_kinematics import make_car

from pwdsim.kinematics import CarKinematics
from pwdsim.physics import (
    AxleFriction,
    BodyRotation,
    Dissipation,
    DissipativeTerm,
    Drag,
    Environment,
    Gravity,
    RollingFriction,
    Translation,
    WheelSpin,
    default_physics,
    full_physics,
    simple_physics,
    wheel_angles,
)
from pwdsim.simulation import EquationsOfMotion
from pwdsim.track import besttrack

# Rear contact locations through the BestTrack curve, clear of the spline knots
S = torch.tensor([1.0, 2.25, 2.5, 2.8, 3.1, 5.0], dtype=torch.float64)
V = torch.tensor([0.5, 2.0, 3.0, 3.5, 4.0, 4.5], dtype=torch.float64)


def car(**overrides):
    # Exaggerate the rotational inertias and friction so that every term matters
    args = dict(
        body_inertia=4e-3,
        rear_wheel_inertia=3e-5,
        front_wheel_inertia=2e-5,
        rear_axle_friction=0.3,
        front_axle_friction=0.2,
        rolling_friction=0.05,
        n_front_wheels=1,
    )
    args.update(overrides)
    return make_car(**args)


@pytest.fixture
def kinematics():
    return CarKinematics(besttrack(), car())


def central(f, h):
    """Fourth-order central difference of f(delta) at delta = 0."""
    return (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)


def unit(i):
    e = torch.zeros(3, 1, dtype=torch.float64)
    e[i] = 1.0
    return e


class Reference:
    """Lagrange's equations with the full physics by finite differences, using only
    the positions of the car: the center of gravity, the pitch angle from the body
    axis, and the wheel angles from the contact locations."""

    def __init__(self, kinematics, env, physics="full"):
        self.kinematics = kinematics
        self.track = kinematics.track
        self.car = kinematics.car
        self.env = env
        self.full = physics == "full"

    def coordinates(self, q):
        """CG position, pitch, wheel angles, and contact locations at
        q = (s, h_r, h_f)."""
        config = self.kinematics.evaluate(q[0], h_r=q[1], h_f=q[2])
        axis = config.axis.value
        rear, front = q[0], config.front_contact.value
        r_r, r_f = self.car.rear_wheel_radius, self.car.front_wheel_radius
        return {
            "cg": config.cg.value,
            "pitch": torch.atan2(axis[..., 1], axis[..., 0])[..., None],
            "rear_wheel": (-(rear - r_r * self.track.angle(rear)) / r_r)[..., None],
            "front_wheel": (-(front - r_f * self.track.angle(front)) / r_f)[..., None],
            "rear_contact": rear[..., None],
            "front_contact": front[..., None],
        }

    def rates(self, q, qdot, eps=1e-4):
        """Rates of change of the coordinates along qdot."""
        values = [
            self.coordinates(q + d * qdot) for d in (2 * eps, eps, -eps, -2 * eps)
        ]
        return {
            name: (
                -values[0][name]
                + 8 * values[1][name]
                - 8 * values[2][name]
                + values[3][name]
            )
            / (12 * eps)
            for name in values[0]
        }

    def weights(self):
        car = self.car
        if not self.full:
            return {"cg": car.mass}
        return {
            "cg": car.mass,
            "pitch": car.body_inertia,
            "rear_wheel": car.n_rear_wheels * car.rear_wheel_inertia,
            "front_wheel": car.n_front_wheels * car.front_wheel_inertia,
        }

    def kinetic(self, q, qdot):
        rates = self.rates(q, qdot)
        return sum(
            0.5 * w * torch.sum(rates[name] ** 2, dim=-1)
            for name, w in self.weights().items()
        )

    def dT_dqdot(self, q, qdot, i):
        # T is quadratic in qdot, so dT/dqdot_i = sum m rho . drho/dqdot_i exactly
        rates, along = self.rates(q, qdot), self.rates(q, unit(i))
        return sum(
            w * torch.sum(rates[name] * along[name], dim=-1)
            for name, w in self.weights().items()
        )

    def potential(self, q):
        return self.car.mass * self.env.g * self.coordinates(q)["cg"][..., 1]

    def dissipation(self, q, qdot, normal):
        """Rayleigh dissipation function, with the normal forces given."""
        if not self.full:
            return torch.zeros(q.shape[1:], dtype=torch.float64)
        car, rates = self.car, self.rates(q, qdot)
        drag = (
            self.env.rho
            * car.drag_coefficient
            * car.frontal_area
            / 6
            * torch.linalg.norm(rates["cg"], dim=-1) ** 3
        )
        axle = 0.0
        for index, wheel, friction, radius in (
            (0, "rear_wheel", car.rear_axle_friction, car.rear_axle_radius),
            (1, "front_wheel", car.front_axle_friction, car.front_axle_radius),
        ):
            relative = (rates[wheel] - rates["pitch"])[..., 0].abs()
            axle = axle + friction * radius * normal[..., index] * relative
        rolling = car.rolling_friction * (
            normal[..., 0] * rates["rear_contact"][..., 0].abs()
            + normal[..., 1] * rates["front_contact"][..., 0].abs()
        )
        return drag + axle + rolling

    def residual(self, q, qdot, qddot, normal, i, delta=1e-3, h=1e-3):
        """d/dt dT/dqdot_i - dT/dq_i + dV/dq_i + dF/dqdot_i."""
        e = unit(i)
        ddt = central(
            lambda d: self.dT_dqdot(
                q + d * qdot + 0.5 * d**2 * qddot, qdot + d * qddot, i
            ),
            delta,
        )
        dT_dq = central(lambda d: self.kinetic(q + d * e, qdot), h)
        dV_dq = central(lambda d: self.potential(q + d * e), h)
        dF_dqdot = central(lambda d: self.dissipation(q, qdot + d * e, normal), 1e-4)
        return ddt - dT_dq + dV_dq + dF_dqdot


def state(a):
    zeros = torch.zeros_like(S)
    q = torch.stack([S, zeros, zeros])
    qdot = torch.stack([V, zeros, zeros])
    qddot = torch.stack([a, zeros, zeros])
    return q, qdot, qddot


@pytest.mark.parametrize("physics", ["simple", "full"])
def test_equations_of_motion_match_lagrange(kinematics, physics):
    """The acceleration and normal forces satisfy Lagrange's equations: no
    generalized force along s, and the normal forces along the lifts."""
    env = Environment()
    terms = simple_physics() if physics == "simple" else full_physics()
    eom = EquationsOfMotion(kinematics, terms, env)
    reference = Reference(kinematics, env, physics)
    with torch.no_grad():
        a, normal = eom.solve(S, V)
        q, qdot, qddot = state(a)
        scale = kinematics.car.mass * env.g
        residual = reference.residual(q, qdot, qddot, normal, 0)
        torch.testing.assert_close(
            residual / scale, torch.zeros_like(S), atol=1e-7, rtol=0
        )
        for i in (1, 2):
            expected = reference.residual(q, qdot, qddot, normal, i)
            torch.testing.assert_close(
                normal[..., i - 1], expected, rtol=1e-6, atol=1e-7
            )


def test_every_term_matters(kinematics):
    env = Environment()
    with torch.no_grad():
        full, _ = EquationsOfMotion(kinematics, full_physics(), env).solve(S, V)
        for name in [term.name for term in full_physics()[2:]]:
            terms = [t for t in full_physics() if t.name != name]
            partial, _ = EquationsOfMotion(kinematics, terms, env).solve(S, V)
            assert (full - partial).abs().max() > 1e-3, name


def test_wheel_angles(kinematics):
    """Wheel angle derivatives against finite differences."""
    reference = Reference(kinematics, Environment())
    c = kinematics.car
    h = 1e-5

    def angles(s, **lifts):
        return wheel_angles(kinematics.evaluate(s, **lifts), c)

    with torch.no_grad():
        center, plus, minus = angles(S), angles(S + h), angles(S - h)
        zeros = torch.zeros_like(S)
        coordinates = reference.coordinates(torch.stack([S, zeros, zeros]))
        for index, name in enumerate(("rear_wheel", "front_wheel")):
            torch.testing.assert_close(center[index].value, coordinates[name][..., 0])
            for lower, higher in (("value", "ds"), ("ds", "dss"), ("dss", "dsss")):
                fd = (getattr(plus[index], lower) - getattr(minus[index], lower)) / (
                    2 * h
                )
                torch.testing.assert_close(
                    getattr(center[index], higher), fd, rtol=1e-5, atol=1e-5
                )
            for lift, key in (("dhr", "h_r"), ("dhf", "h_f")):
                up, down = angles(S, **{key: h}), angles(S, **{key: -h})
                fd = (up[index].value - down[index].value) / (2 * h)
                torch.testing.assert_close(
                    getattr(center[index], lift), fd, rtol=1e-6, atol=1e-6
                )
                fd = (getattr(plus[index], lift) - getattr(minus[index], lift)) / (
                    2 * h
                )
                torch.testing.assert_close(
                    getattr(center[index], f"{lift}_ds"), fd, rtol=1e-5, atol=1e-5
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
            up, down = potential(S, **{key: h}), potential(S, **{key: -h})
            fd = (up.value - down.value) / (2 * h)
            torch.testing.assert_close(getattr(p, lift), fd, rtol=1e-6, atol=1e-8)
            fd = (getattr(plus, lift) - getattr(minus, lift)) / (2 * h)
            torch.testing.assert_close(
                getattr(p, f"{lift}_ds"), fd, rtol=1e-5, atol=1e-6
            )


def test_drag_derivatives(kinematics):
    env, drag, c = Environment(), Drag(), kinematics.car
    h = 1e-6

    def dissipation(s, v):
        return drag.dissipation(kinematics.evaluate(s), v, c, env)

    with torch.no_grad():
        d = dissipation(S, V)
        plus_s, minus_s = dissipation(S + h, V), dissipation(S - h, V)
        plus_v, minus_v = dissipation(S, V + h), dissipation(S, V - h)
        for name in ("s", "hr", "hf"):
            fd_s = (getattr(plus_s, name) - getattr(minus_s, name)) / (2 * h)
            fd_v = (getattr(plus_v, name) - getattr(minus_v, name)) / (2 * h)
            torch.testing.assert_close(
                getattr(d, f"{name}_ds"), fd_s, rtol=1e-5, atol=1e-8
            )
            torch.testing.assert_close(
                getattr(d, f"{name}_dv"), fd_v, rtol=1e-5, atol=1e-8
            )
        # Along s: the drag force on the CG, 1/2 rho Cd A |x_g dot|^2, times the
        # rate of the CG per unit s
        speed = torch.linalg.norm(kinematics.evaluate(S).cg.ds, dim=-1)
        force = 0.5 * env.rho * c.drag_coefficient * c.frontal_area * (speed * V) ** 2
        torch.testing.assert_close(d.s, force * speed)


@pytest.mark.parametrize("physics", ["simple", "full", "friction", "drag"])
def test_jacobian(kinematics, physics):
    terms = {
        "simple": simple_physics(),
        "full": full_physics(),
        "friction": [Gravity(), Translation(), AxleFriction(), RollingFriction()],
        "drag": [Gravity(), Translation(), Drag()],
    }[physics]
    eom = EquationsOfMotion(kinematics, terms, Environment())
    h = 1e-6
    with torch.no_grad():
        a, da_ds, da_dv = eom.acceleration_and_jacobian(S, V)
        fd_s = (eom.acceleration(S + h, V) - eom.acceleration(S - h, V)) / (2 * h)
        fd_v = (eom.acceleration(S, V + h) - eom.acceleration(S, V - h)) / (2 * h)
    torch.testing.assert_close(da_ds, fd_s, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(da_dv, fd_v, rtol=1e-5, atol=1e-6)


def test_jacobian_near_rest(kinematics):
    """The regularized friction sign keeps the Jacobian finite as the car starts."""
    eom = EquationsOfMotion(kinematics, full_physics(), Environment())
    v = torch.tensor([0.0, 1e-4, 1e-3], dtype=torch.float64)
    s = torch.full_like(v, 1.0)
    h = 1e-8
    with torch.no_grad():
        _, _, da_dv = eom.acceleration_and_jacobian(s, v)
        fd_v = (eom.acceleration(s, v + h) - eom.acceleration(s, v - h)) / (2 * h)
    assert torch.all(torch.isfinite(da_dv))
    torch.testing.assert_close(da_dv, fd_v, rtol=1e-4, atol=1e-4)


class LinearDamping(DissipativeTerm):
    """F = c v^2 / 2 in the rear contact speed, to exercise the assembly."""

    name = "damping"

    def __init__(self, c):
        super().__init__()
        self.c = c

    def dissipation(self, config, v, car, env):
        zero = torch.zeros_like(v)
        return Dissipation(
            s=self.c * v,
            s_ds=zero,
            s_dv=self.c + zero,
            hr=zero,
            hr_ds=zero,
            hr_dv=zero,
            hf=zero,
            hf_ds=zero,
            hf_dv=zero,
        )


def test_custom_dissipation(kinematics):
    env = Environment()
    free = EquationsOfMotion(kinematics, simple_physics(), env)
    damped = EquationsOfMotion(
        kinematics, simple_physics() + [LinearDamping(0.05)], env
    )
    with torch.no_grad():
        mass = free.system(S, V).mass
        torch.testing.assert_close(
            damped.acceleration(S, V), free.acceleration(S, V) - 0.05 * V / mass
        )
        torch.testing.assert_close(damped.power(S, V), 0.05 * V**2)


def test_power(kinematics):
    """Dissipated power is v times dF/ds dot."""
    env = Environment()
    eom = EquationsOfMotion(kinematics, full_physics(), env)
    reference = Reference(kinematics, env)
    with torch.no_grad():
        _, normal = eom.solve(S, V)
        q, qdot, _ = state(torch.zeros_like(S))
        dF = central(
            lambda d: reference.dissipation(q, qdot + d * unit(0), normal), 1e-4
        )
        torch.testing.assert_close(eom.power(S, V), V * dF, rtol=1e-6, atol=1e-9)


def test_disabled_terms(kinematics):
    gravity = Gravity(enabled=False)
    eom = EquationsOfMotion(kinematics, [gravity, Translation()], Environment())
    with torch.no_grad():
        kinetic, potential = eom.energy(S, V)
        torch.testing.assert_close(potential, torch.zeros_like(S))
        mass = eom.system(S, V).mass
        torch.testing.assert_close(kinetic, 0.5 * mass * V**2)


def test_needs_kinetic_term(kinematics):
    eom = EquationsOfMotion(kinematics, [Gravity()], Environment())
    with pytest.raises(ValueError, match="kinetic"):
        eom.acceleration(S, V)


def test_presets():
    assert [term.name for term in simple_physics()] == ["gravity", "translation"]
    names = [term.name for term in full_physics()]
    assert names == [
        "gravity",
        "translation",
        "body_rotation",
        "wheel_spin",
        "drag",
        "axle_friction",
        "rolling_friction",
    ]
    assert [term.name for term in default_physics()] == names
    assert isinstance(full_physics()[2], BodyRotation)
    assert isinstance(full_physics()[3], WheelSpin)
