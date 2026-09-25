import math
import warnings

import pytest
import torch
from test_kinematics import make_car

from pwdsim.kinematics import CarKinematics
from pwdsim.physics import (
    AxleFriction,
    BodyRotation,
    Drag,
    Environment,
    Gravity,
    RollingFriction,
    Translation,
    WheelSpin,
    full_physics,
    simple_physics,
)
from pwdsim.simulation import (
    DidNotFinishWarning,
    EquationsOfMotion,
    LiftOffWarning,
    Simulation,
)
from pwdsim.track import SplineTrack, besttrack
from pwdsim.units import INCH

THETA = -0.5
RADIUS = 0.595 * INCH


def incline():
    return SplineTrack([0.0, 3.0], [THETA] * 2, [0.0] * 2, s_start=0.3, s_finish=2.5)


def car(**overrides):
    return make_car(front_wheel_radius=RADIUS, **overrides)


def incline_finish_time():
    g = Environment().g
    track = incline()
    distance = track.s_finish + RADIUS * math.tan(THETA) - track.s_start
    return math.sqrt(2 * distance / (g * math.sin(-THETA)))


def simulate(track, car, **kwargs):
    kwargs.setdefault("dt", 1e-3)
    kwargs.setdefault("duration", 1.2)
    kwargs.setdefault("physics", simple_physics())
    return Simulation(track, car, **kwargs)()


class TestIncline:
    def test_finish_time_converges(self):
        errors = [
            abs(
                simulate(incline(), car(), dt=dt).finish_time.item()
                - incline_finish_time()
            )
            for dt in (2e-3, 1e-3, 5e-4)
        ]
        # Backward Euler is first order
        assert errors[2] < 1e-3
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
            assert coarse / fine == pytest.approx(2.0, rel=0.1)

    def test_integrators_agree(self):
        times = [
            simulate(incline(), car(), dt=2.5e-4, integrator=integrator).finish_time
            for integrator in ("backward-euler", "forward-euler")
        ]
        assert times[0].item() == pytest.approx(times[1].item(), abs=5e-4)
        assert times[1].item() == pytest.approx(incline_finish_time(), abs=5e-4)

    def test_normal_forces(self):
        c = car()
        run = simulate(incline(), c)
        # In the frame accelerating down the incline, only the normal component of
        # gravity remains, split between the axles by the lever rule
        weight = (c.mass * Environment().g * math.cos(THETA)).item()
        xi, w = c.cg[0].item(), c.wheelbase.item()
        forces = run.normal_forces().detach()
        torch.testing.assert_close(
            forces,
            torch.tensor(
                [weight * (w - xi) / w, weight * xi / w], dtype=torch.float64
            ).expand_as(forces),
        )

    def test_lift_off_warning(self):
        c = car(cg=(-0.5 * INCH, 0.4 * INCH))
        with pytest.warns(LiftOffWarning):
            run = simulate(incline(), c)
        assert run.min_normal_force[1] < 0 < run.min_normal_force[0]

    def test_no_gravity(self):
        sim = Simulation(
            incline(), car(), physics=simple_physics(), dt=1e-3, duration=0.1
        )
        sim.disable("gravity")
        with pytest.warns(DidNotFinishWarning):
            run = sim()
        assert torch.isnan(run.finish_time)
        torch.testing.assert_close(run.v, torch.zeros_like(run.v))
        sim.enable("gravity")
        with pytest.warns(DidNotFinishWarning):
            run = sim()
        assert torch.all(run.v[1:] > 0)

    def test_results_use_the_physics_of_the_run(self):
        sim = Simulation(incline(), car(), dt=1e-3, duration=1.2)
        sim.disable("rolling_friction")
        run = sim()
        before = (run.finish_time.item(), run.normal_forces().detach().clone())
        sim.enable("rolling_friction")
        sim.dt = 5e-4
        assert run.finish_time.item() == before[0]
        torch.testing.assert_close(run.normal_forces().detach(), before[1])

    def test_results_after_the_car_changes(self):
        c = car()
        run = simulate(incline(), c)
        assert torch.isfinite(run.finish_time)
        with torch.no_grad():
            c.mass_.fill_(0.2)
        with pytest.raises(RuntimeError, match="simulate the car again"):
            _ = run.finish_time

    def test_batched_matches_single(self):
        cg = torch.tensor([[1.0, 0.4], [0.5, 0.3]]) * INCH
        front_offset = torch.tensor([6.125, 5.0]) * INCH
        batched = simulate(incline(), car(cg=cg, front_offset=front_offset))
        assert batched.finish_time.shape == (2,)
        for i in range(2):
            single = simulate(incline(), car(cg=cg[i], front_offset=front_offset[i]))
            torch.testing.assert_close(batched.finish_time[i], single.finish_time)
            torch.testing.assert_close(batched.states[:, i], single.states)

    def test_unknown_integrator(self):
        with pytest.raises(ValueError, match="integrator"):
            Simulation(incline(), car(), integrator="runge-kutta")


@pytest.fixture(scope="module")
def track35():
    return besttrack(35)


def simulate_besttrack(track, c, dt=1e-3, **kwargs):
    kwargs.setdefault("physics", simple_physics())
    return Simulation(track, c, dt=dt, duration=2.5, **kwargs)()


class TestBestTrack:
    def test_energy(self, track35):
        drifts = []
        for dt in (1e-3, 5e-4):
            energy = (
                simulate_besttrack(track35, car(), dt=dt).energy()["total"].detach()
            )
            drifts.append(((energy[0] - energy[-1]) / energy[0]).item())
        # Backward Euler loses a little energy, converging at first order
        assert 0 < drifts[1] < drifts[0] < 5e-3
        assert drifts[0] / drifts[1] == pytest.approx(2.0, rel=0.05)

    def test_gradients_converge(self, track35):
        """Gradients don't depend on where the time steps fall along the track.

        Jumps in the curvature or its first two derivatives would add errors that
        depend on where the axles cross them within a time step, which shows up as
        noise in the smaller gradients."""
        grads = []
        for dt in (4e-4, 2e-4):
            c = car()
            simulate_besttrack(track35, c, dt=dt).finish_time.backward()
            grads.append(torch.cat([c.cg_.grad, c.wheelbase_.grad[None]]))
        torch.testing.assert_close(grads[0], grads[1], rtol=0.05, atol=0)

    def test_finish_speed(self, track35):
        c = car()
        run = simulate_besttrack(track35, c, dt=2.5e-4)
        kinematics = CarKinematics(track35, c)
        with torch.no_grad():
            start = kinematics.evaluate(kinematics.start_position()).cg.value[1]
            flat = kinematics.evaluate(torch.tensor(8.0)).cg.value[1]
        expected = math.sqrt(2 * Environment().g * (start - flat).item())
        assert run.finish_speed.item() == pytest.approx(expected, rel=1e-3)
        assert run.max_speed.item() >= run.finish_speed.item()

    def test_adjoint_gradients(self, track35):
        c = car()
        run = simulate_besttrack(track35, c)
        run.finish_time.backward()
        adjoint = {name: p.grad.clone() for name, p in c.named_parameters()}

        # Automatic differentiation through the time steps
        c.zero_grad()
        Simulation(track35, c, physics=simple_physics(), dt=1e-3, duration=2.5)(
            adjoint=False
        ).finish_time.backward()
        for name, p in c.named_parameters():
            # Parameters that don't affect the run get no gradient from autograd
            direct = p.grad if p.grad is not None else torch.zeros_like(p)
            torch.testing.assert_close(adjoint[name], direct, msg=name)

        # With these physics terms the mass doesn't matter
        assert adjoint["mass_"].abs().item() < 1e-10

        # Finite differences for the front offset, which moves the start
        h = 1e-6
        times = []
        for delta in (h, -h):
            with torch.no_grad():
                shifted = car(front_offset=c.front_offset.item() + delta)
                times.append(simulate_besttrack(track35, shifted).finish_time.item())
        fd = (times[0] - times[1]) / (2 * h)
        assert adjoint["front_offset_"].item() == pytest.approx(fd, rel=1e-5)

    def test_frozen_parameters(self, track35):
        c = car()
        for p in c.parameters():
            p.requires_grad_(False)
        c.cg_.requires_grad_(True)
        run = simulate_besttrack(track35, c)
        run.finish_time.backward()
        assert c.cg_.grad is not None
        assert c.mass_.grad is None

    def test_no_warnings_for_normal_car(self, track35):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run = simulate_besttrack(track35, car())
        assert torch.all(run.min_normal_force > 0)


class TestTermsOnIncline:
    """Closed form accelerations on a straight incline, one term at a time."""

    S = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64)
    V = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)

    def solve(self, c, *terms):
        eom = EquationsOfMotion(
            CarKinematics(incline(), c),
            [Gravity(), Translation(), *terms],
            Environment(),
        )
        with torch.no_grad():
            return eom.solve(self.S, self.V)

    def check(self, c, expected, *terms):
        a, normal = self.solve(c, *terms)
        torch.testing.assert_close(a, torch.full_like(a, expected))
        # Friction and inertia don't change the total normal force on a straight
        weight = (c.mass * Environment().g * math.cos(THETA)).item()
        torch.testing.assert_close(normal.sum(-1), torch.full_like(a, weight))

    def test_simple(self):
        self.check(car(), Environment().g * math.sin(-THETA))

    def test_body_rotation(self):
        # The car doesn't pitch on a straight track
        self.check(car(), Environment().g * math.sin(-THETA), BodyRotation())

    def test_wheel_spin(self):
        c = car(rear_wheel_inertia=3e-5, front_wheel_inertia=2e-5, n_front_wheels=1)
        M = c.mass.item()
        inertia = (2 * 3e-5 + 1 * 2e-5) / RADIUS**2
        expected = M * Environment().g * math.sin(-THETA) / (M + inertia)
        self.check(c, expected, WheelSpin())

    def test_rolling_friction(self):
        c = car(rolling_friction=0.05)
        g = Environment().g
        expected = g * (math.sin(-THETA) - 0.05 * math.cos(THETA))
        self.check(c, expected, RollingFriction())

    def test_axle_friction(self):
        c = car(rear_axle_friction=0.2, front_axle_friction=0.2)
        g, a = Environment().g, c.rear_axle_radius.item()
        expected = g * (math.sin(-THETA) - 0.2 * a / RADIUS * math.cos(THETA))
        self.check(c, expected, AxleFriction())

    def test_drag(self):
        c = car(frontal_area=0.05)
        env = Environment()
        k = 0.5 * env.rho * c.drag_coefficient.item() * 0.05
        M, g = c.mass.item(), env.g
        run = simulate(
            incline(),
            c,
            physics=[Gravity(), Translation(), Drag()],
            dt=2.5e-4,
            duration=1.5,
        )
        terminal = math.sqrt(M * g * math.sin(-THETA) / k)
        expected = terminal * torch.tanh(g * math.sin(-THETA) * run.times / terminal)
        torch.testing.assert_close(run.v.detach(), expected, rtol=2e-3, atol=1e-4)


class TestFullPhysics:
    def test_energy_balance(self, track35):
        drifts = []
        for dt in (1e-3, 5e-4):
            energy = simulate_besttrack(track35, car(), dt=dt, physics=None).energy()
            total = energy["total"].detach()
            drifts.append(((total[0] - total[-1]) / total[0]).item())
            # Friction and drag dissipate a good part of the energy
            assert energy["dissipated"][-1] > 0.05 * total[0]
        assert 0 < drifts[1] < drifts[0] < 5e-3
        assert drifts[0] / drifts[1] == pytest.approx(2.0, rel=0.1)

    def test_energy_by_term(self, track35):
        run = simulate_besttrack(track35, car(), physics=None)
        with torch.no_grad():
            energy, terms = run.energy(), run.energy_by_term()
        assert list(terms) == [t.name for t in full_physics()]
        kinetic = terms["translation"] + terms["body_rotation"] + terms["wheel_spin"]
        dissipated = terms["drag"] + terms["axle_friction"] + terms["rolling_friction"]
        torch.testing.assert_close(kinetic, energy["kinetic"])
        torch.testing.assert_close(terms["gravity"], energy["potential"])
        torch.testing.assert_close(dissipated, energy["dissipated"])
        # Every term takes some energy during the run (the body only pitches in the
        # curve, so its energy is back to zero at the end)
        assert all(terms[name].max() > 0 for name in terms)

    def test_smooth_min_normal_force(self, track35):
        cg = torch.tensor([[1.0, 0.4], [0.25, 0.4]]) * INCH
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LiftOffWarning)
            run = simulate_besttrack(track35, car(cg=cg), physics=None)
        exact = run.min_normal_force.min(-1).values
        n = 2 * int(run._before_finish().sum(0).max())
        for sharpness in (100.0, 500.0, 2000.0):
            smooth = run.smooth_min_normal_force(sharpness)
            assert smooth.shape == (2,)
            assert torch.all(smooth <= exact)
            assert torch.all(smooth >= exact - math.log(n) / sharpness)
        # It's differentiable, and moving the CG forward loads the front wheels
        smooth = run.smooth_min_normal_force()
        (grad,) = torch.autograd.grad(smooth.sum(), run.equations.car.cg_)
        assert torch.all(grad[:, 0] > 0)

    def test_slower_than_simple(self, track35):
        simple = simulate_besttrack(track35, car())
        full = simulate_besttrack(track35, car(), physics=None)
        assert full.finish_time > simple.finish_time + 0.05
        assert full.finish_speed < simple.finish_speed

    def test_gradients_converge(self, track35):
        grads = []
        for dt in (2e-4, 1e-4):
            c = car()
            simulate_besttrack(track35, c, dt=dt, physics=None).finish_time.backward()
            grads.append(
                torch.cat(
                    [
                        c.cg_.grad,
                        c.wheelbase_.grad[None],
                        c.mass_.grad[None],
                        c.rear_axle_friction_.grad[None],
                    ]
                )
            )
        torch.testing.assert_close(grads[0], grads[1], rtol=0.03, atol=0)

    def test_adjoint_gradients(self, track35):
        c = car()
        simulate_besttrack(track35, c, physics=None).finish_time.backward()
        adjoint = {name: p.grad.clone() for name, p in c.named_parameters()}
        # Every parameter matters with the full physics; heavier cars are faster
        # because drag matters less
        assert all(g.abs().max() > 0 for g in adjoint.values())
        assert adjoint["mass_"] < 0

        c.zero_grad()
        Simulation(track35, c, dt=1e-3, duration=2.5)(
            adjoint=False
        ).finish_time.backward()
        for name, p in c.named_parameters():
            torch.testing.assert_close(adjoint[name], p.grad, msg=name)

        h = 1e-7
        for name in ("mass", "rear_axle_friction"):
            times = []
            for delta in (h, -h):
                with torch.no_grad():
                    shifted = car(**{name: getattr(c, name).item() + delta})
                    run = simulate_besttrack(track35, shifted, physics=None)
                    times.append(run.finish_time.item())
            fd = (times[0] - times[1]) / (2 * h)
            assert adjoint[f"{name}_"].item() == pytest.approx(fd, rel=1e-4), name

    def test_no_warnings_for_normal_car(self, track35):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run = simulate_besttrack(track35, car(), physics=None)
        assert torch.all(run.min_normal_force > 0)
