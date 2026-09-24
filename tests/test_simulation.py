import math
import warnings

import pytest
import torch
from test_kinematics import make_car

from pwdsim.kinematics import CarKinematics
from pwdsim.physics import Environment
from pwdsim.simulation import DidNotFinishWarning, LiftOffWarning, Simulation
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
        sim = Simulation(incline(), car(), dt=1e-3, duration=0.1)
        sim.disable("gravity")
        with pytest.warns(DidNotFinishWarning):
            run = sim()
        assert torch.isnan(run.finish_time)
        torch.testing.assert_close(run.v, torch.zeros_like(run.v))
        sim.enable("gravity")
        with pytest.warns(DidNotFinishWarning):
            run = sim()
        assert torch.all(run.v[1:] > 0)

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
    return Simulation(track, c, dt=dt, duration=2.5, **kwargs)()


class TestBestTrack:
    def test_energy(self, track35):
        drifts = []
        for dt in (1e-3, 2.5e-4):
            energy = (
                simulate_besttrack(track35, car(), dt=dt).energy()["total"].detach()
            )
            drifts.append(((energy[0] - energy[-1]) / energy[0]).item())
        # Backward Euler loses a little energy.  Most of the error comes from the
        # steps where the axles cross the ends of the easements, where the
        # curvature derivative jumps, so the convergence is first order but not
        # smooth in the step size.
        assert 0 < drifts[1] < drifts[0] / 2
        assert drifts[0] < 5e-3

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
        Simulation(track35, c, dt=1e-3, duration=2.5)(
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
