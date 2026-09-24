import math

import numpy as np
import pytest
import torch

from pwdsim.track import SplineTrack, besttrack, ramp_track
from pwdsim.units import DEGREE, INCH


@pytest.fixture
def curvy_track():
    """A general spline track, with curvature everywhere."""
    knots = [0.0, 1.0, 2.5, 3.0, 5.0]
    angles = [-0.6, -0.8, -0.2, 0.1, 0.0]
    curvatures = [0.1, 0.3, 0.5, -0.2, 0.0]
    return SplineTrack(knots, angles, curvatures, s_start=0.2, s_finish=4.5)


@pytest.fixture
def ramp():
    return ramp_track(
        length=10.0,
        ramp_length=2.0,
        ramp_angle=30 * DEGREE,
        radius=1.5,
        s_start=0.2,
        s_finish=9.0,
        easement=0.05,
    )


def reference_position(track, s, n=200001):
    """Position by brute force trapezoid integration of the angle."""
    grid = torch.linspace(0, float(track.length), n, dtype=torch.float64)
    theta = track.angle(grid).numpy()
    grid = grid.numpy()

    def cumulative_trapezoid(f):
        return np.concatenate([[0], np.cumsum(np.diff(grid) * (f[1:] + f[:-1]) / 2)])

    x = cumulative_trapezoid(np.cos(theta))
    y = cumulative_trapezoid(np.sin(theta))
    y -= y[-1]
    return np.interp(s, grid, x), np.interp(s, grid, y)


class TestSplineTrack:
    def test_straight_track(self):
        theta = -0.4
        track = SplineTrack([0.0, 2.0, 3.0], [theta] * 3, [0.0] * 3, 0.0, 3.0)
        s = torch.linspace(0, 3, 11, dtype=torch.float64)
        x, y = track.position(s)
        torch.testing.assert_close(x, s * math.cos(theta))
        torch.testing.assert_close(y, (s - 3) * math.sin(theta))
        torch.testing.assert_close(track.curvature(s), torch.zeros_like(s))

    def test_float64(self, curvy_track):
        x, y = curvy_track.position(1.3)
        assert x.dtype == y.dtype == torch.float64
        assert curvy_track.angle(1.3).dtype == torch.float64

    def test_interpolates_knots(self, curvy_track):
        torch.testing.assert_close(
            curvy_track.angle(curvy_track.knots), curvy_track.angles
        )
        torch.testing.assert_close(
            curvy_track.curvature(curvy_track.knots), curvy_track.curvatures
        )

    def test_end_conditions(self, curvy_track):
        x, y = curvy_track.position(
            torch.stack([torch.tensor(0.0), curvy_track.length])
        )
        assert x[0] == 0
        assert y[1] == pytest.approx(0, abs=1e-15)

    def test_position_matches_reference(self, curvy_track):
        s = np.linspace(0, 5, 37)
        x, y = curvy_track.position(torch.as_tensor(s))
        x_ref, y_ref = reference_position(curvy_track, s)
        np.testing.assert_allclose(x.numpy(), x_ref, atol=1e-9)
        np.testing.assert_allclose(y.numpy(), y_ref, atol=1e-9)

    def test_derivatives_consistent(self, curvy_track):
        s = torch.linspace(0.05, 4.95, 50, dtype=torch.float64, requires_grad=True)
        x, y = curvy_track.position(s)
        theta = curvy_track.angle(s)
        kappa = curvy_track.curvature(s)
        dkappa = curvy_track.curvature_derivative(s)
        dx, dy, dtheta, d_kappa, d_dkappa = (
            torch.autograd.grad(v.sum(), s, retain_graph=True)[0]
            for v in (x, y, theta, kappa, dkappa)
        )
        torch.testing.assert_close(dx, torch.cos(theta))
        torch.testing.assert_close(dy, torch.sin(theta))
        torch.testing.assert_close(dtheta, kappa)
        torch.testing.assert_close(d_kappa, dkappa)
        torch.testing.assert_close(d_dkappa, curvy_track.curvature_second_derivative(s))

    def test_continuity_at_knots(self, curvy_track):
        eps = 1e-9
        interior = curvy_track.knots[1:-1]
        for f in (curvy_track.position, curvy_track.angle, curvy_track.curvature):
            left, right = f(interior - eps), f(interior + eps)
            for a, b in zip(
                left if isinstance(left, tuple) else (left,),
                right if isinstance(right, tuple) else (right,),
                strict=True,
            ):
                torch.testing.assert_close(a, b, atol=1e-7, rtol=0)

    def test_straight_extension(self, curvy_track):
        L = curvy_track.length
        for end, direction in ((torch.tensor(0.0), -1), (L, 1)):
            s = end + direction * torch.tensor([0.5, 1.0], dtype=torch.float64)
            x, y = curvy_track.position(s)
            x0, y0 = curvy_track.position(end)
            theta = curvy_track.angle(end)
            torch.testing.assert_close(x, x0 + (s - end) * torch.cos(theta))
            torch.testing.assert_close(y, y0 + (s - end) * torch.sin(theta))
            torch.testing.assert_close(curvy_track.angle(s), theta.expand(2))
            torch.testing.assert_close(
                curvy_track.curvature(s), torch.zeros(2).double()
            )
            torch.testing.assert_close(
                curvy_track.curvature_derivative(s), torch.zeros(2).double()
            )
            torch.testing.assert_close(
                curvy_track.curvature_second_derivative(s), torch.zeros(2).double()
            )

    def test_batched_input(self, curvy_track):
        s = torch.rand(3, 4, dtype=torch.float64) * 5
        x, y = curvy_track.position(s)
        assert x.shape == y.shape == curvy_track.angle(s).shape == (3, 4)

    def test_gradient_wrt_knot_angles(self):
        angles = torch.tensor(
            [-0.5, -0.5, 0.0], dtype=torch.float64, requires_grad=True
        )
        track = SplineTrack([0.0, 1.0, 2.0], angles, [0.0] * 3, 0.1, 1.9)
        track.height(0.0).backward()
        assert angles.grad is not None
        assert torch.all(torch.isfinite(angles.grad))

    @pytest.mark.parametrize(
        "knots, angles, curvatures, s_start, s_finish",
        [
            ([0.0], [0.0], [0.0], 0.0, 0.0),
            ([0.0, 1.0], [0.0], [0.0, 0.0], 0.0, 1.0),
            ([0.0, 1.0], [0.0, 0.0], [0.0], 0.0, 1.0),
            ([0.5, 1.0], [0.0, 0.0], [0.0, 0.0], 0.6, 1.0),
            ([0.0, 1.0, 1.0], [0.0] * 3, [0.0] * 3, 0.0, 1.0),
            ([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], 0.5, 0.4),
            ([0.0, 1.0], [0.0, 0.0], [0.0, 0.0], 0.0, 1.5),
        ],
    )
    def test_invalid(self, knots, angles, curvatures, s_start, s_finish):
        with pytest.raises(ValueError):
            SplineTrack(knots, angles, curvatures, s_start, s_finish)

    def test_curvature_derivatives_at_knots(self):
        knots = [0.0, 1.0, 2.0]
        dkappa = [0.2, -0.3, 0.1]
        track = SplineTrack(
            knots,
            [0.0, 0.1, 0.0],
            [0.1, 0.0, -0.1],
            0.1,
            1.9,
            curvature_derivatives=dkappa,
        )
        torch.testing.assert_close(
            track.curvature_derivative(track.knots), torch.tensor(dkappa).double()
        )
        # The curvature derivative is continuous across the interior knot
        eps = 1e-9
        torch.testing.assert_close(
            track.curvature_derivative(torch.tensor(1.0 - eps).double()),
            track.curvature_derivative(torch.tensor(1.0 + eps).double()),
            atol=1e-6,
            rtol=0,
        )

    def test_reproduces_septic(self):
        coefficients = [-0.4, 0.3, -0.2, 0.1, -0.05, 0.01, 0.004, -0.001]

        def derivative(s, order):
            """Derivative of the septic polynomial with the given coefficients."""
            total = torch.zeros_like(s)
            for k, c in enumerate(coefficients):
                if k >= order:
                    factor = math.prod(range(k - order + 1, k + 1))
                    total = total + c * factor * s ** (k - order)
            return total

        knots = torch.tensor([0.0, 0.8, 2.0, 3.0], dtype=torch.float64)
        track = SplineTrack(
            knots,
            derivative(knots, 0),
            derivative(knots, 1),
            0.0,
            3.0,
            curvature_derivatives=derivative(knots, 2),
            curvature_second_derivatives=derivative(knots, 3),
        )
        s = torch.linspace(0, 3, 31, dtype=torch.float64)
        torch.testing.assert_close(track.angle(s), derivative(s, 0))
        torch.testing.assert_close(track.curvature(s), derivative(s, 1))
        torch.testing.assert_close(track.curvature_derivative(s), derivative(s, 2))
        torch.testing.assert_close(
            track.curvature_second_derivative(s), derivative(s, 3)
        )

    def test_curvature_second_derivatives_continuous(self):
        track = SplineTrack(
            [0.0, 1.0, 2.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.0, -0.1],
            0.1,
            1.9,
            curvature_derivatives=[0.2, -0.3, 0.1],
            curvature_second_derivatives=[0.0, 0.5, 0.0],
        )
        eps = 1e-9
        s = torch.tensor([1.0 - eps, 1.0 + eps], dtype=torch.float64)
        values = track.curvature_second_derivative(s)
        torch.testing.assert_close(values[0], values[1], atol=1e-6, rtol=0)
        assert values[0].item() == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.parametrize(
        "keyword", ["curvature_derivatives", "curvature_second_derivatives"]
    )
    def test_invalid_curvature_derivatives(self, keyword):
        with pytest.raises(ValueError, match="curvature"):
            SplineTrack([0.0, 1.0], [0.0] * 2, [0.0] * 2, 0.0, 1.0, **{keyword: [0.0]})


class TestInterpolate:
    @pytest.fixture
    def track(self):
        knots = [0.0, 1.0, 1.5, 3.0, 4.0]
        angles = [-0.5, -0.5, -0.3, 0.0, 0.0]
        return SplineTrack.interpolate(
            knots, angles, 0.1, 3.9, end_curvatures=(0.0, 0.1)
        )

    def test_interpolates_angles(self, track):
        torch.testing.assert_close(track.angle(track.knots), track.angles)

    def test_end_curvatures(self, track):
        assert track.curvatures[0] == pytest.approx(0.0, abs=1e-14)
        assert track.curvatures[-1] == pytest.approx(0.1)

    def test_c2_continuity(self, track):
        eps = 1e-10
        interior = track.knots[1:-1]
        torch.testing.assert_close(
            track.curvature_derivative(interior - eps),
            track.curvature_derivative(interior + eps),
            atol=1e-6,
            rtol=0,
        )

    def test_reproduces_cubic(self):
        # A C2 spline exactly reproduces a single cubic with matching end slopes
        def theta(s):
            return 0.1 * s**3 - 0.3 * s**2 + 0.2 * s - 0.4

        def kappa(s):
            return 0.3 * s**2 - 0.6 * s + 0.2

        knots = torch.tensor([0.0, 0.7, 1.1, 2.0, 3.0], dtype=torch.float64)
        track = SplineTrack.interpolate(
            knots, theta(knots), 0.0, 3.0, end_curvatures=(kappa(0.0), kappa(3.0))
        )
        s = torch.linspace(0, 3, 31, dtype=torch.float64)
        torch.testing.assert_close(track.angle(s), theta(s))

    def test_invalid(self):
        with pytest.raises(ValueError):
            SplineTrack.interpolate([0.0, 1.0], [0.0], 0.0, 1.0)


class TestRampTrack:
    def test_easement_is_smooth(self, ramp):
        # The curvature follows a smooth step over each easement, with its first
        # two derivatives continuous everywhere
        s1, s2 = ramp.knots[1].item(), ramp.knots[2].item()
        u = torch.linspace(0, 1, 11, dtype=torch.float64)
        torch.testing.assert_close(
            ramp.curvature(s1 + u * (s2 - s1)),
            (6 * u**5 - 15 * u**4 + 10 * u**3) / 1.5,
        )
        eps = 1e-10
        knots = ramp.knots[1:-1]
        for f in (ramp.curvature_derivative, ramp.curvature_second_derivative):
            torch.testing.assert_close(
                f(knots - eps), f(knots + eps), atol=1e-3, rtol=0
            )

    def test_curvature_profile(self, ramp):
        on_ramp = torch.tensor([0.5, 1.9], dtype=torch.float64)
        on_arc = torch.tensor([2.2, 2.5, 2.7], dtype=torch.float64)
        on_flat = torch.tensor([3.0, 9.5], dtype=torch.float64)
        torch.testing.assert_close(ramp.curvature(on_ramp), torch.zeros(2).double())
        torch.testing.assert_close(
            ramp.curvature(on_arc), torch.full((3,), 1 / 1.5).double()
        )
        torch.testing.assert_close(ramp.curvature(on_flat), torch.zeros(2).double())

    def test_angles(self, ramp):
        assert ramp.angle(0.0) == pytest.approx(-30 * DEGREE)
        assert ramp.angle(ramp.length) == pytest.approx(0.0, abs=1e-14)
        # Total turn through the curve matches the ramp angle
        assert ramp.knots[4] - ramp.knots[1] == pytest.approx(30 * DEGREE * 1.5 + 0.05)

    def test_arc_is_circular(self, ramp):
        # Points on the constant curvature arc are a distance R from its center
        s = torch.linspace(float(ramp.knots[2]), float(ramp.knots[3]), 20).double()
        x, y = ramp.position(s)
        theta = ramp.angle(s)
        xc = x - 1.5 * torch.sin(theta)
        yc = y + 1.5 * torch.cos(theta)
        torch.testing.assert_close(xc, xc[0].expand(20))
        torch.testing.assert_close(yc, yc[0].expand(20))

    def test_heights(self, ramp):
        assert ramp.height(ramp.knots[4]) == pytest.approx(0.0, abs=1e-14)
        assert ramp.height(0.0) > ramp.height(ramp.s_start) > 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"easement": 0.0},
            {"easement": 1.0},
            {"length": 2.5, "s_finish": 2.4},
        ],
    )
    def test_invalid(self, kwargs):
        args = dict(
            length=10.0,
            ramp_length=2.0,
            ramp_angle=30 * DEGREE,
            radius=1.5,
            s_start=0.2,
            s_finish=9.0,
            easement=0.05,
        )
        args.update(kwargs)
        with pytest.raises(ValueError):
            ramp_track(**args)


class TestBestTrack:
    @pytest.mark.parametrize(
        "length_ft, racing_distance_in", [(35, 358), (42, 442), (49, 526)]
    )
    def test_dimensions(self, length_ft, racing_distance_in):
        track = besttrack(length_ft)
        assert track.length == pytest.approx(length_ft * 12 * INCH)
        assert track.s_finish - track.s_start == pytest.approx(
            racing_distance_in * INCH
        )
        # Start gate is about 4 ft above the floor, with the flat track a few
        # inches off the floor
        assert 40 * INCH < track.height(track.s_start) < 48 * INCH

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            besttrack(40)
