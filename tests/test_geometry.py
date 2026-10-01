import pytest
import scipy.integrate
import torch

from pwdsim.geometry import (
    TUNGSTEN,
    VOID,
    BSpline,
    Profile,
    Rectangle,
    Region,
    WeightPocket,
)


def t(*values):
    return torch.tensor(values, dtype=torch.float64)


class TestBSpline:
    @pytest.fixture
    def spline(self):
        return BSpline(-0.5, 2.0, 7)

    def test_partition_of_unity(self, spline):
        basis = spline.basis(torch.linspace(-0.5, 2.0, 201, dtype=torch.float64))
        torch.testing.assert_close(basis.sum(-1), torch.ones(201, dtype=torch.float64))
        assert basis.min() >= 0

    def test_clamped_ends(self, spline):
        c = t(1.0, 3.0, -2.0, 0.5, 4.0, 2.0, -1.0)
        torch.testing.assert_close(spline.basis(t(-0.5, 2.0)) @ c, t(1.0, -1.0))

    def test_reproduces_linear(self, spline):
        x = torch.linspace(-0.5, 2.0, 51, dtype=torch.float64)
        torch.testing.assert_close(spline.basis(x) @ spline.greville(), x)

    def test_convex_hull(self, spline):
        c = t(0.2, 1.0, 0.0, 0.9, 1.0, 0.3, 0.6)
        h = spline.basis(torch.linspace(-0.5, 2.0, 401, dtype=torch.float64)) @ c
        assert h.min() >= 0.0 and h.max() <= 1.0

    def test_derivative(self, spline):
        c = t(0.2, 1.0, 0.0, 0.9, 1.0, 0.3, 0.6)
        x = t(0.1, 0.7, 1.3, 1.9).requires_grad_(True)
        (analytic,) = torch.autograd.grad((spline.basis(x) @ c).sum(), x)
        h = 1e-7
        with torch.no_grad():
            fd = (spline.basis(x + h) @ c - spline.basis(x - h) @ c) / (2 * h)
        torch.testing.assert_close(analytic, fd, rtol=1e-6, atol=1e-6)

    def test_quadrature_exact(self, spline):
        x, w = spline.quadrature()
        # Degree 9 polynomials integrate exactly
        assert torch.sum(w * x**9).item() == pytest.approx((2.0**10 - 0.5**10) / 10)

    def test_too_few_values(self):
        with pytest.raises(ValueError):
            BSpline(0.0, 1.0, 3)


class TestProfile:
    def test_block(self):
        p = Profile.block(-1.0, 6.0, -0.2, 1.25)
        m = p.moments()
        assert m.area.item() == pytest.approx(7 * 1.25, rel=1e-14)
        assert m.x.item() == pytest.approx(7 * 1.25 * 2.5, rel=1e-14)
        assert m.y.item() == pytest.approx(7 * 1.25 * (-0.2 + 1.25 / 2), rel=1e-13)
        polar = 1.25 * (6.0**3 + 1.0**3) / 3 + 7 * ((1.05**3) - (-0.2) ** 3) / 3
        assert m.polar.item() == pytest.approx(polar, rel=1e-13)

    def test_moments_against_quad(self):
        p = Profile(-0.8, 6.2, -0.3, [0.4, 1.2, 0.9, 1.25, 0.5, 0.7, 0.3, 0.6], 1.25)
        m = p.moments()

        def h(x):
            with torch.no_grad():
                return p.height(torch.tensor(x, dtype=torch.float64)).item()

        yb = -0.3
        breaks = p.spline.breaks.numpy()

        def integrate(f):
            return sum(
                scipy.integrate.quad(f, a, b, epsabs=0, epsrel=1e-12)[0]
                for a, b in zip(breaks[:-1], breaks[1:], strict=True)
            )

        expected = [
            integrate(h),
            integrate(lambda x: x * h(x)),
            integrate(lambda x: ((yb + h(x)) ** 2 - yb**2) / 2),
            integrate(lambda x: x**2 * h(x) + ((yb + h(x)) ** 3 - yb**3) / 3),
        ]
        actual = [m.area, m.x, m.y, m.polar]
        for a, e in zip(actual, expected, strict=True):
            assert a.item() == pytest.approx(e, rel=1e-11)

    def test_bounds(self):
        p = Profile(0.0, 7.0, 0.0, [1.0, 1.1, 0.9, 1.2, 1.0, 0.8, 1.0, 1.25], 1.25)
        lower, upper = p.bounds()
        torch.testing.assert_close(lower, torch.zeros(8, dtype=torch.float64))
        torch.testing.assert_close(upper, torch.full((8,), 1.25, dtype=torch.float64))
        lower, upper = p.bounds(fixed=[(2.0, 4.0)], minimum=0.2)
        g = p.spline.greville()
        fixed = (g >= 2.0) & (g <= 4.0)
        assert fixed.any() and not fixed.all()
        torch.testing.assert_close(lower[fixed], upper[fixed])
        torch.testing.assert_close(lower[fixed], p.heights.detach()[fixed])
        assert torch.all(lower[~fixed] == 0.2)

    def test_invalid(self):
        with pytest.raises(ValueError, match="front"):
            Profile(1.0, 0.0, 0.0, [1.0] * 4)
        with pytest.raises(ValueError, match="1D"):
            Profile(0.0, 1.0, 0.0, [[1.0] * 4])


class TestRegions:
    def test_rectangle_moments(self):
        r = Rectangle(t(1.0), t(3.0), t(-0.5), t(0.5), TUNGSTEN)
        m = r.moments()
        assert m.area.item() == pytest.approx(2.0)
        assert m.x.item() == pytest.approx(4.0)
        assert m.y.item() == pytest.approx(0.0)
        # integral of x^2 + y^2 over the rectangle
        expected = 1.0 * (27 - 1) / 3 + 2.0 * (0.125 + 0.125) / 3
        assert m.polar.item() == pytest.approx(expected)

    def test_region(self):
        region = Region(1.0, 2.0, 0.0, 0.5, VOID, depth=0.3)
        (r,) = region.rectangles(bottom=-0.1)
        assert r.density == VOID
        assert r.depth.item() == pytest.approx(0.3)
        assert Region(1.0, 2.0, 0.0, 0.5, VOID).depth is None

    def test_weight_pocket(self):
        pocket = WeightPocket(2.0, 0.5, 1.0, 0.4, TUNGSTEN)
        void, weight = pocket.rectangles(bottom=-0.2)
        assert (void.x0.item(), void.x1.item()) == pytest.approx((1.75, 2.25))
        assert (void.y0.item(), void.y1.item()) == pytest.approx((-0.2, 0.4))
        assert (weight.y0.item(), weight.y1.item()) == pytest.approx((0.4, 0.8))
        assert void.density == VOID and weight.density == TUNGSTEN

    @pytest.mark.parametrize(
        "args", [(2.0, 0.5, 1.0, 1.5), (2.0, 0.5, 1.0, 0.0), (2.0, 0.0, 1.0, 0.5)]
    )
    def test_invalid_pocket(self, args):
        with pytest.raises(ValueError):
            WeightPocket(*args)
