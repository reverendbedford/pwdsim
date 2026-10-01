import math
import warnings

import matplotlib
import numpy as np
import pytest
import scipy.integrate
import torch

import pwdsim
from pwdsim import constraints
from pwdsim.geometry import PINE, TUNGSTEN, VOID, Profile, Region, WeightPocket
from pwdsim.plotting import plot_geometry
from pwdsim.units import GRAM, INCH

matplotlib.use("Agg")

THICKNESS = 1.75 * INCH
BOTTOM = -0.2 * INCH
WHEEL_MASS = 2.6 * GRAM
WHEELBASE = 4.375 * INCH
HEIGHTS = [0.6, 1.2, 1.0, 1.25, 0.9, 0.7, 0.5, 0.4]


def car_args(**overrides):
    wheel_radius = 0.595 * INCH
    args = dict(
        thickness=THICKNESS,
        body_density=PINE,
        wheelbase=WHEELBASE,
        wheel_mass=WHEEL_MASS,
        rear_wheel_inertia=0.58 * WHEEL_MASS * wheel_radius**2,
        front_wheel_inertia=0.58 * WHEEL_MASS * wheel_radius**2,
        rear_wheel_radius=wheel_radius,
        front_wheel_radius=wheel_radius,
        rear_axle_radius=0.0435 * INCH,
        front_axle_radius=0.0435 * INCH,
        rear_axle_friction=0.1,
        front_axle_friction=0.1,
        frontal_area=0.0014,
        drag_coefficient=0.4,
        rolling_friction=0.002,
    )
    args.update(overrides)
    return args


def profile(heights=HEIGHTS):
    return Profile(
        -0.875 * INCH,
        6.125 * INCH,
        BOTTOM,
        torch.tensor(heights, dtype=torch.float64) * INCH,
        1.25 * INCH,
    )


def pocket(**overrides):
    args = dict(
        position=-0.45 * INCH,
        width=0.6 * INCH,
        height=0.5 * INCH,
        length=0.3 * INCH,
        material=TUNGSTEN,
        depth=0.75 * INCH,
    )
    args.update(overrides)
    return WeightPocket(**args)


def void():
    return Region(2.0 * INCH, 3.0 * INCH, 0.0, 0.3 * INCH, VOID)


def make_car(inclusions=None, heights=HEIGHTS, **overrides):
    inclusions = [pocket(), void()] if inclusions is None else inclusions
    return pwdsim.GeometricCar(profile(heights), inclusions, **car_args(**overrides))


def reference_moments(car):
    """Mass moments from scipy quadrature over the profile and rectangle formulas."""
    p = car.profile
    yb = p.bottom

    def h(x):
        with torch.no_grad():
            return p.height(torch.tensor(x, dtype=torch.float64)).item()

    breaks = p.spline.breaks.numpy()

    def integrate(f):
        return sum(
            scipy.integrate.quad(f, a, b, epsabs=0, epsrel=1e-12)[0]
            for a, b in zip(breaks[:-1], breaks[1:], strict=True)
        )

    rho_t = PINE * THICKNESS
    moments = (
        np.array(
            [
                integrate(h),
                integrate(lambda x: x * h(x)),
                integrate(lambda x: ((yb + h(x)) ** 2 - yb**2) / 2),
                integrate(lambda x: x**2 * h(x) + ((yb + h(x)) ** 3 - yb**3) / 3),
            ]
        )
        * rho_t
    )

    def rectangle(x0, x1, y0, y1):
        a = (x1 - x0) * (y1 - y0)
        return np.array(
            [
                a,
                a * (x0 + x1) / 2,
                a * (y0 + y1) / 2,
                (y1 - y0) * (x1**3 - x0**3) / 3 + (x1 - x0) * (y1**3 - y0**3) / 3,
            ]
        )

    with torch.no_grad():
        for inclusion in car.inclusions:
            for r in inclusion.rectangles(yb):
                depth = THICKNESS if r.depth is None else r.depth.item()
                coords = (r.x0.item(), r.x1.item(), r.y0.item(), r.y1.item())
                moments += (r.density - PINE) * depth * rectangle(*coords)
    moments += WHEEL_MASS * np.array(
        [4, 2 * WHEELBASE, 0, 2 * WHEELBASE**2]
    )  # two wheels at each axle
    mass = moments[0]
    cg = moments[1:3] / mass
    return mass, cg, moments[3] - mass * np.sum(cg**2)


class TestMassProperties:
    def test_block(self):
        car = make_car(inclusions=[], heights=[1.25] * 8)
        length, height = 7 * INCH, 1.25 * INCH
        body = PINE * THICKNESS * length * height
        mass = body + 4 * WHEEL_MASS
        assert car.mass.item() == pytest.approx(mass, rel=1e-12)
        cg_x = (body * (6.125 * INCH - length / 2) + 2 * WHEEL_MASS * WHEELBASE) / mass
        cg_y = body * (BOTTOM + height / 2) / mass
        torch.testing.assert_close(
            car.cg.detach(), torch.tensor([cg_x, cg_y], dtype=torch.float64)
        )
        assert car.front_offset.item() == pytest.approx(6.125 * INCH)

    def test_against_reference(self):
        car = make_car()
        mass, cg, inertia = reference_moments(car)
        assert car.mass.item() == pytest.approx(mass, rel=1e-10)
        np.testing.assert_allclose(car.cg.detach().numpy(), cg, rtol=1e-10)
        assert car.body_inertia.item() == pytest.approx(inertia, rel=1e-10)

    def test_against_grid(self):
        """A brute force check: sum the density over a fine grid of cells."""
        car = make_car()
        n = 1500
        x = np.linspace(-0.875 * INCH, 6.125 * INCH, n + 1)
        y = np.linspace(BOTTOM, BOTTOM + 1.25 * INCH, n + 1)
        xc, yc = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2
        X, Y = np.meshgrid(xc, yc, indexing="ij")
        with torch.no_grad():
            top = car.profile.top(torch.tensor(xc)).numpy()[:, None]
        density = np.where(top >= Y, PINE * THICKNESS, 0.0)
        with torch.no_grad():
            for inclusion in car.inclusions:
                for r in inclusion.rectangles(BOTTOM):
                    depth = THICKNESS if r.depth is None else r.depth.item()
                    inside = (
                        (r.x0.item() <= X)
                        & (r.x1.item() >= X)
                        & (r.y0.item() <= Y)
                        & (r.y1.item() >= Y)
                    )
                    density = density + inside * (r.density - PINE) * depth
        dm = density * (x[1] - x[0]) * (y[1] - y[0])
        mass = dm.sum() + 4 * WHEEL_MASS
        cg_x = ((dm * X).sum() + 2 * WHEEL_MASS * WHEELBASE) / mass
        cg_y = (dm * Y).sum() / mass
        assert car.mass.item() == pytest.approx(mass, rel=2e-3)
        assert car.cg[0].item() == pytest.approx(cg_x, rel=2e-3)
        assert car.cg[1].item() == pytest.approx(cg_y, rel=2e-2)

    def test_overrides(self):
        car = make_car(mass=0.14, cg=(0.02, 0.01))
        assert car.overrides == ["mass", "cg"]
        assert car.mass.item() == pytest.approx(0.14)
        torch.testing.assert_close(
            car.cg.detach(), torch.tensor([0.02, 0.01], dtype=torch.float64)
        )
        # The others are still computed from the geometry
        assert car.body_inertia.item() == pytest.approx(make_car().body_inertia.item())
        assert "mass_" in dict(car.named_parameters())

    def test_gradients(self):
        car = make_car()
        p = dict(car.named_parameters())
        names = [
            "profile.heights",
            "inclusions.0.position",
            "inclusions.0.length",
            "inclusions.0.depth_",
        ]
        for output in ("mass", "cg", "body_inertia"):

            def value(output=output):
                v = getattr(car, output)
                return v.sum() if v.ndim else v

            grads = torch.autograd.grad(value(), [p[n] for n in names])
            for name, grad in zip(names, grads, strict=True):
                parameter = p[name]
                flat = parameter.data.view(-1)
                for i in range(flat.numel()):
                    h = 1e-7
                    flat[i] += h
                    plus = value().item()
                    flat[i] -= 2 * h
                    minus = value().item()
                    flat[i] += h
                    fd = (plus - minus) / (2 * h)
                    assert grad.reshape(-1)[i].item() == pytest.approx(
                        fd, rel=1e-5, abs=1e-9
                    ), f"{output} {name}[{i}]"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"inclusions": [pocket(position=-1.0 * INCH)]}, "block length"),
            ({"inclusions": [pocket(depth=2.0 * INCH)]}, "deeper"),
            ({"thickness": -1.0}, "thickness"),
            ({"heights": [1.0, -0.1, 1.0, 1.0]}, "nonnegative"),
        ],
    )
    def test_check(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            make_car(**kwargs)


class TestSimulation:
    def test_matches_simple_car(self):
        geometric = make_car()
        with torch.no_grad():
            simple = pwdsim.SimpleCar(
                **{
                    name: getattr(geometric, name).detach()
                    for name in pwdsim.Car.PROPERTIES
                    if not name.startswith("n_")
                }
            )
        track = pwdsim.besttrack(35)
        times = [
            pwdsim.Simulation(track, c, dt=1e-3, duration=2.6)().finish_time.item()
            for c in (geometric, simple)
        ]
        assert times[0] == pytest.approx(times[1], rel=1e-12)

    def test_adjoint_gradients(self):
        car = make_car()
        sim = pwdsim.Simulation(pwdsim.besttrack(35), car, dt=1e-3, duration=2.6)
        sim().finish_time.backward()
        adjoint = car.profile.heights.grad.clone()
        car.zero_grad()
        sim(adjoint=False).finish_time.backward()
        torch.testing.assert_close(adjoint, car.profile.heights.grad)
        assert adjoint.abs().max() > 0


class TestConstraints:
    def evaluate(self, constraint, car):
        return constraint(car, None)

    def test_inside_body(self):
        constraint = constraints.regions_inside_body()
        values = self.evaluate(constraint, make_car())
        # pocket: top, rear, front, weight in pocket; void region: top, rear,
        # front, bottom
        assert values.shape == (8,)
        assert torch.all(values >= 0)
        # A pocket taller than the body sticks out the top
        tall = make_car(inclusions=[pocket(height=1.3 * INCH, length=0.3 * INCH)])
        assert self.evaluate(constraint, tall)[0] < 0

    def test_inside_body_smooth_min(self):
        car = make_car(inclusions=[pocket(position=1.0 * INCH, width=2.0 * INCH)])
        with torch.no_grad():
            value = self.evaluate(constraints.regions_inside_body(samples=64), car)[0]
            x = torch.linspace(0.0, 2.0, 64, dtype=torch.float64) * INCH
            exact = (car.profile.top(x) - (BOTTOM + 0.5 * INCH)).min()
        assert value <= exact
        assert value >= exact - math.log(64) / 1e4

    def test_inside_body_gradient(self):
        car = make_car()
        constraint = constraints.regions_inside_body()
        heights = car.profile.heights
        (grad,) = torch.autograd.grad(self.evaluate(constraint, car)[0], heights)
        h = 1e-7
        for i in range(len(heights)):
            with torch.no_grad():
                heights[i] += h
                plus = self.evaluate(constraint, car)[0].item()
                heights[i] -= 2 * h
                minus = self.evaluate(constraint, car)[0].item()
                heights[i] += h
            assert grad[i].item() == pytest.approx((plus - minus) / (2 * h), abs=1e-6)

    def test_within_thickness(self):
        constraint = constraints.regions_within_thickness()
        values = self.evaluate(constraint, make_car())
        torch.testing.assert_close(
            values, torch.tensor([THICKNESS - 0.75 * INCH], dtype=torch.float64)
        )
        assert self.evaluate(constraint, make_car(inclusions=[void()])).shape == (0,)

    def test_dont_overlap(self):
        constraint = constraints.regions_dont_overlap()
        values = self.evaluate(constraint, make_car())
        assert values.item() == pytest.approx(2.0 * INCH - (-0.45 + 0.3) * INCH)
        overlapping = make_car(
            inclusions=[pocket(position=2.1 * INCH), void()], heights=[1.25] * 8
        )
        assert self.evaluate(constraint, overlapping).item() < 0

    def test_avoid_axles(self):
        car = make_car()
        constraint = constraints.regions_avoid_axles(car, clearance=0.1 * INCH)
        values = self.evaluate(constraint, car)
        # pocket behind the rear axle; void between the axles
        expected = [-0.1 - (-0.45 + 0.3), 2.0 - 0.1, 4.375 - 0.1 - 3.0]
        np.testing.assert_allclose(values.detach().numpy(), np.array(expected) * INCH)
        with pytest.raises(ValueError, match="axle slot"):
            constraints.regions_avoid_axles(make_car(inclusions=[pocket(position=0.0)]))

    def test_needs_geometric_car(self):
        from test_kinematics import make_car as simple_car

        with pytest.raises(TypeError, match="GeometricCar"):
            constraints.regions_dont_overlap()(simple_car(), None)


def test_optimize_pocket():
    """Optimize a weight pocket's position and length."""
    car = make_car(inclusions=[pocket(length=0.1 * INCH)], heights=[1.25] * 8)
    sim = pwdsim.Simulation(pwdsim.besttrack(35), car, dt=1e-3, duration=2.6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = pwdsim.optimize(
            sim,
            {
                "inclusions.0.position": (-0.8 * INCH, -0.1 * INCH),
                "inclusions.0.length": (0.05 * INCH, 1.2 * INCH),
            },
            constraints=[
                constraints.max_mass(),
                constraints.regions_inside_body(),
                constraints.regions_avoid_axles(car),
            ],
        )
    assert result.success
    assert result.final_objective < result.initial_objective
    # The weight fills the mass budget and moves to the rear of the block
    assert car.mass.item() == pytest.approx(5 * 28.349523125e-3, rel=1e-3)
    assert result.constraints["regions_inside_body"][1].item() < 0.002


def test_plot_geometry():
    fig = plot_geometry(make_car())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "x (in)"
    # body, pocket void and weight, void region, two wheels
    assert len(ax.patches) == 6
