import math

import pytest
import torch
from test_geometric_car import car_args, profile

import pwdsim
from pwdsim.aerodynamics import DragModel
from pwdsim.geometry import Profile
from pwdsim.units import DEGREE, INCH

THICKNESS = torch.tensor(1.75 * INCH, dtype=torch.float64)
RADIUS = torch.tensor(0.595 * INCH, dtype=torch.float64)
X_REAR, X_FRONT, BOTTOM = -0.875 * INCH, 6.125 * INCH, -0.15 * INCH


def make_profile(heights):
    return Profile(
        X_REAR,
        X_FRONT,
        BOTTOM,
        torch.tensor(heights, dtype=torch.float64) * INCH,
        1.25 * INCH,
    )


def breakdown(heights, model=None, wheels=(2, 2)):
    model = model or DragModel()
    return model.breakdown(make_profile(heights), THICKNESS, (RADIUS, RADIUS), wheels)


class TestBlock:
    """An uncut block, where every component has a closed form."""

    @pytest.fixture
    def drag(self):
        return breakdown([1.25] * 9)

    def test_friction(self, drag):
        length, height = 7 * INCH, 1.25 * INCH
        wetted = 2 * length * height + 2 * THICKNESS.item() * length
        reynolds = 4.5 * length / 1.5e-5
        expected = 1.328 / math.sqrt(reynolds) * wetted
        assert drag["friction"].item() == pytest.approx(expected, rel=1e-6)

    def test_forebody(self, drag):
        expected = 0.8 * THICKNESS.item() * 1.25 * INCH
        assert drag["forebody"].item() == pytest.approx(expected, rel=1e-5)

    def test_base(self, drag):
        area = THICKNESS.item() * 1.25 * INCH
        forebody = (drag["friction"] + drag["forebody"]).item() / area
        expected = 0.029 / math.sqrt(forebody) * area
        assert drag["base"].item() == pytest.approx(expected, rel=1e-6)

    def test_wheels(self, drag):
        expected = (2 * 0.5 + 2) * 0.6 * 2 * RADIUS.item() * 0.3 * INCH
        assert drag["wheels"].item() == pytest.approx(expected)

    def test_totals(self, drag):
        area = THICKNESS.item() * 1.25 * INCH
        assert drag["frontal_area"].item() == pytest.approx(area, rel=2e-3)
        assert drag["frontal_area"].item() >= area
        total = sum(drag[k] for k in ("friction", "forebody", "base", "wheels"))
        torch.testing.assert_close(drag["total"], total)
        torch.testing.assert_close(
            drag["drag_coefficient"], total / drag["frontal_area"]
        )
        body = (drag["total"] - drag["wheels"]).item() / area
        assert 0.85 < body < 1.0  # a bluff block


def test_friction_coefficient():
    model = DragModel()
    assert model.friction_coefficient(0.1) == pytest.approx(
        1.328 / math.sqrt(4.5 * 0.1 / 1.5e-5)
    )
    reynolds = 4.5 * 10.0 / 1.5e-5  # turbulent
    expected = 0.074 * reynolds**-0.2 - 1742 / reynolds
    assert model.friction_coefficient(10.0) == pytest.approx(expected)


def test_lower_nose_less_drag():
    block = breakdown([1.25] * 9)
    wedge = breakdown([1.25, 1.2, 1.1, 0.95, 0.8, 0.65, 0.5, 0.35, 0.25])
    assert wedge["forebody"] < block["forebody"] / 2
    assert wedge["drag_coefficient"] < block["drag_coefficient"]


def test_slopes_blunt_or_ramp():
    """Gentle forward-facing slopes count as ramps; steep ones as blunt."""
    model = DragModel(face=1.0, ramp=0.0)
    face = THICKNESS.item() * 0.75 * INCH
    # A front face 0.75 in tall, with a straight ramp of about 5 degrees up to 1.25 in
    ramp = [1.25 - 0.5 * i / 8 for i in range(9)]
    gentle = breakdown(ramp, model)
    assert gentle["forebody"].item() == pytest.approx(face, rel=1e-3)
    # The same drop in a short, steep step counts partly as blunt: the spline
    # smooths the step, and only the parts steeper than separated_angle count fully
    steep = breakdown([1.25, 1.25, 1.25, 1.25, 1.25, 0.75, 0.75, 0.75, 0.75], model)
    step = THICKNESS.item() * 0.5 * INCH
    assert steep["forebody"].item() > face + 0.2 * step


def test_steep_rear_ramp_is_base():
    """A steep drop toward the rear separates like a base; a gentle one doesn't."""
    gentle = breakdown([0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.1, 1.2, 1.25])
    steep = breakdown([0.25, 0.25, 0.25, 0.25, 1.25, 1.25, 1.25, 1.25, 1.25])
    assert steep["base"] > 2 * gentle["base"]


def test_wheel_count():
    four = breakdown([1.25] * 9, wheels=(2, 2))
    three = breakdown([1.25] * 9, wheels=(2, 1))
    per_front_wheel = 0.6 * 2 * RADIUS.item() * 0.3 * INCH
    assert (four["wheels"] - three["wheels"]).item() == pytest.approx(per_front_wheel)


def test_gradients():
    p = make_profile([1.25, 1.2, 1.1, 0.95, 0.8, 0.65, 0.5, 0.35, 0.25])
    model = DragModel()

    def total():
        return model.breakdown(p, THICKNESS, (RADIUS, RADIUS), (2, 2))["total"]

    (grad,) = torch.autograd.grad(total(), p.heights)
    h = 1e-7
    for i in range(len(p.heights)):
        with torch.no_grad():
            p.heights[i] += h
            plus = total().item()
            p.heights[i] -= 2 * h
            minus = total().item()
            p.heights[i] += h
        assert grad[i].item() == pytest.approx(
            (plus - minus) / (2 * h), rel=1e-4, abs=1e-9
        )


def test_angles():
    model = DragModel()
    assert model.attached_angle == pytest.approx(10 * DEGREE)
    assert model.separated_angle == pytest.approx(25 * DEGREE)


class TestGeometricCar:
    def make(self, **overrides):
        args = car_args()
        del args["frontal_area"], args["drag_coefficient"]
        args.update(overrides)
        return pwdsim.GeometricCar(profile(), [], **args)

    def test_computed(self):
        car = self.make()
        drag = car.drag_breakdown()
        assert car.overrides == []
        torch.testing.assert_close(car.frontal_area, drag["frontal_area"])
        torch.testing.assert_close(car.drag_coefficient, drag["drag_coefficient"])

    def test_overridden(self):
        car = self.make(drag_coefficient=0.4)
        assert car.overrides == ["drag_coefficient"]
        assert car.drag_coefficient.item() == pytest.approx(0.4)
        torch.testing.assert_close(
            car.frontal_area, car.drag_breakdown()["frontal_area"]
        )

    def test_drag_model(self):
        clean = self.make(drag_model=DragModel(wheel=0.0))
        assert clean.drag_breakdown()["wheels"].item() == 0.0
        assert clean.drag_coefficient < self.make().drag_coefficient

    def test_simulation_gradients(self):
        car = self.make()
        sim = pwdsim.Simulation(pwdsim.besttrack(35), car, dt=1e-3, duration=2.6)
        sim().finish_time.backward()
        adjoint = car.profile.heights.grad.clone()
        car.zero_grad()
        sim(adjoint=False).finish_time.backward()
        torch.testing.assert_close(adjoint, car.profile.heights.grad)
        # Without the drag term, the heights affect the time only through the mass
        drag_free = self.make()
        sim = pwdsim.Simulation(pwdsim.besttrack(35), drag_free, dt=1e-3, duration=2.6)
        sim.disable("drag")
        sim().finish_time.backward()
        assert not torch.allclose(drag_free.profile.heights.grad, adjoint)
