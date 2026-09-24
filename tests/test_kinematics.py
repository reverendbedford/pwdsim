import math

import pytest
import torch

from pwdsim.car import SimpleCar
from pwdsim.kinematics import CarKinematics
from pwdsim.track import SplineTrack, besttrack
from pwdsim.units import INCH, OUNCE

FIELDS = ("value", "ds", "dss", "dsss")


def make_car(**overrides):
    args = dict(
        cg=(1.0 * INCH, 0.4 * INCH),
        wheelbase=4.375 * INCH,
        front_offset=6.125 * INCH,
        mass=5 * OUNCE,
        body_inertia=3.9e-4,
        rear_wheel_inertia=3.4e-7,
        front_wheel_inertia=3.4e-7,
        rear_wheel_radius=0.595 * INCH,
        front_wheel_radius=0.5 * INCH,
        rear_axle_radius=0.0435 * INCH,
        front_axle_radius=0.0435 * INCH,
        rear_axle_friction=0.1,
        front_axle_friction=0.1,
        frontal_area=0.0014,
        drag_coefficient=0.4,
        rolling_friction=0.002,
    )
    args.update(overrides)
    return SimpleCar(**args)


def curvy_track():
    knots = [0.0, 1.0, 2.5, 3.0, 5.0]
    angles = [-0.6, -0.8, -0.2, 0.1, 0.0]
    curvatures = [2.0, 3.0, 5.0, -2.0, 0.0]
    return SplineTrack(knots, angles, curvatures, s_start=0.5, s_finish=4.5)


TRACKS = {"curvy": curvy_track, "besttrack": besttrack}


def sample_points(track, kinematics, n=40, clearance=1e-3):
    """Rear contact locations with both contacts clear of the spline knots, where
    the curvature derivative jumps."""
    s = torch.cat(
        [
            torch.linspace(0.3, float(track.length) - 0.3, n, dtype=torch.float64),
            # Concentrate points around the curve of the BestTrack
            torch.linspace(2.0, 3.5, n, dtype=torch.float64),
        ]
    )
    with torch.no_grad():
        sigma = kinematics.evaluate(s).front_contact.value

    def near_knot(x):
        return (x[:, None] - track.knots).abs().min(-1).values < clearance

    return s[~(near_knot(s) | near_knot(sigma))]


def quantities(config):
    return {
        "rear_axle": config.rear_axle,
        "axis": config.axis,
        "front_contact": config.front_contact,
        "pitch": config.pitch,
        "cg": config.cg,
        "front": config.front,
        "front_axle": config.front_axle,
    }


@pytest.fixture(params=list(TRACKS))
def setup(request):
    track = TRACKS[request.param]()
    kinematics = CarKinematics(track, make_car())
    return track, kinematics, sample_points(track, kinematics)


def test_s_derivatives(setup):
    _, kinematics, s = setup
    h = 1e-5
    with torch.no_grad():
        center = quantities(kinematics.evaluate(s))
        plus = quantities(kinematics.evaluate(s + h))
        minus = quantities(kinematics.evaluate(s - h))
    for name, q in center.items():
        for lower, higher in zip(FIELDS[:-1], FIELDS[1:], strict=True):
            fd = (getattr(plus[name], lower) - getattr(minus[name], lower)) / (2 * h)
            torch.testing.assert_close(
                getattr(q, higher), fd, rtol=1e-5, atol=1e-5, msg=f"{name}.{higher}"
            )


@pytest.mark.parametrize("lift", ["dhr", "dhf"])
def test_lift_derivatives(setup, lift):
    _, kinematics, s = setup
    h = 1e-6
    kwargs = {"h_r" if lift == "dhr" else "h_f": h}
    kwargs_minus = {k: -v for k, v in kwargs.items()}
    with torch.no_grad():
        center = quantities(kinematics.evaluate(s))
        plus = quantities(kinematics.evaluate(s, **kwargs))
        minus = quantities(kinematics.evaluate(s, **kwargs_minus))
    for name, q in center.items():
        fd = (plus[name].value - minus[name].value) / (2 * h)
        torch.testing.assert_close(
            getattr(q, lift), fd, rtol=1e-6, atol=1e-7, msg=f"{name}.{lift}"
        )


def test_constraints(setup):
    track, kinematics, s = setup
    car = kinematics.car
    with torch.no_grad():
        config = kinematics.evaluate(s)
        rear, front = config.rear_axle.value, config.front_axle.value
        # Wheelbase is maintained
        torch.testing.assert_close(
            torch.linalg.norm(front - rear, dim=-1), car.wheelbase.expand(len(s))
        )
        # Both axles sit one wheel radius off the track at their contacts
        for axle, contact, radius in (
            (rear, s, car.rear_wheel_radius),
            (front, config.front_contact.value, car.front_wheel_radius),
        ):
            theta = track.angle(contact)
            x, y = track.position(contact)
            expected = torch.stack(
                [x - radius * torch.sin(theta), y + radius * torch.cos(theta)], -1
            )
            torch.testing.assert_close(axle, expected)


def test_straight_track():
    theta = -0.5
    track = SplineTrack([0.0, 3.0], [theta] * 2, [0.0] * 2, 0.5, 2.5)
    car = make_car()
    config = CarKinematics(track, car).evaluate(torch.tensor([1.0, 1.5]))
    # Unequal wheel radii tilt the car relative to the track
    tilt = math.asin((car.front_wheel_radius - car.rear_wheel_radius).item() / 0.111125)
    torch.testing.assert_close(
        config.pitch.value, torch.full((2,), theta + tilt, dtype=torch.float64)
    )
    for field in ("ds", "dss", "dsss"):
        torch.testing.assert_close(
            getattr(config.pitch, field), torch.zeros(2).double()
        )
    # The whole car translates along the track
    torch.testing.assert_close(
        config.cg.ds.detach(),
        torch.tensor([[math.cos(theta), math.sin(theta)]] * 2, dtype=torch.float64),
    )


def test_start_position():
    theta = -0.5
    track = SplineTrack([0.0, 3.0], [theta] * 2, [0.0] * 2, 0.5, 2.5)
    car = make_car(front_wheel_radius=0.595 * INCH)
    s0 = CarKinematics(track, car).start_position()
    torch.testing.assert_close(s0, track.s_start - car.front_offset)


def test_start_position_on_curve():
    track = besttrack()
    kinematics = CarKinematics(track, make_car())
    s0 = kinematics.start_position()
    front = kinematics.evaluate(s0).front.value
    theta = track.angle(track.s_start)
    pin = torch.stack(track.position(track.s_start))
    along = torch.dot(front - pin, torch.stack([torch.cos(theta), torch.sin(theta)]))
    assert along.item() == pytest.approx(0.0, abs=1e-12)


def test_parameter_gradients():
    """Gradients through the Newton solves match finite differences."""
    track = besttrack()
    s = torch.tensor([2.3, 2.6, 3.0], dtype=torch.float64)

    def outputs(wheelbase):
        car = make_car(wheelbase=wheelbase)
        kinematics = CarKinematics(track, car)
        config = kinematics.evaluate(s)
        return (
            config.front_contact.value.sum()
            + config.cg.value.sum()
            + kinematics.start_position()
        ), car

    w0 = 4.375 * INCH
    value, car = outputs(w0)
    value.backward()
    h = 1e-7
    with torch.no_grad():
        fd = (outputs(w0 + h)[0] - outputs(w0 - h)[0]) / (2 * h)
    assert car.wheelbase_.grad.item() == pytest.approx(fd.item(), rel=1e-6)


def test_batched_car():
    track = besttrack()
    cg = torch.tensor([[1.0, 0.4], [0.5, 0.2]]) * INCH
    wheelbase = torch.tensor([4.375, 4.0]) * INCH
    batched = CarKinematics(track, make_car(cg=cg, wheelbase=wheelbase))
    s = torch.tensor([[2.3, 2.3], [2.6, 2.6]], dtype=torch.float64)
    with torch.no_grad():
        config = batched.evaluate(s)
        for i in range(2):
            single = CarKinematics(track, make_car(cg=cg[i], wheelbase=wheelbase[i]))
            expected = single.evaluate(s[:, i])
            for field in FIELDS + ("dhr", "dhf"):
                torch.testing.assert_close(
                    getattr(config.cg, field)[:, i], getattr(expected.cg, field)
                )
        s0 = batched.start_position()
        assert s0.shape == (2,)
