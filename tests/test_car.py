import pytest
import torch

from pwdsim.car import Car, SimpleCar
from pwdsim.units import INCH, OUNCE


def car_args(**overrides):
    args = dict(
        cg=(1.0 * INCH, 0.4 * INCH),
        wheelbase=4.375 * INCH,
        front_offset=6.0 * INCH,
        mass=5 * OUNCE,
        body_inertia=4e-4,
        rear_wheel_inertia=3.4e-7,
        front_wheel_inertia=3.4e-7,
        rear_wheel_radius=0.595 * INCH,
        front_wheel_radius=0.595 * INCH,
        rear_axle_radius=0.045 * INCH,
        front_axle_radius=0.045 * INCH,
        rear_axle_friction=0.1,
        front_axle_friction=0.1,
        frontal_area=1.4e-3,
        drag_coefficient=0.4,
        rolling_friction=0.002,
    )
    args.update(overrides)
    return args


@pytest.fixture
def car():
    return SimpleCar(**car_args())


def test_abstract():
    with pytest.raises(TypeError):
        Car()


def test_incomplete_subclass():
    class Incomplete(Car):
        @property
        def mass(self):
            return torch.tensor(1.0)

    with pytest.raises(TypeError):
        Incomplete()


def test_properties(car):
    args = car_args()
    for name in Car.PROPERTIES:
        value = getattr(car, name)
        if name in ("n_rear_wheels", "n_front_wheels"):
            assert value == 2
        else:
            assert value.dtype == torch.float64
            torch.testing.assert_close(
                value, torch.as_tensor(args[name], dtype=torch.float64)
            )
    assert car.cg.shape == (2,)


def test_summary(car):
    summary = car.summary()
    assert list(summary) == list(Car.PROPERTIES)
    assert summary["mass"] == pytest.approx(5 * OUNCE)
    assert summary["cg"] == pytest.approx([1.0 * INCH, 0.4 * INCH])
    assert summary["n_front_wheels"] == 2


def test_properties_read_only(car):
    with pytest.raises(AttributeError):
        car.mass = 1.0


def test_parameters(car):
    # Every property except the wheel counts is a trainable parameter
    params = dict(car.named_parameters())
    assert len(params) == len(Car.PROPERTIES) - 2
    assert all(p.requires_grad and p.dtype == torch.float64 for p in params.values())
    assert params["mass_"] is car.mass


def test_gradient(car):
    energy = car.mass * 9.81 * car.cg[1] + 0.5 * car.rear_wheel_inertia
    energy.backward()
    torch.testing.assert_close(car.mass_.grad, 9.81 * car.cg[1].detach())
    torch.testing.assert_close(car.cg_.grad[1], 9.81 * car.mass.detach())
    assert car.cg_.grad[0] == 0
    assert car.wheelbase_.grad is None


def test_freeze_parameter(car):
    car.mass_.requires_grad_(False)
    trainable = [name for name, p in car.named_parameters() if p.requires_grad]
    assert "mass_" not in trainable
    assert len(trainable) == len(Car.PROPERTIES) - 3


def test_optimizer_step(car):
    optimizer = torch.optim.SGD(car.parameters(), lr=0.1)
    (car.mass**2).backward()
    optimizer.step()
    assert car.mass < 5 * OUNCE


def test_inputs_copied():
    cg = torch.tensor([0.02, 0.01], dtype=torch.float64)
    car = SimpleCar(**car_args(cg=cg))
    cg[0] = 1.0
    assert car.cg[0].item() == pytest.approx(0.02)


def test_three_wheeler():
    car = SimpleCar(**car_args(), n_front_wheels=1)
    assert car.n_front_wheels == 1
    assert car.n_rear_wheels == 2


def test_state_dict_round_trip(car):
    other = SimpleCar(**car_args(mass=3 * OUNCE))
    other.load_state_dict(car.state_dict())
    torch.testing.assert_close(other.mass, car.mass)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mass": 0.0},
        {"wheelbase": -1.0},
        {"rear_wheel_radius": 0.0},
        {"body_inertia": -1e-4},
        {"rolling_friction": -0.1},
        {"n_rear_wheels": 0},
        {"cg": (0.1, 0.2, 0.3)},
        {"front_offset": 3.0 * INCH},
        {"rear_axle_radius": 0.6 * INCH},
        {"front_axle_radius": 0.6 * INCH},
    ],
)
def test_invalid(overrides):
    with pytest.raises(ValueError):
        SimpleCar(**car_args(**overrides))
