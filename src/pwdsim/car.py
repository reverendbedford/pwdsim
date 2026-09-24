"""Car models.

A car is a rigid body riding on a rear and a front axle.  [`Car`][pwdsim.car.Car]
defines the essential parameters the simulation needs, as read-only properties.
Concrete car classes decide where those values come from:
[`SimpleCar`][pwdsim.car.SimpleCar] stores each one directly as a trainable
parameter, while more detailed models can compute them from a description of the
car's geometry and materials.

Positions are given in the body frame, with its origin at the rear axle center, its
first axis pointing toward the front axle center, and its second axis perpendicular
to that, pointing away from the track.  All values are in SI units.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

from pwdsim.types import DTYPE, Scalar


class Car(nn.Module, ABC):
    """Base class for cars.

    Subclasses implement each of the abstract properties, returning a float64
    tensor, except for the wheel counts, which are integers.  The tensors may have
    leading batch dimensions to describe several car designs at once; see
    [`batch_shape`][pwdsim.car.Car.batch_shape].  Because a car is a
    `torch.nn.Module`, its trainable parameters are available
    through `car.parameters()` for optimization.
    """

    @property
    @abstractmethod
    def cg(self) -> torch.Tensor:
        """Position of the center of gravity relative to the rear axle, in the body
        frame, $\\mathbf{x}_g$, as a tensor with shape `(2,)`."""

    @property
    @abstractmethod
    def wheelbase(self) -> torch.Tensor:
        """Distance between the rear and front axle centers, $w$."""

    @property
    @abstractmethod
    def front_offset(self) -> torch.Tensor:
        """Distance from the rear axle to the front of the car, along the body frame
        first axis, $d$."""

    @property
    @abstractmethod
    def mass(self) -> torch.Tensor:
        """Total mass of the car, including the wheels, $M$."""

    @property
    @abstractmethod
    def body_inertia(self) -> torch.Tensor:
        """Moment of inertia of the car about its center of gravity for pitching
        motion, $I_b$."""

    @property
    @abstractmethod
    def n_rear_wheels(self) -> int:
        """Number of rear wheels in contact with the track, $n_r$."""

    @property
    @abstractmethod
    def n_front_wheels(self) -> int:
        """Number of front wheels in contact with the track, $n_f$."""

    @property
    @abstractmethod
    def rear_wheel_inertia(self) -> torch.Tensor:
        """Moment of inertia of each rear wheel about its axle, $I_r$."""

    @property
    @abstractmethod
    def front_wheel_inertia(self) -> torch.Tensor:
        """Moment of inertia of each front wheel about its axle, $I_f$."""

    @property
    @abstractmethod
    def rear_wheel_radius(self) -> torch.Tensor:
        """Radius of the rear wheels, $r_r$."""

    @property
    @abstractmethod
    def front_wheel_radius(self) -> torch.Tensor:
        """Radius of the front wheels, $r_f$."""

    @property
    @abstractmethod
    def rear_axle_radius(self) -> torch.Tensor:
        """Radius of the rear axle, $a_r$."""

    @property
    @abstractmethod
    def front_axle_radius(self) -> torch.Tensor:
        """Radius of the front axle, $a_f$."""

    @property
    @abstractmethod
    def rear_axle_friction(self) -> torch.Tensor:
        """Coefficient of friction between the rear wheels and axle, $\\mu_r$."""

    @property
    @abstractmethod
    def front_axle_friction(self) -> torch.Tensor:
        """Coefficient of friction between the front wheels and axle, $\\mu_f$."""

    @property
    @abstractmethod
    def frontal_area(self) -> torch.Tensor:
        """Cross-sectional area of the car facing the air flow, $A$."""

    @property
    @abstractmethod
    def drag_coefficient(self) -> torch.Tensor:
        """Aerodynamic drag coefficient, $C_d$."""

    @property
    @abstractmethod
    def rolling_friction(self) -> torch.Tensor:
        """Rolling friction coefficient of the wheels on the track, $c_r$."""

    PROPERTIES = (
        "cg",
        "wheelbase",
        "front_offset",
        "mass",
        "body_inertia",
        "n_rear_wheels",
        "n_front_wheels",
        "rear_wheel_inertia",
        "front_wheel_inertia",
        "rear_wheel_radius",
        "front_wheel_radius",
        "rear_axle_radius",
        "front_axle_radius",
        "rear_axle_friction",
        "front_axle_friction",
        "frontal_area",
        "drag_coefficient",
        "rolling_friction",
    )
    """Names of the essential car properties, in order."""

    def summary(self) -> dict[str, float | int | list[float]]:
        """The current values of the essential car properties, by name, as plain
        python numbers for display.  Use the properties themselves in calculations
        that need gradients."""
        return {
            name: value.tolist() if isinstance(value, torch.Tensor) else value
            for name in self.PROPERTIES
            for value in (getattr(self, name),)
        }

    def check(self):
        """Check that the car properties are physically meaningful.

        Raises:
            ValueError: if any property is out of range.
        """
        positive = (
            "wheelbase",
            "front_offset",
            "mass",
            "rear_wheel_radius",
            "front_wheel_radius",
            "rear_axle_radius",
            "front_axle_radius",
        )
        nonnegative = (
            "body_inertia",
            "rear_wheel_inertia",
            "front_wheel_inertia",
            "rear_axle_friction",
            "front_axle_friction",
            "frontal_area",
            "drag_coefficient",
            "rolling_friction",
        )
        for name in positive:
            if not torch.all(getattr(self, name) > 0):
                raise ValueError(f"{name} must be positive")
        for name in nonnegative:
            if not torch.all(getattr(self, name) >= 0):
                raise ValueError(f"{name} must be nonnegative")
        for name in ("n_rear_wheels", "n_front_wheels"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.cg.ndim < 1 or self.cg.shape[-1] != 2:
            raise ValueError("cg must be a vector with two components")
        try:
            _ = self.batch_shape
        except RuntimeError as error:
            raise ValueError("Batch shapes of the car properties differ") from error
        if not torch.all(self.front_offset >= self.wheelbase):
            raise ValueError("The front of the car must be ahead of the front axle")
        if not torch.all(self.rear_axle_radius < self.rear_wheel_radius):
            raise ValueError("The rear axle must be smaller than the rear wheels")
        if not torch.all(self.front_axle_radius < self.front_wheel_radius):
            raise ValueError("The front axle must be smaller than the front wheels")

    @property
    def batch_shape(self) -> torch.Size:
        """Batch shape of the car, broadcast from the shapes of its properties.

        Every property may have leading batch dimensions, to describe several car
        designs at once, as long as the shapes broadcast.  The `cg` has an extra
        trailing dimension of size 2.  A single car has batch shape `()`.
        """
        shapes = [
            getattr(self, name).shape
            for name in self.PROPERTIES
            if name not in ("cg", "n_rear_wheels", "n_front_wheels")
        ]
        return torch.broadcast_shapes(self.cg.shape[:-1], *shapes)


def _parameter(value) -> nn.Parameter:
    return nn.Parameter(torch.as_tensor(value, dtype=DTYPE).clone())


def _stored(attribute: str) -> property:
    """Read-only property returning a stored attribute."""
    return property(lambda self: getattr(self, attribute))


class SimpleCar(Car):
    """A car defined directly by its essential properties.

    Each property except the wheel counts is stored as a float64
    `torch.nn.Parameter`, so all of them are trainable by
    default.  To hold a property fixed during optimization, turn off its gradient,
    e.g. `car.mass_.requires_grad_(False)`.  The parameters are stored with a
    trailing underscore (`mass_`) and exposed through the read-only properties
    defined by [`Car`][pwdsim.car.Car] (`mass`).

    All arguments are in SI units: multiply by the constants in
    [`pwdsim.units`][pwdsim.units] to give them in customary units.

    To describe a batch of car designs, give any of the arguments a leading batch
    dimension, e.g. `mass=torch.tensor([4.0, 5.0]) * OUNCE` or a `cg` with shape
    `(nbatch, 2)`.  Arguments without the batch dimension are shared by every car.

    Args:
        cg: center of gravity relative to the rear axle in the body frame,
            $\\mathbf{x}_g$, as `(along, above)`.
        wheelbase: distance between the axle centers, $w$.
        front_offset: distance from the rear axle to the front of the car, $d$.
        mass: total mass including the wheels, $M$.
        body_inertia: pitching moment of inertia about the center of gravity, $I_b$.
        rear_wheel_inertia: moment of inertia of each rear wheel, $I_r$.
        front_wheel_inertia: moment of inertia of each front wheel, $I_f$.
        rear_wheel_radius: radius of the rear wheels, $r_r$.
        front_wheel_radius: radius of the front wheels, $r_f$.
        rear_axle_radius: radius of the rear axle, $a_r$.
        front_axle_radius: radius of the front axle, $a_f$.
        rear_axle_friction: rear axle friction coefficient, $\\mu_r$.
        front_axle_friction: front axle friction coefficient, $\\mu_f$.
        frontal_area: cross-sectional area, $A$.
        drag_coefficient: drag coefficient, $C_d$.
        rolling_friction: rolling friction coefficient, $c_r$.
        n_rear_wheels: number of rear wheels on the track, $n_r$.
        n_front_wheels: number of front wheels on the track, $n_f$.
    """

    def __init__(
        self,
        cg: tuple[Scalar, Scalar] | torch.Tensor,
        wheelbase: Scalar,
        front_offset: Scalar,
        mass: Scalar,
        body_inertia: Scalar,
        rear_wheel_inertia: Scalar,
        front_wheel_inertia: Scalar,
        rear_wheel_radius: Scalar,
        front_wheel_radius: Scalar,
        rear_axle_radius: Scalar,
        front_axle_radius: Scalar,
        rear_axle_friction: Scalar,
        front_axle_friction: Scalar,
        frontal_area: Scalar,
        drag_coefficient: Scalar,
        rolling_friction: Scalar,
        n_rear_wheels: int = 2,
        n_front_wheels: int = 2,
    ):
        super().__init__()
        self.cg_ = _parameter(cg)
        self.wheelbase_ = _parameter(wheelbase)
        self.front_offset_ = _parameter(front_offset)
        self.mass_ = _parameter(mass)
        self.body_inertia_ = _parameter(body_inertia)
        self.rear_wheel_inertia_ = _parameter(rear_wheel_inertia)
        self.front_wheel_inertia_ = _parameter(front_wheel_inertia)
        self.rear_wheel_radius_ = _parameter(rear_wheel_radius)
        self.front_wheel_radius_ = _parameter(front_wheel_radius)
        self.rear_axle_radius_ = _parameter(rear_axle_radius)
        self.front_axle_radius_ = _parameter(front_axle_radius)
        self.rear_axle_friction_ = _parameter(rear_axle_friction)
        self.front_axle_friction_ = _parameter(front_axle_friction)
        self.frontal_area_ = _parameter(frontal_area)
        self.drag_coefficient_ = _parameter(drag_coefficient)
        self.rolling_friction_ = _parameter(rolling_friction)
        self._n_rear_wheels = int(n_rear_wheels)
        self._n_front_wheels = int(n_front_wheels)
        self.check()

    cg = _stored("cg_")
    wheelbase = _stored("wheelbase_")
    front_offset = _stored("front_offset_")
    mass = _stored("mass_")
    body_inertia = _stored("body_inertia_")
    n_rear_wheels = _stored("_n_rear_wheels")
    n_front_wheels = _stored("_n_front_wheels")
    rear_wheel_inertia = _stored("rear_wheel_inertia_")
    front_wheel_inertia = _stored("front_wheel_inertia_")
    rear_wheel_radius = _stored("rear_wheel_radius_")
    front_wheel_radius = _stored("front_wheel_radius_")
    rear_axle_radius = _stored("rear_axle_radius_")
    front_axle_radius = _stored("front_axle_radius_")
    rear_axle_friction = _stored("rear_axle_friction_")
    front_axle_friction = _stored("front_axle_friction_")
    frontal_area = _stored("frontal_area_")
    drag_coefficient = _stored("drag_coefficient_")
    rolling_friction = _stored("rolling_friction_")
