"""Kinematics of a car riding on a track.

The configuration of a car on the track is described by the generalized coordinates
$q = (s, h_r, h_f)$: the arc length location $s$ of the rear wheel contact point, and
the lifts $h_r$ and $h_f$ of the rear and front axles off the track, along the track
normal.  The simulation always keeps the car on the track, $h_r = h_f = 0$, but the
lift directions define the normal forces (as constraint forces), so we need
derivatives with respect to them.

This module computes the positions of points on the car along with their analytic
derivatives: the first three derivatives with respect to $s$, the first derivatives
with respect to $h_r$ and $h_f$, and the mixed derivatives with respect to $s$ and
the lifts.

The track provides the unit tangent $t = (\\cos\\theta, \\sin\\theta)$ and
normal $n = (-\\sin\\theta, \\cos\\theta)$, with $t' = \\kappa n$ and
$n' = -\\kappa t$.  An axle of radius $r$ (plus lift $h$) rolling on the track has
its center on the offset curve $c(\\sigma) = p(\\sigma) + (r + h)\\, n(\\sigma)$.

The rear axle center is $R(s) = c_r(s)$.  The front axle contact point $\\sigma(s)$ is
found implicitly as the location where the front axle center $c_f(\\sigma)$ is a
distance $w$ (the wheelbase) from the rear axle center.  The car body frame then has
its origin at $R$ and first axis $u = (c_f - R) / w$.
"""

from dataclasses import dataclass

import torch

from pwdsim.car import Car
from pwdsim.track import Track
from pwdsim.types import DTYPE


@dataclass
class Derivatives:
    """A quantity and its analytic derivatives with respect to the generalized
    coordinates, evaluated at a configuration.

    For vector quantities (points) each field has a trailing dimension of size 2.

    Attributes:
        value: the quantity itself.
        ds: first derivative with respect to $s$.
        dss: second derivative with respect to $s$.
        dsss: third derivative with respect to $s$.
        dhr: first derivative with respect to the rear lift $h_r$.
        dhf: first derivative with respect to the front lift $h_f$.
        dhr_ds: mixed derivative with respect to $h_r$ and $s$.
        dhf_ds: mixed derivative with respect to $h_f$ and $s$.
    """

    value: torch.Tensor
    ds: torch.Tensor
    dss: torch.Tensor
    dsss: torch.Tensor
    dhr: torch.Tensor
    dhf: torch.Tensor
    dhr_ds: torch.Tensor
    dhf_ds: torch.Tensor


FIELDS = ("value", "ds", "dss", "dsss", "dhr", "dhf", "dhr_ds", "dhf_ds")
"""Names of the fields of [`Derivatives`][pwdsim.kinematics.Derivatives]."""


@dataclass
class TrackPoint:
    """The track at a contact point.

    Attributes:
        location: arc length location of the point.
        angle: track angle $\\theta$.
        curvature: curvature $\\kappa$.
        dcurvature: curvature derivative $\\kappa'$.
        ddcurvature: curvature second derivative $\\kappa''$.
    """

    location: torch.Tensor
    angle: torch.Tensor
    curvature: torch.Tensor
    dcurvature: torch.Tensor
    ddcurvature: torch.Tensor


def _dot(a, b):
    return torch.sum(a * b, dim=-1)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _rot(a):
    """Rotate vectors by 90 degrees counterclockwise."""
    return torch.stack([-a[..., 1], a[..., 0]], dim=-1)


def _vec(x, y):
    return torch.stack([x, y], dim=-1)


@dataclass
class _OffsetCurve:
    """An offset curve $c(\\sigma) = p(\\sigma) + r n(\\sigma)$ at a point."""

    c: torch.Tensor
    c1: torch.Tensor
    c2: torch.Tensor
    c3: torch.Tensor
    t: torch.Tensor
    n: torch.Tensor
    track: TrackPoint


def _offset_curve(track: Track, sigma, offset) -> _OffsetCurve:
    """Offset curve $c(\\sigma) = p(\\sigma) + r n(\\sigma)$ and its first three
    derivatives with respect to $\\sigma$, along with the track tangent and normal."""
    point = TrackPoint(
        location=sigma,
        angle=track.angle(sigma),
        curvature=track.curvature(sigma),
        dcurvature=track.curvature_derivative(sigma),
        ddcurvature=track.curvature_second_derivative(sigma),
    )
    kappa, dkappa = point.curvature, point.dcurvature
    x, y = track.position(sigma)
    t = _vec(torch.cos(point.angle), torch.sin(point.angle))
    n = _rot(t)

    alpha = 1 - offset * kappa
    dalpha = -offset * dkappa
    ddalpha = -offset * point.ddcurvature

    return _OffsetCurve(
        c=_vec(x, y) + offset[..., None] * n,
        c1=alpha[..., None] * t,
        c2=dalpha[..., None] * t + (alpha * kappa)[..., None] * n,
        c3=(ddalpha - alpha * kappa**2)[..., None] * t
        + (2 * dalpha * kappa + alpha * dkappa)[..., None] * n,
        t=t,
        n=n,
        track=point,
    )


class Configuration:
    """The configuration of a car on the track, with derivatives.

    Created by [`CarKinematics.evaluate`][pwdsim.kinematics.CarKinematics.evaluate].
    Points on the car body are available through
    [`point`][pwdsim.kinematics.Configuration.point], with shortcuts for the
    important ones.

    Attributes:
        rear_axle: center of the rear axle, $R$.
        axis: unit vector along the body frame first axis, $u$.
        front_contact: arc length location of the front wheel contact, $\\sigma$.
        pitch: pitch angle of the car body, $\\phi$, the angle of $u$ from the
            horizontal.
        rear_track: the track at the rear wheel contact, $s$.
        front_track: the track at the front wheel contact, $\\sigma$.
    """

    def __init__(
        self,
        car: Car,
        rear_axle: Derivatives,
        axis: Derivatives,
        front_contact: Derivatives,
        rear_track: TrackPoint,
        front_track: TrackPoint,
    ):
        self.car = car
        self.rear_axle = rear_axle
        self.axis = axis
        self.front_contact = front_contact
        self.rear_track = rear_track
        self.front_track = front_track
        self.pitch = self._pitch()

    def point(self, xi, eta) -> Derivatives:
        """A point fixed to the car body, at body frame coordinates $(\\xi, \\eta)$.

        The point is $P = R + \\xi u + \\eta\\, \\mathrm{rot}(u)$, which is linear in
        $R$ and $u$, so its derivatives follow directly from theirs.
        """
        xi = torch.as_tensor(xi, dtype=DTYPE)[..., None]
        eta = torch.as_tensor(eta, dtype=DTYPE)[..., None]
        R, u = self.rear_axle, self.axis
        return Derivatives(
            *(
                getattr(R, f) + xi * getattr(u, f) + eta * _rot(getattr(u, f))
                for f in FIELDS
            )
        )

    @property
    def front_axle(self) -> Derivatives:
        """Center of the front axle."""
        return self.point(self.car.wheelbase, 0.0)

    @property
    def cg(self) -> Derivatives:
        """Center of gravity of the car."""
        return self.point(self.car.cg[..., 0], self.car.cg[..., 1])

    @property
    def front(self) -> Derivatives:
        """Front of the car, the point used for the start and finish."""
        return self.point(self.car.front_offset, 0.0)

    def _pitch(self) -> Derivatives:
        u = self.axis
        return Derivatives(
            value=torch.atan2(u.value[..., 1], u.value[..., 0]),
            ds=_cross(u.value, u.ds),
            dss=_cross(u.value, u.dss),
            dsss=_cross(u.ds, u.dss) + _cross(u.value, u.dsss),
            dhr=_cross(u.value, u.dhr),
            dhf=_cross(u.value, u.dhf),
            dhr_ds=_cross(u.ds, u.dhr) + _cross(u.value, u.dhr_ds),
            dhf_ds=_cross(u.ds, u.dhf) + _cross(u.value, u.dhf_ds),
        )


class CarKinematics:
    """Positions of a car on a track, with analytic derivatives.

    Args:
        track: the track.
        car: the car.  Car properties with batch dimensions broadcast against the
            trailing dimensions of the arc length locations.
        newton_iterations: number of Newton iterations used to solve for the front
            wheel contact location and the starting position.
    """

    def __init__(self, track: Track, car: Car, newton_iterations: int = 3):
        self.track = track
        self.car = car
        self.newton_iterations = newton_iterations

    def evaluate(
        self,
        s: float | torch.Tensor,
        h_r: float | torch.Tensor = 0.0,
        h_f: float | torch.Tensor = 0.0,
    ) -> Configuration:
        """The configuration of the car with its rear wheel contact at `s`.

        Args:
            s: arc length location of the rear wheel contact.
            h_r: lift of the rear axle off the track.
            h_f: lift of the front axle off the track.
        """
        car = self.car
        s = torch.as_tensor(s, dtype=DTYPE)
        r_rear = car.rear_wheel_radius + h_r
        r_front = car.front_wheel_radius + h_f
        w = car.wheelbase
        shape = torch.broadcast_shapes(s.shape, r_rear.shape, r_front.shape, w.shape)
        s = s.expand(shape)

        rear = _offset_curve(self.track, s, r_rear.expand(shape))
        R, R1, R2, R3 = rear.c, rear.c1, rear.c2, rear.c3
        sigma = self._front_contact(s, R, r_front.expand(shape), w)
        front = _offset_curve(self.track, sigma, r_front.expand(shape))
        C, C1, C2, C3 = front.c, front.c1, front.c2, front.c3

        # Derivatives of the front contact location, from differentiating
        # D . D = w^2 with D = C(sigma(s)) - R(s)
        D = C - R
        DC1 = _dot(D, C1)
        sigma1 = _dot(D, R1) / DC1
        D1 = sigma1[..., None] * C1 - R1
        sigma2 = (-_dot(D1, D1) - sigma1**2 * _dot(D, C2) + _dot(D, R2)) / DC1
        D2 = sigma2[..., None] * C1 + (sigma1**2)[..., None] * C2 - R2
        sigma3 = (
            -3 * _dot(D1, D2)
            - 3 * sigma1 * sigma2 * _dot(D, C2)
            - sigma1**3 * _dot(D, C3)
            + _dot(D, R3)
        ) / DC1
        D3 = (
            sigma3[..., None] * C1
            + (3 * sigma1 * sigma2)[..., None] * C2
            + (sigma1**3)[..., None] * C3
            - R3
        )

        # Lift derivatives, from D . D_h = 0, with R_hr = n(s) and C_hf = n(sigma)
        n_rear, n_front = rear.n, front.n
        sigma_hr = _dot(D, n_rear) / DC1
        sigma_hf = -_dot(D, n_front) / DC1
        D_hr = sigma_hr[..., None] * C1 - n_rear
        D_hf = sigma_hf[..., None] * C1 + n_front

        # Mixed derivatives, from differentiating D . D_h = 0 with respect to s,
        # using n' = -kappa t
        dn_rear = -rear.track.curvature[..., None] * rear.t
        dn_front = -front.track.curvature[..., None] * front.t
        sigma_hr_s = (
            -_dot(D1, D_hr) - sigma_hr * sigma1 * _dot(D, C2) + _dot(D, dn_rear)
        ) / DC1
        sigma_hf_s = (
            -_dot(D1, D_hf)
            - sigma_hf * sigma1 * _dot(D, C2)
            - sigma1 * _dot(D, dn_front)
        ) / DC1
        D_hr_s = (
            sigma_hr_s[..., None] * C1 + (sigma_hr * sigma1)[..., None] * C2 - dn_rear
        )
        D_hf_s = (
            sigma_hf_s[..., None] * C1
            + (sigma_hf * sigma1)[..., None] * C2
            + sigma1[..., None] * dn_front
        )

        w = w[..., None]
        zero = torch.zeros_like(R)
        rear_axle = Derivatives(R, R1, R2, R3, n_rear, zero, dn_rear, zero)
        axis = Derivatives(
            D / w, D1 / w, D2 / w, D3 / w, D_hr / w, D_hf / w, D_hr_s / w, D_hf_s / w
        )
        front_contact = Derivatives(
            sigma, sigma1, sigma2, sigma3, sigma_hr, sigma_hf, sigma_hr_s, sigma_hf_s
        )
        return Configuration(
            car, rear_axle, axis, front_contact, rear.track, front.track
        )

    def _front_contact(self, s, R, r_front, w):
        """Solve $|c_f(\\sigma) - R| = w$ for the front contact location $\\sigma$.

        Newton iterations run without gradients, then one final iteration with
        gradients gives exact first derivatives with respect to the car parameters.
        """

        def newton_step(sigma):
            curve = _offset_curve(self.track, sigma, r_front)
            D = curve.c - R
            return sigma - (_dot(D, D) - w**2) / (2 * _dot(D, curve.c1))

        with torch.no_grad():
            sigma = (s + w).detach()
            for _ in range(self.newton_iterations):
                sigma = newton_step(sigma)
        return newton_step(sigma.detach())

    def start_position(self) -> torch.Tensor:
        """Rear wheel contact location with the front of the car at the start pin.

        The front of the car touches the start line: the line through the track at
        $s_\\mathrm{start}$ along the track normal.
        """
        track = self.track
        theta = track.angle(track.s_start)
        tangent = _vec(torch.cos(theta), torch.sin(theta))
        pin = _vec(*track.position(track.s_start))

        def newton_step(s):
            front = self.evaluate(s).front
            return s - _dot(front.value - pin, tangent) / _dot(front.ds, tangent)

        s = track.s_start - self.car.front_offset
        with torch.no_grad():
            s = s.detach()
            for _ in range(self.newton_iterations):
                s = newton_step(s)
        return newton_step(s.detach())
