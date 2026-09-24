"""Track geometry.

A track is a curve in the plane parameterized by arc length $s \\in [0, L]$, tracing
the path of the wheel contact points.  Tracks provide the angle $\\theta(s)$ of the
track from the horizontal, the curvature $\\kappa(s) = \\theta'(s)$ and its
derivative $\\kappa'(s)$, and the position $(x(s), y(s))$, with $x(0) = 0$ and
$y(L) = 0$.

Outside of $[0, L]$ tracks continue as straight lines at their end angles, so that
geometry calculations near the ends of the track stay well defined.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from pwdsim.types import DTYPE, ArrayLike, Scalar
from pwdsim.units import INCH


def _as_tensor(value, device=None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=DTYPE, device=device)


class Track(ABC):
    """Base class for tracks.

    Args:
        length: total track length $L$.
        s_start: location of the start pin, where the fronts of the cars start.
        s_finish: location of the finish line, with
            $0 \\le s_\\mathrm{start} < s_\\mathrm{finish} \\le L$.
    """

    def __init__(self, length: Scalar, s_start: Scalar, s_finish: Scalar):
        self.length = _as_tensor(length)
        self.s_start = _as_tensor(s_start)
        self.s_finish = _as_tensor(s_finish)
        if not self.length > 0:
            raise ValueError("Track length must be positive")
        if not 0 <= self.s_start < self.s_finish <= self.length:
            raise ValueError("Require 0 <= s_start < s_finish <= length")

    @abstractmethod
    def angle(self, s: float | torch.Tensor) -> torch.Tensor:
        """Angle of the track from the horizontal, $\\theta(s)$."""

    @abstractmethod
    def curvature(self, s: float | torch.Tensor) -> torch.Tensor:
        """Curvature of the track, $\\kappa(s) = \\theta'(s)$."""

    @abstractmethod
    def curvature_derivative(self, s: float | torch.Tensor) -> torch.Tensor:
        """Derivative of the curvature with respect to arc length, $\\kappa'(s)$."""

    @abstractmethod
    def curvature_second_derivative(self, s: float | torch.Tensor) -> torch.Tensor:
        """Second derivative of the curvature with respect to arc length,
        $\\kappa''(s)$."""

    @abstractmethod
    def position(self, s: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Position $(x(s), y(s))$ of the track."""

    def height(self, s: float | torch.Tensor) -> torch.Tensor:
        """Height of the track, $y(s)$."""
        return self.position(s)[1]


class SplineTrack(Track):
    """Track whose angle $\\theta(s)$ is a piecewise cubic Hermite spline.

    The spline is defined by the angle and curvature at each knot, making $\\theta$
    and $\\kappa$ continuous along the track.  Specifying the curvature directly
    makes it easy to build exact straight sections ($\\kappa = 0$) and circular arcs
    (constant $\\kappa$).  Use [`interpolate`][pwdsim.track.SplineTrack.interpolate]
    to instead build a $C^2$ spline from the knot angles alone.

    The position is integrated from the angle with Gauss-Legendre quadrature.

    Args:
        knots: increasing arc length locations of the knots, starting at 0.  The
            last knot is the track length.
        angles: track angle at each knot.
        curvatures: track curvature at each knot.
        s_start: location of the start pin.
        s_finish: location of the finish line.
        n_quad: number of Gauss-Legendre points used per spline segment to
            integrate the position.
    """

    def __init__(
        self,
        knots: ArrayLike,
        angles: ArrayLike,
        curvatures: ArrayLike,
        s_start: Scalar,
        s_finish: Scalar,
        n_quad: int = 16,
    ):
        self.knots = _as_tensor(knots)
        self.angles = _as_tensor(angles, self.knots.device)
        self.curvatures = _as_tensor(curvatures, self.knots.device)

        if self.knots.ndim != 1 or self.knots.shape[0] < 2:
            raise ValueError("Need a 1D array of at least two knots")
        if self.angles.shape != self.knots.shape:
            raise ValueError("Need one angle per knot")
        if self.curvatures.shape != self.knots.shape:
            raise ValueError("Need one curvature per knot")
        if self.knots[0] != 0:
            raise ValueError("The first knot must be at s = 0")
        if not torch.all(torch.diff(self.knots) > 0):
            raise ValueError("Knots must be strictly increasing")

        super().__init__(self.knots[-1], s_start, s_finish)

        xi, w = np.polynomial.legendre.leggauss(n_quad)
        self._quad_points = _as_tensor((xi + 1) / 2, self.knots.device)
        self._quad_weights = _as_tensor(w / 2, self.knots.device)

        # Position at each knot, shifted so that y(L) = 0
        segments = torch.arange(len(self.knots) - 1, device=self.knots.device)
        dx, dy = self._integrate(segments, self.knots.new_ones(segments.shape))
        self._knot_x = torch.cat([dx.new_zeros(1), torch.cumsum(dx, 0)])
        self._knot_y = torch.cat([dy.new_zeros(1), torch.cumsum(dy, 0)])
        self._knot_y = self._knot_y - self._knot_y[-1]

    @classmethod
    def interpolate(
        cls,
        knots: ArrayLike,
        angles: ArrayLike,
        s_start: Scalar,
        s_finish: Scalar,
        end_curvatures: tuple[Scalar, Scalar] = (0.0, 0.0),
        n_quad: int = 16,
    ) -> "SplineTrack":
        """Build a $C^2$ cubic spline through the given knot angles.

        The knot curvatures are chosen so that $\\kappa'$ is also continuous, with
        the curvatures at the two ends of the track fixed (clamped end conditions).

        Args:
            knots: increasing arc length locations of the knots, starting at 0.
            angles: track angle at each knot.
            s_start: location of the start pin.
            s_finish: location of the finish line.
            end_curvatures: curvature at the start and end of the track.
            n_quad: number of Gauss-Legendre points per spline segment.
        """
        knots = _as_tensor(knots)
        angles = _as_tensor(angles, knots.device)
        if knots.ndim != 1 or knots.shape[0] < 2 or angles.shape != knots.shape:
            raise ValueError("Need matching 1D arrays of at least two knots and angles")

        n = knots.shape[0]
        h = torch.diff(knots)
        A = knots.new_zeros((n, n))
        A[0, 0] = 1
        A[-1, -1] = 1
        rhs = [_as_tensor(end_curvatures[0], knots.device)]
        for i in range(1, n - 1):
            A[i, i - 1] = h[i]
            A[i, i] = 2 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i - 1]
            rhs.append(
                3
                * (
                    h[i] * (angles[i] - angles[i - 1]) / h[i - 1]
                    + h[i - 1] * (angles[i + 1] - angles[i]) / h[i]
                )
            )
        rhs.append(_as_tensor(end_curvatures[1], knots.device))
        curvatures = torch.linalg.solve(A, torch.stack(rhs))

        return cls(knots, angles, curvatures, s_start, s_finish, n_quad=n_quad)

    def angle(self, s: float | torch.Tensor) -> torch.Tensor:
        s_clamped = self._clamp(s)
        return self._hermite(*self._locate(s_clamped), derivative=0)

    def curvature(self, s: float | torch.Tensor) -> torch.Tensor:
        s_clamped = self._clamp(s)
        value = self._hermite(*self._locate(s_clamped), derivative=1)
        return torch.where(self._on_track(s), value, 0.0)

    def curvature_derivative(self, s: float | torch.Tensor) -> torch.Tensor:
        s_clamped = self._clamp(s)
        value = self._hermite(*self._locate(s_clamped), derivative=2)
        return torch.where(self._on_track(s), value, 0.0)

    def curvature_second_derivative(self, s: float | torch.Tensor) -> torch.Tensor:
        s_clamped = self._clamp(s)
        value = self._hermite(*self._locate(s_clamped), derivative=3)
        return torch.where(self._on_track(s), value, 0.0)

    def position(self, s: float | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = _as_tensor(s, self.knots.device)
        s_clamped = self._clamp(s)
        i, t = self._locate(s_clamped)
        dx, dy = self._integrate(i, t)
        x = self._knot_x[i] + dx
        y = self._knot_y[i] + dy

        # Straight line extension off the ends of the track
        theta = self._hermite(i, t, derivative=0)
        overshoot = s - s_clamped
        return x + overshoot * torch.cos(theta), y + overshoot * torch.sin(theta)

    def _clamp(self, s):
        return torch.clamp(_as_tensor(s, self.knots.device), 0.0, self.length)

    def _on_track(self, s):
        s = _as_tensor(s, self.knots.device)
        return (s >= 0) & (s <= self.length)

    def _locate(self, s):
        """Segment index and local coordinate $t \\in [0, 1]$ of each point."""
        i = torch.searchsorted(self.knots, s.contiguous(), right=True) - 1
        i = torch.clamp(i, 0, len(self.knots) - 2)
        t = (s - self.knots[i]) / (self.knots[i + 1] - self.knots[i])
        return i, t

    def _hermite(self, i, t, derivative=0):
        """Evaluate the angle spline, or its derivatives with respect to $s$."""
        h = self.knots[i + 1] - self.knots[i]
        a0, a1 = self.angles[i], self.angles[i + 1]
        m0, m1 = h * self.curvatures[i], h * self.curvatures[i + 1]
        t2 = t * t
        if derivative == 0:
            return (
                (2 * t2 * t - 3 * t2 + 1) * a0
                + (t2 * t - 2 * t2 + t) * m0
                + (-2 * t2 * t + 3 * t2) * a1
                + (t2 * t - t2) * m1
            )
        if derivative == 1:
            return (
                (6 * t2 - 6 * t) * a0
                + (3 * t2 - 4 * t + 1) * m0
                + (-6 * t2 + 6 * t) * a1
                + (3 * t2 - 2 * t) * m1
            ) / h
        if derivative == 2:
            return (
                (12 * t - 6) * a0
                + (6 * t - 4) * m0
                + (-12 * t + 6) * a1
                + (6 * t - 2) * m1
            ) / h**2
        if derivative == 3:
            return (12 * a0 + 6 * m0 - 12 * a1 + 6 * m1) / h**3
        raise ValueError(f"Unsupported derivative order {derivative}")

    def _integrate(self, i, t):
        """Integrate $(\\cos\\theta, \\sin\\theta)$ from the start of segment `i` to
        local coordinate `t`."""
        h = self.knots[i + 1] - self.knots[i]
        theta = self._hermite(i[..., None], t[..., None] * self._quad_points, 0)
        scale = h * t
        dx = scale * torch.sum(self._quad_weights * torch.cos(theta), -1)
        dy = scale * torch.sum(self._quad_weights * torch.sin(theta), -1)
        return dx, dy


def ramp_track(
    length: float,
    ramp_length: float,
    ramp_angle: float,
    radius: float,
    s_start: float,
    s_finish: float,
    easement: float = 1 * INCH,
) -> SplineTrack:
    """Build the classic derby track: a straight ramp, a curve, and a flat run.

    The curve is a circular arc joined to the straight sections by short easement
    curves, over which the curvature ramps linearly between zero and $1/R$.  The
    easements keep the curvature continuous, as required by
    [`SplineTrack`][pwdsim.track.SplineTrack].

    Args:
        length: total track length.
        ramp_length: length of the straight ramp, from the top of the track to the
            start of the curve.
        ramp_angle: downhill angle of the ramp from the horizontal (positive).
        radius: radius of the circular arc in the curve.
        s_start: location of the start pin.
        s_finish: location of the finish line.
        easement: length of each easement curve.
    """
    if not easement > 0:
        raise ValueError("The easement length must be positive")
    arc_length = ramp_angle * radius - easement
    if not arc_length > 0:
        raise ValueError("The easements are too long for the curve")
    s1 = ramp_length
    s2 = s1 + easement
    s3 = s2 + arc_length
    s4 = s3 + easement
    if not s4 < length:
        raise ValueError("The track is too short for the ramp and curve")

    theta0 = -ramp_angle
    knots = [0.0, s1, s2, s3, s4, length]
    angles = [
        theta0,
        theta0,
        theta0 + easement / (2 * radius),
        theta0 + (easement / 2 + arc_length) / radius,
        0.0,
        0.0,
    ]
    curvatures = [0.0, 0.0, 1 / radius, 1 / radius, 0.0, 0.0]
    return SplineTrack(knots, angles, curvatures, s_start, s_finish)


_BESTTRACK_RACING_DISTANCE = {35: 358 * INCH, 42: 442 * INCH, 49: 526 * INCH}


def besttrack(length_ft: int = 42) -> SplineTrack:
    """Approximate geometry of a BestTrack aluminum track.

    Based on the manufacturer's published dimensions: a 48 in radius curve section
    41 in long, 84 in straight sections and a 40 in stop section after the curve,
    the start pin 37 ft 9 in from the end of a 42 ft track, and racing distances
    (start pin to timer) of 358, 442, and 526 in for the 35, 42, and 49 ft tracks.
    The ramp angle is inferred by treating the whole curve section as a circular
    arc, $41/48$ rad or about 49 degrees.

    Args:
        length_ft: nominal track length in feet: 35, 42, or 49.
    """
    if length_ft not in _BESTTRACK_RACING_DISTANCE:
        raise ValueError("BestTrack lengths are 35, 42, or 49 ft")
    radius = 48 * INCH
    curve_length = 41 * INCH
    n_straight = {35: 3, 42: 4, 49: 5}[length_ft]
    flat_length = n_straight * 84 * INCH + 40 * INCH
    length = length_ft * 12 * INCH
    ramp_length = length - flat_length - curve_length
    s_start = 42 * 12 * INCH - (37 * 12 + 9) * INCH
    return ramp_track(
        length=length,
        ramp_length=ramp_length,
        ramp_angle=curve_length / radius,
        radius=radius,
        s_start=s_start,
        s_finish=s_start + _BESTTRACK_RACING_DISTANCE[length_ft],
    )


__all__ = ["SplineTrack", "Track", "besttrack", "ramp_track"]
