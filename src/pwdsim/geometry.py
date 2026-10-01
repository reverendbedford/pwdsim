"""Car body geometry.

The car body is a side profile extruded through a thickness: the region between a
flat bottom and a B-spline height profile, described by a
[`Profile`][pwdsim.geometry.Profile].  Inside it, rectangular regions of other
materials, such as weights and voids, replace the wood.  A
[`Region`][pwdsim.geometry.Region] is a general rectangle, and a
[`WeightPocket`][pwdsim.geometry.WeightPocket] is a pocket drilled up from the
bottom with a weight in it.

Coordinates are in the car body frame: the origin at the rear axle center, $x$
along the car toward the front axle, and $y$ up.  All values are in SI units.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from pwdsim.types import DTYPE, Scalar

PINE = 420.0
"""Density of pine, the usual derby block wood, in kg/m³ (approximate)."""

BASSWOOD = 415.0
"""Density of basswood, in kg/m³ (approximate)."""

TUNGSTEN = 19300.0
"""Density of tungsten, in kg/m³.  Tungsten alloy weights are a little lighter."""

LEAD = 11340.0
"""Density of lead, in kg/m³."""

VOID = 0.0
"""Density of an empty region."""


def _tensor(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=DTYPE)


def _parameter(value) -> nn.Parameter:
    return nn.Parameter(_tensor(value).clone())


class BSpline:
    """A clamped, uniform B-spline basis on $[x_0, x_1]$.

    The curve $h(x) = \\sum_i c_i N_i(x)$ starts at $c_0$ and ends at $c_{n-1}$, and
    lies within the range of its control values $c_i$ (the convex hull property).

    Args:
        x0: start of the interval.
        x1: end of the interval.
        n: number of control values, at least `degree + 1`.
        degree: polynomial degree.
    """

    def __init__(self, x0: float, x1: float, n: int, degree: int = 3):
        if n < degree + 1:
            raise ValueError(f"A degree {degree} B-spline needs {degree + 1} values")
        self.x0, self.x1, self.n, self.degree = float(x0), float(x1), n, degree
        interior = np.linspace(x0, x1, n - degree + 1)
        self.knots = _tensor(np.concatenate([[x0] * degree, interior, [x1] * degree]))
        self.breaks = _tensor(interior)

    def basis(self, x) -> torch.Tensor:
        """The basis functions $N_i(x)$, in a trailing dimension of size `n`.

        Points outside $[x_0, x_1]$ are clamped to the interval.  The basis is
        differentiable with respect to `x`.
        """
        x = torch.clamp(_tensor(x), self.x0, self.x1)[..., None]
        t = self.knots
        # Degree 0: indicator of the span, with the last span closed on the right
        left, right = t[:-1], t[1:]
        last = torch.arange(len(left)) == self.n - 1
        N = ((x >= left) & ((x < right) | (last & (x == right)))).to(DTYPE)
        for p in range(1, self.degree + 1):
            count = len(t) - p - 1
            ti, tip = t[:count], t[p : p + count]
            ti1, tip1 = t[1 : count + 1], t[p + 1 : p + 1 + count]
            a = torch.where(tip > ti, (x - ti) / torch.where(tip > ti, tip - ti, 1), 0)
            b = torch.where(
                tip1 > ti1, (tip1 - x) / torch.where(tip1 > ti1, tip1 - ti1, 1), 0
            )
            N = a * N[..., :count] + b * N[..., 1 : count + 1]
        return N

    def greville(self) -> torch.Tensor:
        """The Greville abscissae: where each control value has the most influence."""
        t = self.knots
        return torch.stack(
            [t[i + 1 : i + self.degree + 1].mean() for i in range(self.n)]
        )

    def quadrature(self, points: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        """Gauss-Legendre points and weights on each span between knots, exact for
        polynomials of degree up to `2 * points - 1` on each span."""
        xi, w = np.polynomial.legendre.leggauss(points)
        lo, hi = self.breaks[:-1, None], self.breaks[1:, None]
        x = (lo + hi) / 2 + (hi - lo) / 2 * _tensor(xi)
        weights = (hi - lo) / 2 * _tensor(w)
        return x.reshape(-1), weights.reshape(-1)


@dataclass
class Moments:
    """Integrals over an area: $\\int dA$, $\\int x\\, dA$, $\\int y\\, dA$, and
    $\\int (x^2 + y^2)\\, dA$."""

    area: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    polar: torch.Tensor


class Profile(nn.Module):
    """The side profile of the car body: the region between a flat bottom at
    $y = y_b$ and the top $y = y_b + h(x)$, for $x$ from the rear to the front of the
    block.

    The height $h(x)$ is a cubic B-spline with control heights `heights`, a
    parameter.  Because the spline lies within the range of its control heights,
    keeping the control heights between 0 and the block height keeps the whole body
    inside the block; [`bounds`][pwdsim.geometry.Profile.bounds] gives those bounds
    for [`optimize`][pwdsim.optimization.optimize].

    Args:
        x_rear: position of the rear of the block, relative to the rear axle
            (negative: behind it).
        x_front: position of the front of the block, relative to the rear axle.
        bottom: height of the bottom of the block relative to the axles, $y_b$
            (negative: below them).
        heights: control heights of the profile above the bottom.
        max_height: height of the uncut block.  Defaults to the largest control
            height.
    """

    def __init__(
        self,
        x_rear: float,
        x_front: float,
        bottom: float,
        heights: Sequence[float] | torch.Tensor,
        max_height: float | None = None,
    ):
        super().__init__()
        if not x_front > x_rear:
            raise ValueError("The front of the block must be ahead of the rear")
        self.heights = _parameter(heights)
        if self.heights.ndim != 1:
            raise ValueError("The control heights must be a 1D sequence")
        self.x_rear, self.x_front, self.bottom = float(x_rear), float(x_front), bottom
        self.max_height = (
            float(self.heights.detach().max()) if max_height is None else max_height
        )
        self.spline = BSpline(x_rear, x_front, len(self.heights))
        points, self._weights = self.spline.quadrature()
        self._points = points
        self._basis = self.spline.basis(points)

    @classmethod
    def block(
        cls,
        x_rear: float,
        x_front: float,
        bottom: float,
        height: float,
        n: int = 8,
    ) -> "Profile":
        """An uncut rectangular block with `n` control heights."""
        return cls(
            x_rear,
            x_front,
            bottom,
            torch.full((n,), float(height), dtype=DTYPE),
            height,
        )

    @property
    def length(self) -> float:
        return self.x_front - self.x_rear

    def height(self, x) -> torch.Tensor:
        """The height of the profile above the bottom, $h(x)$."""
        return self.spline.basis(x) @ self.heights

    def top(self, x) -> torch.Tensor:
        """The $y$ coordinate of the top of the profile, $y_b + h(x)$."""
        return self.bottom + self.height(x)

    def moments(self) -> Moments:
        """Area moments of the profile, exact for the cubic spline."""
        x, w = self._points, self._weights
        h = self._basis @ self.heights
        yb, top = self.bottom, self.bottom + h
        return Moments(
            area=torch.sum(w * h),
            x=torch.sum(w * x * h),
            y=torch.sum(w * (top**2 - yb**2) / 2),
            polar=torch.sum(w * (x**2 * h + (top**3 - yb**3) / 3)),
        )

    def bounds(
        self,
        fixed: Sequence[tuple[float, float]] = (),
        minimum: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bounds on the control heights for optimization: between `minimum` and
        the block height.

        Args:
            fixed: ranges of $x$, as `(start, end)` pairs, over which the profile is
                held fixed.  Control heights whose Greville abscissae (where each has
                the most influence) fall in a range get equal lower and upper bounds,
                which fixes them at their current values.
            minimum: the smallest allowed control height.
        """
        current = self.heights.detach()
        lower = torch.full_like(current, minimum)
        upper = torch.full_like(current, self.max_height)
        g = self.spline.greville()
        for start, end in fixed:
            mask = (g >= start) & (g <= end)
            lower[mask] = current[mask]
            upper[mask] = current[mask]
        return lower, upper


@dataclass
class Rectangle:
    """A rectangle of material inside the body.

    Attributes:
        x0: rear edge.
        x1: front edge.
        y0: bottom edge.
        y1: top edge.
        density: density of the material.
        depth: depth through the car, or `None` for the full body thickness.
    """

    x0: torch.Tensor
    x1: torch.Tensor
    y0: torch.Tensor
    y1: torch.Tensor
    density: float
    depth: torch.Tensor | None = None

    def moments(self) -> Moments:
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        area = dx * dy
        return Moments(
            area=area,
            x=area * (self.x0 + self.x1) / 2,
            y=area * (self.y0 + self.y1) / 2,
            polar=dy * (self.x1**3 - self.x0**3) / 3
            + dx * (self.y1**3 - self.y0**3) / 3,
        )


class Inclusion(nn.Module, ABC):
    """A region of the body made of other materials, described by rectangles."""

    @abstractmethod
    def rectangles(self, bottom: float) -> list[Rectangle]:
        """The rectangles of material, for a body with its bottom at `bottom`."""

    @property
    @abstractmethod
    def depth(self) -> torch.Tensor | None:
        """Depth through the car, or `None` for the full body thickness."""


class Region(Inclusion):
    """A rectangle of a single material, such as a weight or a void.

    Args:
        x0: rear edge.
        x1: front edge.
        y0: bottom edge.
        y1: top edge.
        density: density of the material, e.g. [`TUNGSTEN`][pwdsim.geometry.TUNGSTEN]
            or [`VOID`][pwdsim.geometry.VOID].
        depth: depth through the car.  `None`, the default, is the full body
            thickness.
    """

    def __init__(
        self,
        x0: Scalar,
        x1: Scalar,
        y0: Scalar,
        y1: Scalar,
        density: float,
        depth: Scalar | None = None,
    ):
        super().__init__()
        self.x0, self.x1 = _parameter(x0), _parameter(x1)
        self.y0, self.y1 = _parameter(y0), _parameter(y1)
        self.density = float(density)
        self.depth_ = None if depth is None else _parameter(depth)

    @property
    def depth(self):
        return self.depth_

    def rectangles(self, bottom):
        return [
            Rectangle(self.x0, self.x1, self.y0, self.y1, self.density, self.depth_)
        ]


class WeightPocket(Inclusion):
    """A pocket drilled up into the car from its bottom, holding a weight at its top.

    The pocket is centered at `position` along the car and is `width` long.  It is
    empty from the bottom of the car up to `height - length`, and filled with the
    weight from there up to `height`.

    Args:
        position: position of the center of the pocket along the car.
        width: length of the pocket along the car.
        height: total height of the pocket above the bottom of the car.
        length: length of the weight, from the top of the pocket down.
        material: density of the weight, e.g.
            [`TUNGSTEN`][pwdsim.geometry.TUNGSTEN].
        depth: depth of the pocket through the car.  `None`, the default, is the
            full body thickness.
    """

    def __init__(
        self,
        position: Scalar,
        width: Scalar,
        height: Scalar,
        length: Scalar,
        material: float = TUNGSTEN,
        depth: Scalar | None = None,
    ):
        super().__init__()
        self.position, self.width = _parameter(position), _parameter(width)
        self.height, self.length = _parameter(height), _parameter(length)
        self.material = float(material)
        self.depth_ = None if depth is None else _parameter(depth)
        if not (0 < self.length.item() <= self.height.item()):
            raise ValueError("The weight must fit in the pocket: 0 < length <= height")
        if not self.width.item() > 0:
            raise ValueError("The pocket width must be positive")

    @property
    def depth(self):
        return self.depth_

    def rectangles(self, bottom):
        x0 = self.position - self.width / 2
        x1 = self.position + self.width / 2
        top = bottom + self.height
        split = top - self.length
        return [
            Rectangle(x0, x1, bottom + 0 * top, split, VOID, self.depth_),
            Rectangle(x0, x1, split, top, self.material, self.depth_),
        ]
