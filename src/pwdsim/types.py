"""Common types and the floating point precision used throughout pwdsim."""

from collections.abc import Sequence

import numpy as np
import torch

DTYPE = torch.float64
"""Floating point type for all calculations."""

Scalar = float | torch.Tensor
"""A scalar value, as a python float or 0-d tensor."""

ArrayLike = Sequence[float] | np.ndarray | torch.Tensor
"""A 1D array of values."""
