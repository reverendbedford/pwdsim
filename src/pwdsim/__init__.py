"""pwdsim: a pytorch Pinewood Derby simulator and optimizer."""

from importlib.metadata import version

from pwdsim import constraints
from pwdsim.car import Car, SimpleCar
from pwdsim.optimization import DesignProblem, OptimizationResult, optimize
from pwdsim.physics import (
    AxleFriction,
    BodyRotation,
    Drag,
    Environment,
    Gravity,
    RollingFriction,
    Translation,
    WheelSpin,
    default_physics,
    full_physics,
    simple_physics,
)
from pwdsim.simulation import (
    DidNotFinishWarning,
    LiftOffWarning,
    Run,
    Simulation,
)
from pwdsim.track import SplineTrack, Track, besttrack, ramp_track

__version__ = version("pwdsim")

__all__ = [
    "AxleFriction",
    "BodyRotation",
    "Car",
    "DesignProblem",
    "DidNotFinishWarning",
    "Drag",
    "Environment",
    "Gravity",
    "LiftOffWarning",
    "OptimizationResult",
    "RollingFriction",
    "Run",
    "Simulation",
    "SimpleCar",
    "SplineTrack",
    "Track",
    "Translation",
    "WheelSpin",
    "besttrack",
    "constraints",
    "default_physics",
    "full_physics",
    "optimize",
    "ramp_track",
    "simple_physics",
]
