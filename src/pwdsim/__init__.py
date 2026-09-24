"""pwdsim: a pytorch Pinewood Derby simulator and optimizer."""

from importlib.metadata import version

from pwdsim.car import Car, SimpleCar
from pwdsim.physics import Environment, Gravity, Translation, default_physics
from pwdsim.simulation import (
    DidNotFinishWarning,
    LiftOffWarning,
    Run,
    Simulation,
)
from pwdsim.track import SplineTrack, Track, besttrack, ramp_track

__version__ = version("pwdsim")

__all__ = [
    "Car",
    "DidNotFinishWarning",
    "Environment",
    "Gravity",
    "LiftOffWarning",
    "Run",
    "Simulation",
    "SimpleCar",
    "SplineTrack",
    "Track",
    "Translation",
    "besttrack",
    "default_physics",
    "ramp_track",
]
