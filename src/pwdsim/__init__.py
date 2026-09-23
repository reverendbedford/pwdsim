"""pwdsim: a pytorch Pinewood Derby simulator and optimizer."""

from importlib.metadata import version

from pwdsim.track import SplineTrack, Track, besttrack, ramp_track

__version__ = version("pwdsim")

__all__ = ["SplineTrack", "Track", "besttrack", "ramp_track"]
