# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Forward model
#
# This example walks through the pieces of a pwdsim forward simulation: defining the
# track, defining the car, and simulating a run down the track.  So far only the
# track is implemented.

# %%
import matplotlib.pyplot as plt

import pwdsim
from pwdsim.plotting import plot_track
from pwdsim.units import DEGREE, FOOT, INCH

# %% [markdown]
# ## The track
#
# pwdsim describes a track by its angle from the horizontal, $\theta(s)$, as a
# function of arc length $s$ along the path of the wheel contacts.  The curvature is
# $\kappa = \theta'$, and the track profile comes from integrating
# $x' = \cos\theta$ and $y' = \sin\theta$.
#
# Our standard track approximates a 42 ft
# [BestTrack](https://www.besttrack.com/) aluminum track, one of the most common
# commercial tracks.  From the manufacturer's
# [FAQ](https://www.besttrack.com/faq.htm) and
# [specifications](https://www.besttrack.com/track_specs.html):
#
# - the curve section is 41 in long with a smooth 48 in radius,
# - after the curve come four 84 in straight sections and a 40 in stop section,
# - the start pin is about 37 ft 9 in from the end of the track,
# - the racing distance from the start pin to the timer is 442 in, and
# - the starting gate is approximately 4 ft from the floor.
#
# Treating the whole curve section as a circular arc gives a ramp angle of
# $41/48$ rad, about 49 degrees.  Short easement curves join the arc to the
# straight sections so that the curvature is continuous.

# %%
track = pwdsim.besttrack(42)

print(f"Track length:        {track.length / FOOT:.1f} ft")
print(f"Start pin:           {track.s_start / INCH:.1f} in from the top")
print(f"Racing distance:     {(track.s_finish - track.s_start) / INCH:.1f} in")
print(f"Ramp angle:          {-track.angle(0.0) / DEGREE:.1f} deg")
print(f"Height at start pin: {track.height(track.s_start) / INCH:.1f} in")

# %% [markdown]
# The start pin sits about 44 in above the flat part of the track.  That matches the
# quoted 4 ft gate height if the flat sections stand a few inches off the floor.
#
# The profile below exaggerates the vertical scale.  The angle and curvature
# along the track show the straight ramp, the curve, and the flat run.

# %%
fig = plot_track(track, unit="ft")
plt.show()

# %% [markdown]
# Zooming in on the curve, at true scale, shows the easements where the curvature
# ramps up to $1/R$ over one inch of track at each end of the circular arc.

# %%
s_curve = track.knots[[1, 4]]
x_curve, y_curve = track.position(s_curve)
s_curve, x_curve, y_curve = (v.numpy() for v in (s_curve, x_curve, y_curve))
margin = 6 * INCH

fig = plot_track(track, unit="in", equal_aspect=True)
ax_profile, ax_angle = fig.axes[:2]
ax_profile.set_xlim((x_curve[0] - margin) / INCH, (x_curve[1] + margin) / INCH)
ax_profile.set_ylim((y_curve[1] - margin) / INCH, (y_curve[0] + margin) / INCH)
ax_angle.set_xlim((s_curve[0] - margin) / INCH, (s_curve[1] + margin) / INCH)
plt.show()
