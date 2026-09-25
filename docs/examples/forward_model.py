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
import warnings

import matplotlib.pyplot as plt
import torch

import pwdsim
from pwdsim.plotting import plot_car, plot_energy, plot_run, plot_runs, plot_track
from pwdsim.units import DEGREE, FOOT, GRAM, INCH, OUNCE

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
# straight sections so that the curvature changes smoothly.

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

# %% [markdown]
# ## The car
#
# The simulator needs a handful of essential car properties: the mass and its
# distribution, the axle and wheel geometry, and the friction and drag coefficients.
# `SimpleCar` takes these values directly and stores each one as a trainable torch
# parameter.  The *Cars* page of the docs describes the car model in detail.
#
# The values below describe a legal, reasonably well-built car made from the
# standard BSA kit.  They are estimates, mostly from the
# [Wikibooks derby physics page](https://en.wikibooks.org/wiki/How_To_Build_a_Pinewood_Derby_Car/Physics):
#
# - **Geometry:** the kit's axle slots are about $4\tfrac{3}{8}$ in apart, with
#   the rear slot about $\tfrac{7}{8}$ in from the back of the 7 in block.
# - **Mass:** the 5 oz limit, with the center of gravity 1 in ahead of the rear
#   axle, a common target for fast cars.  The body's pitching inertia is estimated
#   as a uniform 7 in by 1.25 in block, $I_b \approx M (l^2 + h^2) / 12$.
# - **Wheels:** 95.0 mm around, so a radius of 0.595 in, weighing 2.6 g each with
#   $I \approx 0.58\, m r^2$.  The axles are 0.087 in in diameter.
# - **Friction and drag:** a lubricated axle friction coefficient of 0.1, a rolling
#   resistance coefficient of 0.002, a drag coefficient of 0.4, and the
#   $0.0014\ \mathrm{m}^2$ frontal area of an uncut block.

# %%
mass = 5 * OUNCE
wheel_radius = 0.595 * INCH
wheel_inertia = 0.58 * (2.6 * GRAM) * wheel_radius**2
body_inertia = mass * ((7 * INCH) ** 2 + (1.25 * INCH) ** 2) / 12

car_args = dict(
    cg=(1.0 * INCH, 0.4 * INCH),
    wheelbase=4.375 * INCH,
    front_offset=(7 - 0.875) * INCH,
    mass=mass,
    body_inertia=body_inertia,
    rear_wheel_inertia=wheel_inertia,
    front_wheel_inertia=wheel_inertia,
    rear_wheel_radius=wheel_radius,
    front_wheel_radius=wheel_radius,
    rear_axle_radius=0.087 / 2 * INCH,
    front_axle_radius=0.087 / 2 * INCH,
    rear_axle_friction=0.1,
    front_axle_friction=0.1,
    frontal_area=0.0014,
    drag_coefficient=0.4,
    rolling_friction=0.002,
)
car = pwdsim.SimpleCar(**car_args)

for name, value in car.summary().items():
    values = value if isinstance(value, list) else [value]
    print(f"{name:20s}", *(f"{v:.4g}" for v in values))

# %% [markdown]
# All values are in SI units.  Every property except the wheel counts is a
# trainable parameter, ready for a torch optimizer:

# %%
for name, parameter in car.named_parameters():
    print(f"{name:21s} requires_grad={parameter.requires_grad}")

# %% [markdown]
# ## Simulating a run
#
# A `Simulation` puts the car on the track, with the front of the car against the
# start pin, and integrates its equation of motion in time.  The equation of motion
# is assembled from modular physics terms, and by default a simulation uses all of
# them:
#
# | Term | Physics |
# | --- | --- |
# | `gravity` | potential energy of the car, $M g\, y_g$ |
# | `translation` | kinetic energy of the car moving with its center of gravity |
# | `body_rotation` | kinetic energy of the car body pitching through the curve |
# | `wheel_spin` | kinetic energy of the spinning wheels, rolling without slip |
# | `drag` | aerodynamic drag, $\tfrac{1}{2} \rho C_d A v^2$ |
# | `axle_friction` | friction between the wheels and axles, a torque $\mu N a$ |
# | `rolling_friction` | rolling resistance of the wheels, a force $c_r N$ |
#
# The simulation uses backward Euler time steps of 0.1 ms.

# %%
sim = pwdsim.Simulation(track, car)
print("Physics terms:", list(sim.physics))

run = sim()
print(f"Finish time:  {run.finish_time.item():.4f} s")
print(f"Finish speed: {run.finish_speed.item():.3f} m/s")
print(f"Max speed:    {run.max_speed.item():.3f} m/s")
print(f"Min normal forces (rear, front): {run.min_normal_force.detach().numpy()} N")

# %% [markdown]
# The car speeds up down the ramp and through the curve, reaching its top speed as
# it enters the flat run, where friction and drag slowly bleed off speed.  The
# normal forces jump up in the curve, where the track has to turn the car's
# momentum upwards.  If either normal force went negative, the simulation would warn
# that a wheel lifted off the track.

# %%
fig = plot_run(run, unit="ft")
plt.show()

# %% [markdown]
# Here's the car at true scale as it goes through the curve.  The car pitches up
# as it follows the track, and the wheels ride one radius off the track surface.

# %%
fig = plot_car(sim, torch.tensor([75, 90, 105, 120, 135]) * INCH)
plt.show()

# %% [markdown]
# ## Simple vs. full physics
#
# How much does each piece of physics matter?  The simple physics has only gravity
# and the translational kinetic energy: a car rolling without friction or drag, with
# no rotational inertia.  Here is the same car with the simple and the full physics.

# %%
simple_run = pwdsim.Simulation(track, car, physics=pwdsim.simple_physics())()

for label, r in (("simple", simple_run), ("full", run)):
    print(
        f"{label:>6}: finish {r.finish_time.item():.4f} s, "
        f"finish speed {r.finish_speed.item():.3f} m/s, "
        f"max speed {r.max_speed.item():.3f} m/s"
    )

fig = plot_runs([simple_run, run], labels=["simple", "full"], unit="ft")
plt.show()

# %% [markdown]
# With the full physics, part of the car's potential energy goes into spinning the
# wheels and pitching the body, and drag and friction steadily dissipate energy.
# The energy budget shows where it goes; the total of the mechanical and dissipated
# energy stays constant.

# %%
fig = plot_energy(run)
plt.show()

# %% [markdown]
# To see what each term costs, switch it off and race again.  Each row is the time
# the car would save without that piece of physics.  The
# [which physics matters?](../physics_study/) example looks at this in more detail.

# %%
with torch.no_grad():
    full_time = run.finish_time.item()
    for name in list(sim.physics)[2:]:
        sim.disable(name)
        saved = full_time - sim().finish_time.item()
        sim.enable(name)
        print(f"{name:>16}: {saved * 1000:5.1f} ms")

# %% [markdown]
# ## Sensitivities
#
# The finish time is differentiable with respect to every car parameter.  The
# adjoint method provides the gradients at about the cost of one more simulation.

# %%
run.finish_time.backward()
dt_dcg = car.cg_.grad * INCH * 1000
print(f"Moving the CG forward 1 in:     {dt_dcg[0].item():+.1f} ms")
print(f"Moving the CG up 1 in:          {dt_dcg[1].item():+.1f} ms")
print(f"Adding 1 oz:                    {car.mass_.grad.item() * OUNCE * 1000:+.1f} ms")
print(
    "Axle friction coefficient +0.01: "
    f"{(car.rear_axle_friction_.grad + car.front_axle_friction_.grad).item() * 10:+.1f}"
    " ms"
)

# %% [markdown]
# What matters most is how far the center of gravity drops between the start and
# the flat run.  Moving the center of gravity back raises it on the ramp, where the
# car is tilted, but not on the flat, so it drops further and the car is faster.
# Raising the center of gravity does the opposite: on the ramp a point 1 in above
# the axles is only $\cos 49^\circ \approx 0.66$ in higher, while on the flat it is
# the full inch higher, so the drop gets smaller.
#
# With the simple physics the mass doesn't matter at all, since every force is
# proportional to it.  With the full physics a heavier car is faster: friction still
# scales with the weight, but drag and the rotational inertia of the wheels don't,
# so they slow a heavy car less.

# %% [markdown]
# ## Comparing designs
#
# Car properties can have a batch dimension to simulate several designs in one
# run.  Here we sweep the center of gravity from right over the rear axle to 2 in
# ahead of it, and look at both the finish time and the smallest normal force on
# the front wheels.

# %%
cg_along = torch.linspace(0.0, 2.0, 9) * INCH
cg_sweep = torch.stack([cg_along, torch.full_like(cg_along, 0.4 * INCH)], dim=-1)
cars = pwdsim.SimpleCar(**(car_args | {"cg": cg_sweep}))

with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    sweep = pwdsim.Simulation(track, cars)()
    finish_times = sweep.finish_time
    min_front_force = sweep.min_normal_force[:, 1]
for warning in caught:
    print(f"{warning.category.__name__}: {warning.message}")

lifted = min_front_force < 0
fig, (ax_time, ax_force) = plt.subplots(
    2, 1, figsize=(8, 6), sharex=True, constrained_layout=True
)
ax_time.plot(cg_along / INCH, finish_times * 1000, "o-")
ax_time.plot(cg_along[lifted] / INCH, finish_times[lifted] * 1000, "rx", ms=10)
ax_time.set_ylabel("finish time (ms)")
ax_force.plot(cg_along / INCH, min_front_force, "o-")
ax_force.axhline(0, color="k", lw=0.8)
ax_force.set_xlabel("center of gravity ahead of the rear axle (in)")
ax_force.set_ylabel("min front normal force (N)")
plt.show()

# %% [markdown]
# Moving the center of gravity back keeps paying off, but only up to a point.  As
# the car comes out of the curve it stops pitching up, and stopping the rotation of
# the car body takes a torque that unloads the front wheels.  With the center of
# gravity much closer than about 3/4 in to the rear axle, the front normal force
# goes negative (red crosses): the front wheels would lift off the track, which the
# model doesn't allow.  The simulation warns about this, and reports the minimum
# normal forces so an optimizer can avoid these designs.
#
# How far forward the limit sits depends on the pitching moment of inertia of the
# body, $I_b$, which here is a rough estimate.
