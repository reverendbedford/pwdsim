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
# # Optimizing a car
#
# The [which physics matters?](../physics_study/) example showed that the position
# of the center of gravity and the mass are two of the biggest levers on a car's
# speed.  Here we let pwdsim tune them.
#
# Running an optimization takes three things:
#
# 1. the system: a `Simulation` of the car on the track, with its physics;
# 2. the car parameters to tune, with bounds on each;
# 3. any constraints, like the race rules.
#
# The optimizer adjusts the chosen parameters to minimize the finish time, holding
# everything else fixed.  It also keeps the wheels on the track by default.

# %%
import warnings

import matplotlib.pyplot as plt
import torch

import pwdsim
from pwdsim.plotting import plot_car, plot_runs
from pwdsim.units import GRAM, INCH, OUNCE

# %% [markdown]
# ## The starting car
#
# The stock car from the other examples, but only 4 oz, so the optimizer has
# somewhere to go with the mass.  (The pitching inertia of the body is left at its
# 5 oz value, and stays fixed as the mass changes.)
#
# The simulation uses a time step of 0.2 ms, where the gradients are accurate to
# about 1%, to keep the optimization quick.

# %%
track = pwdsim.besttrack(42)

wheel_radius = 0.595 * INCH
wheel_inertia = 0.58 * (2.6 * GRAM) * wheel_radius**2
car_args = dict(
    cg=(1.0 * INCH, 0.4 * INCH),
    wheelbase=4.375 * INCH,
    front_offset=(7 - 0.875) * INCH,
    mass=4 * OUNCE,
    body_inertia=5 * OUNCE * ((7 * INCH) ** 2 + (1.25 * INCH) ** 2) / 12,
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


def simulation(car):
    return pwdsim.Simulation(track, car, dt=2e-4, duration=3.5)


car = pwdsim.SimpleCar(**car_args)
start = simulation(car)()
print(f"Starting finish time: {start.finish_time.item():.4f} s")

# %% [markdown]
# ## Setting up the optimization
#
# **Variables.**  We tune the center of gravity (both its position along the car
# and its height) and the mass.  Each variable gets `(lower, upper)` bounds in SI
# units, one value per component for the center of gravity.  Every other car
# parameter stays fixed.
#
# The bounds here are deliberately loose: the center of gravity can go from 1 in
# behind the rear axle to 1 in past the front axle, and the mass from 3 to 6 oz.
# The race rules come in as constraints instead.
#
# **Constraints.**  `pwdsim.constraints.bsa_rules()` is a standard set with the
# common Scout rules that apply to these parameters: the center of gravity between
# the axles, and a mass of at most 5 oz.
#
# **No lift-off.**  The optimizer also keeps the normal forces on both axles
# positive all the way to the finish, so the wheels stay on the track.  This is on
# by default.

# %%
variables = {
    # (along, up): from 1 in behind the rear axle to 1 in past the front, and
    # from 0.2 in to 1 in above the axles
    "cg": ((-1.0 * INCH, 0.2 * INCH), (5.375 * INCH, 1.0 * INCH)),
    "mass": (3 * OUNCE, 6 * OUNCE),
}
constraints = pwdsim.constraints.bsa_rules()
units = {"cg_": (INCH, "in"), "mass_": (OUNCE, "oz")}

result = pwdsim.optimize(simulation(car), variables, constraints)
print(result.summary(units))

# %% [markdown]
# The objective is the finish time in milliseconds.  The optimizer uses SciPy's
# `trust-constr` method, which builds up an approximation to the curvature of the
# finish time from its gradients, which come from the adjoint method.  Each
# iteration costs one simulation.

# %%
history = result.history
iterations = [h["iteration"] for h in history]
fig, (ax_time, ax_cg, ax_mass) = plt.subplots(
    3, 1, figsize=(8, 7), sharex=True, constrained_layout=True
)
ax_time.plot(iterations, [h["objective"] for h in history], "o-")
ax_time.set_ylabel("finish time (ms)")
cg = torch.stack([h["variables"]["cg_"] for h in history]) / INCH
ax_cg.plot(iterations, cg[:, 0], "o-", label="along")
ax_cg.plot(iterations, cg[:, 1], "o-", label="up")
ax_cg.set_ylabel("center of gravity (in)")
ax_cg.legend()
mass_history = [h["variables"]["mass_"].item() / OUNCE for h in history]
ax_mass.plot(iterations, mass_history, "o-")
ax_mass.set_ylabel("mass (oz)")
ax_mass.set_xlabel("iteration")
plt.show()

# %% [markdown]
# ## What limits the design
#
# Each tuned variable ends up against a limit, or at a balance between competing
# effects:
#
# - **The mass** goes straight to the 5 oz limit of the `max_mass` constraint.
#   With the full physics a heavier car is always faster, as we already knew.
# - **The center of gravity moves back**, but not all the way to the rear axle,
#   where `cg_between_axles` would stop it.  The lift-off constraint stops it first:
#   any further back and the front wheels would leave the track as the car comes
#   out of the curve.
# - **Its height** goes down to its lower bound: a lower center of gravity drops
#   further between the ramp and the flat run.
#
# The constraint values in the summary show which are active: the lift-off
# constraint sits at zero, and the mass at its limit.  The distances from the
# center of gravity to the axles are both positive, so `cg_between_axles` isn't
# active.
#
# The optimizer changes the car in place, and keeps a copy of the starting car in
# `result.initial_car`.  A run's results are computed from the car's current
# parameters, so to compare with the starting car we simulate that copy again.

# %%
start = simulation(result.initial_car)()
optimized = simulation(car)()
fig = plot_runs(
    [start, optimized],
    labels=["starting car", "optimized car"],
    unit="ft",
)
plt.show()

# %% [markdown]
# The optimized car is faster everywhere, and its front normal force (dashed) just
# touches zero as the car comes out of the curve.  Here it is at true scale through
# the curve:

# %%
fig = plot_car(simulation(car), torch.tensor([75, 95, 115, 135]) * INCH)
plt.show()

# %% [markdown]
# ## Why the lift-off constraint matters
#
# Without it, the optimizer pushes the center of gravity right back to the rear
# axle.  The simulation of that car warns that its front wheels leave the track,
# which the model can't represent, so its finish time can't be trusted.

# %%
reckless = pwdsim.SimpleCar(**car_args)
reckless_result = pwdsim.optimize(
    simulation(reckless), variables, constraints, lift_off_tolerance=None
)
print(reckless_result.summary(units))

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    run = simulation(reckless)()
for warning in caught:
    print(f"{warning.category.__name__}: {warning.message}")
print(f"Min normal forces (rear, front): {run.min_normal_force.detach().numpy()} N")

# %% [markdown]
# ## Robust designs
#
# A design that just barely keeps its wheels on the track in the simulation might
# not in real life: the car, the track, and the model are all imperfect.  The
# `lift_off_tolerance` asks for a margin, the smallest normal force allowed on
# either axle.  A margin costs some speed, since the center of gravity has to stay
# further forward.

# %%
robust = pwdsim.SimpleCar(**car_args)
robust_result = pwdsim.optimize(
    simulation(robust), variables, constraints, lift_off_tolerance=0.1
)
print(robust_result.summary(units))
cost = robust_result.final_objective - result.final_objective
print(f"\nA 0.1 N margin costs {cost:.1f} ms")

# %% [markdown]
# ## Next steps
#
# The same call tunes any of the car's parameters: add them to `variables` with
# their bounds.  Custom constraints are functions of the car and its run; see the
# [optimization docs](../../optimization/).
