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
# # Which physics matters?
#
# A pinewood derby car loses time to many things: spinning up its wheels, friction
# in its axles, air drag, and more.  Before optimizing a car it helps to know which
# of these actually matter, and which car parameters control them.
#
# pwdsim builds its equation of motion from modular physics terms, which can be
# switched on and off.  This example races the same car with different physics to
# measure what each term costs, follows where the car's energy goes, and uses the
# gradients of the finish time to rank the car parameters.

# %%
import matplotlib.pyplot as plt
import torch

import pwdsim
from pwdsim.units import GRAM, INCH, OUNCE

# %% [markdown]
# ## The car and track
#
# The same stock car on the 42 ft BestTrack as in the
# [forward model example](../forward_model/): a 5 oz car built from the BSA kit,
# with the center of gravity 1 in ahead of the rear axle.

# %%
track = pwdsim.besttrack(42)

mass = 5 * OUNCE
wheel_radius = 0.595 * INCH
wheel_inertia = 0.58 * (2.6 * GRAM) * wheel_radius**2
car_args = dict(
    cg=(1.0 * INCH, 0.4 * INCH),
    wheelbase=4.375 * INCH,
    front_offset=(7 - 0.875) * INCH,
    mass=mass,
    body_inertia=mass * ((7 * INCH) ** 2 + (1.25 * INCH) ** 2) / 12,
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


def race(car, physics=None):
    """Finish time of a car, in seconds, with the given physics terms."""
    with torch.no_grad():
        return pwdsim.Simulation(track, car, physics=physics)(adjoint=False).finish_time


# %% [markdown]
# ## Simple vs. full physics
#
# The *simple* physics has only gravity and the translational kinetic energy of the
# car: a car rolling without friction or drag, with no rotational inertia.  This is
# the fastest the car could possibly be.  The *full* physics adds the pitching of the
# car body, the spin of the wheels, drag, and axle and rolling friction.

# %%
simple_time = race(car, pwdsim.simple_physics()).item()
full_time = race(car).item()
print(f"Simple physics: {simple_time:.4f} s")
print(f"Full physics:   {full_time:.4f} s")
print(f"Difference:     {(full_time - simple_time) * 1000:.1f} ms")

# %% [markdown]
# Close races are decided by a few milliseconds, so the ~150 ms the real physics
# costs is enormous.  Where does it go?
#
# ## The cost of each term
#
# There are two natural ways to measure what a term costs:
#
# - **switch it off**: race with the full physics minus that term, and see how much
#   time the car saves;
# - **switch it on**: race with the simple physics plus that term, and see how much
#   time the car loses.
#
# If the terms were independent the two would agree.  They don't quite, because the
# terms interact, as the table shows.

# %%
names = [term.name for term in pwdsim.full_physics()][2:]


def physics_with(included):
    return [t for t in pwdsim.full_physics() if t.name in included]


cost_off, cost_on = {}, {}
for name in names:
    without = physics_with(set(names) - {name} | {"gravity", "translation"})
    cost_off[name] = (full_time - race(car, without).item()) * 1000
    with_only = physics_with({name, "gravity", "translation"})
    cost_on[name] = (race(car, with_only).item() - simple_time) * 1000

print(f"{'term':>16}  {'off (ms)':>9}  {'on (ms)':>8}")
for name in names:
    print(f"{name:>16}  {cost_off[name]:9.1f}  {cost_on[name]:8.1f}")
print(f"{'sum':>16}  {sum(cost_off.values()):9.1f}  {sum(cost_on.values()):8.1f}")
print(f"{'total':>16}  {(full_time - simple_time) * 1000:9.1f}")

# %%
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
y = torch.arange(len(names))
ax.barh(y - 0.2, [cost_off[n] for n in names], height=0.4, label="switched off")
ax.barh(y + 0.2, [cost_on[n] for n in names], height=0.4, label="switched on")
ax.set_yticks(y, names)
ax.invert_yaxis()
ax.set_xlabel("time cost (ms)")
ax.legend()
plt.show()

# %% [markdown]
# Three terms dominate, and the two measures agree on the ranking:
#
# - **Wheel spin.**  The car has to spin up its wheels as well as move itself.
#   Four stock wheels add about 4% to the car's effective mass, so it accelerates
#   down the ramp about 4% more slowly.
# - **Axle friction.**  The wheels rub on their axles for the whole run, and most of
#   the run is the long flat section, where friction is the main thing slowing the
#   car.
# - **Drag**, at about half the cost of each of the first two.
#
# Rolling friction is smaller, and the pitching of the car body through the curve
# costs almost nothing.
#
# The losses compound.  A car slowed by one loss spends longer on the track, which
# gives the others more time to act.  So each term costs a little more when it's
# switched off from the full physics than when it's added to the simple physics, and
# the costs measured by switching terms off add up to slightly more than the total
# difference.

# %% [markdown]
# ## Where the energy goes
#
# The car starts with potential energy, and by the finish line it has converted it
# into kinetic energy, which is what makes it fast, and into losses.  The energy
# breakdown shows each term's share.  The kinetic energy is split between the car
# moving (translation), its wheels spinning, and its body pitching.

# %%
sim = pwdsim.Simulation(track, car)
run = sim()
with torch.no_grad():
    terms = run.energy_by_term()
    finish = int(torch.searchsorted(run.times, run.finish_time))
    released = (terms["gravity"][0] - terms["gravity"][finish]).item()

order = ["translation", "wheel_spin", "body_rotation"] + names[2:]
print(f"Potential energy released by the finish: {released * 1000:.0f} mJ")
for name in order:
    share = terms[name][finish].item()
    print(f"{name:>16}: {share * 1000:6.1f} mJ ({share / released:5.1%})")

# %%
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
with torch.no_grad():
    ax.stackplot(
        run.times[: finish + 1],
        *(terms[name][: finish + 1] * 1000 for name in order),
        labels=order,
    )
ax.set_xlabel("time (s)")
ax.set_ylabel("energy (mJ)")
ax.set_title("Where the released potential energy goes, up to the finish")
ax.legend(loc="upper left")
plt.show()

# %% [markdown]
# More than 80% of the energy ends up as the car's speed, which is the goal.  The
# spinning wheels hold a few percent, which the car never gets back before the
# finish.  The losses to friction and drag are small on the short ramp, and grow
# steadily along the flat.  Axle friction is the largest loss.
#
# ## Which car parameters matter
#
# The physics terms map onto car parameters: wheel spin onto the wheel inertia and
# radius, axle friction onto the friction coefficient and axle radius, and so on.
# The gradient of the finish time with respect to every parameter comes from one
# adjoint calculation.  To compare parameters with different units, the chart below
# shows the effect of a 10% increase in each one.

# %%
run.finish_time.backward()


def effects(car):
    """Change in finish time, in ms, for a 10% increase in each parameter."""
    result = {}
    for name, parameter in car.named_parameters():
        change = 0.1 * parameter.detach() * parameter.grad * 1000
        if name == "cg_":
            result["cg (along)"] = change[0].item()
            result["cg (up)"] = change[1].item()
        else:
            result[name.rstrip("_")] = change.item()
    return result


full_effects = effects(car)

simple_car = pwdsim.SimpleCar(**car_args)
simple_run = pwdsim.Simulation(track, simple_car, physics=pwdsim.simple_physics())()
simple_run.finish_time.backward()
simple_effects = effects(simple_car)

ranked = sorted(full_effects, key=lambda n: abs(full_effects[n]))
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
y = torch.arange(len(ranked))
ax.barh(y + 0.2, [full_effects[n] for n in ranked], height=0.4, label="full physics")
ax.barh(
    y - 0.2, [simple_effects[n] for n in ranked], height=0.4, label="simple physics"
)
ax.set_yticks(y, ranked)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("change in finish time for a 10% increase (ms)")
ax.legend()
plt.show()

# %% [markdown]
# Negative bars make the car faster.  With the simple physics only the geometry
# matters: where the center of gravity is, and how long the car is.  The full
# physics brings in almost every parameter:
#
# - **The front offset**, the length of the car ahead of the rear axle, is the
#   biggest.  A longer car's nose starts at the same pin but crosses the finish line
#   sooner.  The race rules cap the car's length.
# - **The wheel radius** comes next.  Larger wheels spin more slowly for the same
#   speed, so they take less energy, and the axle friction has less leverage over
#   them.  Note that this holds the wheel inertia fixed, which a real larger wheel
#   wouldn't, and most races require the stock wheels anyway.
# - **Mass** helps: every loss except friction is independent of the mass, so a
#   heavier car loses less of its speed to them.  The rules cap the mass too.
# - **Axle friction** matters about three times as much at the rear axle as at the
#   front, because the rear axle carries most of the car's weight.  The friction
#   coefficient and the axle radius have identical effects, since the friction
#   torque is $\mu N a$.  These are the classic derby tricks: thin, polished,
#   lubricated axles.
# - **Drag and wheel inertia** follow, then rolling friction.
# - **The center of gravity** looks small here, but only because 10% of 1 in is
#   0.1 in.  Moving it by a whole inch, which is easy to do with the ballast weights,
#   is worth about 18 ms.  It is limited by the front wheels lifting off the track
#   when it gets too close to the rear axle (see the
#   [forward model example](../forward_model/)).
#
# Scaling each parameter by 10% is only one way to compare them.  What matters in
# practice is how far each one can actually be changed, which the race rules and the
# builder's skill decide.
#
# ## A discrete choice: three wheels
#
# A popular trick is to raise one front wheel so that it never touches the track.
# The number of wheels isn't a continuous parameter, so there's no gradient for it,
# but it's easy to race both cars.  In this model the car saves the energy of
# spinning one wheel: a quarter of the cost of wheel spin above.  Axle friction
# depends on the load on each axle, which doesn't change when fewer wheels share
# it.

# %%
three_wheels = pwdsim.SimpleCar(**car_args, n_front_wheels=1)
three_time = race(three_wheels).item()
print(f"Four wheels:  {full_time:.4f} s")
print(f"Three wheels: {three_time:.4f} s ({(three_time - full_time) * 1000:+.1f} ms)")

# %% [markdown]
# ## The physics changes the answer
#
# The choice of physics doesn't just shift the finish time; it can change which
# designs are fastest.  Mass is the clearest example.  With the simple physics every
# force is proportional to the mass, so the mass doesn't matter at all.  With the
# full physics, a heavier car is faster: here by about 30 ms per ounce for a light
# car, falling to about 17 ms per ounce at the 5 oz limit.  (The sweep scales the
# body's moment of inertia with its mass.)

# %%
masses = torch.linspace(3.0, 5.0, 5) * OUNCE
# The body's moment of inertia scales with its mass
inertias = masses * car_args["body_inertia"] / mass
cars = pwdsim.SimpleCar(**(car_args | {"mass": masses, "body_inertia": inertias}))
simple_sweep = race(cars, pwdsim.simple_physics())
full_sweep = race(cars)

fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
ax.plot(masses / OUNCE, simple_sweep * 1000, "o-", label="simple physics")
ax.plot(masses / OUNCE, full_sweep * 1000, "o-", label="full physics")
ax.set_xlabel("mass (oz)")
ax.set_ylabel("finish time (ms)")
ax.legend()
plt.show()

# %% [markdown]
# ## Takeaways
#
# For this car, the physics that matters most is the spin of the wheels and the
# friction in the axles, followed by drag.  The car parameters that matter most
# are:
#
# 1. the length of the car and its mass, both up to the limits in the race rules;
# 2. the position of the center of gravity, as far back as the front wheels allow;
# 3. the friction and radius of the axles, especially the rear one;
# 4. the size and inertia of the wheels, where the rules allow changing them.
#
# Several of these pull against each other or against the rules: moving the center
# of gravity back risks lifting the front wheels, and the length and mass are
# capped.  Tuning a car is an optimization problem with constraints, which is where
# pwdsim goes next.
