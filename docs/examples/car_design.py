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
# # Designing a car body
#
# The other examples describe a car by its properties: its mass, the position of its
# center of gravity, and so on.  A real builder doesn't pick those directly.  They
# cut a block of wood into a shape and drill pockets for weights, and the properties
# follow.
#
# A `GeometricCar` works the same way.  Its body is a side profile, cut from the
# block and extruded through the block's thickness, and it can hold inclusions of
# other materials, like a tungsten weight in a pocket.  The mass, center of
# gravity, moment of inertia, and drag all come from the geometry, so the optimizer
# can tune the shape of the car and the placement of its weight directly.

# %%
import copy

import matplotlib.pyplot as plt

import pwdsim
from pwdsim import constraints
from pwdsim.geometry import PINE, TUNGSTEN, Profile, WeightPocket
from pwdsim.plotting import plot_geometry, plot_runs
from pwdsim.units import GRAM, INCH, OUNCE

# %% [markdown]
# ## The block and the weight
#
# The standard BSA kit block is 7 in long, 1.25 in tall, and 1.75 in thick (across
# the car), with axle slots $4\tfrac{3}{8}$ in apart, the rear one $\tfrac{7}{8}$ in
# from the back.  The body frame has its origin at the rear axle, so the block runs
# from $x = -0.875$ in to $6.125$ in.  Its bottom sits 0.15 in below the axles.
#
# The `Profile` is the shape of the top of the body: a B-spline through 9 control
# heights above the bottom.  Here it starts as the uncut block.
#
# The weight is a tungsten `WeightPocket` drilled up from the bottom of the car,
# behind the rear axle: a pocket 0.5 in long (along the car) and `height` deep, with
# the weight filling its top `length`.  Its `depth` across the car is less than the
# full thickness, like a real tungsten bar.

# %%
track = pwdsim.besttrack(42)

profile = Profile.block(
    x_rear=-0.875 * INCH,
    x_front=6.125 * INCH,
    bottom=-0.15 * INCH,
    height=1.25 * INCH,
    n=9,
)
weight = WeightPocket(
    position=-0.5 * INCH,
    width=0.5 * INCH,
    height=0.6 * INCH,
    length=0.15 * INCH,
    material=TUNGSTEN,
    depth=0.5 * INCH,
)

wheel_radius = 0.595 * INCH
wheel_mass = 2.6 * GRAM
car_args = dict(
    thickness=1.75 * INCH,
    body_density=PINE,
    wheelbase=4.375 * INCH,
    wheel_mass=wheel_mass,
    rear_wheel_inertia=0.58 * wheel_mass * wheel_radius**2,
    front_wheel_inertia=0.58 * wheel_mass * wheel_radius**2,
    rear_wheel_radius=wheel_radius,
    front_wheel_radius=wheel_radius,
    rear_axle_radius=0.087 / 2 * INCH,
    front_axle_radius=0.087 / 2 * INCH,
    rear_axle_friction=0.1,
    front_axle_friction=0.1,
    rolling_friction=0.002,
)
car = pwdsim.GeometricCar(profile, [weight], **car_args)


def properties(car):
    print(f"Mass:              {car.mass.item() / OUNCE:.3f} oz")
    cg = car.cg.detach() / INCH
    print(f"Center of gravity: ({cg[0]:.3f}, {cg[1]:.3f}) in")
    print(f"Pitch inertia:     {car.body_inertia.item():.3e} kg m²")
    print(f"Frontal area:      {car.frontal_area.item() / INCH**2:.3f} in²")
    print(f"Drag coefficient:  {car.drag_coefficient.item():.3f}")


def drag(car):
    """The drag areas of the components, in in²."""
    breakdown = car.drag_breakdown()
    for name in ("friction", "forebody", "base", "wheels", "total"):
        print(f"{name:>9}: {breakdown[name].item() / INCH**2:.3f} in²")


properties(car)
fig = plot_geometry(car)
plt.show()

# %% [markdown]
# The uncut block with a little tungsten weighs about 4.5 oz, with its center of
# gravity well forward, about 2.3 in ahead of the rear axle: most of the mass is
# still the wood.
#
# ## Drag from the shape
#
# The drag comes from a semi-empirical model of the body and wheels (see the
# [car geometry docs](../../geometry/#drag-from-the-geometry)).  It adds up the drag
# areas $C_d A$ of the skin friction on the body, the pressure on its blunt front
# face and steep forward-facing slopes, the suction behind its rear face, and the
# wheels.  The uncut block is a bluff body: most of its drag comes from the flat
# front face.

# %%
drag(car)
#
# ## Mixing in numbers
#
# Any of the computed properties can be replaced by a number.  That's useful for
# checking the geometry against a measured car, or for holding one property fixed
# while the geometry sets the others.  Here the mass and drag coefficient are given,
# while everything else still comes from the geometry:

# %%
weighed = pwdsim.GeometricCar(
    copy.deepcopy(profile),
    [copy.deepcopy(weight)],
    **car_args,
    mass=5 * OUNCE,
    drag_coefficient=0.4,
)
print("Given:", weighed.overrides)
properties(weighed)

# %% [markdown]
# ## Optimizing the design
#
# Now let the optimizer shape the body and place the weight.  The design variables
# are the geometry's parameters, by name:
#
# - **`profile.heights`**: the 9 control heights of the top of the body.
#   `Profile.bounds` keeps them between 0.25 in and the height of the block, which
#   keeps the whole body inside the block.  It also fixes the control heights over
#   the two axle slots, leaving wood around the axles.
# - **`inclusions.0.position`**, **`height`**, **`length`**, and **`depth`**: where
#   the weight pocket is, how deep it goes, how much tungsten it holds, and how
#   thick the tungsten is across the car.
#
# The constraints:
#
# - `bsa_rules()`: a mass of at most 5 oz, and the center of gravity between the
#   axles;
# - `regions_inside_body`: the weight stays inside the body, with at least 1/16 in
#   of wood above it;
# - `regions_within_thickness`: the weight is no thicker than the car;
# - `regions_avoid_axles`: the pocket stays clear of the axle slots;
# - and, by default, the wheels stay on the track.
#
# With this many variables the optimizer gets most of the way in about 20
# iterations, then creeps the last fraction of a millisecond as the body shape
# approaches its bounds, at a few hundredths of a millisecond per iteration.  To
# keep this example quick, the `maxiter` option caps it at 30 iterations.

# %%
heights = car.profile.bounds(
    fixed=[(-0.5 * INCH, 0.5 * INCH), (3.875 * INCH, 4.875 * INCH)],
    minimum=0.25 * INCH,
)
variables = {
    "profile.heights": heights,
    "inclusions.0.position": (-0.875 * INCH, 6.125 * INCH),
    "inclusions.0.height": (0.1 * INCH, 1.25 * INCH),
    "inclusions.0.length": (0.02 * INCH, 1.25 * INCH),
    "inclusions.0.depth": (0.1 * INCH, 2.5 * INCH),
}
design_constraints = constraints.bsa_rules() + [
    constraints.regions_inside_body(margin=INCH / 16),
    constraints.regions_within_thickness(),
    constraints.regions_avoid_axles(car),
]

sim = pwdsim.Simulation(track, car, dt=2e-4, duration=3.5)
result = pwdsim.optimize(sim, variables, design_constraints, options={"maxiter": 30})
print(
    result.summary(
        {
            "profile.heights": (INCH, "in"),
            "inclusions.0.position": (INCH, "in"),
            "inclusions.0.height": (INCH, "in"),
            "inclusions.0.length": (INCH, "in"),
            "inclusions.0.depth_": (INCH, "in"),
        }
    )
)

# %% [markdown]
# ## The optimized car

# %%
fig, (ax_before, ax_after) = plt.subplots(
    2, 1, figsize=(10, 6), constrained_layout=True
)
plot_geometry(result.initial_car, ax=ax_before)
ax_before.set_title("Starting design")
plot_geometry(car, ax=ax_after)
ax_after.set_title("Optimized design")
plt.show()
properties(car)
print()
drag(car)

# %%
fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
ax.plot([h["objective"] for h in result.history], "o-")
ax.set_xlabel("iteration")
ax.set_ylabel("finish time (ms)")
plt.show()

# %%
start = pwdsim.Simulation(track, result.initial_car, dt=2e-4, duration=3.5)()
optimized = pwdsim.Simulation(track, car, dt=2e-4, duration=3.5)()
print(f"Starting design:  {start.finish_time.item():.4f} s")
print(f"Optimized design: {optimized.finish_time.item():.4f} s")
fig = plot_runs([start, optimized], labels=["starting", "optimized"], unit="ft")
plt.show()

# %% [markdown]
# ## What the optimizer did
#
# The optimized car is about 90 ms faster.  Its shape now matters twice over:
# through where its mass is, and through its drag.  The optimizer:
#
# - **shapes the body into a long, gentle wedge**, from nearly full height at the
#   rear down to the 0.25 in minimum at the nose.  The low nose shrinks the blunt
#   front face, which dominated the drag of the uncut block, and the slope stays
#   gentle enough for the flow to stay attached: steep slopes would count as blunt
#   too.  The drag area falls by almost half, and the drag coefficient from about
#   1.2 to 0.7;
# - **keeps wood at the rear**, which helps move the center of gravity back, but
#   rounds off the top of the rear face a little, trading a little base drag;
# - **moves the tungsten to the very back of the block** and fills the pocket with
#   it, sized to bring the car up to exactly 5 oz.
#
# The summary shows which constraints are active: `max_mass`, and
# `regions_inside_body` (the weight is against the back of the block).  The
# center of gravity ends up about 0.6 in ahead of the rear axle, close to, but not
# at, the lift-off limit.
#
# The wheels now make up almost half of the remaining drag, and the shape of the
# body can't change that.  The drag model is semi-empirical, so treat the exact
# numbers with some caution, but the trends (a low nose, gentle slopes, and weight
# at the back) match what fast derby cars look like.
