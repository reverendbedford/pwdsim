"""Forward simulation of a car running down a track.

The state of the car is $x = (s, v)$, the location of the rear wheel contact along
the track and its rate $v = \\dot{s}$.  The enabled [physics
terms][pwdsim.physics] are assembled into the equation of motion

$$
M(s)\\, a + C(s)\\, v^2 + V_s + D_s = 0,
$$

which gives the acceleration $a = \\ddot{s}$.  [pyzag](https://github.com/applied-material-modeling/pyzag)
integrates the resulting ODE $\\dot{x} = (v, a)$ in time, with gradients of the
results with respect to the car parameters from the adjoint method.
"""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from pyzag import nonlinear, ode
from pyzag.chunktime import ChunkNewtonRaphson
from torch import nn

from pwdsim.car import Car
from pwdsim.kinematics import CarKinematics, Configuration
from pwdsim.physics import (
    DissipativeTerm,
    Environment,
    KineticTerm,
    PhysicsTerm,
    PotentialTerm,
    default_physics,
)
from pwdsim.track import Track
from pwdsim.types import DTYPE


class LiftOffWarning(UserWarning):
    """A wheel lifted off the track: the normal force at an axle went negative."""


class DidNotFinishWarning(UserWarning):
    """The car did not reach the finish line in the simulated time."""


@dataclass
class GeneralizedForces:
    """The assembled terms of the equation of motion at a set of states.

    Attributes:
        mass: effective mass $M(s) = \\sum m J_s \\cdot J_s$.
        dmass: $M'(s)$.
        coriolis: $C(s) = \\sum m J_s \\cdot J_s'$, so the inertial force is
            $M a + C v^2$.
        dcoriolis: $C'(s)$.
        potential: $V_s$, the potential energy gradient.
        dpotential: $V_{ss}$.
        dissipation: $D_s = \\partial F / \\partial \\dot s$.
        ddissipation_ds: $\\partial D_s / \\partial s$.
        ddissipation_dv: $\\partial D_s / \\partial v$.
    """

    mass: torch.Tensor
    dmass: torch.Tensor
    coriolis: torch.Tensor
    dcoriolis: torch.Tensor
    potential: torch.Tensor
    dpotential: torch.Tensor
    dissipation: torch.Tensor
    ddissipation_ds: torch.Tensor
    ddissipation_dv: torch.Tensor


def _dot(a, b):
    return torch.sum(a * b, dim=-1)


class EquationsOfMotion:
    """Assembles the enabled physics terms into the equation of motion.

    Args:
        kinematics: the car on the track.
        terms: the physics terms.  Disabled terms are skipped.
        env: the global parameters.
    """

    def __init__(
        self, kinematics: CarKinematics, terms: Iterable[PhysicsTerm], env: Environment
    ):
        self.kinematics = kinematics
        self.terms = [term for term in terms if term.enabled]
        self.env = env

    @property
    def car(self) -> Car:
        return self.kinematics.car

    def _of_kind(self, kind):
        return [term for term in self.terms if isinstance(term, kind)]

    def _rates(self, config):
        return [
            rate
            for term in self._of_kind(KineticTerm)
            for rate in term.rates(config, self.car, self.env)
        ]

    def _potentials(self, config):
        return [
            term.potential(config, self.car, self.env)
            for term in self._of_kind(PotentialTerm)
        ]

    def _dissipations(self, config, v):
        return [
            term.dissipation(config, v, self.car, self.env)
            for term in self._of_kind(DissipativeTerm)
        ]

    def generalized_forces(self, config: Configuration, v) -> GeneralizedForces:
        """The terms of the equation of motion along $s$."""
        zero = torch.zeros_like(config.front_contact.value)

        def total(values):
            return sum(values, zero)

        rates = self._rates(config)
        potentials = self._potentials(config)
        dissipations = self._dissipations(config, v)
        return GeneralizedForces(
            mass=total(r.weight * _dot(r.ds, r.ds) for r in rates),
            dmass=total(2 * r.weight * _dot(r.ds, r.dss) for r in rates),
            coriolis=total(r.weight * _dot(r.ds, r.dss) for r in rates),
            dcoriolis=total(
                r.weight * (_dot(r.dss, r.dss) + _dot(r.ds, r.dsss)) for r in rates
            ),
            potential=total(p.ds for p in potentials),
            dpotential=total(p.dss for p in potentials),
            dissipation=total(d.s for d in dissipations),
            ddissipation_ds=total(d.s_ds for d in dissipations),
            ddissipation_dv=total(d.s_dv for d in dissipations),
        )

    def acceleration(self, s, v) -> torch.Tensor:
        """The acceleration $a = \\ddot{s}$."""
        return self.acceleration_and_jacobian(s, v)[0]

    def acceleration_and_jacobian(
        self, s, v
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The acceleration $a$ and its derivatives $\\partial a / \\partial s$ and
        $\\partial a / \\partial v$."""
        if not self._of_kind(KineticTerm):
            raise ValueError("The equation of motion needs a kinetic energy term")
        f = self.generalized_forces(self.kinematics.evaluate(s), v)
        a = -(f.coriolis * v**2 + f.potential + f.dissipation) / f.mass
        da_dv = -(2 * f.coriolis * v + f.ddissipation_dv) / f.mass
        da_ds = (
            -(f.dcoriolis * v**2 + f.dpotential + f.ddissipation_ds + a * f.dmass)
            / f.mass
        )
        return a, da_ds, da_dv

    def normal_forces(self, s, v) -> torch.Tensor:
        """Normal forces at the rear and front axles, in a trailing dimension.

        The normal forces are the constraint forces holding the axle lifts at zero,
        from the Euler-Lagrange equations along $h_r$ and $h_f$.  A positive force
        pushes the car away from the track.
        """
        config = self.kinematics.evaluate(s)
        a = self.acceleration(s, v)
        rates = self._rates(config)
        potentials = self._potentials(config)
        dissipations = self._dissipations(config, v)
        forces = []
        for lift, dissipation in (("dhr", "hr"), ("dhf", "hf")):
            # Inertial force m J_h . (J_s a + J_s' v^2) for each rate
            force = sum(
                r.weight
                * _dot(
                    getattr(r, lift), r.ds * a[..., None] + r.dss * (v**2)[..., None]
                )
                for r in rates
            )
            force = force + sum(getattr(p, lift) for p in potentials)
            force = force + sum(getattr(d, dissipation) for d in dissipations)
            forces.append(force + torch.zeros_like(a))
        return torch.stack(forces, dim=-1)

    def energy(self, s, v) -> tuple[torch.Tensor, torch.Tensor]:
        """Kinetic and potential energy."""
        config = self.kinematics.evaluate(s)
        f = self.generalized_forces(config, v)
        potential = sum(
            (p.value for p in self._potentials(config)), torch.zeros_like(f.mass)
        )
        return 0.5 * f.mass * v**2, potential


class CarODE(nn.Module):
    """The equation of motion as a pyzag ODE, $\\dot{x} = (v, a)$ for $x = (s, v)$.

    Args:
        car: the car, registered as a submodule so pyzag finds its parameters.
        equations: the equations of motion.
    """

    def __init__(self, car: Car, equations: EquationsOfMotion):
        super().__init__()
        self.car = car
        self.equations = equations

    def forward(self, t, x):
        s, v = x[..., 0], x[..., 1]
        a, da_ds, da_dv = self.equations.acceleration_and_jacobian(s, v)
        x_dot = torch.stack([v, a], dim=-1)
        jacobian = torch.zeros(x.shape + (2,), dtype=x.dtype, device=x.device)
        jacobian[..., 0, 1] = 1.0
        jacobian[..., 1, 0] = da_ds
        jacobian[..., 1, 1] = da_dv
        return x_dot, jacobian


_INTEGRATORS = {
    "backward-euler": ode.BackwardEulerODE,
    "forward-euler": ode.ForwardEulerODE,
}


class Simulation(nn.Module):
    """A car running down a track.

    Args:
        track: the track.
        car: the car.  Cars with batch dimensions simulate several designs at once.
        physics: the physics terms, by default all of the implemented terms.
        env: the global parameters.
        dt: time step.
        duration: total simulated time, which must be long enough for the car to
            finish.
        integrator: time integration scheme, `"backward-euler"` or
            `"forward-euler"`.
        block_size: number of time steps pyzag solves together.
    """

    def __init__(
        self,
        track: Track,
        car: Car,
        physics: Iterable[PhysicsTerm] | None = None,
        env: Environment | None = None,
        dt: float = 1e-4,
        duration: float = 4.0,
        integrator: str = "backward-euler",
        block_size: int = 1000,
    ):
        super().__init__()
        if integrator not in _INTEGRATORS:
            raise ValueError(f"Unknown integrator {integrator!r}")
        self.track = track
        self.car = car
        terms = default_physics() if physics is None else list(physics)
        self.physics = nn.ModuleDict({term.name: term for term in terms})
        self.env = Environment() if env is None else env
        self.dt = dt
        self.duration = duration
        self.integrator = integrator
        self.block_size = block_size

    @property
    def kinematics(self) -> CarKinematics:
        return CarKinematics(self.track, self.car)

    @property
    def equations(self) -> EquationsOfMotion:
        return EquationsOfMotion(self.kinematics, self.physics.values(), self.env)

    def enable(self, *names: str):
        """Switch on the named physics terms."""
        for name in names:
            self.physics[name].enabled = True

    def disable(self, *names: str):
        """Switch off the named physics terms."""
        for name in names:
            self.physics[name].enabled = False

    @property
    def batch_shape(self) -> torch.Size:
        """Batch shape of the simulation, `(1,)` for a single car."""
        return self.car.batch_shape or torch.Size((1,))

    def forward(self, adjoint: bool = True) -> "Run":
        """Simulate the run.

        Args:
            adjoint: use the adjoint method for gradients.  Otherwise gradients
                come from automatic differentiation through the time steps, which
                takes much more memory.

        Returns:
            The results of the run.
        """
        n = int(round(self.duration / self.dt)) + 1
        times = torch.arange(n, dtype=DTYPE) * self.dt
        shape = self.batch_shape
        forces = times.reshape((n,) + (1,) * len(shape) + (1,)).expand((n, *shape, 1))

        s0 = self.kinematics.start_position().expand(shape)
        y0 = torch.stack([s0, torch.zeros_like(s0)], dim=-1)

        with warnings.catch_warnings():
            # pyzag warns about a torch.compile setting that doesn't affect us
            warnings.filterwarnings("ignore", message="pyzag lowered")
            solver = nonlinear.RecursiveNonlinearEquationSolver(
                _INTEGRATORS[self.integrator](CarODE(self.car, self.equations)),
                step_generator=nonlinear.StepGenerator(self.block_size),
                predictor=nonlinear.PreviousStepsPredictor(),
                nonlinear_solver=ChunkNewtonRaphson(
                    rtol=1e-12, atol=1e-12, throw_on_fail=True
                ),
            )
        if adjoint:
            # Only pass the trainable parameters: pyzag fails on frozen ones
            params = [p for p in solver.parameters() if p.requires_grad]
            states = nonlinear.AdjointWrapper.apply(solver, y0, n, [forces], *params)
        else:
            states = nonlinear.solve(solver, y0, n, forces)

        # Drop the batch dimension added for a single car
        states = states.reshape((n, *self.car.batch_shape, 2))
        run = Run(self, times, states)
        run.check()
        return run


class Run:
    """The results of a simulated run.

    Quantities are per car in the batch, with the time steps in the leading
    dimension for time series.  Scalar results like the finish time are
    differentiable with respect to the car parameters.

    Args:
        simulation: the simulation that produced the run.
        times: the times, with shape `(n,)`.
        states: the states $(s, v)$, with shape `(n, *batch, 2)`.
    """

    def __init__(
        self, simulation: Simulation, times: torch.Tensor, states: torch.Tensor
    ):
        self.simulation = simulation
        self.times = times
        self.states = states

    @property
    def s(self) -> torch.Tensor:
        """Location of the rear wheel contact along the track."""
        return self.states[..., 0]

    @property
    def v(self) -> torch.Tensor:
        """Rate of the rear wheel contact along the track, $\\dot{s}$."""
        return self.states[..., 1]

    def configuration(self) -> Configuration:
        """The car configuration at every time step."""
        return self.simulation.kinematics.evaluate(self.s)

    def speed(self) -> torch.Tensor:
        """Speed of the car's center of gravity."""
        cg = self.configuration().cg
        return torch.linalg.norm(cg.ds, dim=-1) * self.v.abs()

    def _finish_step(self):
        """Index of the last step before the front of the car crosses the finish
        line, and the fraction of the following step at which it crosses."""
        track = self.simulation.track
        x_front = self.configuration().front.value[..., 0]
        x_finish = track.position(track.s_finish)[0]
        crossed = x_front >= x_finish
        finished = crossed.any(dim=0)
        step = torch.clamp(torch.argmax(crossed.to(torch.int8), dim=0) - 1, min=0)
        x0 = torch.gather(x_front, 0, step[None])[0]
        x1 = torch.gather(x_front, 0, (step + 1)[None])[0]
        fraction = (x_finish - x0) / (x1 - x0)
        return step, fraction, finished

    @property
    def finished(self) -> torch.Tensor:
        """Whether each car crossed the finish line."""
        return self._finish_step()[2]

    @property
    def finish_time(self) -> torch.Tensor:
        """Time when the front of the car crosses the finish line, NaN for cars
        that don't finish."""
        step, fraction, finished = self._finish_step()
        time = self.times[step] + fraction * self.simulation.dt
        return torch.where(finished, time, torch.nan)

    @property
    def finish_speed(self) -> torch.Tensor:
        """Speed of the center of gravity as the car crosses the finish line."""
        step, fraction, finished = self._finish_step()
        speed = self.speed()
        v0 = torch.gather(speed, 0, step[None])[0]
        v1 = torch.gather(speed, 0, (step + 1)[None])[0]
        return torch.where(finished, v0 + fraction * (v1 - v0), torch.nan)

    def _before_finish(self) -> torch.Tensor:
        """Mask of the time steps up to the finish."""
        step, _, finished = self._finish_step()
        last = torch.where(finished, step + 1, len(self.times) - 1)
        index = torch.arange(len(self.times)).reshape((-1,) + (1,) * last.ndim)
        return index <= last

    @property
    def max_speed(self) -> torch.Tensor:
        """Maximum speed of the center of gravity before the finish."""
        speed = torch.where(self._before_finish(), self.speed(), -torch.inf)
        return speed.max(dim=0).values

    def normal_forces(self) -> torch.Tensor:
        """Normal forces at the rear and front axles, in a trailing dimension."""
        return self.simulation.equations.normal_forces(self.s, self.v)

    @property
    def min_normal_force(self) -> torch.Tensor:
        """Minimum normal force at the rear and front axles before the finish, in a
        trailing dimension.  Negative values mean a wheel lifted off the track."""
        forces = torch.where(
            self._before_finish()[..., None], self.normal_forces(), torch.inf
        )
        return forces.min(dim=0).values

    def energy(self) -> dict[str, torch.Tensor]:
        """Kinetic, potential, and total mechanical energy at each time step."""
        kinetic, potential = self.simulation.equations.energy(self.s, self.v)
        return {
            "kinetic": kinetic,
            "potential": potential,
            "total": kinetic + potential,
        }

    def check(self):
        """Warn if a car didn't finish or lifted a wheel off the track."""
        with torch.no_grad():
            if not torch.all(self.finished):
                warnings.warn(
                    "The car did not reach the finish line; increase the duration",
                    DidNotFinishWarning,
                    stacklevel=3,
                )
            if torch.any(self.min_normal_force < 0):
                warnings.warn(
                    "A wheel lifted off the track (negative normal force) before "
                    "the finish; the results are not physical",
                    LiftOffWarning,
                    stacklevel=3,
                )
