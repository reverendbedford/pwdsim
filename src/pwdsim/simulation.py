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


def _dot(a, b):
    return torch.sum(a * b, dim=-1)


@dataclass
class LinearSystem:
    """The equations of motion on the track as a linear system $K x = b$ for
    $x = (a, N_r, N_f)$, along with the derivatives of $K$ and $b$ with respect to
    $s$ and $v$.

    The first row is the equation of motion along $s$ and the others give the normal
    forces, the constraint forces along the lifts.  Friction proportional to the
    normal forces couples them.

    Attributes:
        K: the matrix, with shape `(..., 3, 3)`.
        b: the right hand side, with shape `(..., 3)`.
        K_ds: $\\partial K / \\partial s$.
        b_ds: $\\partial b / \\partial s$.
        K_dv: $\\partial K / \\partial v$.
        b_dv: $\\partial b / \\partial v$.
        mass: the effective mass $M(s) = \\sum m J_s \\cdot J_s$.
        potential: the potential energy.
        dissipation: the dissipative force along $s$ that doesn't depend on the
            normal forces, $D^0_s$.
        friction: the friction coefficients along $s$, $c_k$, so the total
            dissipative force along $s$ is $D^0_s + \\sum_k c_k N_k$, with shape
            `(..., 2)`.
    """

    K: torch.Tensor
    b: torch.Tensor
    K_ds: torch.Tensor
    b_ds: torch.Tensor
    K_dv: torch.Tensor
    b_dv: torch.Tensor
    mass: torch.Tensor
    potential: torch.Tensor
    dissipation: torch.Tensor
    friction: torch.Tensor


LIFTS = (("dhr", "dhr_ds", "hr"), ("dhf", "dhf_ds", "hf"))


def _friction_force(friction, v):
    """The force of a normal force friction along s per unit normal force,
    $c |J_s| \\tanh(v / v_\\mathrm{reg})$."""
    return (
        friction.coefficient
        * friction.rate.ds.abs()
        * torch.tanh(v / friction.regularization)
    )


class EquationsOfMotion:
    """Assembles the enabled physics terms into the equations of motion.

    The equation of motion along $s$ is

    $$
    M a + C v^2 + V_s + D^0_s + \\sum_k c_k N_k = 0,
    $$

    and the normal forces, the constraint forces along the lifts, are

    $$
    N_j = A_j a + B_j + \\sum_k e_{jk} N_k,
    $$

    where the $N_k$ terms come from friction proportional to the normal forces.
    Together these are a linear system for $(a, N_r, N_f)$ at each state; see
    [`system`][pwdsim.simulation.EquationsOfMotion.system].

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

    def system(self, s, v) -> LinearSystem:
        """Assemble the linear system for $(a, N_r, N_f)$ at the states $(s, v)$."""
        if not self._of_kind(KineticTerm):
            raise ValueError("The equation of motion needs a kinetic energy term")
        s = torch.as_tensor(s, dtype=DTYPE)
        v = torch.as_tensor(v, dtype=DTYPE)
        config = self.kinematics.evaluate(s)
        car, env = self.car, self.env
        rates = [
            r for t in self._of_kind(KineticTerm) for r in t.rates(config, car, env)
        ]
        potentials = [
            t.potential(config, car, env) for t in self._of_kind(PotentialTerm)
        ]
        dissipative = self._of_kind(DissipativeTerm)
        dissipations = [
            d for t in dissipative if (d := t.dissipation(config, v, car, env))
        ]
        frictions = [f for t in dissipative for f in t.frictions(config, car, env)]

        zero = torch.zeros_like(config.front_contact.value * v)

        def total(values):
            return sum(values, zero)

        def weighted(rate, a, b):
            return rate.weight * _dot(a, b)

        # Along s: M a + C v^2 + V_s + D_s = 0
        mass = total(weighted(r, r.ds, r.ds) for r in rates)
        dmass = total(2 * weighted(r, r.ds, r.dss) for r in rates)
        coriolis = total(weighted(r, r.ds, r.dss) for r in rates)
        dcoriolis = total(
            weighted(r, r.dss, r.dss) + weighted(r, r.ds, r.dsss) for r in rates
        )
        b_s = -(
            coriolis * v**2
            + total(p.ds for p in potentials)
            + total(d.s for d in dissipations)
        )
        b_s_ds = -(
            dcoriolis * v**2
            + total(p.dss for p in potentials)
            + total(d.s_ds for d in dissipations)
        )
        b_s_dv = -(2 * coriolis * v + total(d.s_dv for d in dissipations))

        # Along the lifts: N_j = A_j a + B_j + sum_k e_jk N_k
        A, A_ds, B, B_ds, B_dv = [], [], [], [], []
        for lift, lift_ds, dissipation in LIFTS:
            A.append(total(weighted(r, getattr(r, lift), r.ds) for r in rates))
            A_ds.append(
                total(
                    weighted(r, getattr(r, lift_ds), r.ds)
                    + weighted(r, getattr(r, lift), r.dss)
                    for r in rates
                )
            )
            inertia = total(weighted(r, getattr(r, lift), r.dss) for r in rates)
            dinertia = total(
                weighted(r, getattr(r, lift_ds), r.dss)
                + weighted(r, getattr(r, lift), r.dsss)
                for r in rates
            )
            B.append(
                inertia * v**2
                + total(getattr(p, lift) for p in potentials)
                + total(getattr(d, dissipation) for d in dissipations)
            )
            B_ds.append(
                dinertia * v**2
                + total(getattr(p, lift_ds) for p in potentials)
                + total(getattr(d, f"{dissipation}_ds") for d in dissipations)
            )
            B_dv.append(
                2 * inertia * v
                + total(getattr(d, f"{dissipation}_dv") for d in dissipations)
            )

        # Friction proportional to the normal forces, F = coef N_k |rho|, with
        # sgn(rho) = sgn(J_s) tanh(v / v_reg)
        c = [[zero, zero, zero] for _ in range(2)]  # c_k and its s, v derivatives
        e = [[[zero, zero, zero] for _ in range(2)] for _ in range(2)]  # e_jk
        for f in frictions:
            rate = f.rate
            sign = torch.sign(rate.ds)
            tanh = torch.tanh(v / f.regularization)
            dtanh = (1 - tanh**2) / f.regularization
            k = f.axle
            c[k][0] = c[k][0] + _friction_force(f, v)
            c[k][1] = c[k][1] + f.coefficient * sign * rate.dss * tanh
            c[k][2] = c[k][2] + f.coefficient * rate.ds.abs() * dtanh
            for j, (lift, lift_ds, _) in enumerate(LIFTS):
                J, J_ds = getattr(rate, lift), getattr(rate, lift_ds)
                e[j][k][0] = e[j][k][0] + f.coefficient * sign * tanh * J
                e[j][k][1] = e[j][k][1] + f.coefficient * sign * tanh * J_ds
                e[j][k][2] = e[j][k][2] + f.coefficient * sign * dtanh * J

        def matrix(i):
            """K (i = 0) or its derivative with respect to s (i = 1) or v (i = 2)."""
            first = (mass, dmass, zero)[i]
            rows = [[first, c[0][i], c[1][i]]]
            for j in range(2):
                a_term = (-A[j], -A_ds[j], zero)[i]
                identity = [(1.0 if (k == j and i == 0) else 0.0) for k in range(2)]
                rows.append(
                    [a_term] + [identity[k] - e[j][k][i] + zero for k in range(2)]
                )
            return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)

        return LinearSystem(
            K=matrix(0),
            b=torch.stack([b_s, *B], dim=-1),
            K_ds=matrix(1),
            b_ds=torch.stack([b_s_ds, *B_ds], dim=-1),
            K_dv=matrix(2),
            b_dv=torch.stack([b_s_dv, *B_dv], dim=-1),
            mass=mass,
            potential=total(p.value for p in potentials),
            dissipation=total(d.s for d in dissipations),
            friction=torch.stack([c[0][0], c[1][0]], dim=-1),
        )

    def solve(self, s, v) -> tuple[torch.Tensor, torch.Tensor]:
        """The acceleration $a$ and the normal forces $(N_r, N_f)$, in a trailing
        dimension."""
        system = self.system(s, v)
        x = torch.linalg.solve(system.K, system.b)
        return x[..., 0], x[..., 1:]

    def acceleration(self, s, v) -> torch.Tensor:
        """The acceleration $a = \\ddot{s}$."""
        return self.solve(s, v)[0]

    def acceleration_and_jacobian(
        self, s, v
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The acceleration $a$ and its derivatives $\\partial a / \\partial s$ and
        $\\partial a / \\partial v$.

        Differentiating $K x = b$ gives $x_{,s} = K^{-1}(b_{,s} - K_{,s} x)$, and
        likewise for $v$.
        """
        system = self.system(s, v)
        x = torch.linalg.solve(system.K, system.b)
        rhs = torch.stack(
            [
                system.b_ds - (system.K_ds @ x[..., None])[..., 0],
                system.b_dv - (system.K_dv @ x[..., None])[..., 0],
            ],
            dim=-1,
        )
        dx = torch.linalg.solve(system.K, rhs)
        return x[..., 0], dx[..., 0, 0], dx[..., 0, 1]

    def normal_forces(self, s, v) -> torch.Tensor:
        """Normal forces at the rear and front axles, in a trailing dimension.

        The normal forces are the constraint forces holding the axle lifts at zero,
        from the Euler-Lagrange equations along $h_r$ and $h_f$.  A positive force
        pushes the car away from the track.
        """
        return self.solve(s, v)[1]

    def energy(self, s, v) -> tuple[torch.Tensor, torch.Tensor]:
        """Kinetic and potential energy."""
        system = self.system(s, v)
        v = torch.as_tensor(v, dtype=DTYPE)
        return 0.5 * system.mass * v**2, system.potential + torch.zeros_like(
            system.mass
        )

    def power(self, s, v) -> torch.Tensor:
        """Power dissipated by the dissipative terms, $v D_s$."""
        system = self.system(s, v)
        x = torch.linalg.solve(system.K, system.b)
        v = torch.as_tensor(v, dtype=DTYPE)
        return v * (system.dissipation + _dot(system.friction, x[..., 1:]))

    def by_term(self, s, v) -> dict[str, torch.Tensor]:
        """The contribution of each enabled term, by name: the kinetic energy of a
        kinetic term, the potential energy of a potential term, and the power
        dissipated by a dissipative term.

        The friction terms use the normal forces from the full equations of motion,
        so the dissipated powers add up to
        [`power`][pwdsim.simulation.EquationsOfMotion.power].
        """
        s = torch.as_tensor(s, dtype=DTYPE)
        v = torch.as_tensor(v, dtype=DTYPE)
        _, normal = self.solve(s, v)
        config = self.kinematics.evaluate(s)
        car, env = self.car, self.env
        zero = torch.zeros_like(normal[..., 0])
        contributions = {}
        for term in self.terms:
            if isinstance(term, KineticTerm):
                value = sum(
                    0.5 * r.weight * _dot(r.ds, r.ds) * v**2
                    for r in term.rates(config, car, env)
                )
            elif isinstance(term, PotentialTerm):
                value = term.potential(config, car, env).value
            else:
                dissipation = term.dissipation(config, v, car, env)
                force = zero if dissipation is None else dissipation.s
                for f in term.frictions(config, car, env):
                    force = force + _friction_force(f, v) * normal[..., f.axle]
                value = v * force
            contributions[term.name] = value + zero
        return contributions


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

        equations = self.equations
        with warnings.catch_warnings():
            # pyzag warns about a torch.compile setting that doesn't affect us
            warnings.filterwarnings("ignore", message="pyzag lowered")
            solver = nonlinear.RecursiveNonlinearEquationSolver(
                _INTEGRATORS[self.integrator](CarODE(self.car, equations)),
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
        run = Run(self, equations, times, states)
        run.check()
        return run


class Run:
    """The results of a simulated run.

    Quantities are per car in the batch, with the time steps in the leading
    dimension for time series.  Scalar results like the finish time are
    differentiable with respect to the car parameters.

    Args:
        simulation: the simulation that produced the run.
        equations: the equations of motion the run was simulated with.  Results
            are computed with these, so switching physics terms on or off after the
            run doesn't change them.
        times: the times, with shape `(n,)`.
        states: the states $(s, v)$, with shape `(n, *batch, 2)`.
    """

    def __init__(
        self,
        simulation: Simulation,
        equations: EquationsOfMotion,
        times: torch.Tensor,
        states: torch.Tensor,
    ):
        self.simulation = simulation
        self._equations = equations
        self._versions = self._parameter_versions()
        self.times = times
        self.states = states

    def _parameter_versions(self):
        return [p._version for p in self._equations.car.parameters()]

    @property
    def equations(self) -> EquationsOfMotion:
        """The equations of motion the run was simulated with.

        The results of a run are computed from its states and the car's parameters,
        so they can be differentiated with respect to those parameters.  If the
        parameters change after the run, for example in an optimization, the
        results would be wrong, so this raises an error instead.
        """
        if self._parameter_versions() != self._versions:
            raise RuntimeError(
                "The car's parameters changed after this run, so its results would "
                "be wrong: simulate the car again"
            )
        return self._equations

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
        return self.equations.kinematics.evaluate(self.s)

    def speed(self) -> torch.Tensor:
        """Speed of the car's center of gravity."""
        cg = self.configuration().cg
        return torch.linalg.norm(cg.ds, dim=-1) * self.v.abs()

    def _finish_step(self):
        """Index of the last step before the front of the car crosses the finish
        line, and the fraction of the following step at which it crosses."""
        track = self.equations.kinematics.track
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
        time = self.times[step] + fraction * (self.times[step + 1] - self.times[step])
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
        return self.equations.normal_forces(self.s, self.v)

    @property
    def min_normal_force(self) -> torch.Tensor:
        """Minimum normal force at the rear and front axles before the finish, in a
        trailing dimension.  Negative values mean a wheel lifted off the track."""
        forces = torch.where(
            self._before_finish()[..., None], self.normal_forces(), torch.inf
        )
        return forces.min(dim=0).values

    def smooth_min_normal_force(self, sharpness: float = 500.0) -> torch.Tensor:
        """A smooth version of the minimum normal force over both axles before the
        finish, for use as an optimization constraint.

        Uses the Kreisselmeier-Steinhauser aggregate
        $-\\frac{1}{\\rho} \\log \\sum_i e^{-\\rho N_i}$ over the time steps and
        axles.  It is never larger than the exact minimum, and at most
        $\\ln(n) / \\rho$ smaller for $n$ values, so it errs on the safe side.
        Unlike the exact minimum, it's differentiable everywhere.

        Args:
            sharpness: the parameter $\\rho$, in 1/N.  Larger values follow the exact
                minimum more closely but make the aggregate less smooth.
        """
        forces = torch.where(
            self._before_finish()[..., None], self.normal_forces(), torch.inf
        )
        forces = forces.movedim(-1, 1).flatten(0, 1)  # steps and axles together
        return -torch.logsumexp(-sharpness * forces, dim=0) / sharpness

    def energy(self) -> dict[str, torch.Tensor]:
        """The energy budget at each time step.

        Returns the kinetic and potential energy, their sum (the mechanical energy),
        the energy dissipated so far by drag and friction, and the total of the
        mechanical and dissipated energy, which stays constant up to the error of
        the time integration.
        """
        equations = self.equations
        kinetic, potential = equations.energy(self.s, self.v)
        dissipated = self._integrate(equations.power(self.s, self.v))
        mechanical = kinetic + potential
        return {
            "kinetic": kinetic,
            "potential": potential,
            "mechanical": mechanical,
            "dissipated": dissipated,
            "total": mechanical + dissipated,
        }

    def energy_by_term(self) -> dict[str, torch.Tensor]:
        """The energy of each physics term at each time step, by name.

        Kinetic and potential terms give their energy.  Dissipative terms give the
        energy they have dissipated so far.  The kinetic terms add up to the kinetic
        energy of [`energy`][pwdsim.simulation.Run.energy], and likewise for the
        potential and dissipated energy.
        """
        contributions = self.equations.by_term(self.s, self.v)
        for term in self.equations.terms:
            if isinstance(term, DissipativeTerm):
                contributions[term.name] = self._integrate(contributions[term.name])
        return contributions

    def _integrate(self, power):
        """Cumulative trapezoid rule integral of a power over time."""
        steps = (
            0.5
            * (power[1:] + power[:-1])
            * torch.diff(self.times).reshape((-1,) + (1,) * (power.ndim - 1))
        )
        return torch.cat([torch.zeros_like(power[:1]), torch.cumsum(steps, 0)])

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
