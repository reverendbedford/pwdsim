"""Design optimization.

[`optimize`][pwdsim.optimization.optimize] tunes chosen car parameters to minimize
the finish time, subject to bounds and constraints.  It uses SciPy's `trust-constr`
method, a trust region interior point method that builds a BFGS approximation to the
Hessian from the gradients, which come from the adjoint method.

The optimizer works with the design variables scaled to $[0, 1]$ between their
bounds, $p = l + x (u - l)$, which puts parameters of very different sizes on an
equal footing.
"""

import copy
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize
import torch
from torch import nn

from pwdsim.car import Car
from pwdsim.constraints import Constraint, lift_off
from pwdsim.simulation import DidNotFinishWarning, LiftOffWarning, Run, Simulation
from pwdsim.types import DTYPE

BoundValue = float | Sequence[float] | torch.Tensor
"""A bound for a design variable: a scalar or one value per element."""

Bounds = tuple[BoundValue, BoundValue]
"""Lower and upper bounds for a design variable."""

FEASIBLE = 1e-6
"""The largest constraint violation, in the scaled units of each constraint, for a
design to count as feasible when checking whether the optimization has stalled."""

DEFAULT_OPTIONS = {
    "trust-constr": {
        "maxiter": 50,
        "gtol": 1e-3,
        "xtol": 1e-4,
        # A small barrier lets the interior point method converge onto the bounds
        "initial_barrier_parameter": 1e-3,
    },
    "SLSQP": {"maxiter": 50, "ftol": 1e-6},
}
"""Default optimizer settings for each method."""


@dataclass
class Variable:
    """A design variable: a car parameter with bounds.

    Elements whose lower and upper bounds are equal are fixed at that value, so
    only some elements of a parameter can be tuned, e.g. one component of `cg`.

    Attributes:
        name: name of the parameter on the car, e.g. `"cg_"`.
        parameter: the parameter.
        lower: lower bound, with the shape of the parameter.
        upper: upper bound, with the shape of the parameter.
    """

    name: str
    parameter: nn.Parameter
    lower: torch.Tensor
    upper: torch.Tensor

    @property
    def free(self) -> torch.Tensor:
        """Mask of the elements that are tuned, flattened."""
        return (self.upper > self.lower).reshape(-1)

    @property
    def size(self) -> int:
        """Number of tuned elements."""
        return int(self.free.sum())


def _resolve(car: Car, name: str) -> tuple[str, nn.Parameter]:
    """Find the car parameter for a variable name, allowing the property names of
    [`SimpleCar`][pwdsim.car.SimpleCar] (`"cg"` for the `cg_` parameter)."""
    parameters = dict(car.named_parameters())
    for candidate in (name, f"{name}_"):
        if candidate in parameters:
            return candidate, parameters[candidate]
    raise ValueError(
        f"The car has no parameter {name!r}; its parameters are "
        + ", ".join(parameters)
    )


def _bounds(name: str, parameter: nn.Parameter, bounds: Bounds | None):
    value = parameter.detach()
    if bounds is None:
        if torch.any(value == 0):
            raise ValueError(
                f"Can't choose default bounds for {name!r}, which has a zero value: "
                "give its bounds explicitly"
            )
        lower = torch.minimum(value / 2, value * 2)
        upper = torch.maximum(value / 2, value * 2)
    else:
        lower, upper = (
            torch.as_tensor(b, dtype=DTYPE).expand(value.shape).clone() for b in bounds
        )
    if torch.any(lower > upper):
        raise ValueError(
            f"The lower bounds of {name!r} must not exceed the upper bounds"
        )
    if torch.all(lower == upper):
        raise ValueError(f"The bounds of {name!r} fix every element; nothing to tune")
    if torch.any(value < lower) or torch.any(value > upper):
        raise ValueError(f"The current value of {name!r} is outside its bounds")
    return lower, upper


@dataclass
class _Evaluation:
    x: np.ndarray
    objective: float
    gradient: np.ndarray
    constraints: list[np.ndarray]
    jacobians: list[np.ndarray]


@dataclass
class OptimizationResult:
    """The results of an optimization.

    Attributes:
        success: whether the optimizer converged.
        message: the optimizer's description of how it finished.
        car: the optimized car, the same object that was optimized.
        initial_car: a copy of the car before the optimization.
        initial: the design variables before, in SI units, by parameter name.
        final: the design variables after, in SI units, by parameter name.
        initial_objective: the objective before (by default the finish time in ms).
        final_objective: the objective after.
        constraints: the values of the constraints at the optimum, by name, in the
            units of each constraint.
        history: the objective, design variables, and constraint values at each
            iteration.
        scipy_result: the full result from `scipy.optimize.minimize`.
    """

    success: bool
    message: str
    car: Car
    initial_car: Car
    initial: dict[str, torch.Tensor]
    final: dict[str, torch.Tensor]
    initial_objective: float
    final_objective: float
    constraints: dict[str, torch.Tensor]
    history: list[dict] = field(default_factory=list)
    scipy_result: scipy.optimize.OptimizeResult | None = None

    def summary(self, units: Mapping[str, tuple[float, str]] | None = None) -> str:
        """A table of the design variables before and after, and the objective.

        Args:
            units: optional units to report each variable in, by parameter name, as
                `(size of the unit in SI, label)`, e.g. `{"cg_": (INCH, "in")}`.
        """
        units = dict(units or {})
        lines = [f"{'variable':>24}  {'before':>20}  {'after':>20}"]
        for name, before in self.initial.items():
            scale, label = units.get(name, (1.0, "SI"))
            after = self.final[name]

            def fmt(v, scale=scale):
                return ", ".join(f"{x:.4g}" for x in (v / scale).reshape(-1).tolist())

            lines.append(
                f"{name + ' (' + label + ')':>24}  {fmt(before):>20}  {fmt(after):>20}"
            )
        lines.append(
            f"{'objective':>24}  {self.initial_objective:>20.4f}  "
            f"{self.final_objective:>20.4f}"
        )
        for name, value in self.constraints.items():
            values = ", ".join(f"{v:.4g}" for v in value.reshape(-1).tolist())
            lines.append(f"{'constraint ' + name:>24}  {'':>20}  {values:>20}")
        lines.append(f"{'converged':>24}  {'':>20}  {str(self.success):>20}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


class DesignProblem:
    """The problem of minimizing an objective over some car parameters, subject to
    bounds and constraints.

    Most users will call [`optimize`][pwdsim.optimization.optimize] instead, which
    sets up the problem and solves it.

    Args:
        simulation: the simulation of the car on the track, with its physics.
        variables: the car parameters to tune, as a list of names (with default
            bounds) or a dict of names and bounds.  See
            [`optimize`][pwdsim.optimization.optimize].
        constraints: constraints on the design.
        lift_off_tolerance: the smallest allowed normal force on the wheels, in
            newtons, for the [`lift_off`][pwdsim.constraints.lift_off] constraint,
            or `None` to leave it out.
        objective: the function of the car and its run to minimize.  By default the
            finish time, in milliseconds.
    """

    def __init__(
        self,
        simulation: Simulation,
        variables: Iterable[str] | Mapping[str, Bounds | None],
        constraints: Iterable[Constraint] = (),
        lift_off_tolerance: float | None = 0.0,
        objective: Callable[[Car, Run], torch.Tensor] | None = None,
    ):
        self.simulation = simulation
        self.car = simulation.car
        if self.car.batch_shape != ():
            raise ValueError("Optimization needs a single car, not a batch of cars")
        if not isinstance(variables, Mapping):
            variables = dict.fromkeys(variables)
        if not variables:
            raise ValueError("Choose at least one design variable")
        self.variables = []
        for name, bounds in variables.items():
            parameter_name, parameter = _resolve(self.car, name)
            lower, upper = _bounds(parameter_name, parameter, bounds)
            self.variables.append(Variable(parameter_name, parameter, lower, upper))
        self.constraints = list(constraints)
        if lift_off_tolerance is not None:
            self.constraints.append(lift_off(lift_off_tolerance))
        self.objective = objective or (lambda car, run: run.finish_time * 1000)
        self.evaluations = 0
        self._cache: _Evaluation | None = None

    @property
    def size(self) -> int:
        """Number of scalar design variables."""
        return sum(v.size for v in self.variables)

    def to_x(self) -> np.ndarray:
        """The current design variables, scaled to $[0, 1]$ between their bounds."""
        return np.concatenate(
            [
                ((v.parameter.detach() - v.lower) / (v.upper - v.lower))
                .reshape(-1)[v.free]
                .numpy()
                for v in self.variables
            ]
        )

    def set_x(self, x: np.ndarray):
        """Set the car parameters from the scaled design variables."""
        offset = 0
        with torch.no_grad():
            for v in self.variables:
                scaled = torch.zeros(v.parameter.numel(), dtype=DTYPE)
                scaled[v.free] = torch.as_tensor(
                    x[offset : offset + v.size], dtype=DTYPE
                )
                value = v.lower + scaled.reshape(v.lower.shape) * (v.upper - v.lower)
                v.parameter.copy_(value)
                offset += v.size

    def _scaled_gradient(self, value, retain_graph):
        """Gradient of a scalar with respect to the scaled design variables."""
        grads = torch.autograd.grad(
            value,
            [v.parameter for v in self.variables],
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return np.concatenate(
            [
                (
                    (torch.zeros_like(v.parameter) if g is None else g)
                    * (v.upper - v.lower)
                )
                .reshape(-1)[v.free]
                .detach()
                .numpy()
                for v, g in zip(self.variables, grads, strict=True)
            ]
        )

    def evaluate(self, x: np.ndarray) -> _Evaluation:
        """Simulate the design at `x` and compute the objective, the constraints,
        and their gradients.  The results are cached, so asking for the objective
        and constraints at the same `x` runs one simulation."""
        x = np.asarray(x, dtype=float)
        if self._cache is not None and np.array_equal(self._cache.x, x):
            return self._cache
        self.set_x(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LiftOffWarning)
            warnings.simplefilter("error", DidNotFinishWarning)
            try:
                run = self.simulation()
            except DidNotFinishWarning as error:
                raise RuntimeError(
                    "The car didn't finish during the optimization: increase the "
                    "duration of the simulation"
                ) from error
        self.evaluations += 1

        objective = self.objective(self.car, run)
        if not torch.isfinite(objective):
            raise RuntimeError(f"The objective is not finite: {objective.item()}")
        outputs = [objective] + [c(self.car, run) / c.scale for c in self.constraints]
        count = 1 + sum(o.numel() for o in outputs[1:])
        rows, done = [], 0
        for output in outputs:
            for value in output.reshape(-1):
                done += 1
                rows.append(self._scaled_gradient(value, retain_graph=done < count))
        gradient, rows = rows[0], rows[1:]

        jacobians, offset = [], 0
        for output in outputs[1:]:
            jacobians.append(np.array(rows[offset : offset + output.numel()]))
            offset += output.numel()
        self._cache = _Evaluation(
            x=x.copy(),
            objective=objective.item(),
            gradient=gradient,
            constraints=[o.detach().reshape(-1).numpy() for o in outputs[1:]],
            jacobians=jacobians,
        )
        return self._cache

    def _record(self, x, history):
        evaluation = self.evaluate(x)
        violation = self.violation(x)
        best = history[-1]["best_feasible"] if history else math.inf
        if violation <= FEASIBLE:
            best = min(best, evaluation.objective)
        history.append(
            {
                "iteration": len(history),
                "objective": evaluation.objective,
                "violation": violation,
                "best_feasible": best,
                "variables": {
                    v.name: v.parameter.detach().clone() for v in self.variables
                },
                "constraints": {
                    c.name: torch.as_tensor(values * c.scale)
                    for c, values in zip(
                        self.constraints, evaluation.constraints, strict=True
                    )
                },
            }
        )

    def _scipy_constraints(self, method):
        constraints = []
        for i, c in enumerate(self.constraints):
            lower, upper = c.lower / c.scale, c.upper / c.scale

            def fun(x, i=i):
                return self.evaluate(x).constraints[i]

            def jac(x, i=i):
                return self.evaluate(x).jacobians[i]

            if method == "trust-constr":
                if c.linear:

                    def hess(x, v):
                        return np.zeros((self.size, self.size))

                else:
                    hess = scipy.optimize.BFGS()
                constraints.append(
                    scipy.optimize.NonlinearConstraint(
                        fun, lower, upper, jac=jac, hess=hess
                    )
                )
                continue
            if lower == upper:
                constraints.append(
                    {
                        "type": "eq",
                        "fun": lambda x, f=fun, b=lower: f(x) - b,
                        "jac": jac,
                    }
                )
                continue
            if math.isfinite(lower):
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, f=fun, b=lower: f(x) - b,
                        "jac": jac,
                    }
                )
            if math.isfinite(upper):
                constraints.append(
                    {
                        "type": "ineq",
                        "fun": lambda x, f=fun, b=upper: b - f(x),
                        "jac": lambda x, j=jac: -j(x),
                    }
                )
        return constraints

    def violation(self, x: np.ndarray) -> float:
        """The largest violation of any constraint at `x`, in the scaled units of
        each constraint."""
        evaluation = self.evaluate(x)
        worst = 0.0
        for c, values in zip(self.constraints, evaluation.constraints, strict=True):
            if values.size:
                below = np.max(c.lower / c.scale - values)
                above = np.max(values - c.upper / c.scale)
                worst = max(worst, below, above)
        return float(worst)

    def solve(
        self,
        method: str = "trust-constr",
        options: Mapping | None = None,
        stall_tolerance: float | None = 0.01,
        stall_iterations: int = 5,
    ) -> OptimizationResult:
        """Run the optimizer from the current design.

        Args:
            method: `"trust-constr"` (the default) or `"SLSQP"`.
            options: optimizer settings, passed to `scipy.optimize.minimize`,
                overriding [`DEFAULT_OPTIONS`][pwdsim.optimization.DEFAULT_OPTIONS].
            stall_tolerance: stop once the objective (the finish time in ms, by
                default) has improved by less than this over the last
                `stall_iterations` iterations, with the constraints satisfied.
                `None` leaves the stopping to the optimizer's own tolerances.
            stall_iterations: the number of iterations to look back over.

        Returns:
            The results.  The optimum is also written into the car.
        """
        if method not in DEFAULT_OPTIONS:
            raise ValueError(f"Unknown method {method!r}")
        options = DEFAULT_OPTIONS[method] | dict(options or {})
        initial_car = copy.deepcopy(self.car)

        # Only the design variables need gradients
        flags = {name: p.requires_grad for name, p in self.car.named_parameters()}
        design = {v.name for v in self.variables}
        for name, p in self.car.named_parameters():
            p.requires_grad_(name in design)

        history = []
        try:
            x0 = self.to_x()
            self._record(x0, history)
            initial_objective = history[0]["objective"]

            stalled = False

            def callback(x, *args):
                nonlocal stalled
                self._record(x, history)
                if stall_tolerance is None or len(history) <= stall_iterations:
                    return
                # Compare the best feasible objective now with the best feasible
                # objective stall_iterations ago
                best = [h["best_feasible"] for h in history]
                before, now = best[-(stall_iterations + 1)], best[-1]
                if (
                    math.isfinite(before)
                    and before - now < stall_tolerance
                    and history[-1]["violation"] <= FEASIBLE
                ):
                    stalled = True
                    raise StopIteration

            kwargs = {}
            if method == "trust-constr":
                kwargs["hess"] = scipy.optimize.BFGS()
            result = scipy.optimize.minimize(
                lambda x: self.evaluate(x).objective,
                x0,
                jac=lambda x: self.evaluate(x).gradient,
                method=method,
                bounds=scipy.optimize.Bounds(np.zeros(self.size), np.ones(self.size)),
                constraints=self._scipy_constraints(method),
                options=options,
                callback=callback,
                **kwargs,
            )
            final = self.evaluate(np.clip(result.x, 0.0, 1.0))
            self.set_x(final.x)
        finally:
            for name, p in self.car.named_parameters():
                p.requires_grad_(flags[name])

        if stalled:
            success = True
            message = (
                f"The objective improved by less than {stall_tolerance} over the "
                f"last {stall_iterations} iterations"
            )
        else:
            success, message = bool(result.success), str(result.message)
        return OptimizationResult(
            success=success,
            message=message,
            car=self.car,
            initial_car=initial_car,
            initial={
                v.name: dict(initial_car.named_parameters())[v.name].detach().clone()
                for v in self.variables
            },
            final={v.name: v.parameter.detach().clone() for v in self.variables},
            initial_objective=initial_objective,
            final_objective=final.objective,
            constraints={
                c.name: torch.as_tensor(values * c.scale)
                for c, values in zip(self.constraints, final.constraints, strict=True)
            },
            history=history,
            scipy_result=result,
        )


def optimize(
    simulation: Simulation,
    variables: Iterable[str] | Mapping[str, Bounds | None],
    constraints: Iterable[Constraint] = (),
    lift_off_tolerance: float | None = 0.0,
    objective: Callable[[Car, Run], torch.Tensor] | None = None,
    method: str = "trust-constr",
    options: Mapping | None = None,
    stall_tolerance: float | None = 0.01,
    stall_iterations: int = 5,
) -> OptimizationResult:
    """Tune car parameters to minimize the finish time.

    ```python
    result = pwdsim.optimize(
        pwdsim.Simulation(track, car),
        variables={"cg": ((0.0, 0.1 * INCH), (4.0 * INCH, 1.0 * INCH)), "mass": None},
        constraints=pwdsim.constraints.bsa_rules(),
    )
    print(result.summary())
    ```

    Args:
        simulation: the simulation of the car on the track, with its physics and
            time step.  The car must be a single car, not a batch.
        variables: the car parameters to tune; every other parameter stays fixed.
            Either a list of names, or a dict of names and their bounds.  Names are
            the car's parameter names, and for a
            [`SimpleCar`][pwdsim.car.SimpleCar] its property names (`"cg"`,
            `"mass"`, ...).  Bounds are `(lower, upper)` in SI units, each a scalar
            or one value per element of the parameter (e.g. the two components of
            `cg`).  An element with equal lower and upper bounds is fixed at that
            value, which tunes only part of a parameter.  Variables without bounds,
            or with `None`, get default bounds of half to twice their current
            values.
        constraints: constraints on the design, e.g. from
            [`pwdsim.constraints`][pwdsim.constraints].
        lift_off_tolerance: the smallest allowed normal force on the wheels, in
            newtons, keeping them on the track.  Zero by default; a positive value
            gives a more robust design.  `None` removes the constraint.
        objective: the function of the car and its run to minimize.  By default the
            finish time, in milliseconds.
        method: the SciPy method, `"trust-constr"` (the default) or `"SLSQP"`.
        options: optimizer settings passed to `scipy.optimize.minimize`, overriding
            [`DEFAULT_OPTIONS`][pwdsim.optimization.DEFAULT_OPTIONS], e.g.
            `{"maxiter": 20}`.
        stall_tolerance: stop once the objective has improved by less than this
            (in ms, for the default objective) over the last `stall_iterations`
            iterations, with the constraints satisfied.  The gradients are only
            accurate to about 1%, so the optimizer's own tolerances can be out of
            reach for larger problems, while the finish time has long since stopped
            improving.  `None` switches this off.
        stall_iterations: the number of iterations to look back over.

    Returns:
        The results.  The optimum is also written into the car.
    """
    problem = DesignProblem(
        simulation, variables, constraints, lift_off_tolerance, objective
    )
    return problem.solve(
        method=method,
        options=options,
        stall_tolerance=stall_tolerance,
        stall_iterations=stall_iterations,
    )
