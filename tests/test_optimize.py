import numpy as np
import pytest
import torch
from test_simulation import car

import pwdsim
from pwdsim import constraints
from pwdsim.optimization import DesignProblem
from pwdsim.units import INCH, OUNCE

# A CG bound box, and the same with the height fixed so only the CG along varies
CG_BOUNDS = ((0.0, 0.2 * INCH), (4.0 * INCH, 0.8 * INCH))
CG_ALONG = ((0.25 * INCH, 0.4 * INCH), (2.0 * INCH, 0.4 * INCH))


@pytest.fixture(scope="module")
def track35():
    return pwdsim.besttrack(35)


def simulation(track, c, **kwargs):
    return pwdsim.Simulation(track, c, dt=1e-3, duration=2.6, **kwargs)


def names(problem):
    return [c.name for c in problem.constraints]


class TestSetup:
    def test_resolve_names(self, track35):
        problem = DesignProblem(simulation(track35, car()), ["cg", "mass_"])
        assert [v.name for v in problem.variables] == ["cg_", "mass_"]
        with pytest.raises(ValueError, match="no parameter"):
            DesignProblem(simulation(track35, car()), ["color"])

    def test_default_bounds(self, track35):
        c = car()
        problem = DesignProblem(simulation(track35, c), {"mass": None})
        (v,) = problem.variables
        torch.testing.assert_close(v.lower, c.mass.detach() / 2)
        torch.testing.assert_close(v.upper, c.mass.detach() * 2)

    @pytest.mark.parametrize(
        "variables, message",
        [
            ({"cg": None}, None),  # fine: both components are nonzero
            ({"cg": ((0.0, 0.0), (0.5 * INCH, 1.0 * INCH))}, "outside its bounds"),
            ({"mass": (6 * OUNCE, 4 * OUNCE)}, "must not exceed"),
            ({"mass": (5 * OUNCE, 5 * OUNCE)}, "fix every element"),
            ({}, "at least one"),
        ],
    )
    def test_bounds_checks(self, track35, variables, message):
        sim = simulation(track35, car(mass=5 * OUNCE))
        if message is None:
            DesignProblem(sim, variables)
        else:
            with pytest.raises(ValueError, match=message):
                DesignProblem(sim, variables)

    def test_zero_value_needs_bounds(self, track35):
        sim = simulation(track35, car(cg=(1.0 * INCH, 0.0)))
        with pytest.raises(ValueError, match="zero value"):
            DesignProblem(sim, ["cg"])

    def test_batch_car(self, track35):
        c = car(mass=torch.tensor([4.0, 5.0]) * OUNCE)
        with pytest.raises(ValueError, match="single car"):
            DesignProblem(simulation(track35, c), ["mass"])

    def test_scaling_round_trip(self, track35):
        c = car()
        problem = DesignProblem(
            simulation(track35, c), {"cg": CG_ALONG, "mass": (3 * OUNCE, 6 * OUNCE)}
        )
        assert problem.size == 2  # the CG height is fixed
        x = problem.to_x()
        np.testing.assert_allclose(
            x, [(1.0 - 0.25) / 1.75, (c.mass.item() / OUNCE - 3) / 3]
        )
        problem.set_x(np.array([0.0, 1.0]))
        torch.testing.assert_close(
            c.cg.detach(), torch.tensor([0.25, 0.4], dtype=torch.float64) * INCH
        )
        assert c.mass.item() == pytest.approx(6 * OUNCE)
        problem.set_x(x)
        torch.testing.assert_close(problem.to_x(), x, check_dtype=False)

    def test_lift_off_constraint(self, track35):
        sim = simulation(track35, car())
        assert names(DesignProblem(sim, ["mass"])) == ["lift_off"]
        assert names(DesignProblem(sim, ["mass"], lift_off_tolerance=None)) == []
        problem = DesignProblem(
            sim, ["mass"], constraints.bsa_rules(), lift_off_tolerance=0.2
        )
        assert names(problem) == ["max_mass", "cg_between_axles", "lift_off"]
        assert problem.constraints[-1].lower == 0.2


class TestEvaluation:
    @pytest.fixture
    def problem(self, track35):
        c = car(mass=4 * OUNCE)
        for p in c.parameters():
            p.requires_grad_(False)
        c.cg_.requires_grad_(True)
        c.mass_.requires_grad_(True)
        return DesignProblem(
            simulation(track35, c),
            {"cg": CG_BOUNDS, "mass": (3 * OUNCE, 6 * OUNCE)},
            constraints.bsa_rules(),
        )

    def test_gradients(self, problem):
        """Objective and constraint gradients against finite differences in x."""
        x = problem.to_x()
        evaluation = problem.evaluate(x)
        gradient = evaluation.gradient.copy()
        jacobians = [j.copy() for j in evaluation.jacobians]
        h = 1e-6
        for i in range(problem.size):
            step = np.zeros_like(x)
            step[i] = h
            plus, minus = problem.evaluate(x + step), problem.evaluate(x - step)
            fd = (plus.objective - minus.objective) / (2 * h)
            assert gradient[i] == pytest.approx(fd, rel=1e-4, abs=1e-6)
            for k, jacobian in enumerate(jacobians):
                fd = (plus.constraints[k] - minus.constraints[k]) / (2 * h)
                np.testing.assert_allclose(jacobian[:, i], fd, rtol=1e-4, atol=1e-6)

    def test_cache(self, problem):
        x = problem.to_x()
        problem.evaluate(x)
        problem.evaluate(x.copy())
        assert problem.evaluations == 1
        problem.evaluate(x + 0.01)
        assert problem.evaluations == 2

    def test_did_not_finish(self, track35):
        sim = pwdsim.Simulation(track35, car(), dt=1e-3, duration=0.5)
        problem = DesignProblem(sim, ["mass"])
        with pytest.raises(RuntimeError, match="duration"):
            problem.evaluate(problem.to_x())


class TestOptimize:
    def test_cg_and_mass(self, track35):
        c = car(mass=4 * OUNCE)
        c.body_inertia_.requires_grad_(False)
        before = {name: p.detach().clone() for name, p in c.named_parameters()}
        result = pwdsim.optimize(
            simulation(track35, c),
            {"cg": CG_BOUNDS, "mass": (3 * OUNCE, 6 * OUNCE)},
            constraints=constraints.bsa_rules(),
        )
        assert result.success
        assert result.final_objective < result.initial_objective - 10
        # The mass goes to the rule limit, and the CG back to the lift-off limit
        assert c.mass.item() == pytest.approx(5 * OUNCE, rel=1e-3)
        assert result.constraints["lift_off"].item() == pytest.approx(0.0, abs=0.01)
        assert 0.25 * INCH < c.cg[0].item() < 1.0 * INCH
        # Everything else is unchanged, and the gradient flags are restored
        for name, p in c.named_parameters():
            if name not in ("cg_", "mass_"):
                torch.testing.assert_close(p.detach(), before[name])
        assert not c.body_inertia_.requires_grad
        assert c.wheelbase_.requires_grad
        # The result keeps the starting car and the history
        assert result.initial_car.mass.item() == pytest.approx(4 * OUNCE)
        assert result.history[0]["objective"] == result.initial_objective
        assert len(result.history) > 2
        assert "mass_" in result.summary({"mass_": (OUNCE, "oz")})

    def test_without_lift_off(self, track35):
        c = car()
        pwdsim.optimize(
            simulation(track35, c), {"cg": CG_ALONG}, lift_off_tolerance=None
        )
        # Nothing stops the CG going all the way back
        assert c.cg[0].item() == pytest.approx(0.25 * INCH, abs=1e-3 * INCH)
        assert c.cg[1].item() == pytest.approx(0.4 * INCH)

    def test_lift_off_tolerance(self, track35):
        cgs = []
        for tolerance in (0.0, 0.1):
            c = car()
            result = pwdsim.optimize(
                simulation(track35, c), {"cg": CG_ALONG}, lift_off_tolerance=tolerance
            )
            assert result.constraints["lift_off"].item() >= tolerance - 1e-3
            cgs.append(c.cg[0].item())
        # A bigger margin keeps the CG further forward
        assert cgs[1] > cgs[0] + 0.05 * INCH

    def test_slsqp(self, track35):
        c = car()
        result = pwdsim.optimize(
            simulation(track35, c), {"cg": CG_ALONG}, method="SLSQP"
        )
        assert result.success
        assert result.constraints["lift_off"].item() == pytest.approx(0.0, abs=0.01)

    @pytest.mark.parametrize("method", ["trust-constr", "SLSQP"])
    def test_equality_constraint(self, track35, method):
        c = car(mass=4 * OUNCE)
        exact_mass = constraints.Constraint(
            fun=lambda car, run: car.mass,
            lower=4.5 * OUNCE,
            upper=4.5 * OUNCE,
            name="exact_mass",
            scale=OUNCE,
            linear=True,
        )
        result = pwdsim.optimize(
            simulation(track35, c),
            {"mass": (3 * OUNCE, 6 * OUNCE)},
            constraints=[exact_mass],
            lift_off_tolerance=None,
            method=method,
        )
        assert c.mass.item() == pytest.approx(4.5 * OUNCE, rel=1e-3)
        assert result.constraints["exact_mass"].item() == pytest.approx(
            4.5 * OUNCE, rel=1e-3
        )

    def test_unknown_method(self, track35):
        with pytest.raises(ValueError, match="method"):
            pwdsim.optimize(simulation(track35, car()), ["mass"], method="genetic")
