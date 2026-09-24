import matplotlib
import pytest
import torch
from test_kinematics import make_car

from pwdsim.plotting import plot_car, plot_energy, plot_run, plot_runs, plot_track
from pwdsim.simulation import Simulation
from pwdsim.track import besttrack
from pwdsim.units import INCH

matplotlib.use("Agg")


def test_plot_track():
    fig = plot_track(besttrack(), unit="ft", equal_aspect=True)
    ax_profile = fig.axes[0]
    assert ax_profile.get_xlabel() == "x (ft)"
    assert len(fig.axes) == 3


def test_plot_track_bad_unit():
    with pytest.raises(ValueError):
        plot_track(besttrack(), unit="furlong")


@pytest.fixture(scope="module")
def run():
    cg = torch.tensor([[1.0, 0.4], [1.5, 0.4]]) * INCH
    sim = Simulation(besttrack(35), make_car(cg=cg), dt=2e-3, duration=2.5)
    return sim()


def test_plot_run(run):
    fig = plot_run(run, labels=["back", "forward"], unit="ft")
    assert len(fig.axes) == 2
    with pytest.raises(ValueError):
        plot_run(run, labels=["one"])


def test_plot_energy(run):
    fig = plot_energy(run, index=1)
    assert fig.axes[0].get_ylabel() == "energy (J)"


def test_plot_car(run):
    fig = plot_car(run.simulation, [2.2, 2.6, 3.0], index=1)
    assert len(fig.axes[0].patches) == 6


def test_plot_runs(run):
    fig = plot_runs([run, run], labels=["a", "b", "c", "d"])
    assert len(fig.axes[0].get_lines()) == 8  # speeds and finish lines
    with pytest.raises(ValueError):
        plot_runs([run, run], labels=["a", "b"])
