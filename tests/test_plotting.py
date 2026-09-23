import matplotlib
import pytest

from pwdsim.plotting import plot_track
from pwdsim.track import besttrack

matplotlib.use("Agg")


def test_plot_track():
    fig = plot_track(besttrack(), unit="ft", equal_aspect=True)
    ax_profile = fig.axes[0]
    assert ax_profile.get_xlabel() == "x (ft)"
    assert len(fig.axes) == 3


def test_plot_track_bad_unit():
    with pytest.raises(ValueError):
        plot_track(besttrack(), unit="furlong")
