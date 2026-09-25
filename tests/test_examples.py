"""Run the example notebooks (stored as jupytext percent scripts) end to end.

These are slow, so they only run with ``pytest --runslow``.
"""

import runpy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

matplotlib.use("Agg")

EXAMPLES = sorted((Path(__file__).parents[1] / "docs" / "examples").glob("*.py"))


@pytest.mark.slow
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_runs(path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    runpy.run_path(str(path), run_name="__main__")
    plt.close("all")
