import matplotlib.pyplot as plt
import numpy as np

from scCCA.plots.utils import set_up_subplots


def test_set_up_subplots_single_plot():
    fig, ax = set_up_subplots(num_plots=1)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)


def test_set_up_subplots_multiple_plots():
    num_plots = 8
    fig, axes = set_up_subplots(num_plots=num_plots)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2, 4)  # Adjust based on the expected number of rows and columns


def test_set_up_subplots_less_than_ncols():
    num_plots = 3
    fig, axes = set_up_subplots(num_plots=num_plots)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (3,)  # Adjust based on the expected number of rows and columns
