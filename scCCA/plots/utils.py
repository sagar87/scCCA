from typing import Callable, List, Union

import matplotlib.colors as co
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData


def set_up_cmap(array: np.ndarray, cmap: str = "RdBu"):
    vmin = array.min()
    vmax = array.max()

    if vmin < 0 and vmax > 0:
        norm = co.TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0)
    elif vmin < 0 and vmax < 0:
        # print('min color')
        norm = co.Normalize(vmin=vmin, vmax=0)
        cmap = co.LinearSegmentedColormap.from_list("name", [cmap(-0.001), "w"])
    else:
        # print('max color')
        cmap = co.LinearSegmentedColormap.from_list("name", ["w", cmap(1.001)])
        norm = co.Normalize(vmin=0, vmax=vmax)

    return cmap, norm


def rand_jitter(arr, stdev=1):
    # stdev = .01 * (max(arr) - min(arr))
    # print(stdev)
    return arr + np.random.randn(len(arr)) * stdev


def set_up_subplots(num_plots, ncols=4, width=4, height=3):
    """
    Set up subplots for plotting multiple factors.

    Parameters
    ----------
    num_plots : int
        The total number of plots to be created.
    ncols : int, optional
        The number of columns in the subplot grid. Default is 4.
    width : int, optional
        The width factor for each subplot. Default is 4.
    height : int, optional
        The height factor for each subplot. Default is 3.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object representing the entire figure.
    axes : numpy.ndarray of matplotlib.axes.Axes
        An array of Axes objects representing the subplots. The shape of the
        array is determined by the number of rows and columns.

    Notes
    -----
    - If `num_plots` is 1, a single subplot is created and returned as `fig` and `ax`.
    - If `num_plots` is less than `ncols`, a single row of subplots is created.
    - If `num_plots` is greater than or equal to `ncols`, a grid of subplots is created.
    - The `axes` array may contain empty subplots if the number of plots is less than the total available subplots.

    Examples
    --------
    # Create a single subplot
    fig, ax = set_up_subplots(num_plots=1)

    # Create a grid of subplots with 2 rows and 4 columns
    fig, axes = set_up_subplots(num_plots=8)

    # Create a single row of subplots with 1 row and 3 columns
    fig, axes = set_up_subplots(num_plots=3, ncols=3)
    """

    if num_plots == 1:
        fig, ax = plt.subplots()
        return fig, ax

    nrows, reminder = divmod(num_plots, ncols)

    if num_plots < ncols:
        nrows = 1
        ncols = num_plots
    else:
        nrows, reminder = divmod(num_plots, ncols)

        if nrows == 0:
            nrows = 1
        if reminder > 0:
            nrows += 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows))
    _ = [ax.axis("off") for ax in axes.flatten()[num_plots:]]
    return fig, axes


def set_up_plot(
    adata: AnnData,
    model_key: str,
    instances: Union[int, List[int], None],
    func: Callable,
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    ax: Union[plt.Axes, None] = None,
    **kwargs
):
    if isinstance(instances, list):
        num_plots = len(instances)
        fig, ax = set_up_subplots(num_plots, ncols=ncols, width=width, height=height)
    elif isinstance(instances, int):
        num_plots = 1
        if ax is None:
            fig, ax = plt.subplots(1, 1)
    else:
        model_dict = adata.uns[model_key]
        if model_key == "pca":
            num_plots = model_dict["variance"].shape[0]
        else:
            num_plots = model_dict["model"]["num_factors"]

        instances = [i for i in range(num_plots)]
        fig, ax = set_up_subplots(num_plots, ncols=ncols, width=width, height=height)

    if num_plots == 1:
        if isinstance(instances, list):
            instances = instances[0]

        func(adata, model_key, instances, ax=ax, **kwargs)
    else:
        for i, ax_i in zip(instances, ax.flatten()):
            func(adata, model_key, i, ax=ax_i, **kwargs)

    return ax
