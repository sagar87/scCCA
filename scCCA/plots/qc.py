from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import variation
from sklearn.metrics import mean_squared_error

from ..utils import get_protein_counts, get_rna_counts


def _extract_counts(adata, layers_key, protein_obsm_key):
    if protein_obsm_key is not None:
        counts = get_protein_counts(adata, protein_obsm_key)
    else:
        counts = get_rna_counts(adata, layers_key)

    return counts


def disp(
    adata,
    model_key: str = "scpca",
    layers_key: Union[str, None] = None,
    protein_obsm_key: Union[str, None] = None,
    cmap: str = "viridis",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the fitted dispersion against the coefficient of variation RNA or
    protein counts.

    Parameters
    ----------
    adata: AnnData
        AnnData object.
    model_key: str, optional (default: "scpca")
        Key for the fitted model.
    layers_key: str, optional (default: None)
        If `layers_key` is None, then the raw counts are extracted from `adata.X`.
        Otherwise, the counts are extracted from `adata.layers[layers_key]`.
    protein_obsm_key: str, optional (default: None)
        Key for protein counts in `adata.obsm`. Providing `protein_obsm_key`
        overrides `layers_key`, i.e. protein counts are plotted.
    cmap: str, optional (default: "viridis")
        Colormap for the scatterplot. Color represents the mean of the counts.
    ax: matplotlib.axes.Axes, optional (default: None)
        Axes to plot on. If None, then a new figure is created.

    Returns
    -------
    ax: matplotlib.axes.Axes
    """
    # Extract counts
    counts = _extract_counts(adata, layers_key, protein_obsm_key)
    posterior_key = "α_prot" if protein_obsm_key is not None else "α_rna"

    if ax is None:
        plt.scatter(
            adata.uns[model_key]["posterior"][posterior_key],
            variation(counts, axis=0),
            c=counts.mean(0),
            cmap="viridis",
            norm=LogNorm(),
        )
        ax = plt.gca()
    else:
        ax.scatter(
            adata.uns[model_key]["posterior"][posterior_key],
            variation(counts, axis=0),
            c=counts.mean(0),
            cmap="viridis",
            norm=LogNorm(),
        )

    plt.colorbar()
    # ax.plot(np.linspace(1, 60), np.linspace(1, 60), color='C1')
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("CV")
    return ax


def qc_hist(
    adata,
    model_key: str = "scpca",
    layers_key: Union[str, None] = None,
    protein_obsm_key: Union[str, None] = None,
    cmap: str = "viridis",
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plots a 2D histogram of the predicted counts against the true counts.

    Parameters
    ----------
    adata: AnnData
        AnnData object.
    model_key: str, optional (default: "scpca")
        Key for the fitted model.
    layers_key: str, optional (default: None)
        If `layers_key` is None, then the raw counts are extracted from `adata.X`.
        Otherwise, the counts are extracted from `adata.layers[layers_key]`.
    protein_obsm_key: str, optional (default: None)
        Key for protein counts in `adata.obsm`. Providing `protein_obsm_key`
        overrides `layers_key`, i.e. protein counts are plotted.
    cmap: str, optional (default: "viridis")
        Colormap for the scatterplot. Color represents the mean of the counts.
    ax: matplotlib.axes.Axes, optional (default: None)
        Axes to plot on. If None, then a new figure is created.

    Returns
    -------
    ax: matplotlib.axes.Axes
    """
    # Extract counts
    counts = _extract_counts(adata, layers_key, protein_obsm_key)
    posterior_key = "μ_prot" if protein_obsm_key is not None else "μ_rna"
    predicted_counts = adata.uns[model_key]["posterior"][posterior_key]

    if ax is None:
        plt.hist2d(
            np.log10(counts.reshape(-1) + 1),
            np.log10(predicted_counts.reshape(-1) + 1),
            bins=50,
            norm=LogNorm(),
            cmap=cmap,
        )
        ax = plt.gca()
    else:
        ax.hist2d(
            np.log10(counts.reshape(-1) + 1),
            np.log10(predicted_counts.reshape(-1) + 1),
            bins=50,
            norm=LogNorm(),
            cmap=cmap,
        )
    plt.colorbar()
    max_val = np.max([*ax.get_xlim(), *ax.get_ylim()])
    min_val = np.min([*ax.get_xlim(), *ax.get_ylim()])
    # print(max_val)
    ax.set_xlim([min_val, max_val])
    ax.set_ylim([min_val, max_val])
    ax.plot(
        np.linspace(min_val, max_val),
        np.linspace(min_val, max_val),
        color="w",
        linewidth=2,
    )
    ax.set_aspect("equal")
    ax.set_ylabel(r"Predicted count ($\log_{10}(x+1)$ scaled)", fontsize=12)
    ax.set_xlabel(r"True count ($\log_{10}(x+1)$ scaled)", fontsize=12)
    rmse = mean_squared_error(counts, predicted_counts)
    ax.set_title(f"RMSE {rmse:.2f}")
    return ax
