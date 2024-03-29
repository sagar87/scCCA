from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gamma, variation
from sklearn.metrics import mean_squared_error

from ..utils import extract_counts


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
    counts = extract_counts(adata, layers_key, protein_obsm_key)
    posterior_key = "α_prot" if protein_obsm_key is not None else "α_rna"

    if ax is None:
        plt.scatter(
            adata.uns[model_key][posterior_key],
            variation(counts, axis=0),
            c=counts.mean(0),
            cmap="viridis",
            norm=LogNorm(),
        )
        ax = plt.gca()
    else:
        ax.scatter(
            adata.uns[model_key][posterior_key],
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
    colorbar_pos="right",
    colorbar_width="3%",
    orientation="vertical",
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
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = plt.gcf()

    # Extract counts
    counts = extract_counts(adata, layers_key, protein_obsm_key)
    posterior_key = "μ_prot" if protein_obsm_key is not None else "μ_rna"
    if posterior_key == "μ_rna":
        predicted_counts = adata.layers[f"{model_key}_{posterior_key}"]
    else:
        predicted_counts = adata.obsm[f"{model_key}_{posterior_key}"]

    im = ax.hist2d(
        np.log10(counts.reshape(-1) + 1),
        np.log10(predicted_counts.reshape(-1) + 1),
        bins=50,
        norm=LogNorm(),
        cmap=cmap,
    )
    divider = make_axes_locatable(ax)

    cax = divider.append_axes(colorbar_pos, size=colorbar_width, pad=0.1)
    fig.colorbar(im[3], cax=cax, orientation=orientation)

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
    ax.set_ylabel(r"Predicted count ($\log_{10}(x+1)$ scaled)")
    ax.set_xlabel(r"True count ($\log_{10}(x+1)$ scaled)")
    rmse = mean_squared_error(counts, predicted_counts)
    ax.set_title(f"RMSE {rmse:.2f}")
    return ax


def mean_var(
    adata,
    model_key: Union[str, None] = None,
    layers_key: Union[str, None] = None,
    protein_obsm_key: Union[str, None] = None,
    highest: Union[int, None] = None,
    β_rna_mean: float = 3,
    β_rna_sd: float = 1,
    alpha: float = 1.0,
    repel: float = 0.1,
    margin: float = 0.01,
    max_distance: float = 0.1,
    ax: plt.Axes = None,
):

    if ax is None:
        plt.figure()
        ax = plt.gca()

    counts = extract_counts(adata, layers_key, protein_obsm_key)

    def vart(concentration, mean):
        """Computes the expected variance of the Gamma Poission distribution."""
        return concentration / (concentration / mean) ** 2 * (1 + concentration / mean)

    if model_key is not None:
        model_dict = adata.uns[model_key]
        params = model_dict["model"]
        # prior_mean = params["β_rna_mean"]
        β_rna_mean = params["β_rna_mean"]
        β_rna_sd = params["β_rna_sd"]

    a = β_rna_mean**2 / β_rna_sd**2
    b = β_rna_mean / β_rna_sd**2
    -4

    upper = gamma(a, scale=1 / b).ppf(0.975)
    lower = gamma(a, scale=1 / b).ppf(0.025)

    true_mean = np.mean(counts, axis=0)
    true_var = np.var(counts, axis=0)
    theoretical = vart(β_rna_mean, np.logspace(-4, 3, 1000))
    expectation = vart(β_rna_mean, true_mean)

    ax.fill_between(
        np.logspace(-4, 3, 1000),
        vart(lower, np.logspace(-4, 3, 1000)),
        vart(upper, np.logspace(-4, 3, 1000)),
        color="C3",
        alpha=0.2,
    )
    im = ax.scatter(
        true_mean,
        true_var,
        alpha=alpha,
        s=10,
        c=adata.varm[f"{model_key}_α_rna"] if model_key is not None else None,
        cmap="viridis",
    )

    ax.plot(np.logspace(-4, 3), np.logspace(-4, 3), color="C3", label="Identity")
    ax.plot(
        np.logspace(-4, 3, 1000),
        theoretical,
        color="C3",
        linestyle="--",
        label=f"Prior mean {β_rna_mean:.2f}",
    )
    # ax.plot(np.logspace(-3, 3, 1000), vart(upper, np.logspace(-3, 3, 1000)), color='C3', linestyle='--')
    # ax.plot(np.logspace(-3, 3, 1000), vart(lower, np.logspace(-3, 3, 1000)), color='C3', linestyle='--')

    ax.legend()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("Variance")
    ax.set_xlabel("Mean")

    cax = plt.colorbar(im)
    cax.set_label("α")

    if highest is not None:
        deviation = np.abs((true_var - expectation) / expectation)
        highest_genes = np.argsort(deviation)[-highest:]
        genes = adata.var_names[highest_genes]

        texts = [ax.text(true_mean[h], true_var[h], adata.var_names[h], fontsize=10) for h in highest_genes]
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))

        print(true_mean[highest_genes])
        # ta.allocate_text(
        #     fig,
        #     ax,
        #     true_mean[highest_genes],
        #     true_var[highest_genes],
        #     genes,
        #     x_scatter=true_mean[highest_genes],
        #     y_scatter=true_var[highest_genes],
        #     textsize=10,
        #     linecolor="grey",
        #     min_distance=repel,
        #     max_distance=max_distance,
        #     margin=margin,
        # )
        # txt_height = np.log(0.04*(ax.get_ylim()[1] - ax.get_ylim()[0]))
        # txt_width = np.log(0.02*(ax.get_xlim()[1] - ax.get_xlim()[0]))
        # text_positions = get_text_positions(true_mean[highest_genes], true_var[highest_genes], txt_width, txt_height)
        # text_plotter(true_mean[highest_genes], true_var[highest_genes], text_positions, ax, txt_width, txt_height)

        print(genes)

    return expectation, true_var
