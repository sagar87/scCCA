from typing import List, Union

import pandas as pd
import seaborn as sns

from .utils import set_up_plot


def factor_enrichment(
    adata,
    model_key: str,
    factor: Union[int, List[int]],
    cluster_key: str,
    highlight: Union[str, List[str]] = None,
    hue_key: Union[str, None] = None,
    swap_axes=False,
    sign=1.0,
    kind="strip",
    ncols: int = 4,
    width: int = 4,
    height: int = 3,
    ax=None,
    **kwargs,
):
    ax = set_up_plot(
        adata,
        model_key,
        factor,
        _factor_enrichment,
        cluster_key=cluster_key,
        highlight=highlight,
        hue_key=hue_key,
        sign=sign,
        kind=kind,
        swap_axes=swap_axes,
        sharex=True,
        width=width,
        height=height,
        ncols=ncols,
        ax=ax,
        **kwargs,
    )
    return ax


def _factor_enrichment(
    adata,
    model_key,
    factor,
    cluster_key,
    hue_key=None,
    highlight=None,
    sign=1.0,
    kind="strip",
    swap_axes=False,
    ax=None,
    **kwargs,
):
    plot_funcs = {
        "strip": sns.stripplot,
        "box": sns.boxplot,
    }

    df = pd.DataFrame(sign * adata.obsm[f"X_{model_key}"]).assign(cluster=adata.obs[cluster_key].tolist())

    if highlight is not None:
        df = df.assign(highlight=lambda df: df.cluster.apply(lambda x: x if x in highlight else "other"))

    groupby_vars = ["cluster"]

    if hue_key is not None:
        df[hue_key] = adata.obs[hue_key].tolist()
        groupby_vars.append(hue_key)

    if hue_key is None and highlight is not None:
        hue_key = "highlight"
        groupby_vars.append(hue_key)

    df = df.melt(groupby_vars, var_name="factor")
    # import pdb; pdb.set_trace()
    if swap_axes:
        g = plot_funcs[kind](y="cluster", x="value", hue=hue_key, data=df[df["factor"] == factor], ax=ax, **kwargs)
        # g.axes.tick_params(axis="x", rotation=90)
        g.axes.axvline(0, color="k", linestyle="-", lw=0.5)
        g.axes.xaxis.grid(True)
        g.axes.set_xlabel("Factor weight")
    else:
        g = plot_funcs[kind](x="cluster", y="value", hue=hue_key, data=df[df["factor"] == factor], ax=ax, **kwargs)
        g.axes.tick_params(axis="x", rotation=90)
        g.axes.axhline(0, color="k", linestyle="-", lw=0.5)
        g.axes.yaxis.grid(True)
        g.axes.set_ylabel("Factor weight")
    g.axes.spines["top"].set_visible(False)
    g.axes.spines["bottom"].set_visible(False)
    g.axes.spines["right"].set_visible(False)
    g.axes.spines["left"].set_visible(False)
    g.axes.set_title(f"Factor {factor}")

    return df
