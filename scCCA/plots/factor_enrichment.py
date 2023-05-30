from typing import List, Union

import pandas as pd
import seaborn as sns

from .utils import set_up_plot


def factor_enrichment(
    adata,
    model_key: str,
    factor: Union[int, List[int]],
    cluster_key: str,
    hue_key: Union[str, None] = None,
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
        hue_key=hue_key,
        sign=sign,
        kind=kind,
        sharex=True,
        ax=ax,
        **kwargs,
    )
    return ax


def _factor_enrichment(adata, model_key, factor, cluster_key, hue_key=None, sign=1.0, kind="strip", ax=None, **kwargs):
    plot_funcs = {
        "strip": sns.stripplot,
        "box": sns.boxplot,
    }

    df = pd.DataFrame(sign * adata.obsm[f"X_{model_key}"]).assign(cluster=adata.obs[cluster_key].tolist())

    hue = None
    groupby_vars = ["cluster"]

    if hue_key is not None:
        df = df.assign(hue=adata.obs[hue_key].tolist())
        groupby_vars.append("hue")
        hue = "hue"
    df = df.melt(groupby_vars, var_name="factor")

    g = plot_funcs[kind](x="cluster", y="value", hue=hue, data=df[df["factor"] == factor], ax=ax, **kwargs)

    g.axes.tick_params(axis="x", rotation=90)
    g.axes.axhline(0, color="k", linestyle="--", lw=0.5)
