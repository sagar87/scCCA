from typing import List, Union

import scanpy as sc
from anndata import AnnData

from ..utils import get_diff_genes
from ..utils.data import _validate_sign


def cluster_enrichment(
    adata: AnnData,
    model_key: str,
    states: Union[str, List[str]],
    factor: int,
    state_key: str,
    cluster_key: str,
    groups: Union[str, List[str]],
    dot: bool = False,
    sign: Union[int, float] = 1,
    highest: int = 0,
    lowest: int = 0,
    threshold: float = 1.96,
    return_var_names: bool = False,
    **kwargs,
):
    """
    Enrichment analysis of clusters based on differential gene expression.

    Parameters
    ----------
    adata :
        Anndata object.
    model_key :
        Key to access the model in the adata object.
    states :
        States for comparison. If not pair of states is provided, "Intercept" is assumed to be the base state.
    factor :
        Factor index to consider for the differential calculation.
    state_key :
        Key for state in adata.
    cluster_key :
        Key for cluster in adata.
    groups :
        Groups to consider for enrichment.
    dot :
        If True, a dot plot is generated. Otherwise, a heatmap is generated. Default is False.
    sign :
        Sign to adjust the difference, by default 1.
    highest :
        Number of highest differential genes to retrieve, by default 0.
    lowest :
        Number of lowest differential genes to retrieve, by default 0.
    threshold :
        Threshold for significance, by default 1.96.
    return_var_names :
        If True, returns variable names. Otherwise, returns the plot axes. Default is False.
    **kwargs
        Additional keyword arguments passed to the plotting function.

    Returns
    -------
    Union[dict, matplotlib.axes.Axes]
        If `return_var_names` is True, returns a dictionary of variable names. Otherwise, returns the plot axes.

    Notes
    -----
    This function performs enrichment analysis of clusters based on differential gene expression.
    It first validates the sign and retrieves differential genes. Depending on the `dot` parameter, either a
    dot plot or a heatmap is generated to visualize the enrichment.
    """
    sign = _validate_sign(sign)

    if isinstance(groups, str):
        groups = [groups]
    if isinstance(state_key, str):
        state_key = [state_key]
    df = get_diff_genes(adata, model_key, states, factor, sign=sign, highest=adata.shape[1], threshold=threshold)
    if highest > 0 or lowest > 0:
        var_names = {
            f"Up in {states[1]}": df.head(highest).gene.tolist(),
            f"Down in {states[1]}": df.tail(lowest).gene.tolist(),
        }
    else:
        var_names = {
            f"Up in {states[1]}": df[df.significant & (df["diff"] > 0)].gene.tolist(),
            f"Down in {states[1]}": df[df.significant & (df["diff"] < 0)].gene.tolist(),
        }

    for k, v in var_names.copy().items():
        if len(v) == 0:
            del var_names[k]

    if dot:
        axes = sc.pl.dotplot(
            adata[adata.obs[cluster_key].isin(groups)],
            groupby=[cluster_key, *state_key],
            var_names=var_names,
            show=False,
            **kwargs,
        )
        labels = [
            item.get_text() + f" ({df[df.gene==item.get_text()]['diff'].item():.2f})"
            for item in axes["mainplot_ax"].get_xticklabels()
        ]
        axes["mainplot_ax"].set_xticklabels(labels)
    else:
        axes = sc.pl.heatmap(
            adata[adata.obs[cluster_key].isin(groups)],
            groupby=[cluster_key, *state_key],
            var_names=var_names,
            show=False,
            **kwargs,
        )
        labels = [
            item.get_text() + f" ({df[df.gene==item.get_text()]['diff'].item():.2f})"
            for item in axes["heatmap_ax"].get_xticklabels()
        ]
        axes["heatmap_ax"].set_xticklabels(labels)
        # axes['heatmap_ax'].get_xticklabels()

    if return_var_names:
        return var_names
    return axes
