from typing import List, Union

import gseapy as gp
from anndata import AnnData

from .design import get_diff_genes, get_ordered_genes


def get_factor_enrichment(
    adata: AnnData,
    model_key: str,
    state: str,
    factor: int,
    highest: int = 50,
    lowest: int = 0,
    sign: Union[int, float] = 1.0,
    geneset: str = "GO_Biological_Process_2021",
    organism: str = "human",
):
    if (highest > 0) and (lowest > 0):
        raise ValueError("Plese provide either highest or lower by setting one of them to zero.")

    ordered_genes = get_ordered_genes(
        adata,
        model_key=model_key,
        state=state,
        factor=factor,
        highest=highest,
        lowest=lowest,
        ascending=True if lowest > 0 else False,
    )
    enr = gp.enrichr(gene_list=ordered_genes["gene"].tolist(), gene_sets=geneset, organism=organism, outdir=None)
    return enr.results


def get_diff_enrichment(
    adata: AnnData,
    model_key: str,
    states: List[str],
    factor: int = 0,
    highest: int = 50,
    lowest: int = 0,
    sign: Union[int, float] = 1.0,
    geneset: str = "GO_Biological_Process_2021",
    organism: str = "human",
):
    # assert (highest > 0 and lowest == 0) or (highest == 0 and lowest > 0), "Either highest or lowest must be greater zero but not both."
    if (highest > 0) and (lowest > 0):
        raise ValueError("Plese provide either highest or lower by setting one of them to zero.")
    diff_genes = get_diff_genes(
        adata,
        model_key,
        states=states,
        factor=factor,
        highest=highest,
        lowest=lowest,
        sign=sign,
        ascending=True if lowest > 0 else False,
    )

    enr = gp.enrichr(
        gene_list=diff_genes["gene"].tolist(),  # or "./tests/data/gene_list.txt",
        gene_sets=geneset,
        organism=organism,  # don't forget to set organism to the one you desired! e.g. Yeast
        outdir=None,  # don't write to disk
    )

    enr.results["Regulation"] = "Down" if lowest > 0 else "Up"
    res = enr.results
    return res
