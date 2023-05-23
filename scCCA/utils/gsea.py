import gseapy as gp
import pandas as pd

from .design import get_diff_genes


def geneset_enrichment_diff(
    adata,
    model_key,
    states,
    factor=0,
    highest=50,
    lowest=0,
    sign=1.0,
    gene_sets=["GO_Biological_Process_2021"],
    organism="human",
):
    # assert (highest > 0 and lowest == 0) or (highest == 0 and lowest > 0), "Either highest or lowest must be greater zero but not both."
    
    if (highest > 0 and lowest == 0) or (highest == 0 and lowest > 0):
        df = get_diff_genes(
            adata,
            model_key,
            state=states,
            factor=factor,
            highest=highest,
            lowest=lowest,
            sign=sign,
            ascending=True if lowest > 0 else False,
        )

        enr = gp.enrichr(
            gene_list=df["gene"].tolist(),  # or "./tests/data/gene_list.txt",
            gene_sets=gene_sets,
            organism=organism,  # don't forget to set organism to the one you desired! e.g. Yeast
            outdir=None,  # don't write to disk
        )
        enr.results['Regulation'] = 'Down' if lowest > 0 else 'Up'
        res = enr.results
    else:
        df_up = get_diff_genes(
            adata,
            model_key,
            state=states,
            factor=factor,
            highest=highest,
            lowest=0,
            sign=sign,
            ascending=False,
        )

        enr_up = gp.enrichr(
            gene_list=df_up["gene"].tolist(),  # or "./tests/data/gene_list.txt",
            gene_sets=gene_sets,
            organism=organism,  # don't forget to set organism to the one you desired! e.g. Yeast
            outdir=None,  # don't write to disk
        )
        
        df_down = get_diff_genes(
            adata,
            model_key,
            state=states,
            factor=factor,
            highest=0,
            lowest=lowest,
            sign=sign,
            ascending=True,
        )

        enr_down = gp.enrichr(
            gene_list=df_down["gene"].tolist(),  # or "./tests/data/gene_list.txt",
            gene_sets=gene_sets,
            organism=organism,  # don't forget to set organism to the one you desired! e.g. Yeast
            outdir=None,  # don't write to disk
        )
        
        enr_up.results['Regulation'] = 'Up'
        enr_down.results['Regulation'] = 'Down'
        res = pd.concat([enr_up.results, enr_down.results])
        
    return res