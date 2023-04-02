import scanpy as sc


def umap(adata, basis, neighbors_kwargs={}, umap_kwargs={}):
    sc.pp.neighbors(adata, use_rep=f"{basis}", key_added=f"{basis}", **neighbors_kwargs)
    sc.tl.umap(adata, neighbors_key=f"{basis}", **umap_kwargs)
    adata.obsm[f"{basis}_umap"] = adata.obsm["X_umap"]
    del adata.obsm["X_umap"]
