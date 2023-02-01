import scanpy as sc


def umap(adata, model_key, neighbors_kwargs={}, umap_kwargs={}):
    sc.pp.neighbors(adata, use_rep=f"X_{model_key}", key_added=f"neighbors_{model_key}", **neighbors_kwargs)
    sc.tl.umap(adata, neighbors_key=f"neighbors_{model_key}", **umap_kwargs)
    adata.obsm[f"X_umap_{model_key}"] = adata.obsm["X_umap"]
    del adata.obsm["X_umap"]
