from typing import Union

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse


def get_rna_counts(adata: AnnData, layers_key: Union[str, None] = None) -> np.ndarray:
    """
    Extracts RNA counts from AnnData object.

    Parameters
    ----------
    adata: AnnData
        AnnData object.
    layers_key: str, optional (default: None)
        If layers_key is None, then the raw counts are extracted from adata.X.
        Otherwise, the counts are extracted from adata.layers[layers_key].

    Returns
    -------
    X: np.ndarray
        RNA counts matrix.
    """
    if layers_key is None:
        X = adata.X
    else:
        X = adata.layers[layers_key]

    if issparse(X):
        X = X.toarray()

    return X.astype(np.float32)
