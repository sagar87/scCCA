import numpy as np

from scCCA.utils import get_rna_counts


def test_get_rna_counts(test_sparse_anndata):
    X = get_rna_counts(test_sparse_anndata)

    assert type(X) == np.ndarray
    assert X.shape == (100, 2000)
