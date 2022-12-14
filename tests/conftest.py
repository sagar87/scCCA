import anndata as ad
import numpy as np
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture(scope="module")
def test_sparse_anndata():
    counts = csr_matrix(np.random.poisson(1, size=(100, 2000)), dtype=np.float32)
    adata = ad.AnnData(counts)
    return adata
