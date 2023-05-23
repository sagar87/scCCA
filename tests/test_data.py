import numpy as np

from scCCA.utils import get_ordered_genes, get_rna_counts


def test_get_rna_counts(test_sparse_anndata):
    X = get_rna_counts(test_sparse_anndata)

    assert type(X) == np.ndarray
    assert X.shape == (100, 2000)


def test_get_ordered_genes(test_anndata):
    # test that the right order of genes is returned
    model_key = "m2"
    state = "Intercept"
    factor = 0
    highest = 50
    lowest = 0
    vector = "W_rna"

    df = get_ordered_genes(
        test_anndata, model_key=model_key, state=state, factor=factor, highest=highest, lowest=lowest, vector=vector
    )

    state_index = test_anndata.uns[model_key]["design"][state]
    highest_index = np.argsort(test_anndata.varm[f"{model_key}_{vector}"][..., state_index, factor])[-highest:]

    assert df["gene"].tolist() == test_anndata.var_names[highest_index][::-1].tolist()
    assert np.all(
        df["value"].to_numpy()
        == test_anndata.varm[f"{model_key}_{vector}"][..., state_index, factor][highest_index][::-1]
    )
