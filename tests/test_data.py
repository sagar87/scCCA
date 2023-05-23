import numpy as np
import pytest

from scCCA.utils import get_ordered_genes, get_rna_counts


def test_get_rna_counts(test_sparse_anndata):
    X = get_rna_counts(test_sparse_anndata)

    assert type(X) == np.ndarray
    assert X.shape == (100, 2000)


@pytest.mark.parametrize(
    "model_key, state, factor, highest, lowest, vector",
    [
        ("m2", "Intercept", 0, 50, 0, "W_rna"),
        ("m2", "Intercept", 5, 50, 0, "W_rna"),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna"),
        ("m2", "Intercept", 0, 0, 50, "W_rna"),
        ("m2", "Intercept", 5, 0, 50, "W_rna"),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna"),
    ],
)
def test_get_ordered_genes(model_key, state, factor, highest, lowest, vector, test_anndata):
    # test that the right order of genes is returned

    df = get_ordered_genes(
        test_anndata, model_key=model_key, state=state, factor=factor, highest=highest, lowest=lowest, vector=vector
    )

    state_index = test_anndata.uns[model_key]["design"][state]
    order = np.argsort(test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_index])

    if highest == 0:
        gene_idx = order[:lowest]
    else:
        gene_idx = np.concatenate([order[:lowest], order[-highest:]])

    assert df["gene"].tolist() == test_anndata.var_names[gene_idx][::-1].tolist()
    assert np.all(
        df["value"].to_numpy() == test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_index][gene_idx][::-1]
    )
