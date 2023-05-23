import numpy as np
import pytest

from scCCA.utils import get_diff_genes, get_ordered_genes, get_rna_counts
from scCCA.utils.design import _get_gene_idx


def test_get_rna_counts(test_sparse_anndata):
    X = get_rna_counts(test_sparse_anndata)

    assert type(X) == np.ndarray
    assert X.shape == (100, 2000)


def test_get_gene_idx():
    test = np.random.randn(50)
    arr = _get_gene_idx(test, highest=10, lowest=0)
    assert np.all(arr == np.argsort(test)[-10:])


@pytest.mark.parametrize(
    "model_key, state, factor, highest, lowest, vector",
    [
        ("m2", "Intercept", 0, 50, 0, "W_rna"),
        ("m2", "Intercept", 5, 50, 0, "W_rna"),
        ("m2", "label[T.stim]", 0, 50, 0, "W_rna"),
        ("m2", "Intercept", 0, 0, 50, "W_rna"),
        ("m2", "Intercept", 5, 0, 50, "W_rna"),
        ("m2", "label[T.stim]", 0, 0, 50, "W_rna"),
        ("m2", "Intercept", 0, 25, 25, "W_rna"),
        ("m2", "Intercept", 5, 25, 25, "W_rna"),
        ("m2", "label[T.stim]", 0, 25, 25, "W_rna"),
    ],
)
def test_get_ordered_genes(model_key, state, factor, highest, lowest, vector, test_anndata):
    # test that the right order of genes is returned

    df = get_ordered_genes(
        test_anndata, model_key=model_key, state=state, factor=factor, highest=highest, lowest=lowest, vector=vector
    )

    state_index = test_anndata.uns[model_key]["design"][state]
    factor_weights = test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_index]
    gene_idx = _get_gene_idx(factor_weights, highest, lowest)

    assert df["gene"].tolist() == test_anndata.var_names[gene_idx][::-1].tolist()
    assert np.all(
        df["value"].to_numpy() == test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_index][gene_idx][::-1]
    )


@pytest.mark.parametrize(
    "model_key, state, factor, highest, lowest, vector, sign",
    [
        ("m2", ["Intercept", "label[T.stim]"], 0, 50, 0, "W_rna", 1.0),
        ("m2", ["Intercept", "label[T.stim]"], 5, 50, 0, "W_rna", 1.0),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0),
        ("m2", ["Intercept", "label[T.stim]"], 5, 0, 50, "W_rna", 1.0),
        ("m2", ["Intercept", "label[T.stim]"], 0, 0, 50, "W_rna", 1.0),
    ],
)
def test_get_diff_genes(model_key, state, factor, highest, lowest, vector, sign, test_anndata):

    model_dict = test_anndata.uns[model_key]
    model_design = model_dict["design"]
    state_a, state_b = model_design[state[0]], model_design[state[1]]
    diff = sign * (
        test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_b]
        - test_anndata.varm[f"{model_key}_{vector}"][..., factor, state_a]
    )
    gene_idx = _get_gene_idx(diff, highest, lowest)

    df = get_diff_genes(
        test_anndata, model_key, state, factor, highest=highest, lowest=lowest, vector=vector, sign=sign
    )

    assert df["gene"].tolist() == test_anndata.var_names[gene_idx][::-1].tolist()
    assert np.all(df["diff"].to_numpy() == diff[gene_idx][::-1])
