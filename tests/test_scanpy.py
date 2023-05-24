import pytest
from anndata import AnnData

from scCCA.utils.scanpy import _get_model_design


def test_get_model_design_existing_key():
    adata = AnnData()
    model_key = "my_model"
    design_mapping = {"Intercept": 0, "stim": 1}
    adata.uns[model_key] = {"design": design_mapping}

    result = _get_model_design(adata, model_key)

    assert result == design_mapping


def test_get_model_design_missing_key():
    adata = AnnData()
    model_key = "my_model"

    with pytest.raises(ValueError):
        _get_model_design(adata, model_key)


def test_get_model_design_missing_design_mapping():
    adata = AnnData()
    model_key = "my_model"
    adata.uns[model_key] = {}

    with pytest.raises(ValueError):
        _get_model_design(adata, model_key)
