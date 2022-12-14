from functools import partial
from typing import Union

import numpy as np
import torch
from anndata import AnnData
from patsy.design_info import DesignMatrix
from torch.types import Device

from .model import guide, model
from .train import SUBSAMPLE, SVILocalHandler
from .utils import get_formula, get_rna_counts, get_state_loadings, get_states


class scPCA(object):
    """
    scPCA model.

    Parameters
    ----------
    adata: anndata.AnnData
        Anndata object with the single-cell data.
    num_factors: int (default: 15)
        Number of factors to fit.
    layers_key: str or None (default: None)
        Key to extract single-cell count matrix from adata.layers. If layers_key is None,
        scPCA will try to extract the count matrix from the adata.X.
    batch_formula: str or None (default: None)
        R style formula to extract batch information from adata.obs. If batch_formula is None,
        scPCA assumes a single batch. Usually `batch_column - 1`.
    design_formula: str or None (default: None)
        R style formula to construct the design matrix from adata.obs. If design_formula is None,
        scPCA fits a normal PCA.
    subsampling: int (default: 4096)
        Number of cells to subsample for training. A larger number will result in a more accurate
        computation of the gradients, but will also increase the training time and memory usage.
    device: torch.device (default: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        Device to run the model on. A GPU is highly recommended.
    model_key: str (default: "scpca")
        Key to store the model in the AnnData object.
    model_kwargs: dict
        Model parameters. See sccca.model.model for more details.
    training_kwargs: dict
        Training parameters. See sccca.handler for more details.
    """

    def __init__(
        self,
        adata: AnnData,
        num_factors: int,
        layers_key: Union[str, None] = None,
        batch_formula: Union[str, None] = None,
        design_formula: Union[str, None] = None,
        subsampling: int = 4096,
        device: Device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model_key: str = "scpca",
        model_kwargs: dict = {
            "β_rna_sd": 0.01,
            "β_rna_mean": 3,
            "intercept": True,
            "batch_beta": False,
            "horseshoe": False,
        },
        training_kwargs: dict = SUBSAMPLE,
    ):
        self.adata = adata
        self.num_factors = num_factors
        self.layers_key = layers_key
        self.batch_formula = batch_formula
        self.design_formula = design_formula
        self.model_key = model_key

        self.subsampling = min([subsampling, adata.shape[0]])
        self.device = device

        # prepare design and batch matrix
        self.batch_matrix = get_formula(self.adata, self.batch_formula)
        self.design_matrix = get_formula(self.adata, self.design_formula)

        #
        self.model_kwargs = model_kwargs
        self.training_kwargs = training_kwargs

        # setup data
        self.data = self._setup_data()
        self.handler = self._setup_handler()

    def _to_torch(self, data):
        """
        Helper method to convert numpy arrays of a dict to torch tensors.
        """
        return {
            k: torch.tensor(v, device=self.device) if isinstance(v, np.ndarray) else v
            for k, v in data.items()
        }

    def _setup_data(self):
        """
        Sets up the data.
        """
        X = get_rna_counts(self.adata, self.layers_key)
        X_size = np.log(X.sum(axis=1, keepdims=True))
        batch = np.asarray(self.batch_matrix).astype(np.float32)
        design = np.asarray(self.design_matrix).astype(np.float32)

        num_genes = X.shape[1]
        num_cells = X.shape[0]
        num_batches = batch.shape[1]
        idx = np.arange(num_cells)

        data = dict(
            X=X,
            X_size=X_size,
            Y=None,
            Y_size=None,
            batch=batch,
            idx=idx,
            num_genes=num_genes,
            num_proteins=None,
            num_batches=num_batches,
            num_cells=num_cells,
            design=design,
        )
        return self._to_torch(data)

    def _setup_handler(self):
        """
        Sets up the handler for training the model.
        """
        train_model = partial(
            model,
            num_factors=self.num_factors,
            subsampling=self.subsampling,
            minibatches=False,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        train_guide = partial(
            guide,
            num_factors=self.num_factors,
            subsampling=self.subsampling,
            minibatches=False,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        idx = self.data.pop("idx")

        predict_model = partial(
            model,
            num_factors=self.num_factors,
            subsampling=0,
            minibatches=True,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        predict_guide = partial(
            guide,
            num_factors=self.num_factors,
            subsampling=0,
            minibatches=True,
            device=self.device,
            **self.data,
            **self.model_kwargs,
        )

        return SVILocalHandler(
            model=train_model,
            guide=train_guide,
            predict_model=predict_model,
            predict_guide=predict_guide,
            idx=idx,
            **self.training_kwargs,
        )

    def fit(self, *args, **kwargs):
        self.handler.fit(*args, **kwargs)

    def to_anndata(self, adata=None, model_key=None, num_samples=25):
        if model_key is None:
            model_key = self.model_key

        if adata is None:
            adata = self.adata

        adata.obsm[f"X_{model_key}"] = self.handler.predict_local_variable(
            "z", num_samples=num_samples
        ).mean(0)
        adata.layers[f"{model_key}_pred_rna"] = self.handler.predict_local_variable(
            "μ_rna", num_samples=num_samples
        ).mean(0)

        adata.varm[f"{model_key}"] = (
            self.handler.predict_global_variable("W_fac", num_samples=num_samples)
            .mean(0)
            .T
        )

        if isinstance(self.design_matrix, DesignMatrix):
            adata.uns[f"{model_key}"] = {"design": get_states(self.design_matrix)}

        state_loadings = get_state_loadings(adata, model_key)
        adata.uns[f"{model_key}"]["states"] = state_loadings
        adata.uns[f"{model_key}"]["α_rna"] = self.handler.predict_global_variable(
            "α_rna"
        ).mean(0)
