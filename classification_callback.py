from typing import Dict

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import xgboost
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
from tqdm import tqdm

import models
from datamodules import Task1Datamodule


class ClassificationCallback(pl.Callback):
    def __init__(
        self,
        datamodule: Task1Datamodule,
        early_stopping: int = 10,
        max_depth: int = 6,
        n_estimators: int = 100,
        pca_n_components: int = -1,
        val_n_splits: int = 5,
        val_n_repeats: int = 10,
        xgb_device: str = "cuda",
        every_n_epochs: int = 1,
        skip_sanity_check: bool = True,
    ):
        super().__init__()
        self.datamodule = datamodule
        self.early_stopping = early_stopping
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.pca_n_components = pca_n_components
        self.val_n_splits = val_n_splits
        self.val_n_repeats = val_n_repeats
        self.xgb_device = xgb_device
        self.every_n_epochs = every_n_epochs
        self.skip_sanity_check = skip_sanity_check

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        self.datamodule.setup(stage)

    @torch.no_grad()
    def get_datamodule_features(self, backbone):
        X = []
        y = []
        for batch, labels in self.datamodule.train_dataloader():
            X.append(backbone(batch["image"].to(backbone.device)).to("cpu"))
            y.append(labels)
        for batch, labels in self.datamodule.val_dataloader()[0]:
            X.append(backbone(batch["image"].to(backbone.device)).to("cpu"))
            y.append(labels)
        X = torch.cat(X, dim=0).numpy()
        y = torch.cat(y, dim=0).numpy()
        return X, y

    def eval_model(self, X_train, X_test, y_train, y_test) -> Dict[str, float]:
        scaler = StandardScaler()
        X_train_transformed = scaler.fit_transform(X_train)
        X_test_transformed = scaler.transform(X_test)

        if self.pca_n_components != -1:
            pca = PCA(n_components=self.pca_n_components)
            X_train_transformed = pca.fit_transform(X_train_transformed)
            X_test_transformed = pca.transform(X_test_transformed)

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

        xgb = xgboost.XGBClassifier(
            early_stopping_rounds=self.early_stopping,
            tree_method="hist",
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            device=self.xgb_device,
        )

        xgb.fit(
            X_train_transformed,
            y_train,
            eval_set=[(X_test_transformed, y_test)],
            sample_weight=sample_weights,
            verbose=False,
        )

        y_score = xgb.predict_proba(X_test_transformed)
        y_pred = np.argmax(y_score, axis=1)

        return {
            "acc": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, average="macro"),
            "auc": roc_auc_score(y_test, y_score, multi_class="ovr"),
            "ap": average_precision_score(y_test, y_score),
        }

    def kfold_eval_model(self, X, y):
        kfold = RepeatedStratifiedKFold(
            n_repeats=self.val_n_repeats, n_splits=self.val_n_splits
        )
        results = []
        for train_index, test_index in tqdm(
            kfold.split(X, y), total=self.val_n_splits * self.val_n_repeats
        ):
            X_train = X[train_index, :]
            X_test = X[test_index, :]
            y_train = y[train_index]
            y_test = y[test_index]
            results.append(self.eval_model(X_train, X_test, y_train, y_test))
        results = {k: np.array([r[k] for r in results]) for k in results[0].keys()}

        def compute_stat(d, f_name, f):
            return {f"{k}_{f_name}": f(v) for k, v in d.items()}

        return (
            compute_stat(results, "mean", np.mean)
            | compute_stat(results, "std", np.std)
            | compute_stat(results, "min", np.min)
            | compute_stat(results, "max", np.max)
        )

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        if trainer.current_epoch == 0 and self.skip_sanity_check:
            return
        if trainer.current_epoch % self.every_n_epochs != 0:
            return

        if isinstance(pl_module, models.ByolModel):

            class _Backbone(nn.Module):
                def __init__(self, timm_model: models.TimmModel, device):
                    super().__init__()
                    self.backbone = timm_model.model
                    self.device = device

                def forward(self, x):
                    x = self.backbone.forward_features(x)
                    x = self.backbone.global_pool(x)
                    return x

            backbone = _Backbone(pl_module.online_backbone, pl_module.device)
            # backbone_output_size = pl_module.num_features
        else:
            raise Exception("Unsupported module")

        X, y = self.get_datamodule_features(backbone)
        logs = self.kfold_eval_model(X, y)

        pl_module.log_dict({f"classification/{k}": v for k, v in logs.items()})
        print(f"Classification {logs}")
