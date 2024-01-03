import lightning.pytorch as pl
import numpy as np
import timm
import torch
import torch.nn as nn
from torchmetrics import SumMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchmetrics.classification.average_precision import MulticlassAveragePrecision

import utils

NUM_CLASSES = 100


class TimmModel(nn.Module):
    def __init__(
        self,
        model_name: str = None,
        pretrained: bool = False,
        model: nn.Module = None,
        ckpt_path: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        if ckpt_path is not None:
            raise Exception("not implemented")
        elif model is not None:
            self.model = model
        else:
            self.model = self._init_timm_model()

    def clone_uninitialized(self):
        return TimmModel(
            model_name=self.model_name,
            pretrained=self.pretrained,
            model=self._init_timm_model(),
        )

    def _init_timm_model(self):
        return timm.create_model(
            self.model_name, pretrained=self.pretrained, num_classes=NUM_CLASSES
        )

    def forward(self, images):
        return self.model(images)


def new_metrics_dict():
    return nn.ModuleDict(
        {
            "train_0": nn.ModuleDict(
                {
                    "loss": SumMetric(),
                    "acc": MulticlassAccuracy(NUM_CLASSES),
                    "ap": MulticlassAveragePrecision(NUM_CLASSES),
                }
            ),
            "val_0": nn.ModuleDict(
                {
                    "loss": SumMetric(),
                    "acc": MulticlassAccuracy(NUM_CLASSES),
                    "ap": MulticlassAveragePrecision(NUM_CLASSES),
                }
            ),
            "val_1": nn.ModuleDict(
                {
                    "loss": SumMetric(),
                    "acc": MulticlassAccuracy(NUM_CLASSES),
                    "ap": MulticlassAveragePrecision(NUM_CLASSES),
                }
            ),
        }
    )


def log_metrics(module: pl.LightningModule, key, logits, labels, loss, **kwargs):
    metrics = module.metrics[key]
    for metric_name, metric in metrics.items():
        if metric_name == "loss":
            metric(loss)
        else:
            metric(logits, labels)
        module.log(f"{key}/{metric_name}", metric, **kwargs)


class SupervisedModel(pl.LightningModule):
    def __init__(self, backbone: TimmModel):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.loss = nn.CrossEntropyLoss()
        self.metrics = new_metrics_dict()

    def forward(self, images):
        return self.backbone(images)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        images = data["image"]
        logits = self.forward(images)
        loss = self.loss(logits, labels)
        log_metrics(self, "train_0", logits, labels, loss.detach())
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, labels = batch
        images = data["image"]
        logits = self.forward(images)
        loss = self.loss(logits, labels)
        log_metrics(self, f"val_{dataloader_idx}", logits, labels, loss.detach())


def make_byol_mlp(in_size: int, hidden_size: int, out_size: int):
    return nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size),
    )


def freeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


class ByolModel(pl.LightningModule):
    def __init__(
        self,
        backbone: TimmModel,
        base_tau: float = 0.99,
        mlp_hidden_size: int = 4096,
        mlp_out_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1.5e-6,
        warmup_epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.online_backbone = backbone
        self.target_backbone = backbone.clone_uninitialized()
        self.num_features = backbone.model.num_features
        self.global_pool = backbone.model.global_pool

        self.online_projector = make_byol_mlp(
            self.num_features, mlp_hidden_size, mlp_out_size
        )
        self.online_predictor = make_byol_mlp(
            mlp_out_size, mlp_hidden_size, mlp_out_size
        )
        self.target_projector = make_byol_mlp(
            self.num_features, mlp_hidden_size, mlp_out_size
        )

        freeze(self.target_backbone)
        freeze(self.target_projector)

    def get_progress(self):
        return self.trainer.current_epoch / self.trainer.max_epochs

    @torch.no_grad()
    def update_target(self):
        tau = (
            1
            - (1 - self.hparams.base_tau)
            * (np.cos(np.pi * self.get_progress()) + 1.0)
            / 2.0
        )
        online_params = {m: t for m, t in self.online_backbone.named_parameters()}
        for m, t in self.target_backbone.named_parameters():
            t.set_(t * tau + (1 - tau) * online_params[m])

    @staticmethod
    def regression_loss(x, y):
        x_norm = torch.linalg.norm(x, ord=2, dim=1)
        y_norm = torch.linalg.norm(y, ord=2, dim=1)
        return 2 - 2 * torch.mean(torch.sum(x * y, dim=1) / (x_norm * y_norm))

    def forward(self, batch_data):
        images = batch_data["pair"]
        bs = images.shape[0] // 2
        online_out = self.online_predictor(
            self.online_projector(
                self.global_pool(self.online_backbone.model.forward_features(images))
            )
        )
        with torch.no_grad():
            target_out = self.target_projector(
                self.global_pool(self.target_backbone.model.forward_features(images))
            )
        online_out_0 = online_out[:bs]
        online_out_1 = online_out[bs:]
        target_out_0 = target_out[:bs]
        target_out_1 = target_out[bs:]
        loss = self.regression_loss(online_out_0, target_out_1) + self.regression_loss(
            online_out_1, target_out_0
        )
        return loss

    def training_step(self, batch, batch_idx):
        data, labels = batch
        loss = self.forward(data)
        self.log("train_0/loss", loss.detach(), batch_size=labels.numel())
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, labels = batch
        loss = self.forward(data)
        self.log(
            f"val_{dataloader_idx}/loss",
            loss.detach(),
            add_dataloader_idx=False,
            batch_size=labels.numel(),
        )
        return loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.update_target()

    def configure_optimizers(self):
        params = utils.add_weight_decay(
            [self.online_backbone, self.online_projector, self.online_predictor],
            weight_decay=self.hparams.weight_decay,
        )
        opt = torch.optim.AdamW(params, lr=self.hparams.lr)
        warmup_lr = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        cos_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, self.trainer.max_epochs - self.hparams.warmup_epochs
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt, [warmup_lr, cos_lr], milestones=[self.hparams.warmup_epochs]
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
