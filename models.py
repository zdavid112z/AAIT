from typing import Any, List

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import timm
import timm.optim.lars
import torch
import torch.nn as nn
from torchmetrics import SumMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy
from lightning.pytorch.core.optimizer import LightningOptimizer
import torch.nn.functional as F

import utils

NUM_CLASSES = 100

TASK1_IMAGENET64_CLASSES = [
    633,
    974,
    542,
    846,
    146,
    929,
    269,
    726,
    265,
    237,
    353,
    838,
    728,
    136,
    526,
    88,
    212,
    879,
    120,
    644,
    895,
    565,
    628,
    593,
    124,
    769,
    653,
    322,
    58,
    835,
    816,
    10,
    600,
    985,
    721,
    865,
    922,
    737,
    970,
    950,
    800,
    599,
    680,
    267,
    845,
    674,
    360,
    366,
    544,
    95,
    547,
    498,
    493,
    704,
    944,
    682,
    719,
    648,
    475,
    283,
    9,
    51,
    449,
    173,
    312,
    309,
    776,
    787,
    990,
    679,
    293,
    631,
    829,
    541,
    712,
    365,
    907,
    715,
    577,
    734,
    594,
    325,
    610,
    813,
    284,
    750,
    590,
    928,
    532,
    798,
    314,
    667,
    606,
    634,
    789,
    844,
    602,
    670,
    617,
    569,
]

TASK1_IMAGENET_CLASSES = [
    314,
    932,
    508,
    414,
    341,
    917,
    627,
    877,
    705,
    625,
    683,
    448,
    781,
    372,
    704,
    158,
    105,
    655,
    354,
    325,
    400,
    739,
    308,
    862,
    207,
    614,
    114,
    954,
    187,
    474,
    457,
    286,
    69,
    806,
    716,
    652,
    733,
    938,
    570,
    910,
    496,
    73,
    839,
    511,
    747,
    849,
    970,
    978,
    517,
    367,
    480,
    32,
    25,
    424,
    842,
    425,
    525,
    109,
    50,
    675,
    283,
    349,
    1,
    281,
    526,
    765,
    440,
    737,
    438,
    821,
    612,
    311,
    964,
    744,
    500,
    975,
    430,
    458,
    761,
    945,
    678,
    957,
    79,
    619,
    734,
    774,
    470,
    482,
    447,
    542,
    532,
    760,
    75,
    315,
    887,
    411,
    71,
    567,
    123,
    758,
]

TASK2_IMAGENET_CLASSES = [
    347,
    99,
    779,
    935,
    301,
    720,
    967,
    850,
    329,
    338,
    707,
    604,
    621,
    890,
    845,
    319,
    406,
    462,
    323,
    435,
    924,
    899,
    568,
    576,
    421,
    879,
    605,
    291,
    950,
    677,
    951,
    115,
    281,
    873,
    963,
    427,
    149,
    731,
    445,
    888,
    146,
    466,
    61,
    811,
    471,
    962,
    398,
    221,
    492,
    208,
    557,
    309,
    682,
    73,
    463,
    768,
    909,
    923,
    387,
    101,
    645,
    113,
    294,
    947,
    235,
    900,
    313,
    145,
    973,
    509,
    853,
    826,
    874,
    32,
    786,
    345,
    929,
    353,
    488,
    107,
    122,
    801,
    988,
    817,
    836,
    562,
    128,
    635,
    687,
    866,
    808,
    972,
    437,
    923,
    365,
    543,
    467,
    735,
    573,
    565,
]

TASK2_IMAGENET64_CLASSES = [
    164,
    419,
    961,
    733,
    621,
    900,
    946,
    785,
    657,
    100,
    842,
    524,
    373,
    935,
    530,
    638,
    676,
    850,
    642,
    883,
    812,
    818,
    756,
    235,
    717,
    219,
    979,
    189,
    318,
    584,
    319,
    654,
    173,
    677,
    947,
    904,
    192,
    377,
    984,
    681,
    440,
    886,
    485,
    514,
    538,
    805,
    546,
    116,
    761,
    175,
    994,
    629,
    698,
    604,
    819,
    875,
    671,
    792,
    6,
    213,
    597,
    652,
    60,
    745,
    210,
    794,
    632,
    439,
    364,
    707,
    988,
    527,
    881,
    500,
    558,
    107,
    967,
    11,
    826,
    646,
    616,
    506,
    326,
    273,
    573,
    711,
    422,
    531,
    332,
    288,
    924,
    358,
    732,
    753,
    82,
    999,
    706,
    854,
    274,
    255,
]


def reset_optimizer_state(optimizer):
    if isinstance(optimizer, LightningOptimizer):
        optimizer = optimizer.optimizer

    # assert isinstance(
    #     optimizer, (torch.optim.Adam, torch.optim.AdamW)
    # ), "untested for any other optimizer"

    state = optimizer.state_dict()
    state["state"] = {}
    optimizer.load_state_dict(state)


def classifier_name(timm_model_name: str):
    if "resnet" in timm_model_name:
        return "fc"
    if "efficientnet" in timm_model_name:
        return "classifier"
    if "vit" in timm_model_name:
        return "head"
    return "fc"


class Ensemble(nn.Module):
    def __init__(self, models: List[nn.Module], learnable_weights: bool = False):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_weights = nn.Parameter(
            torch.zeros(len(models)), requires_grad=learnable_weights
        )
        self.classifiers = nn.ModuleList(
            [model.get_classifier() for model in self.models]
        )

    def forward(self, images):
        if len(self.models) == 1:
            return self.models[0](images)
        scores = torch.stack([model(images) for model in self.models], dim=0)
        weights = F.softmax(self.ensemble_weights, dim=0).view(-1, 1, 1)
        return torch.sum(scores * weights, dim=0)

    def get_classifier(self):
        return self.classifiers


class TimmModel(nn.Module):
    def __init__(
        self,
        model_name: str = None,
        pretrained: bool = False,
        model: nn.Module = None,
        byol_ckpt_path: str = None,
        timm_ckpt_path: str = None,
        finetuned_ckpt_path: str = None,
        use_hq_images: bool = False,
        task_number: int = 2,
    ):
        super().__init__()
        assert task_number == 1 or task_number == 2
        self.model_name = model_name
        self.pretrained = pretrained
        self.use_hq_images = use_hq_images
        self.task_number = task_number
        model_state_dict = None
        if byol_ckpt_path is not None:
            ckpt = torch.load(byol_ckpt_path, map_location="cpu")
            model_state_dict = {
                k.removeprefix("online_backbone.model."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("online_backbone.model.")
            }
        elif finetuned_ckpt_path is not None:
            ckpt = torch.load(finetuned_ckpt_path, map_location="cpu")
            model_state_dict = {
                k.removeprefix("backbone.models.0.model."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("backbone.models.0.model.")
            }
        elif timm_ckpt_path is not None:
            ckpt = torch.load(timm_ckpt_path, map_location="cpu")
            model_state_dict = ckpt["state_dict"]
            cls_name = classifier_name(model_name)
            classes = (
                TASK1_IMAGENET64_CLASSES
                if task_number == 1
                else TASK2_IMAGENET64_CLASSES
            )
            model_state_dict[f"{cls_name}.weight"] = model_state_dict[
                f"{cls_name}.weight"
            ][classes, :]
            model_state_dict[f"{cls_name}.bias"] = model_state_dict[f"{cls_name}.bias"][
                classes
            ]
        if model is not None:
            self.model = model
        else:
            self.model = self._init_timm_model()
        if model_state_dict is not None:
            ok = True
            try:
                self.model.load_state_dict(model_state_dict)
            except Exception as e:
                ok = False
                print(f"Failed to load state dict: {e}")
                print("Trying again without classifier")
                cls_name = classifier_name(model_name)
                model_state_dict.pop(f"{cls_name}.weight")
                model_state_dict.pop(f"{cls_name}.bias")
            if not ok:
                self.model.load_state_dict(model_state_dict, strict=False)

    def clone_uninitialized(self, pretrained: bool = None):
        return TimmModel(
            model_name=self.model_name,
            pretrained=pretrained if pretrained is not None else self.pretrained,
            model=self._init_timm_model(pretrained),
            use_hq_images=self.use_hq_images,
            task_number=self.task_number,
        )

    def _init_timm_model(self, pretrained: bool = None):
        pretrained = pretrained if pretrained is not None else self.pretrained
        kwargs = {}
        if not pretrained:
            kwargs["num_classes"] = NUM_CLASSES
        model = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            **kwargs,
        )
        if pretrained:
            classes = (
                TASK1_IMAGENET_CLASSES
                if self.task_number == 1
                else TASK2_IMAGENET_CLASSES
            )
            cls_name = classifier_name(self.model_name)
            model_state_dict = model.state_dict()
            model_state_dict[f"{cls_name}.weight"] = model_state_dict[
                f"{cls_name}.weight"
            ][classes, :]
            model_state_dict[f"{cls_name}.bias"] = model_state_dict[f"{cls_name}.bias"][
                classes
            ]
            model = timm.create_model(
                self.model_name,
                num_classes=NUM_CLASSES,
            )
            model.load_state_dict(model_state_dict)
        return model

    def forward(self, data):
        images = data["image"] if not self.use_hq_images else data["image_hq"]
        return self.model(images)

    def get_classifier(self):
        return self.model.get_classifier()


def new_metrics_dict():
    return nn.ModuleDict(
        {
            "train_0": nn.ModuleDict(
                {
                    "loss": SumMetric(),
                    "acc": MulticlassAccuracy(NUM_CLASSES),
                    # "ap": MulticlassAveragePrecision(NUM_CLASSES),
                }
            ),
            "val_0": nn.ModuleDict(
                {
                    "loss": SumMetric(),
                    "acc": MulticlassAccuracy(NUM_CLASSES),
                    # "ap": MulticlassAveragePrecision(NUM_CLASSES),
                }
            ),
            "val_1": nn.ModuleDict(
                {
                    "loss": SumMetric(),
                    "acc": MulticlassAccuracy(NUM_CLASSES),
                    # "ap": MulticlassAveragePrecision(NUM_CLASSES),
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
    def __init__(
        self,
        backbone: nn.Module,
        unfreeze_backbone_at_epoch: int = 0,
        unfreeze_only_ensemble_weights: bool = False,
        warmup_epochs: int = 5,
        lr: float = 0.2,
        momentum: float = 0.9,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.loss = nn.CrossEntropyLoss()
        self.metrics = new_metrics_dict()
        if unfreeze_backbone_at_epoch > 0:
            if hasattr(self.backbone, "ensemble_weights"):
                ensemble_weights_req_grad = self.backbone.ensemble_weights.requires_grad
            freeze(self.backbone)
            unfreeze(self.backbone.get_classifier())
            if hasattr(self.backbone, "ensemble_weights"):
                self.backbone.ensemble_weights.requires_grad_(ensemble_weights_req_grad)
            assert unfreeze_backbone_at_epoch > warmup_epochs
        self.test_classes = []

    def forward(self, images):
        return self.backbone(images)

    def on_train_epoch_start(self):
        if (
            self.hparams.unfreeze_backbone_at_epoch > 0
            and self.hparams.unfreeze_backbone_at_epoch == self.current_epoch
        ):
            unfreeze(self.backbone)
            reset_optimizer_state(self.optimizers())

    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.forward(data)
        loss = self.loss(logits, labels)
        log_metrics(self, "train_0", logits, labels, loss.detach())
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        data, labels = batch
        logits = self.forward(data)
        loss = self.loss(logits, labels)
        log_metrics(self, f"val_{dataloader_idx}", logits, labels, loss.detach())

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        data, labels = batch
        logits = self.forward(data)
        classes = torch.argmax(logits, dim=1).to("cpu")
        self.test_classes.append(classes)

    def on_test_end(self):
        classes = torch.cat(self.test_classes).numpy()
        samples = sorted([f"{i}.jpeg" for i in range(len(classes))])
        df = pd.DataFrame(data={"sample": samples, "label": classes})
        df["sample_idx"] = df["sample"].apply(lambda x: int(x[:-5]))
        df = df.sort_values(by=["sample_idx"])
        df = df.drop(["sample_idx"], axis=1)
        df.to_csv("submission.csv", index=False)

    def configure_optimizers(self):
        params = utils.add_weight_decay(
            self.backbone,
            weight_decay=self.hparams.weight_decay,
        )
        opt = timm.optim.Lamb(
            params,
            lr=self.hparams.lr,
            # momentum=self.hparams.momentum
        )
        num_steps = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        if self.hparams.unfreeze_backbone_at_epoch > 0:
            lp_warmup_lr = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=self.hparams.warmup_epochs * num_steps,
            )
            lp_cos_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                (self.hparams.unfreeze_backbone_at_epoch - self.hparams.warmup_epochs)
                * num_steps,
            )
        train_warmup_lr = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs * num_steps,
        )
        train_cos_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            (
                self.trainer.max_epochs
                - self.hparams.unfreeze_backbone_at_epoch
                - self.hparams.warmup_epochs
            )
            * num_steps,
        )
        if self.hparams.unfreeze_backbone_at_epoch > 0:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt,
                [lp_warmup_lr, lp_cos_lr, train_warmup_lr, train_cos_lr],
                milestones=[
                    self.hparams.warmup_epochs * num_steps,
                    self.hparams.unfreeze_backbone_at_epoch * num_steps,
                    (
                        self.hparams.warmup_epochs
                        + self.hparams.unfreeze_backbone_at_epoch
                    )
                    * num_steps,
                ],
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt,
                [train_warmup_lr, train_cos_lr],
                milestones=[
                    self.hparams.warmup_epochs * num_steps,
                ],
            )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def make_lr_scheduler(optimizer, num_warmup_epochs, max_epochs):
    warmup_lr = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=num_warmup_epochs,
    )
    cos_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_epochs - num_warmup_epochs
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_lr, cos_lr], milestones=[num_warmup_epochs]
    )
    return lr_scheduler


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


def unfreeze(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(True)


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
        self.online_backbone = backbone.clone_uninitialized(pretrained=None)
        self.target_backbone = backbone.clone_uninitialized(pretrained=None)
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
        self.log("train_0/tau", tau)
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
