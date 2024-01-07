from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torchvision.io
import torchvision.transforms.v2 as v2
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, ConcatDataset, DataLoader


@lru_cache(maxsize=None)
def read_image(path):
    return torchvision.io.read_image(str(path))


def read_image_no_cache(path):
    return torchvision.io.read_image(str(path))


class TransformDataset:
    def __init__(self, dataset, transforms, final_transforms_to_all):
        self.dataset = dataset
        self.transforms = transforms
        self.final_transforms_to_all = final_transforms_to_all

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        lq_image_path = item["lq_image_path"]
        label = item["label"]
        lq_image = read_image(lq_image_path)

        hq_image = None
        if "hq_image_path" in item:
            hq_image_path = item["hq_image_path"]
            hq_image = read_image_no_cache(hq_image_path)
        return {
            k: (
                self.final_transforms_to_all(t(lq_image))
                if not k.endswith("_hq")
                else self.final_transforms_to_all(t(hq_image))
            )
            for k, t in self.transforms.items()
        }, label


class Collator:
    def __init__(self):
        pass

    def __call__(self, batch):
        data = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch])
        result_data = {}
        if "pair_0" in data[0] and "pair_1" in data[0]:
            result_data["pair"] = torch.stack(
                [b["pair_0"] for b in data] + [b["pair_1"] for b in data]
            )
        result_data.update(
            {
                k: torch.stack([b[k] for b in data])
                for k in data[0].keys()
                if k not in ["pair_0", "pair_1"]
            }
        )
        return result_data, labels


class ImageDataset(Dataset):
    def __init__(
        self,
        folder_path: Union[str, Path],
        hq_folder_path: Optional[Union[str, Path]] = None,
        labels_csv_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        folder_path = Path(folder_path)
        if labels_csv_path is not None:
            labels_df = pd.read_csv(labels_csv_path)
            labels_df["sample_name"] = labels_df["sample"].apply(
                lambda x: x.split("/")[-1]
            )
            labels_df = labels_df.set_index("sample_name")
            self.data = [
                {
                    "lq_image_path": x,
                    "label": int(labels_df.loc[x.name]["label"]),
                }
                for x in sorted(folder_path.iterdir())
            ]
        else:
            self.data = [
                {
                    "lq_image_path": x,
                    "label": -1,
                }
                for x in sorted(folder_path.iterdir())
            ]
        if hq_folder_path is not None:
            hq_folder_path = Path(hq_folder_path)
            self.data = [
                {
                    "lq_image_path": item["lq_image_path"],
                    "hq_image_path": hq_folder_path
                    / item["lq_image_path"].with_suffix(".png").name,
                    "label": item["label"],
                }
                for item in self.data
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class Task1Datamodule(LightningDataModule):
    def __init__(
        self,
        path: str = "C:/Data/AAIT/task1",
        hq_path: str = "C:/Data/AAIT_upsampled/task1",
        load_hq_images: bool = False,
        num_train_workers: int = 4,
        num_val_workers: int = 2,
        num_test_workers: int = 2,
        batch_size: int = 64,
        labeled: bool = True,
        unlabeled: bool = True,
        val_size: float = 0.2,
        no_train_augmentations: bool = False,
        byol: bool = False,
        train_dataset_replicas: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert (
            sum([no_train_augmentations, byol]) <= 1
        ), "At most one of 'no_train_augmentations', 'byol' and 'dino' can be enabled"
        assert (
            not load_hq_images or not byol
        ), "Dual lq-hq data not implemented for BYOL"
        self.train_transform = None
        self.val_transform = None
        self.train_dataset = None
        self.val_datasets = None
        self.test_dataset = None
        self.final_transforms = None
        self.collator = None

    def setup(self, stage: str):
        self.collator = Collator()
        path = Path(self.hparams.path)
        hq_path = (
            Path(self.hparams.hq_path) if self.hparams.hq_path is not None else None
        )

        self.final_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if self.hparams.byol:

            def byol_transform(blur_p: float, solarization_p: float):
                return v2.Compose(
                    [
                        v2.RandomResizedCrop(
                            56,
                            scale=(0.2, 1.0),
                            interpolation=v2.InterpolationMode.BICUBIC,
                            antialias=True,
                        ),
                        v2.RandomHorizontalFlip(p=0.5),
                        v2.RandomApply(
                            [
                                v2.ColorJitter(
                                    brightness=0.4,
                                    contrast=0.4,
                                    saturation=0.2,
                                    hue=0.1,
                                )
                            ],
                            p=0.8,
                        ),
                        v2.RandomGrayscale(p=0.2),
                        v2.RandomApply(
                            [v2.GaussianBlur(23, sigma=(0.1, 2.0))], p=blur_p
                        ),
                        v2.RandomSolarize(128, p=solarization_p),
                    ]
                )

            self.train_transform = {
                "pair_0": byol_transform(blur_p=1, solarization_p=0),
                "pair_1": byol_transform(blur_p=0.1, solarization_p=0.2),
            }
            self.val_transform = {
                "pair_0": byol_transform(blur_p=1, solarization_p=0),
                "pair_1": byol_transform(blur_p=0.1, solarization_p=0.2),
                "image": v2.CenterCrop(56),
            }
        else:
            if self.hparams.no_train_augmentations:
                self.train_transform = {"image": v2.CenterCrop(56)}
                if self.hparams.load_hq_images:
                    self.train_transform["image_hq"] = v2.CenterCrop(224)
            else:

                def gen_train_transform(img_size: int):
                    return v2.Compose(
                        [
                            v2.RandomResizedCrop(
                                img_size,
                                scale=(0.4, 1.0),
                                interpolation=v2.InterpolationMode.BICUBIC,
                                antialias=True,
                            ),
                            v2.RandomHorizontalFlip(p=0.5),
                            v2.RandomApply(
                                [
                                    v2.ColorJitter(
                                        brightness=0.4,
                                        contrast=0.4,
                                        saturation=0.2,
                                        hue=0.1,
                                    )
                                ],
                                p=0.2,
                            ),
                            v2.RandomGrayscale(p=0.1),
                            v2.RandomSolarize(128, p=0.1),
                        ]
                    )

                self.train_transform = {
                    "image": gen_train_transform(56),
                }
                if self.hparams.load_hq_images:
                    self.train_transform["image_hq"] = gen_train_transform(224)

            self.val_transform = {
                "image": v2.CenterCrop(56),
            }
            if self.hparams.load_hq_images:
                self.val_transform["image_hq"] = v2.CenterCrop(224)

        train_datasets = []
        val_datasets = []
        if self.hparams.labeled:
            labeled_dataset = ImageDataset(
                folder_path=path / "train_data" / "images" / "labeled",
                hq_folder_path=hq_path / "train_data" / "images" / "labeled"
                if hq_path is not None and self.hparams.load_hq_images
                else None,
                labels_csv_path=path / "train_data" / "annotations.csv",
            )
            if self.hparams.val_size > 0:
                labeled_train, labeled_val = train_test_split(
                    labeled_dataset, random_state=42, test_size=self.hparams.val_size
                )
                train_datasets.append(labeled_train)
                val_datasets.append(labeled_val)
            else:
                train_datasets.append(labeled_dataset)
        if self.hparams.unlabeled:
            unlabeled_dataset = ImageDataset(
                folder_path=path / "train_data" / "images" / "unlabeled",
                hq_folder_path=hq_path / "train_data" / "images" / "unlabeled"
                if hq_path is not None and self.hparams.load_hq_images
                else None,
            )
            if self.hparams.val_size > 0:
                unlabeled_train, unlabeled_val = train_test_split(
                    unlabeled_dataset, random_state=42, test_size=self.hparams.val_size
                )
                train_datasets.append(unlabeled_train)
                val_datasets.append(unlabeled_val)
            else:
                train_datasets.append(unlabeled_dataset)

        train_dataset = ConcatDataset(train_datasets)
        train_dataset = ConcatDataset(
            [train_dataset] * self.hparams.train_dataset_replicas
        )

        test_dataset = ImageDataset(
            folder_path=path / "val_data",
            hq_folder_path=hq_path / "val_data"
            if hq_path is not None and self.hparams.load_hq_images
            else None,
        )

        self.train_dataset = TransformDataset(
            train_dataset, self.train_transform, self.final_transforms
        )
        self.val_datasets = [
            TransformDataset(ds, self.val_transform, self.final_transforms)
            for ds in val_datasets
        ]
        self.test_dataset = TransformDataset(
            test_dataset, self.val_transform, self.final_transforms
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_train_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.hparams.batch_size,
                shuffle=False,
                num_workers=self.hparams.num_val_workers,
                pin_memory=True,
                collate_fn=self.collator,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_test_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )
