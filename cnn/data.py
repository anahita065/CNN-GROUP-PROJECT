"""Dataset loading with stratified sub-sampling for dataset-size experiments."""

import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple


# Per-dataset normalisation statistics (computed on full training split)
DATASET_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std":  (0.2023, 0.1994, 0.2010),
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std":  (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
    },
    "tiny_imagenet": {
        "mean": (0.4802, 0.4481, 0.3975),
        "std":  (0.2770, 0.2691, 0.2821),
        "num_classes": 200,
    },
}


# ------------------------------------------------------------------
# Transforms
# ------------------------------------------------------------------

def get_transforms(dataset_name: str, image_size: int, train: bool) -> transforms.Compose:
    stats = DATASET_STATS[dataset_name]
    normalize = transforms.Normalize(mean=stats["mean"], std=stats["std"])
    if train:
        pad = max(4, image_size // 8)
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=pad),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


# ------------------------------------------------------------------
# Tiny ImageNet val-set reorganisation
# ------------------------------------------------------------------

def _prepare_tiny_imagenet_val(root: str) -> str:
    """Reorganise the flat Tiny ImageNet val directory into ImageFolder layout.

    Tiny ImageNet's val split stores all images in a single folder plus
    a val_annotations.txt mapping filenames to class IDs.  ImageFolder
    needs one sub-directory per class, so we copy images on first call.
    """
    base = os.path.join(root, "tiny-imagenet-200")
    organised = os.path.join(base, "val_organised")
    if os.path.isdir(organised):
        return organised

    ann_file = os.path.join(base, "val", "val_annotations.txt")
    img_dir  = os.path.join(base, "val", "images")

    if not os.path.isfile(ann_file):
        raise FileNotFoundError(
            f"Tiny ImageNet val annotations not found at {ann_file}.\n"
            "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
            "and extract under the configured data root."
        )

    print("  Reorganising Tiny ImageNet val split (one-time)…")
    with open(ann_file) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            img_file, class_id = parts[0], parts[1]
            class_dir = os.path.join(organised, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(img_dir, img_file)
            dst = os.path.join(class_dir, img_file)
            if os.path.isfile(src) and not os.path.isfile(dst):
                shutil.copy2(src, dst)

    return organised


# ------------------------------------------------------------------
# Dataset factory
# ------------------------------------------------------------------

def get_dataset(dataset_name: str, root: str, train: bool, image_size: int):
    transform = get_transforms(dataset_name, image_size, train)
    os.makedirs(root, exist_ok=True)

    if dataset_name == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True, transform=transform)
    if dataset_name == "cifar100":
        return datasets.CIFAR100(root, train=train, download=True, transform=transform)
    if dataset_name == "tiny_imagenet":
        if train:
            data_dir = os.path.join(root, "tiny-imagenet-200", "train")
        else:
            data_dir = _prepare_tiny_imagenet_val(root)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Tiny ImageNet not found at {data_dir}.\n"
                "Download from http://cs231n.stanford.edu/tiny-imagenet-200.zip "
                "and extract under the configured data root."
            )
        return datasets.ImageFolder(data_dir, transform=transform)
    raise ValueError(f"Unknown dataset: {dataset_name!r}")


# ------------------------------------------------------------------
# Stratified sub-sampling
# ------------------------------------------------------------------

def get_stratified_subset(dataset, fraction: float, seed: int) -> Subset:
    """Return a class-balanced subset covering `fraction` of the training data."""
    if fraction >= 1.0:
        return dataset  # type: ignore[return-value]

    rng = np.random.RandomState(seed)

    # Collect targets — works for both torchvision datasets and Subset wrappers
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    else:
        targets = np.array([dataset[i][1] for i in range(len(dataset))])

    selected: list = []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        n = max(1, int(len(idx) * fraction))
        chosen = rng.choice(idx, n, replace=False)
        selected.extend(chosen.tolist())

    rng.shuffle(selected)
    return Subset(dataset, selected)


# ------------------------------------------------------------------
# DataLoader builder
# ------------------------------------------------------------------

def get_dataloaders(
    dataset_name: str,
    root: str,
    fraction: float,
    seed: int,
    batch_size: int,
    num_workers: int,
    image_size: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = get_dataset(dataset_name, root, train=True,  image_size=image_size)
    val_ds   = get_dataset(dataset_name, root, train=False, image_size=image_size)

    train_subset = get_stratified_subset(train_ds, fraction, seed)
    import torch
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader
