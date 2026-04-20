from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _flatten_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )


def _mnist_transform(jitter_pixels: int = 0):
    tfs = []
    if jitter_pixels > 0:
        translate = float(jitter_pixels) / 28.0
        tfs.append(transforms.RandomAffine(degrees=0, translate=(translate, translate)))
    tfs.extend([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    return transforms.Compose(tfs)


def _build_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def build_mnist_loaders(
    data_dir: str,
    batch_size: int,
    val_size: int = 10000,
    seed: int | None = 42,
    jitter_pixels: int = 0,
    train_subset: int | None = None,
):
    transform = _mnist_transform(jitter_pixels=jitter_pixels)
    train_full = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=data_dir, train=False, download=True, transform=_flatten_transform())

    if train_subset is not None and train_subset > 0:
        train_subset = min(train_subset, len(train_full))
        remainder = len(train_full) - train_subset
        train_full, _ = random_split(
            train_full,
            [train_subset, remainder],
            generator=_build_generator(seed),
        )

    effective_val_size = min(val_size, max(1, len(train_full) // 5))
    train_size = len(train_full) - effective_val_size
    if train_size <= 0:
        raise ValueError("Training split is empty. Increase train_subset or reduce val_size.")
    train_ds, val_ds = random_split(
        train_full,
        [train_size, effective_val_size],
        generator=_build_generator(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=_build_generator(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def build_cifar10_loaders(
    data_dir: str,
    batch_size: int,
    val_size: int = 5000,
    seed: int | None = 42,
    train_subset: int | None = None,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )

    train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    if train_subset is not None and train_subset > 0:
        train_subset = min(train_subset, len(train_full))
        remainder = len(train_full) - train_subset
        train_full, _ = random_split(
            train_full,
            [train_subset, remainder],
            generator=_build_generator(seed),
        )

    effective_val_size = min(val_size, max(1, len(train_full) // 5))
    train_size = len(train_full) - effective_val_size
    if train_size <= 0:
        raise ValueError("Training split is empty. Increase train_subset or reduce val_size.")
    train_ds, val_ds = random_split(
        train_full,
        [train_size, effective_val_size],
        generator=_build_generator(seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=_build_generator(seed),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
