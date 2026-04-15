from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.loaders import build_cifar10_loaders, build_mnist_loaders
from src.data.preprocessing import overlay_label_one_hot, sample_incorrect_labels
from src.goodness.spatial import SpatialBlockGoodness
from src.utils.logging import append_metrics_csv
from src.utils.seed import set_seed


def _to_image(x: torch.Tensor, dataset: str) -> torch.Tensor:
    if dataset == "mnist":
        return x.view(x.shape[0], 1, 28, 28)
    if dataset == "cifar10":
        return x.view(x.shape[0], 3, 32, 32)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _build_loader(args):
    if args.dataset == "mnist":
        train_loader, _, _ = build_mnist_loaders(
            args.data_dir,
            args.batch_size,
            seed=args.seed,
            jitter_pixels=args.jitter_pixels,
            train_subset=args.train_subset if args.train_subset > 0 else None,
        )
        return train_loader
    train_loader, _, _ = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        seed=args.seed,
        train_subset=args.train_subset if args.train_subset > 0 else None,
    )
    return train_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--jitter-pixels", type=int, default=2)
    parser.add_argument("--train-subset", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--squared", action="store_true")
    parser.add_argument("--output", type=str, default="results/logs/spatial_local_goodness.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    loader = _build_loader(args)
    module = SpatialBlockGoodness(block_size=args.block_size, stride=args.stride, squared=args.squared)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for step_idx, (images, labels) in enumerate(loader, start=1):
        if step_idx > args.steps:
            break

        labels = labels.long()
        pos = overlay_label_one_hot(images, labels)
        neg_labels = sample_incorrect_labels(labels)
        neg = overlay_label_one_hot(images, neg_labels)

        pos_img = _to_image(pos, args.dataset)
        neg_img = _to_image(neg, args.dataset)

        pos_block = module(pos_img)
        neg_block = module(neg_img)

        pos_mean = float(pos_block.mean().item())
        neg_mean = float(neg_block.mean().item())
        margin = pos_mean - neg_mean

        append_metrics_csv(
            str(out_path),
            {
                "dataset": args.dataset,
                "step": step_idx,
                "block_size": args.block_size,
                "stride": args.stride,
                "squared": int(args.squared),
                "jitter_pixels": args.jitter_pixels if args.dataset == "mnist" else 0,
                "pos_block_mean": pos_mean,
                "neg_block_mean": neg_mean,
                "pos_neg_margin": margin,
                "num_blocks": pos_block.shape[1],
            },
        )

    print(f"wrote diagnostics to {args.output}")


if __name__ == "__main__":
    main()
