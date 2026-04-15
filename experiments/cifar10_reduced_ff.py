from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.data.loaders import build_cifar10_loaders
from src.data.preprocessing import overlay_label_one_hot, sample_incorrect_labels
from src.goodness.squared_sum import SquaredSumGoodness
from src.goodness.unsquared_sum import UnsquaredSumGoodness
from src.models.ff_network import FFNetwork
from src.models.layers import build_local_receptive_mask
from src.training.ff_trainer import FFTrainer
from src.utils.logging import append_metrics_csv
from src.utils.seed import set_seed


def evaluate(model: FFNetwork, data_loader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            scores = []
            for cls in range(10):
                cls_labels = torch.full_like(labels, fill_value=cls)
                x = overlay_label_one_hot(images, cls_labels)
                scores.append(model.total_goodness(x))
            score_mat = torch.stack(scores, dim=1)
            preds = torch.argmax(score_mat, dim=1)
            total += labels.numel()
            correct += (preds == labels).sum().item()

    return 100.0 * (1.0 - (correct / total))


def resolve_norm_mode(goodness: str) -> str:
    return "l1" if goodness == "unsquared" else "l2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goodness", type=str, default="squared", choices=["squared", "unsquared"])
    parser.add_argument("--train-subset", type=int, default=10000)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--receptive-field", type=int, default=11)
    parser.add_argument("--rf-channels", type=int, default=16)
    parser.add_argument("--run-name", type=str, default="cifar_local_rf")
    parser.add_argument("--output", type=str, default="results/logs/ablation_cifar10_local_rf.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    train_loader, val_loader, test_loader = build_cifar10_loaders(
        args.data_dir,
        args.batch_size,
        seed=args.seed,
        train_subset=args.train_subset,
    )

    first_hidden = args.grid_size * args.grid_size * args.rf_channels
    hidden_dims = [first_hidden] + [args.hidden_dim] * max(0, args.layers - 1)

    goodness_cls = SquaredSumGoodness if args.goodness == "squared" else UnsquaredSumGoodness
    goodness_fns = [goodness_cls() for _ in hidden_dims]

    first_mask = build_local_receptive_mask(
        in_channels=3,
        height=32,
        width=32,
        grid_size=args.grid_size,
        receptive_field=args.receptive_field,
        channels_per_location=args.rf_channels,
    )
    layer_masks = [first_mask] + [None] * (len(hidden_dims) - 1)

    model = FFNetwork(
        input_dim=32 * 32 * 3,
        hidden_dims=hidden_dims,
        goodness_fns=goodness_fns,
        activation_name="relu",
        norm_mode=resolve_norm_mode(args.goodness),
        detach_between_layers=True,
        layer_connection_masks=layer_masks,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = FFTrainer(model=model, optimizer=optimizer, threshold=args.threshold)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    convergence_epoch = None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        running_loss = 0.0
        running_grad_norm = 0.0
        batch_count = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            pos = overlay_label_one_hot(images, labels)
            neg = overlay_label_one_hot(images, sample_incorrect_labels(labels))
            step = trainer.train_step(pos, neg)
            running_loss += step.loss
            running_grad_norm += step.grad_norm
            batch_count += 1

        train_loss = running_loss / max(1, batch_count)
        grad_norm = running_grad_norm / max(1, batch_count)
        val_err = evaluate(model, val_loader, device)
        test_err = evaluate(model, test_loader, device)
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - start
        if convergence_epoch is None and test_err <= 60.0:
            convergence_epoch = epoch

        append_metrics_csv(
            str(out_path),
            {
                "run_name": args.run_name,
                "seed": args.seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "grad_norm": grad_norm,
                "val_error_pct": val_err,
                "test_error_pct": test_err,
                "dataset": "cifar10",
                "goodness": args.goodness,
                "norm_mode": resolve_norm_mode(args.goodness),
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "threshold": args.threshold,
                "train_subset": args.train_subset,
                "grid_size": args.grid_size,
                "receptive_field": args.receptive_field,
                "rf_channels": args.rf_channels,
                "epoch_time_sec": epoch_time,
                "elapsed_sec": elapsed,
                "convergence_epoch": convergence_epoch or 0,
            },
        )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} grad_norm={grad_norm:.3f} "
            f"val_error={val_err:.2f}% test_error={test_err:.2f}% epoch_time={epoch_time:.1f}s"
        )


if __name__ == "__main__":
    main()
