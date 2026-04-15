from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.data.loaders import build_mnist_loaders
from src.data.preprocessing import overlay_label_one_hot, sample_incorrect_labels
from src.goodness.squared_sum import SquaredSumGoodness
from src.goodness.unsquared_sum import UnsquaredSumGoodness
from src.models.ff_network import FFNetwork
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


def resolve_norm_mode(goodness: str, norm_mode: str) -> str:
    if norm_mode != "auto":
        if goodness == "unsquared" and norm_mode != "l1":
            raise ValueError("Unsquared goodness requires l1 normalization")
        return norm_mode
    return "l1" if goodness == "unsquared" else "l2"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goodness", type=str, default="squared", choices=["squared", "unsquared"])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--norm-mode", type=str, default="auto", choices=["auto", "l1", "l2"])
    parser.add_argument("--student-df", type=float, default=2.0)
    parser.add_argument("--student-scale", type=float, default=1.0)
    parser.add_argument("--jitter-pixels", type=int, default=0)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--error-threshold", type=float, default=15.0)
    parser.add_argument("--run-name", type=str, default="baseline")
    parser.add_argument("--output", type=str, default="results/logs/baseline_mnist.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")

    norm_mode = resolve_norm_mode(args.goodness, args.norm_mode)
    train_subset = args.train_subset if args.train_subset > 0 else None
    train_loader, val_loader, test_loader = build_mnist_loaders(
        args.data_dir,
        args.batch_size,
        seed=args.seed,
        jitter_pixels=args.jitter_pixels,
        train_subset=train_subset,
    )

    hidden_dims = [args.hidden_dim] * args.layers
    goodness_cls = SquaredSumGoodness if args.goodness == "squared" else UnsquaredSumGoodness
    goodness_fns = [goodness_cls() for _ in hidden_dims]

    model = FFNetwork(
        input_dim=28 * 28,
        hidden_dims=hidden_dims,
        goodness_fns=goodness_fns,
        activation_name=args.activation,
        activation_kwargs={"df": args.student_df, "scale": args.student_scale},
        norm_mode=norm_mode,
        detach_between_layers=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = FFTrainer(model=model, optimizer=optimizer, threshold=args.threshold)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    convergence_epoch = None
    run_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        running_loss = 0.0
        running_grad_norm = 0.0
        batch_count = 0
        non_finite_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            pos = overlay_label_one_hot(images, labels)
            neg_labels = sample_incorrect_labels(labels)
            neg = overlay_label_one_hot(images, neg_labels)

            step = trainer.train_step(pos, neg)
            running_loss += step.loss
            running_grad_norm += step.grad_norm
            if step.had_non_finite:
                non_finite_batches += 1
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)
        mean_grad_norm = running_grad_norm / max(batch_count, 1)
        val_err = evaluate(model, val_loader, device)
        test_err = evaluate(model, test_loader, device)
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - run_start
        if convergence_epoch is None and test_err <= args.error_threshold:
            convergence_epoch = epoch

        append_metrics_csv(
            str(out_path),
            {
                "run_name": args.run_name,
                "seed": args.seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "grad_norm": mean_grad_norm,
                "non_finite_batches": non_finite_batches,
                "val_error_pct": val_err,
                "test_error_pct": test_err,
                "activation": args.activation,
                "goodness": args.goodness,
                "norm_mode": norm_mode,
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "threshold": args.threshold,
                "jitter_pixels": args.jitter_pixels,
                "train_subset": train_subset or 0,
                "epoch_time_sec": epoch_time,
                "elapsed_sec": elapsed,
                "error_threshold": args.error_threshold,
                "convergence_epoch": convergence_epoch or 0,
            },
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"grad_norm={mean_grad_norm:.3f} val_error={val_err:.2f}% test_error={test_err:.2f}% "
            f"epoch_time={epoch_time:.1f}s"
        )


if __name__ == "__main__":
    main()
