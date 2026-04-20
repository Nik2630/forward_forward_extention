from __future__ import annotations

import torch


def overlay_label_one_hot(images: torch.Tensor, labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """Overlay one-hot label into the first num_classes pixels of flattened images."""
    if images.ndim != 2:
        raise ValueError("Expected flattened image tensor of shape [B, D]")
    out = images.clone()
    out[:, :num_classes] = 0.0
    out[torch.arange(images.shape[0]), labels] = 1.0
    return out


def sample_incorrect_labels(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    rand = torch.randint(0, num_classes - 1, labels.shape, device=labels.device)
    return rand + (rand >= labels).long()
