from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multiclass focal loss for logits and integer class targets."""

    def __init__(
        self,
        *,
        gamma: float = 2.0,
        alpha: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "none",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = str(reduction)
        self.eps = float(eps)
        self.weight = weight

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(list(alpha), dtype=torch.float32)
        else:
            self.alpha = float(alpha)

        if self.reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction={self.reduction}. Expected one of: none|mean|sum")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce.clamp_min(0.0))
        focal = (1.0 - pt).clamp_min(self.eps).pow(self.gamma) * ce

        if self.alpha is not None:
            if isinstance(self.alpha, float):
                focal = focal * self.alpha
            else:
                alpha = self.alpha.to(device=targets.device, dtype=focal.dtype)
                if alpha.ndim != 1:
                    raise ValueError("alpha must be a scalar or a 1D tensor/list of per-class weights")
                if targets.dtype != torch.long:
                    targets = targets.long()
                safe_targets = targets.clamp_min(0)
                alpha_targets = alpha.gather(0, safe_targets)
                alpha_targets = torch.where(
                    targets == self.ignore_index,
                    torch.zeros_like(alpha_targets),
                    alpha_targets,
                )
                focal = focal * alpha_targets

        if self.reduction == "none":
            return focal

        valid = targets != self.ignore_index
        focal_valid = focal[valid]
        if self.reduction == "sum":
            return focal_valid.sum()
        return focal_valid.mean() if focal_valid.numel() > 0 else focal.sum() * 0.0
