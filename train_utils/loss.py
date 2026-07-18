import torch


def build_token_loss_fn(args, device):
    """
    Build per-token loss function with `reduction='none'` for token prediction.

    Notes:
    - We keep `reduction='none'` because training applies its own masking/weighting reductions.
    - `disable_masking_loss` preserves previous behavior: no class weights, no ignore-mask reduction.
    """
    loss_type = str(getattr(args, "loss_type", "ce")).lower()

    class_weights = None
    if (
        (not bool(getattr(args, "disable_masking_loss", False)))
        and (not bool(getattr(args, "disable_all_weighting", False)))
        and bool(getattr(args, "use_precomputed_weights", False))
        and hasattr(args, "class_weights")
    ):
        class_weights = args.class_weights.to(device)

    if loss_type in {"ce", "cross_entropy", "crossentropy"}:
        return torch.nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    if loss_type in {"focal", "focal_loss"}:
        from losses import FocalLoss  # local package import (RemoteVAR scope)

        gamma = float(getattr(args, "focal_gamma", 2.0))
        return FocalLoss(gamma=gamma, alpha=None, weight=class_weights, reduction="none")

    raise ValueError(f"Unknown loss_type='{loss_type}'. Expected: 'ce' or 'focal'.")

