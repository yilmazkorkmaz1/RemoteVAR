"""
Mask evaluation helpers (binary change detection).

This module is intentionally lightweight and depends only on torch/numpy and `utils/metrics.py`.
It provides:
- Robust conversion from model/GT tensors (often in [-1,1]) to binary masks {0,1}
- Confusion-matrix accumulation for n_cl=2
- Convenience wrappers for distributed reduction (Accelerate) and score computation
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

from utils.metrics import compute_score


def to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor to [0,1] range for visualization/thresholding.
    - If values look like [-1,1], map via (x+1)/2
    - Otherwise clamp to [0,1]
    """
    if x.numel() == 0:
        return x
    x_min = float(x.min().detach().cpu().item())
    x_max = float(x.max().detach().cpu().item())
    if x_min < -0.05 or x_max > 1.05:
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)


def extract_change_map(images: torch.Tensor, *, image_size: int) -> torch.Tensor:
    """
    RemoteVAR can return concatenated images along height: [img1, img2, img3] where img3 is the change map.
    If the height equals 3*image_size, return only the last third; otherwise return as-is.
    """
    if images.dim() != 4:
        raise ValueError(f"images must be 4D (B,C,H,W), got shape={tuple(images.shape)}")
    _, _, h, _ = images.shape
    if h == 3 * int(image_size):
        return images[:, :, 2 * image_size :, :]
    return images


def _otsu_threshold_from_uint8(img_u8: np.ndarray) -> int:
    """
    Classic Otsu threshold for a single-channel uint8 image.
    Returns threshold T in [0,255]. Foreground is img > T.
    """
    # Histogram
    hist = np.bincount(img_u8.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0

    # Probabilities
    p = hist / total
    omega = np.cumsum(p)  # class probabilities
    mu = np.cumsum(p * np.arange(256))  # class means (unnormalized)
    mu_t = mu[-1]

    # Between-class variance: (mu_t*omega - mu)^2 / (omega*(1-omega))
    denom = omega * (1.0 - omega)
    # Avoid division by zero
    denom = np.where(denom > 1e-12, denom, np.nan)
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    # Pick the threshold that maximizes between-class variance
    t = int(np.nanargmax(sigma_b2))
    if not np.isfinite(t):
        return 0
    return t


def rgb01_to_binary_01(rgb01: torch.Tensor, *, thr_01: Optional[float] = None) -> torch.Tensor:
    """
    Convert an RGB (or multi-channel) tensor in [0,1] to a binary mask {0,1}.
    If thr_01 is provided, threshold max-channel at thr_01.
    If thr_01 is None, use per-sample Otsu threshold over max-channel grayscale.
    Accepts (B,C,H,W) and returns (B,H,W) uint8-like (float/int).
    """
    if rgb01.dim() != 4:
        raise ValueError(f"rgb01 must be 4D (B,C,H,W), got shape={tuple(rgb01.shape)}")
    m = rgb01.max(dim=1).values  # (B,H,W)
    if thr_01 is not None:
        return (m > float(thr_01)).to(dtype=torch.long)

    # Otsu per sample (CPU numpy)
    m_u8 = (m.detach().cpu().clamp(0, 1) * 255.0).to(torch.uint8).numpy()
    out = np.zeros((m_u8.shape[0], m_u8.shape[1], m_u8.shape[2]), dtype=np.int64)
    for i in range(m_u8.shape[0]):
        t = _otsu_threshold_from_uint8(m_u8[i])
        out[i] = (m_u8[i] > t).astype(np.int64)
    # Return on the same device as input to avoid CPU/GPU mismatches downstream.
    return torch.from_numpy(out).to(device=rgb01.device, dtype=torch.long)


@torch.no_grad()
def confusion_from_pred_and_gt(
    *,
    pred_images: torch.Tensor,
    gt_masks: torch.Tensor,
    image_size: int,
    pred_thr_01: Optional[float] = 0.1,
    gt_thr_01: Optional[float] = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build confusion matrix (2x2), labeled count, and correct count for binary masks.

    Args:
        pred_images: model output images (B,3,H,W) OR concatenated (B,3,3H,W)
        gt_masks: ground truth masks (B,1/3,H,W) possibly in [-1,1]
        image_size: expected H/W (used for split when concatenated)
        pred_thr_01: threshold in [0,1] for predicted foreground. Default is 0.1.
        gt_thr_01: threshold in [0,1] for GT foreground. Default is 0.1 to:
                  - correctly handle RGB location-coded masks (foreground max-channel is typically >= ~0.25)
                  - ignore tiny reconstruction noise near black (important for VQVAE reconstructions)
    Returns:
        hist: (2,2) float tensor
        labeled: scalar float tensor
        correct: scalar float tensor
    """
    # Pred: take only change-map part, then threshold.
    pred_change = extract_change_map(pred_images, image_size=image_size)
    pred01 = to_01(pred_change)
    pred_bin = rgb01_to_binary_01(pred01, thr_01=pred_thr_01)  # (B,H,W) long 0/1

    # GT: robust to 1ch/3ch; treat non-black as foreground.
    gt01 = to_01(gt_masks)
    gt_bin = rgb01_to_binary_01(gt01, thr_01=gt_thr_01)  # (B,H,W) long 0/1

    # Confusion matrix in torch
    n_cl = 2
    k = (gt_bin >= 0) & (gt_bin < n_cl)
    labeled = k.sum().to(dtype=torch.float32)
    correct = (pred_bin[k] == gt_bin[k]).sum().to(dtype=torch.float32)
    idx = (n_cl * gt_bin[k] + pred_bin[k]).to(dtype=torch.int64)
    hist = torch.bincount(idx, minlength=n_cl**2).reshape(n_cl, n_cl).to(dtype=torch.float32)
    return hist, labeled, correct


def scores_from_confusion(
    *,
    hist: torch.Tensor,
    labeled: torch.Tensor,
    correct: torch.Tensor,
) -> Dict[str, Any]:
    """
    Compute IoU / precision / recall / pixel-acc metrics from confusion data.
    Uses `utils.metrics.compute_score` for consistency with the project.
    """
    h = hist.detach().cpu().numpy().astype(np.float64)
    labeled_n = float(labeled.detach().cpu().item())
    correct_n = float(correct.detach().cpu().item())
    iou, recall_1, precision_1, mean_IoU, mean_IoU_no_back, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(
        h, correct_n, labeled_n
    )
    # iou is np array, typically length 2 for binary
    out: Dict[str, Any] = {
        "iou_bg": float(iou[0]) if len(iou) > 0 else float("nan"),
        "iou_fg": float(iou[1]) if len(iou) > 1 else float("nan"),
        "mean_iou": float(mean_IoU),
        "mean_pixel_acc": float(mean_pixel_acc),
        "pixel_acc": float(pixel_acc),
        "precision_fg": float(precision_1),
        "recall_fg": float(recall_1),
        "freq_iou": float(freq_IoU),
        "mean_iou_no_back": float(mean_IoU_no_back),
        "labeled": float(labeled_n),
        "correct": float(correct_n),
    }
    return out


