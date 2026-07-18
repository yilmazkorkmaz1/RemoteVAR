import torch
import argparse
import os
import copy
from models import build_remote_var
from models.vqvae import VQVAE
from models.vae_modules import ConditionedDecoder
from models.remote_var import RemoteVAR
from ruamel.yaml import YAML
from safetensors.torch import load_file
from remotevar_datasets import create_dataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import datetime
from tqdm import tqdm

from utils.mask_metrics import (
    confusion_from_pred_and_gt,
    scores_from_confusion,
    to_01,
    extract_change_map,
    rgb01_to_binary_01,
)


def _load_arial_font(size: int = 20):
    """
    Best-effort Arial font loader for PIL labels.

    Arial exists by default on macOS, but may be missing on Linux servers. We try common locations and
    fall back to DejaVuSans-Bold (or PIL default) so inference never crashes.
    """
    # Optional override
    env_path = os.environ.get("ARIAL_FONT_PATH") or os.environ.get("INFERENCE_FONT_PATH")
    if env_path:
        try:
            if os.path.exists(env_path):
                return ImageFont.truetype(env_path, int(size))
        except Exception:
            pass

    home = os.path.expanduser("~")
    candidates = [
        # macOS user fonts
        os.path.join(home, "Library", "Fonts", "Arial.ttf"),
        os.path.join(home, "Library", "Fonts", "Arial Bold.ttf"),
        # macOS system fonts
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        # Common Linux installs (msttcorefonts / mscorefonts)
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/microsoft/Arial.ttf",
        "/usr/share/fonts/truetype/microsoft/arial.ttf",
        "/usr/share/fonts/truetype/ttf-mscorefonts-installer/Arial.ttf",
    ]
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return ImageFont.truetype(p, int(size))
        except Exception:
            continue

    # Sometimes FreeType can resolve by name
    for name in ["Arial.ttf", "Arial Bold.ttf", "Arial", "arial.ttf", "arial"]:
        try:
            return ImageFont.truetype(name, int(size))
        except Exception:
            continue

    # Fallback
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(size))
    except Exception:
        try:
            return ImageFont.load_default()
        except Exception:
            return None

def create_comparison_image(
    pre_images,
    post_images,
    generated_change_maps,
    ground_truth_masks,
    batch_size,
    image_size,
    confidence_maps=None,
    samples_per_row: int = None,
    pred_thr_01: float = 0.1,
    gt_thr_01: float = 0.1,
    entropy_roi: str = "none",
    entropy_roi_norm: str = "none",
    entropy_roi_q_low: float = 0.05,
    entropy_roi_q_high: float = 0.95,
    entropy_roi_gamma: float = 1.0,
):
    """
    Create comprehensive comparison visualization showing Pre, Post, Generated Change, optional confidence, and Ground Truth.

    Args:
        pre_images: Pre-event images (batch_size, 3, H, W)
        post_images: Post-event images (batch_size, 3, H, W)
        generated_change_maps: Generated change maps - when multi_cond=True, this is concatenated (batch_size, 3, 3*H, W)
        ground_truth_masks: Ground truth change masks (batch_size, 1, H, W)
        confidence_maps: Optional confidence maps for mask tokens (batch_size, 1 or 3, H, W) in [0,1]
        batch_size: Batch size
        image_size: Image size
        samples_per_row: If provided and batch_size>1, how many samples to place per row in the final figure.
                         Use 1 to stack samples as new rows.

    Returns:
        PIL.Image: The comparison visualization image
    """
    # Ensure all tensors are on CPU for visualization
    pre_images = pre_images.cpu()
    post_images = post_images.cpu()
    generated_change_maps = generated_change_maps.cpu()
    ground_truth_masks = ground_truth_masks.cpu()
    if confidence_maps is not None:
        confidence_maps = confidence_maps.cpu()

    # Ensure generated change maps have 3 channels for visualization consistency.
    # (Decoder-refiner may output 1-channel logits/probabilities; we visualize it as grayscale RGB.)
    if generated_change_maps.shape[1] == 1:
        generated_change_maps = generated_change_maps.repeat(1, 3, 1, 1)
    elif generated_change_maps.shape[1] != 3:
        if generated_change_maps.shape[1] > 3:
            generated_change_maps = generated_change_maps[:, :3, :, :]
        else:
            generated_change_maps = generated_change_maps[:, 0:1, :, :].repeat(1, 3, 1, 1)

    # Convert ground truth masks to 3-channel for consistency
    # Handle different input formats (1-channel, 3-channel, or other)
    if ground_truth_masks.shape[1] == 1:
        # Single channel mask - repeat to 3 channels
        gt_masks_rgb = ground_truth_masks.repeat(1, 3, 1, 1)  # (batch_size, 3, H, W)
    elif ground_truth_masks.shape[1] == 3:
        # Already 3-channel
        gt_masks_rgb = ground_truth_masks
    else:
        # Other number of channels - take first 3 or repeat first channel
        if ground_truth_masks.shape[1] >= 3:
            gt_masks_rgb = ground_truth_masks[:, :3, :, :]  # Take first 3 channels
        else:
            gt_masks_rgb = ground_truth_masks[:, 0:1, :, :].repeat(1, 3, 1, 1)  # Repeat first channel

    # Split generated change maps if they're concatenated (when multi_cond=True)
    # The model returns concatenated images: [img1, img2, img3] along height dimension
    _, _, H_concat, W = generated_change_maps.shape
    if H_concat == 3 * image_size:  # Check if concatenated (3*256 = 768 for 256x256 images)
        # Split into 3 separate images along the height dimension
        # RemoteVAR returns: [img1, img2, img3] where img3 is the generated change map
        gen_change_1 = generated_change_maps[:, :, :image_size, :]          # img1: First part
        gen_change_2 = generated_change_maps[:, :, image_size:2*image_size, :]  # img2: Second part
        gen_change_3 = generated_change_maps[:, :, 2*image_size:, :]        # img3: Third part (change map)
        # For change detection, we use img3 as the generated change map
        generated_change_maps = gen_change_3
    # Otherwise, assume it's already the single change map

    def _to_01(x: torch.Tensor) -> torch.Tensor:
        # If already in [0,1], keep; else assume [-1,1] and map.
        if x.min().item() < -0.05 or x.max().item() > 1.05:
            return ((x + 1) / 2).clamp(0, 1)
        return x.clamp(0, 1)

    def _rgb_to_gray01(rgb01: torch.Tensor, thr: float = 0.10) -> torch.Tensor:
        # rgb01: (1,3,H,W) in [0,1]
        # Use max channel and a non-trivial threshold to avoid turning near-black noise into foreground.
        fg = (rgb01.max(dim=1, keepdim=True).values > thr).float()
        return fg.repeat(1, 3, 1, 1)

    def _conf_to_rgb01(conf: torch.Tensor) -> torch.Tensor:
        # conf: (1,1,H,W) or (1,3,H,W), expected in [0,1]
        # Predictive entropy in [0,1]; brighter means more uncertain.
        if conf is None:
            return None
        if conf.dim() != 4:
            raise ValueError(f"confidence_maps must be 4D (B,C,H,W), got shape={tuple(conf.shape)}")
        if conf.shape[1] == 1:
            conf_rgb = conf.repeat(1, 3, 1, 1)
        elif conf.shape[1] >= 3:
            conf_rgb = conf[:, :3, :, :]
        else:
            conf_rgb = conf[:, 0:1, :, :].repeat(1, 3, 1, 1)
        return conf_rgb.clamp(0, 1)

    def _entropy_to_heatmap_rgb01(ent01: torch.Tensor) -> torch.Tensor:
        """
        Convert entropy map in [0,1] to an RGB heatmap biased towards red/yellow.
        ent01: (1,1,H,W) or (1,3,H,W) in [0,1]
        Returns: (1,3,H,W) in [0,1]
        """
        ent_rgb = _conf_to_rgb01(ent01)
        if ent_rgb is None:
            return None
        e = ent_rgb[:, :1, :, :].clamp(0, 1)  # (1,1,H,W)
        # Simple red->yellow ramp:
        # - low entropy: black
        # - mid entropy: red
        # - high entropy: yellow
        r = e
        g = (2.0 * e - 1.0).clamp(0.0, 1.0)
        b = torch.zeros_like(e)
        return torch.cat([r, g, b], dim=1).clamp(0, 1)

    def _overlay(base_rgb01: torch.Tensor, heat_rgb01: torch.Tensor, alpha01: torch.Tensor) -> torch.Tensor:
        """
        Alpha blend heatmap onto base: out = (1-a)*base + a*heat
        base_rgb01: (1,3,H,W), heat_rgb01: (1,3,H,W), alpha01: (1,1,H,W) in [0,1]
        """
        a = alpha01.clamp(0, 1)
        if a.shape[1] != 1:
            a = a[:, :1, :, :]
        return ((1.0 - a) * base_rgb01 + a * heat_rgb01).clamp(0, 1)

    def _normalize_entropy_within_roi(
        ent01_1ch: torch.Tensor,
        roi01_1ch: torch.Tensor,
        *,
        mode: str,
        q_low: float,
        q_high: float,
        gamma: float,
    ) -> torch.Tensor:
        """
        Contrast-stretch entropy within a region-of-interest (ROI) to reveal uncertainty *within* change regions.
        Returns a 1-channel tensor in [0,1], masked outside ROI.
        """
        if ent01_1ch is None or roi01_1ch is None:
            return ent01_1ch
        if ent01_1ch.dim() != 4 or roi01_1ch.dim() != 4:
            return ent01_1ch
        if ent01_1ch.shape[-2:] != roi01_1ch.shape[-2:]:
            return ent01_1ch

        m = (roi01_1ch[:, :1, :, :] > 0.5)
        out = torch.zeros_like(ent01_1ch[:, :1, :, :]).clamp(0, 1)
        if int(m.sum().item()) <= 0:
            return out

        v = ent01_1ch[:, :1, :, :][m].float()
        if v.numel() == 0:
            return out

        mode = str(mode or "none").strip().lower()
        if mode in {"none", "mask"}:
            out[m] = ent01_1ch[:, :1, :, :].clamp(0, 1)[m]
            return out

        # Robust quantile defaults
        ql = float(q_low)
        qh = float(q_high)
        if not (0.0 <= ql <= 1.0):
            ql = 0.05
        if not (0.0 <= qh <= 1.0):
            qh = 0.95
        if qh < ql:
            ql, qh = qh, ql

        if mode == "minmax":
            lo = v.min()
            hi = v.max()
        else:  # "quantile"
            lo = torch.quantile(v, ql)
            hi = torch.quantile(v, qh)

        denom = (hi - lo).abs().clamp_min(1e-8)
        scaled = ((ent01_1ch[:, :1, :, :].float() - lo) / denom).clamp(0, 1)

        g = float(gamma)
        if g > 0 and abs(g - 1.0) > 1e-6:
            scaled = scaled.pow(g)

        out[m] = scaled[m]
        return out.clamp(0, 1)

    # Process each sample in the batch
    all_comparison_images = []

    for i in range(batch_size):
        # Extract individual images for this sample
        pre_img = pre_images[i:i+1]      # (1, 3, H, W)
        post_img = post_images[i:i+1]    # (1, 3, H, W)
        gen_change = generated_change_maps[i:i+1]  # (1, 3, H, W)
        gt_mask = gt_masks_rgb[i:i+1]    # (1, 3, H, W)
        conf = confidence_maps[i:i+1] if confidence_maps is not None else None

        # Normalize to [0,1] for visualization
        pre_img_norm = _to_01(pre_img)
        post_img_norm = _to_01(post_img)
        gen_change_norm = _to_01(gen_change)
        gt_mask_norm = _to_01(gt_mask)
        conf_norm = _conf_to_rgb01(conf) if conf is not None else None

        # Back-convert RGB masks to grayscale/binary (non-black -> white)
        # Match metric thresholds for visualization too (binary BW only).
        gen_change_gray = _rgb_to_gray01(gen_change_norm, thr=float(pred_thr_01))
        gt_mask_gray = _rgb_to_gray01(gt_mask_norm, thr=float(gt_thr_01))

        # Overlay predictive entropy heatmap on top of predicted grayscale mask.
        # Use entropy as alpha so high entropy regions are highlighted more strongly.
        heat_rgb = _entropy_to_heatmap_rgb01(conf_norm) if conf_norm is not None else None
        if heat_rgb is not None:
            # Optionally "focus" entropy contrast within change regions (helpful for highly imbalanced datasets):
            # - default: no change (global normalization as returned by the model)
            # - focus: mask + re-normalize entropy within ROI (pred/gt/union/tp), so you can see which
            #          parts of a change blob are uncertain.
            ent01 = conf_norm[:, :1, :, :].clamp(0, 1)  # (1,1,H,W)

            roi_mode = str(entropy_roi or "none").strip().lower()
            roi01 = None
            base_for_overlay = gen_change_gray
            if roi_mode != "none":
                if roi_mode == "pred":
                    roi01 = gen_change_gray[:, :1, :, :]
                    base_for_overlay = gen_change_gray
                elif roi_mode == "gt":
                    roi01 = gt_mask_gray[:, :1, :, :]
                    base_for_overlay = gt_mask_gray
                elif roi_mode == "union":
                    roi01 = ((gen_change_gray[:, :1, :, :] > 0.5) | (gt_mask_gray[:, :1, :, :] > 0.5)).float()
                    base_for_overlay = roi01.repeat(1, 3, 1, 1)
                elif roi_mode == "tp":
                    roi01 = ((gen_change_gray[:, :1, :, :] > 0.5) & (gt_mask_gray[:, :1, :, :] > 0.5)).float()
                    base_for_overlay = roi01.repeat(1, 3, 1, 1)

                if roi01 is not None:
                    ent01 = _normalize_entropy_within_roi(
                        ent01,
                        roi01,
                        mode=str(entropy_roi_norm or "quantile"),
                        q_low=float(entropy_roi_q_low),
                        q_high=float(entropy_roi_q_high),
                        gamma=float(entropy_roi_gamma),
                    )

            heat_rgb = _entropy_to_heatmap_rgb01(ent01) if ent01 is not None else None
            alpha = ent01.clamp(0, 1) if ent01 is not None else conf_norm[:, :1, :, :].clamp(0, 1)
            alpha = (alpha * 0.85).clamp(0, 0.85)  # cap overlay strength
            pred_gray_entropy_overlay = _overlay(base_for_overlay, heat_rgb, alpha)
        else:
            pred_gray_entropy_overlay = gen_change_gray

        # Create a row with images side by side
        # Columns: Pre | Post | Pred(RGB) | Pred(gray) | Pred(gray)+EntropyHeat | GT(RGB) | GT(gray)
        row_images = [
            pre_img_norm.squeeze(0),
            post_img_norm.squeeze(0),
            gen_change_norm.squeeze(0),
            gen_change_gray.squeeze(0),
            pred_gray_entropy_overlay.squeeze(0),
            gt_mask_norm.squeeze(0),
            gt_mask_gray.squeeze(0),
        ]

        # Use make_grid with the correctly shaped tensors
        row_grid = make_grid(row_images, nrow=7, padding=10, pad_value=1.0)  # White padding

        # Convert to PIL Image for adding text labels
        row_np = row_grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        row_pil = Image.fromarray(row_np)

        # Add text labels
        draw = ImageDraw.Draw(row_pil)

        font = _load_arial_font(size=20)

        # Label positions (centered above each image section)
        img_width = image_size
        ent_label = "Pred + EntropyHeat"
        try:
            if str(entropy_roi or "none").strip().lower() != "none":
                ent_label = f"ROI({str(entropy_roi).strip()}) + EntropyHeat"
        except Exception:
            ent_label = "Pred + EntropyHeat"
        labels = ["Pre-Image", "Post-Image", "Pred (RGB)", "Pred (gray)", ent_label, "GT (RGB)", "GT (gray)"]
        for j, label in enumerate(labels):
            x_pos = j * (img_width + 10) + img_width // 2  # Center of each image
            if font:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                draw.text((x_pos - text_width//2, 5), label, fill=(255, 255, 255), font=font,
                         stroke_fill=(0, 0, 0), stroke_width=2)
            else:
                draw.text((x_pos - 40, 10), label, fill=(255, 255, 255))

        # Convert back to tensor for grid creation
        row_tensor = torch.from_numpy(np.array(row_pil)).permute(2, 0, 1).float() / 255.0
        all_comparison_images.append(row_tensor)

    # Create final grid with all samples (one row per sample)
    if len(all_comparison_images) > 1:
        nrow = len(all_comparison_images) if samples_per_row is None else int(samples_per_row)
        nrow = max(1, nrow)
        final_grid = make_grid(all_comparison_images, nrow=nrow, padding=20, pad_value=0.5)  # Gray padding between samples
    else:
        final_grid = all_comparison_images[0]

    # Convert to PIL Image
    final_np = final_grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    final_image = Image.fromarray(final_np)

    return final_image


def create_entropy_scales_image(
    *,
    generated_change_maps: torch.Tensor,
    image_size: int,
    confidence_maps_per_stage: list,
    confidence_map_agg: torch.Tensor = None,
    ground_truth_masks: torch.Tensor = None,
    patch_nums: list = None,
    pred_thr_01: float = 0.1,
    gt_thr_01: float = 0.1,
    entropy_roi: str = "none",
    entropy_roi_norm: str = "none",
    entropy_roi_q_low: float = 0.05,
    entropy_roi_q_high: float = 0.95,
    entropy_roi_gamma: float = 1.0,
    label_h: int = 34,
    padding: int = 10,
) -> Image.Image:
    """
    Visualize predictive entropy at EACH autoregressive stage (per-scale) plus an aggregated map.

    Each tile is: Pred(BW) with entropy heatmap overlaid (alpha = entropy).
    Layout (single row):
      Pred(BW) | Ent@pn1 | Ent@pn2 | ... | Ent@pnN | Ent@Agg

    Args:
        generated_change_maps: model output (B,3,H,W) OR concatenated (B,3,3H,W)
        confidence_maps_per_stage: list of tensors (B,1,H,W) in [0,1] (already upsampled to pixel space)
        confidence_map_agg: optional tensor (B,1,H,W) in [0,1]
    """
    if generated_change_maps is None:
        raise ValueError("generated_change_maps is required")
    if not isinstance(confidence_maps_per_stage, list) or len(confidence_maps_per_stage) == 0:
        raise ValueError("confidence_maps_per_stage must be a non-empty list")

    # CPU for visualization
    gen = generated_change_maps.cpu()
    gen = extract_change_map(gen, image_size=int(image_size))
    gen01 = to_01(gen)
    if gen01.dim() != 4:
        raise ValueError(f"generated_change_maps must be 4D (B,C,H,W), got {tuple(gen01.shape)} after extract/to_01")

    # Ensure RGB for pred mask derivation
    if int(gen01.shape[1]) == 1:
        gen01 = gen01.repeat(1, 3, 1, 1)
    elif int(gen01.shape[1]) > 3:
        gen01 = gen01[:, :3, :, :]
    elif int(gen01.shape[1]) != 3:
        gen01 = gen01[:, 0:1, :, :].repeat(1, 3, 1, 1)

    # Binary BW pred mask in RGB (non-black -> white)
    pred_bw = (gen01.max(dim=1, keepdim=True).values > float(pred_thr_01)).float().repeat(1, 3, 1, 1)
    pred_roi01 = pred_bw[:, :1, :, :]

    gt_roi01 = None
    if ground_truth_masks is not None:
        try:
            gt = ground_truth_masks.cpu()
            gt01 = to_01(gt)
            if gt01.dim() == 4:
                gt_roi01 = (gt01.max(dim=1, keepdim=True).values > float(gt_thr_01)).float()
        except Exception:
            gt_roi01 = None

    def _conf_to_rgb01(conf: torch.Tensor) -> torch.Tensor:
        if conf is None:
            return None
        if conf.dim() != 4:
            raise ValueError(f"confidence map must be 4D (B,C,H,W), got {tuple(conf.shape)}")
        c = conf
        if c.shape[1] != 1:
            c = c[:, :1, :, :]
        return c.clamp(0, 1).repeat(1, 3, 1, 1)

    def _entropy_to_heatmap_rgb01(ent01: torch.Tensor) -> torch.Tensor:
        ent_rgb = _conf_to_rgb01(ent01)
        if ent_rgb is None:
            return None
        e = ent_rgb[:, :1, :, :].clamp(0, 1)  # (B,1,H,W)
        r = e
        g = (2.0 * e - 1.0).clamp(0.0, 1.0)
        b = torch.zeros_like(e)
        return torch.cat([r, g, b], dim=1).clamp(0, 1)

    def _overlay(base_rgb01: torch.Tensor, heat_rgb01: torch.Tensor, alpha01: torch.Tensor) -> torch.Tensor:
        a = alpha01.clamp(0, 1)
        if a.shape[1] != 1:
            a = a[:, :1, :, :]
        return ((1.0 - a) * base_rgb01 + a * heat_rgb01).clamp(0, 1)

    def _normalize_entropy_within_roi(
        ent01_1ch: torch.Tensor,
        roi01_1ch: torch.Tensor,
        *,
        mode: str,
        q_low: float,
        q_high: float,
        gamma: float,
    ) -> torch.Tensor:
        if ent01_1ch is None or roi01_1ch is None:
            return ent01_1ch
        m = (roi01_1ch[:, :1, :, :] > 0.5)
        out = torch.zeros_like(ent01_1ch[:, :1, :, :]).clamp(0, 1)
        if int(m.sum().item()) <= 0:
            return out
        v = ent01_1ch[:, :1, :, :][m].float()
        if v.numel() == 0:
            return out
        mode = str(mode or "none").strip().lower()
        if mode in {"none", "mask"}:
            out[m] = ent01_1ch[:, :1, :, :].clamp(0, 1)[m]
            return out
        ql = float(q_low)
        qh = float(q_high)
        if not (0.0 <= ql <= 1.0):
            ql = 0.05
        if not (0.0 <= qh <= 1.0):
            qh = 0.95
        if qh < ql:
            ql, qh = qh, ql
        if mode == "minmax":
            lo = v.min()
            hi = v.max()
        else:
            lo = torch.quantile(v, ql)
            hi = torch.quantile(v, qh)
        denom = (hi - lo).abs().clamp_min(1e-8)
        scaled = ((ent01_1ch[:, :1, :, :].float() - lo) / denom).clamp(0, 1)
        g = float(gamma)
        if g > 0 and abs(g - 1.0) > 1e-6:
            scaled = scaled.pow(g)
        out[m] = scaled[m]
        return out.clamp(0, 1)

    roi_mode = str(entropy_roi or "none").strip().lower()
    roi01 = None
    base_for_overlay = pred_bw
    if roi_mode != "none":
        if roi_mode == "pred":
            roi01 = pred_roi01
            base_for_overlay = pred_bw
        elif roi_mode == "gt" and gt_roi01 is not None:
            roi01 = gt_roi01
            base_for_overlay = roi01.repeat(1, 3, 1, 1)
        elif roi_mode == "union" and gt_roi01 is not None:
            roi01 = ((pred_roi01 > 0.5) | (gt_roi01 > 0.5)).float()
            base_for_overlay = roi01.repeat(1, 3, 1, 1)
        elif roi_mode == "tp" and gt_roi01 is not None:
            roi01 = ((pred_roi01 > 0.5) & (gt_roi01 > 0.5)).float()
            base_for_overlay = roi01.repeat(1, 3, 1, 1)
    # Build tiles for the FIRST sample only (B should be 1 in inference.py, but keep robust)
    b0 = 0
    tiles = []
    labels = []

    tiles.append(pred_bw[b0].cpu())
    labels.append("Pred (BW)")

    pn = list(patch_nums) if patch_nums is not None else None
    if pn is None:
        pn = [None for _ in range(len(confidence_maps_per_stage))]
    if len(pn) != len(confidence_maps_per_stage):
        m = min(len(pn), len(confidence_maps_per_stage))
        pn = pn[:m]
        confidence_maps_per_stage = confidence_maps_per_stage[:m]

    for pni, cm in zip(pn, confidence_maps_per_stage):
        if cm is None:
            tiles.append(pred_bw[b0].cpu())
            labels.append(f"Ent pn={int(pni) if pni is not None else '?'} (missing)")
            continue
        c = cm.cpu()
        if c.dim() == 3:
            c = c.unsqueeze(1)
        if c.shape[1] != 1:
            c = c[:, :1, :, :]
        if roi01 is not None:
            c = _normalize_entropy_within_roi(
                c,
                roi01,
                mode=str(entropy_roi_norm or "quantile"),
                q_low=float(entropy_roi_q_low),
                q_high=float(entropy_roi_q_high),
                gamma=float(entropy_roi_gamma),
            )
        heat = _entropy_to_heatmap_rgb01(c)
        alpha = (c[:, :1, :, :].clamp(0, 1) * 0.85).clamp(0, 0.85)
        ov = _overlay(base_for_overlay.cpu(), heat, alpha)
        tiles.append(ov[b0].cpu())
        labels.append(f"Ent pn={int(pni)}" if pni is not None else "Ent pn=?")

    if confidence_map_agg is not None:
        c = confidence_map_agg.cpu()
        if c.dim() == 3:
            c = c.unsqueeze(1)
        if c.shape[1] != 1:
            c = c[:, :1, :, :]
        if roi01 is not None:
            c = _normalize_entropy_within_roi(
                c,
                roi01,
                mode=str(entropy_roi_norm or "quantile"),
                q_low=float(entropy_roi_q_low),
                q_high=float(entropy_roi_q_high),
                gamma=float(entropy_roi_gamma),
            )
        heat = _entropy_to_heatmap_rgb01(c)
        alpha = (c[:, :1, :, :].clamp(0, 1) * 0.85).clamp(0, 0.85)
        ov = _overlay(base_for_overlay.cpu(), heat, alpha)
        tiles.append(ov[b0].cpu())
        labels.append("Ent Agg")

    # Label band above each tile
    font = _load_arial_font(size=20)

    def _tile_with_label(tile01_chw: torch.Tensor, label: str) -> torch.Tensor:
        arr = tile01_chw.clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        tile_pil = Image.fromarray(arr)
        w, h = tile_pil.size
        canvas = Image.new("RGB", (w, h + int(label_h)), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        if font:
            bbox = d.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = max(0, (w - tw) // 2)
            y = max(0, (int(label_h) - th) // 2)
            d.text((x, y), label, fill=(0, 0, 0), font=font)
        else:
            d.text((5, 5), label, fill=(0, 0, 0))
        canvas.paste(tile_pil, (0, int(label_h)))
        return torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0

    labeled_tiles = [_tile_with_label(t, lab) for t, lab in zip(tiles, labels)]
    nrow = len(labeled_tiles) if len(labeled_tiles) > 0 else 1
    grid = make_grid(labeled_tiles, nrow=nrow, padding=int(padding), pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(grid_np)


def create_intermediate_recon_image(
    pre_images: torch.Tensor,
    post_images: torch.Tensor,
    intermediate_masks: list,
    ground_truth_masks: torch.Tensor,
    image_size: int,
    patch_nums: list = None,
    tiles_per_row: int = 8,
    padding: int = 10,
    pred_thr_01: float = 0.1,
    gt_thr_01: float = 0.1,
) -> Image.Image:
    """
    Visualize intermediate (per-stage) reconstructions during autoregressive generation.

    Layout (grid):
      Pre | Post | GT(BW) | Pred@pn1(BW) | Pred@pn2(BW) | ... | Pred@pnN(BW)

    Notes:
    - `intermediate_masks` are expected in [0,1] (either 1ch prob/logits-sigmoid or 3ch RGB).
    - We render GT/pred masks as **binary black/white** using max-channel thresholding (same style as other viz):
      non-black -> white.
    """
    # Ensure CPU for visualization
    pre_images = pre_images.cpu()
    post_images = post_images.cpu()
    ground_truth_masks = ground_truth_masks.cpu()
    intermediate_masks = [t.cpu() if isinstance(t, torch.Tensor) else t for t in (intermediate_masks or [])]

    def _to_01(x: torch.Tensor) -> torch.Tensor:
        if x.min().item() < -0.05 or x.max().item() > 1.05:
            return ((x + 1) / 2).clamp(0, 1)
        return x.clamp(0, 1)

    def _ensure_rgb01(x: torch.Tensor) -> torch.Tensor:
        x01 = _to_01(x)
        if x01.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape={tuple(x01.shape)}")
        if x01.shape[1] == 3:
            return x01
        if x01.shape[1] == 1:
            return x01.repeat(1, 3, 1, 1)
        if x01.shape[1] > 3:
            return x01[:, :3, :, :]
        # unexpected: C==2 etc
        return x01[:, 0:1, :, :].repeat(1, 3, 1, 1)

    def _mask_to_gray_rgb01(x: torch.Tensor) -> torch.Tensor:
        # Binary BW mask in RGB: non-black -> white.
        x01 = _to_01(x)
        if x01.shape[1] == 1:
            # 1ch probabilities in [0,1] (sigmoid logits); use 0.5 by default.
            fg = (x01[:, :1, :, :] > float(pred_thr_01)).float()
        else:
            # RGB location-coded masks; threshold max channel to avoid near-black noise.
            fg = (x01[:, :3, :, :].max(dim=1, keepdim=True).values > float(pred_thr_01)).float()
        return fg.repeat(1, 3, 1, 1)

    pre01 = _ensure_rgb01(pre_images)
    post01 = _ensure_rgb01(post_images)
    gt01 = _to_01(ground_truth_masks)
    if gt01.shape[1] == 1:
        gt_fg = (gt01[:, :1, :, :] > float(gt_thr_01)).float()
    else:
        gt_fg = (gt01[:, :3, :, :].max(dim=1, keepdim=True).values > float(gt_thr_01)).float()
    gt_bw01 = gt_fg.repeat(1, 3, 1, 1)

    tiles = [pre01.squeeze(0), post01.squeeze(0), gt_bw01.squeeze(0)]
    labels = ["Pre-Image", "Post-Image", "GT-Mask"]

    pn = list(patch_nums) if patch_nums is not None else None
    if pn is None:
        pn = [None for _ in range(len(intermediate_masks))]
    if len(pn) != len(intermediate_masks):
        # best-effort: align lengths
        m = min(len(pn), len(intermediate_masks))
        pn = pn[:m]
        intermediate_masks = intermediate_masks[:m]

    for pni, m in zip(pn, intermediate_masks):
        if not isinstance(m, torch.Tensor):
            continue
        tiles.append(_mask_to_gray_rgb01(m).squeeze(0))
        labels.append(f"PN: {int(pni)} x {int(pni)}" if pni is not None else "pn? (BW)")

    # Render labels WITHOUT overwriting any pixel:
    # allocate a label band above each tile, then draw into that band.
    font = _load_arial_font(size=20)
    label_h = 34  # px reserved above each tile

    def _tile_with_label(tile01_chw: torch.Tensor, label: str) -> torch.Tensor:
        # tile01_chw: (3,H,W) in [0,1]
        arr = tile01_chw.clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        tile_pil = Image.fromarray(arr)
        w, h = tile_pil.size
        canvas = Image.new("RGB", (w, h + label_h), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        if font:
            bbox = d.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = max(0, (w - tw) // 2)
            y = max(0, (label_h - th) // 2)
            d.text((x, y), label, fill=(0, 0, 0), font=font)
        else:
            d.text((5, 5), label, fill=(0, 0, 0))
        canvas.paste(tile_pil, (0, label_h))
        return torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0

    labeled_tiles = [_tile_with_label(t, lab) for t, lab in zip(tiles, labels)]
    nrow = min(int(tiles_per_row), len(labeled_tiles)) if len(labeled_tiles) > 0 else 1
    grid = make_grid(labeled_tiles, nrow=nrow, padding=int(padding), pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(grid_np)


def create_intermediate_recon_image_rgb(
    pre_images: torch.Tensor,
    post_images: torch.Tensor,
    intermediate_masks: list,
    ground_truth_masks: torch.Tensor,
    image_size: int,
    patch_nums: list = None,
    tiles_per_row: int = 8,
    padding: int = 10,
) -> Image.Image:
    """
    Visualize intermediate (per-stage) reconstructions during autoregressive generation, keeping RGB masks.

    Layout (grid):
      Pre | Post | GT(RGB) | Pred@pn1(RGB) | Pred@pn2(RGB) | ... | Pred@pnN(RGB)
    """
    # Ensure CPU for visualization
    pre_images = pre_images.cpu()
    post_images = post_images.cpu()
    ground_truth_masks = ground_truth_masks.cpu()
    intermediate_masks = [t.cpu() if isinstance(t, torch.Tensor) else t for t in (intermediate_masks or [])]

    def _to_01(x: torch.Tensor) -> torch.Tensor:
        if x.min().item() < -0.05 or x.max().item() > 1.05:
            return ((x + 1) / 2).clamp(0, 1)
        return x.clamp(0, 1)

    def _ensure_rgb01(x: torch.Tensor) -> torch.Tensor:
        x01 = _to_01(x)
        if x01.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got shape={tuple(x01.shape)}")
        if x01.shape[1] == 3:
            return x01
        if x01.shape[1] == 1:
            return x01.repeat(1, 3, 1, 1)
        if x01.shape[1] > 3:
            return x01[:, :3, :, :]
        return x01[:, 0:1, :, :].repeat(1, 3, 1, 1)

    pre01 = _ensure_rgb01(pre_images)
    post01 = _ensure_rgb01(post_images)
    gt_rgb01 = _ensure_rgb01(ground_truth_masks)

    tiles = [pre01.squeeze(0), post01.squeeze(0), gt_rgb01.squeeze(0)]
    labels = ["Pre-Image", "Post-Image", "GT-Mask"]

    pn = list(patch_nums) if patch_nums is not None else None
    if pn is None:
        pn = [None for _ in range(len(intermediate_masks))]
    if len(pn) != len(intermediate_masks):
        m = min(len(pn), len(intermediate_masks))
        pn = pn[:m]
        intermediate_masks = intermediate_masks[:m]

    for pni, m in zip(pn, intermediate_masks):
        if not isinstance(m, torch.Tensor):
            continue
        tiles.append(_ensure_rgb01(m).squeeze(0))
        labels.append(f"pn: {int(pni)} (RGB)" if pni is not None else "pn? (RGB)")

    # Render labels WITHOUT overwriting any pixel:
    # allocate a label band above each tile, then draw into that band.
    font = _load_arial_font(size=20)
    label_h = 34  # px reserved above each tile

    def _tile_with_label(tile01_chw: torch.Tensor, label: str) -> torch.Tensor:
        arr = tile01_chw.clamp(0, 1).mul(255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        tile_pil = Image.fromarray(arr)
        w, h = tile_pil.size
        canvas = Image.new("RGB", (w, h + label_h), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        if font:
            bbox = d.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x = max(0, (w - tw) // 2)
            y = max(0, (label_h - th) // 2)
            d.text((x, y), label, fill=(0, 0, 0), font=font)
        else:
            d.text((5, 5), label, fill=(0, 0, 0))
        canvas.paste(tile_pil, (0, label_h))
        return torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 255.0

    labeled_tiles = [_tile_with_label(t, lab) for t, lab in zip(tiles, labels)]
    nrow = min(int(tiles_per_row), len(labeled_tiles)) if len(labeled_tiles) > 0 else 1
    grid = make_grid(labeled_tiles, nrow=nrow, padding=int(padding), pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(grid_np)


def create_vqvae_mask_recon_image(gt_masks: torch.Tensor, recon_masks: torch.Tensor, image_size: int) -> Image.Image:
    """
    Create a simple visualization for VQVAE-only mask reconstruction:
      GT(RGB) | GT(gray) | Recon(RGB) | Recon(gray)
    """
    gt_masks = gt_masks.cpu()
    recon_masks = recon_masks.cpu()

    def _to_01(x: torch.Tensor) -> torch.Tensor:
        if x.min().item() < -0.05 or x.max().item() > 1.05:
            return ((x + 1) / 2).clamp(0, 1)
        return x.clamp(0, 1)

    def _rgb_to_gray01(rgb01: torch.Tensor, thr: float = 0.1) -> torch.Tensor:
        fg = (rgb01.max(dim=1, keepdim=True).values > thr).float()
        return fg.repeat(1, 3, 1, 1)

    gt01 = _to_01(gt_masks)
    rec01 = _to_01(recon_masks)
    # Use a modest threshold to ignore tiny near-black reconstruction noise, while keeping all location-coded
    # foreground colors (typically >= ~0.25 in [0,1] when auto-levels are used).
    gt_gray = _rgb_to_gray01(gt01, thr=0.1)
    rec_gray = _rgb_to_gray01(rec01, thr=0.1)

    row_images = [
        gt01.squeeze(0),
        gt_gray.squeeze(0),
        rec01.squeeze(0),
        rec_gray.squeeze(0),
    ]
    row_grid = make_grid(row_images, nrow=4, padding=10, pad_value=1.0)
    row_np = row_grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    row_pil = Image.fromarray(row_np)

    # add labels
    draw = ImageDraw.Draw(row_pil)
    font = _load_arial_font(size=20)

    labels = ["GT (RGB)", "GT (gray)", "Recon (RGB)", "Recon (gray)"]
    img_width = int(image_size)
    for j, label in enumerate(labels):
        x_pos = j * (img_width + 10) + img_width // 2
        if font:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((x_pos - text_width // 2, 5), label, fill=(255, 255, 255), font=font,
                      stroke_fill=(0, 0, 0), stroke_width=2)
        else:
            draw.text((x_pos - 40, 10), label, fill=(255, 255, 255))

    return row_pil

def save_images(images, output_dir, batch_size, image_size, num_epochs, run_name_with_batch):
    """
    Legacy function - kept for compatibility. Use save_comparison_images instead.
    """
    # Create a grid of images
    grid = make_grid(images, nrow=int(batch_size**0.5), padding=2, pad_value=1.0)

    # Convert to numpy array and PIL Image
    grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    # Convert to PIL Image
    image = Image.fromarray(grid_np)

    # Create filename with relevant parameters
    filename = f"{run_name_with_batch}_inference_bs{batch_size}_img{image_size}.png"
    filepath = os.path.join(output_dir, filename)

    # Save the image
    image.save(filepath)
    print(f"Saved inference results to: {filepath}")


def save_prediction_pngs(
    *,
    pred_images: torch.Tensor,
    output_dir: str,
    image_size: int,
    pred_thr_01: float = 0.1,
    fns=None,
    global_offset: int = 0,
    prefix: str = "pred",
) -> None:
    """
    Save per-sample prediction PNGs for the whole test set.

    This writes the **change map** portion of the model output (if concatenated along height),
    mapped into [0,1] via `to_01`, and saved as uint8 RGB PNG.

    Filenames:
      - If `fns` is provided (list/tuple of strings), we use it as the base name.
      - We always append a global index to avoid collisions (e.g., expanded crops / duplicate stems).
    """
    if pred_images is None:
        return
    os.makedirs(output_dir, exist_ok=True)

    # IMPORTANT: match metric computation exactly:
    #   pred_change = extract_change_map(pred_images)
    #   pred01      = to_01(pred_change)
    #   pred_bin    = rgb01_to_binary_01(pred01, thr_01=pred_thr_01)
    # so saved PNGs are strictly binary (no intermediate gray values).
    pred_change = extract_change_map(pred_images, image_size=int(image_size))
    pred01 = to_01(pred_change)
    if pred01.dim() != 4:
        raise ValueError(f"pred_images must be 4D (B,C,H,W), got {tuple(pred01.shape)} after extract/to_01")
    pred_bin = rgb01_to_binary_01(pred01, thr_01=float(pred_thr_01))  # (B,H,W) long 0/1
    if pred_bin.dim() != 3:
        raise ValueError(f"rgb01_to_binary_01 must return (B,H,W), got {tuple(pred_bin.shape)}")

    B = int(pred_bin.shape[0])
    # Normalize/clean fns into a list of strings or None
    fn_list = None
    if fns is not None:
        try:
            if isinstance(fns, (list, tuple)):
                fn_list = [str(x) for x in fns]
            else:
                # DataLoader might collate into something else; best-effort cast.
                fn_list = [str(x) for x in list(fns)]
        except Exception:
            fn_list = None

    for i in range(B):
        gidx = int(global_offset) + int(i)
        base = f"{prefix}_sample"
        if fn_list is not None and i < len(fn_list):
            base = os.path.splitext(str(fn_list[i]))[0] or base
        # Avoid path separators and other oddities in stems
        base = base.replace(os.sep, "_").replace("\\", "_").strip() or "sample"
        filename = f"{base}_{gidx:06d}.png"
        out_path = os.path.join(output_dir, filename)

        arr = (pred_bin[i].detach().cpu().to(torch.uint8).numpy() * 255)  # (H,W) uint8 {0,255}
        Image.fromarray(arr, mode="L").save(out_path)


@torch.no_grad()
def pix_cond_inference(images_pre, images_post, masks, conditions, cond_type, device, B, var, vqvae, c_mask, c_img,
                       guidance_scale, top_k, top_p, seed, args, deterministic=False, return_confidence: bool = False,
                       return_confidence_all: bool = False,
                       confidence_agg: str = "mean",
                       return_intermediate: bool = False,
                       context=None, c_img_pre_idxBl=None, c_img_post_idxBl=None):
    types = {'mask': 0, 'canny': 1, 'depth': 2, 'normal': 3, 'none': 4}
    images_pre = images_pre.to(device)
    images_post = images_post.to(device)
    masks = masks.to(device)
    if isinstance(conditions, int):
        conditions = torch.tensor([conditions for _ in range(B)]).to(device)
    else:
        conditions = conditions.to(device)  # cls
    if isinstance(cond_type, str):
        cond_type = torch.tensor([types[cond_type] for _ in range(B)], device=var.device)
    else:
        cond_type = cond_type.to(device)

    with torch.no_grad():
        # Optional speed path: reuse precomputed VQVAE token IDs for pre/post images if provided
        # (e.g., validation already tokenized pre/post for teacher-forcing loss).
        if c_img_pre_idxBl is None:
            c_img_pre = vqvae.img_to_idxBl(images_pre, v_patch_nums=args.v_patch_nums)
        else:
            c_img_pre = [t.to(device) if isinstance(t, torch.Tensor) else t for t in c_img_pre_idxBl]
        if c_img_post_idxBl is None:
            c_img_post = vqvae.img_to_idxBl(images_post, v_patch_nums=args.v_patch_nums)
        else:
            c_img_post = [t.to(device) if isinstance(t, torch.Tensor) else t for t in c_img_post_idxBl]

        # Handle DDP: access the underlying module if model is wrapped
        model = var.module if hasattr(var, 'module') else var

        # Only compute context (and thus use fusion modules) if cross-attention is enabled.
        # Otherwise, pass context=None end-to-end.
        # If a precomputed context is provided (e.g., during validation), reuse it to avoid duplicate work.
        disable_ca = bool(getattr(args, "disable_cross_attention", False)) or bool(getattr(model, "disable_cross_attention", False))
        if context is None and not disable_ca:
            context = model.encode_context_with_fusion([images_pre, images_post])
        if disable_ca:
            context = None

        # Optional: UNet-style decoder skips from fusion modules (BCHW per level).
        # We only compute these when explicitly requested or when a conditioned decoder is detected.
        decoder_skips = None
        want_decoder_skips = bool(getattr(args, "use_decoder_skips", False)) or hasattr(getattr(vqvae, "decoder", None), "skip_fuse")
        if want_decoder_skips and (not disable_ca):
            # NOTE: this recomputes encoder+fusion (context was computed in BLC form above).
            # If a decoder-refiner checkpoint provided finetuned fusion modules, we use them ONLY for decoder skips by default,
            # while keeping the original fusion modules for transformer cross-attention context (avoids shifting VAR behavior).
            fm_skips = getattr(model, "fusion_modules_for_skips", None)
            if fm_skips is not None:
                fm_ctx = getattr(model, "fusion_modules", None)
                try:
                    model.fusion_modules = fm_skips
                    decoder_skips = model.encode_context_with_fusion_2d([images_pre, images_post])
                finally:
                    model.fusion_modules = fm_ctx
            else:
                decoder_skips = model.encode_context_with_fusion_2d([images_pre, images_post])

        # Set parameters for deterministic generation
        if deterministic:
            # IMPORTANT: `RemoteVAR.conditional_infer_cfg` uses CFG weights `t` in:
            #   logits = (1 + t) * logits_cond - t * logits_uncond
            # so **no guidance** means t=0 (not 1).
            cfg = [0.0, 0.0, 0.0]  # No CFG guidance
            top_k = 1  # Argmax (only top 1 token)
            top_p = 0.0  # No nucleus sampling
        else:
            cfg = [guidance_scale, guidance_scale, guidance_scale]

        # Handle DDP: access the underlying module if model is wrapped in DDP
        model = var.module if hasattr(var, 'module') else var
        out = model.conditional_infer_cfg(
            B=B,
            label_B=conditions,
            cfg=cfg,
            top_k=top_k,
            top_p=top_p,
            g_seed=seed,
            c_mask=None,
            c_img_pre=c_img_pre,
            c_img_post=c_img_post,
            cond_type=cond_type,
            context=context,
            return_confidence=return_confidence,
            return_confidence_all=return_confidence_all,
            confidence_agg=str(confidence_agg or "mean"),
            return_intermediate=return_intermediate,
            decoder_skips=decoder_skips,
        )
    return out

def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config YAML. If omitted, inference will try to load "
            "<epoch_dir>/codes/change_detection.yaml OR <run_dir>/codes/change_detection.yaml, "
            "falling back to configs/change_detection.yaml."
        ),
    )

    # data
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.environ.get("DATASET_ROOT", "data"),
        help="Root folder containing change-detection datasets (default: $DATASET_ROOT or ./data).",
    )
    # Backward compatibility (deprecated): treat --data_dir as dataset_root if dataset_root is unset.
    parser.add_argument("--data_dir", type=str, default=None, help="DEPRECATED alias for --dataset_root")
    # NOTE: dataset_name may come from the run config (often cd_union). For inference we allow selecting
    # the TEST dataset independently via --test_dataset_name (default: whu_cd).
    parser.add_argument("--dataset_name", type=str, default="whu_cd", help="(training) dataset name from config; not used as default test dataset")
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default="whu_cd",
        help="Test dataset name (e.g., whu_cd, levircd, levircdplus, s2looking, cd_union). Default: whu_cd.",
    )
    parser.add_argument(
        "--cd_union_datasets",
        type=str,
        nargs="+",
        default=["whu_cd", "levircd", "levircdplus", "s2looking"],
        help="List of datasets to include in cd_union (options: whu_cd, levircd, levircdplus, s2looking)",
    )
    parser.add_argument("--image_size", type=int, default=256, help="image size")
    parser.add_argument("--batch_size", type=int, default=8, help="per gpu batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="batch size")

    # training
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--run_name", type=str, default=None, help="run_name")
    parser.add_argument("--output_dir", type=str, default="experiments", help="output folder")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_scheduler", type=str, default='lin0', help='lr scheduler')
    parser.add_argument("--log_interval", type=int, default=500, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1, help='validation interval for epochs')
    parser.add_argument("--save_interval", type=str, default='10', help='save interval: number for every N epochs, "epoch" for every epoch')
    parser.add_argument("--mixed_precision", type=str, default='bf16', help='mixed precision', choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')
    parser.add_argument("--lora", type=bool, default=False, help='use lora to train linear layers only')
    parser.add_argument("--clip", type=float, default=2., help='gradient clip, set to -1 if not used')
    parser.add_argument("--wp0", type=float, default=0.005, help='initial lr ratio at the begging of lr warm up')
    parser.add_argument("--wpe", type=float, default=0.01, help='final lr ratio at the end of training')
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--weight_decay_end", type=float, default=0, help='final lr ratio at the end of training')
    parser.add_argument("--resume", type=bool, default=False, help='resume')
    # vqvae
    parser.add_argument("--vocab_size", type=int, default=4096, nargs='+', help="codebook size")
    parser.add_argument("--z_channels", type=int, default=32, help="latent size of vqvae")
    parser.add_argument("--ch", type=int, default=160, help="channel size of vqvae")
    parser.add_argument("--vqvae_pretrained_path", type=str, default='pretrained/vae_ch160v4096z32.pth', help="vqvae pretrained path")
    parser.add_argument("--var_pretrained_path", type=str, default='pretrained/d16.pth', help="var pretrained path")
    # vpq model
    parser.add_argument("--v_patch_nums", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], help="number of patch numbers of each scale")
    parser.add_argument("--v_patch_layers", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], help="index of layers for predicting each scale")
    parser.add_argument("--depth", type=int, default=16, help="depth of vpq model")
    parser.add_argument("--embed_dim", type=int, default=1024, help="embedding dimension of vpq model")
    parser.add_argument("--num_heads", type=int, default=16, help="number of heads of vpq model")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="mlp ratio of vpq model")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="drop rate of vpq model")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="attn drop rate of vpq model")
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="drop path rate of vpq model")
    parser.add_argument("--cross_attn_inner_dim", type=int, default=1024, help="inner dimension of transformer cross-attention")
    parser.add_argument("--mask_type", type=str, default='change_append', help="[interleave_append, replace, change_append]")
    parser.add_argument("--uncond", type=bool, default=False, help="uncond gen")
    parser.add_argument("--bidirectional", type=bool, default=False, help="shuffle mask and image order in each stage")
    parser.add_argument("--separate_decoding", type=bool, default=False, help="separate decode mask and image in each stage")
    parser.add_argument("--separator", type=bool, default=False, help="use special tokens as separator")
    parser.add_argument("--type_pos", type=bool, default=False, help="use type pos embed")
    parser.add_argument("--interpos", type=bool, default=False, help="interpolate positional encoding")
    parser.add_argument("--mpos", type=bool, default=False, help="minus positional encoding")
    parser.add_argument("--indep", type=bool, default=False, help="indep separate decoding")
    parser.add_argument("--multi_cond", type=bool, default=True, help="multi-type conditions")
    parser.add_argument("--disable_cross_attention", action="store_true", default=False, help="disable cross-attention layers")
    parser.add_argument("--enable_current_scale_tokens", action="store_true", default=False, help="inject current-scale pre/post token embeddings at each stage")
    # context and fusion
    parser.add_argument("--use_high_res_context_levels", type=lambda x: str(x).lower() == 'true', default=False, help="include 256x256 and 128x128 context levels")
    parser.add_argument("--fusion_downsample_ratios", type=int, nargs='+', default=[1, 1, 1, 1], help="downsample ratios for each fusion module")
    parser.add_argument("--fusion_num_heads", type=int, default=8, help="fusion module heads (scalar; per-level via YAML list)")
    parser.add_argument("--fusion_num_layers", type=int, default=1, help="fusion module CrossPath layers (scalar; per-level via YAML list)")
    parser.add_argument("--fusion_cross_inner_dim", type=int, default=None, help="fusion module CrossPath inner dim (defaults to dim; per-level via YAML list)")
    parser.add_argument(
        "--fusion_use_feature_rectify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="enable FeatureRectifyModule inside fusion modules (scalar; per-level via YAML list)",
    )
    parser.add_argument(
        "--fusion_downsample_first",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="if true, apply fusion downsample before token mixing (recommended for high-res context levels)",
    )
    # condition model
    parser.add_argument("--condition_model", type=str, default="class_embedder", help="condition model")
    parser.add_argument("--num_classes", type=int, default=1000, help="number of classes for condition model")
    parser.add_argument("--cond_drop_rate", type=float, default=0.1, help="drop rate of condition model")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Trained RemoteVAR checkpoint (required unless --vqvae_only is set).",
    )

    # Inference-specific arguments
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="guidance scale for inference")
    parser.add_argument("--top_k", type=int, default=900, help="top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p sampling")
    parser.add_argument("--deterministic", action="store_true", default=True, help="use deterministic generation (argmax, no CFG)")
    parser.add_argument(
        "--visualize_intermediate",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, also save a per-stage intermediate reconstruction figure for the first few saved samples. "
            "This visualizes how the predicted mask evolves across patch scales during autoregressive generation."
        ),
    )
    parser.add_argument(
        "--visualize_entropy_all_scales",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, also compute predictive entropy maps at EACH autoregressive stage (per patch scale), "
            "plus an aggregated entropy map, and save extra entropy visualizations for the first few saved samples. "
            "Default false to preserve original behavior/perf."
        ),
    )
    parser.add_argument(
        "--entropy_agg_mode",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Aggregation for per-scale entropy when --visualize_entropy_all_scales is enabled.",
    )
    parser.add_argument(
        "--entropy_roi",
        type=str,
        default="none",
        choices=["none", "pred", "gt", "union", "tp"],
        help=(
            "When visualizing entropy, optionally focus contrast *within* a region of interest (ROI) "
            "to reveal uncertainty inside change blobs. "
            "ROI options: pred (predicted change), gt (GT change), union (pred|gt), tp (pred&gt)."
        ),
    )
    parser.add_argument(
        "--entropy_roi_norm",
        type=str,
        default="quantile",
        choices=["none", "minmax", "quantile"],
        help=(
            "How to rescale entropy values within the ROI (only used when --entropy_roi != none). "
            "quantile is most robust for unbalanced masks."
        ),
    )
    parser.add_argument("--entropy_roi_q_low", type=float, default=0.05, help="Lower quantile for ROI normalization (default 0.05).")
    parser.add_argument("--entropy_roi_q_high", type=float, default=0.95, help="Upper quantile for ROI normalization (default 0.95).")
    parser.add_argument(
        "--entropy_roi_gamma",
        type=float,
        default=1.0,
        help="Gamma applied after ROI normalization (e.g., 0.5 boosts contrast in high-entropy regions).",
    )
    parser.add_argument(
        "--use_decoder_skips",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, compute fusion-module BCHW features and pass them into a skip-conditioned VQVAE decoder (UNet-style).",
    )
    parser.add_argument(
        "--vqvae_only",
        action="store_true",
        default=False,
        help="If set, evaluate VQVAE-only mask reconstruction: GT mask -> VQVAE -> reconstructed mask -> metrics.",
    )
    parser.add_argument(
        "--vqvae_thr_sweep",
        action="store_true",
        default=False,
        help="If set with --vqvae_only, sweep binarization thresholds for the VQVAE reconstruction and save metrics per threshold.",
    )
    parser.add_argument(
        "--var_thr_sweep",
        action="store_true",
        default=False,
        help=(
            "If set (and NOT --vqvae_only), sweep binarization thresholds for VAR predicted change maps and save metrics per threshold. "
            "Uses the same threshold range as the VQVAE sweep args (--vqvae_thr_sweep_min/max/step)."
        ),
    )
    parser.add_argument(
        "--var_thr_sweep_max_samples",
        type=int,
        default=20,
        help="When --var_thr_sweep is enabled, only evaluate the first N test samples (default: 20) to keep the sweep fast.",
    )
    parser.add_argument("--vqvae_thr_sweep_min", type=float, default=0.0, help="Min threshold (inclusive) for VQVAE sweep, in [0,1].")
    parser.add_argument("--vqvae_thr_sweep_max", type=float, default=0.5, help="Max threshold (inclusive) for VQVAE sweep, in [0,1].")
    parser.add_argument("--vqvae_thr_sweep_step", type=float, default=0.01, help="Step for VQVAE sweep thresholds.")
    parser.add_argument(
        "--vqvae_thr_sweep_gt_thr",
        type=float,
        default=0.1,
        help="GT binarization threshold used during VQVAE sweep. Default 0.1 (same as everywhere else).",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=-1,
        help="How many test batches to process. Use -1 to process the full test set.",
    )

    parser.add_argument(
        "--save_predictions_png_dir",
        type=str,
        default=None,
        help=(
            "If set, save ALL test predictions (one PNG per sample) into this directory. "
            "This saves the extracted change-map output in [0,1] (uint8 RGB). "
            "Filenames use dataset-provided 'fn' when available."
        ),
    )

    parser.add_argument(
        "--decoder_refiner_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional checkpoint produced by `train_decoder_refiner.py` (e.g., best_decoder_refiner.pth). "
            "If set, replace VQVAE.decoder with a skip-conditioned decoder and optionally load finetuned fusion modules "
            "for decoder skips."
        ),
    )

    # First parse: we need checkpoint path to locate per-run config.
    args = parser.parse_args()

    # Auto-select run-specific config (precedence: epoch_dir > run_dir):
    # - epoch_dir: .../epoch_XX/codes/change_detection.yaml (if you copied codes into the epoch folder)
    # - run_dir:  .../<run>/codes/change_detection.yaml (this is where train_remote_var.py saves codes/)
    inferred_config = None
    if args.checkpoint:
        try:
            ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
            candidate_epoch = os.path.join(ckpt_dir, "codes", "change_detection.yaml")
            candidate_run = os.path.join(os.path.dirname(ckpt_dir), "codes", "change_detection.yaml")
            if os.path.exists(candidate_epoch):
                inferred_config = candidate_epoch
            elif os.path.exists(candidate_run):
                inferred_config = candidate_run
        except Exception:
            inferred_config = None

    # Load YAML config (precedence: explicit --config > inferred per-run config > repo default)
    config_path = args.config or inferred_config or "configs/change_detection.yaml"

    # If a config file is specified, load it and set defaults
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml = YAML(typ='safe')
            with open(config_path, 'r', encoding='utf-8') as file:
                config_args = yaml.load(file)
                print(f"[Inference] Config file: {file.name}")
            parser.set_defaults(**config_args)
        # keep config path for downstream debugging
        parser.set_defaults(config=config_path)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()
    if not args.vqvae_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --vqvae_only is set.")

    return args

if __name__ == '__main__':
    args = parse_args()
    args.deterministic = True
    args.batch_size = 1

    if bool(getattr(args, "vqvae_only", False)) and bool(getattr(args, "var_thr_sweep", False)):
        raise ValueError("--var_thr_sweep cannot be used with --vqvae_only (VAR model is not used in vqvae-only mode).")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vqvae_vocab = args.vocab_size[0] if isinstance(args.vocab_size, (list, tuple)) else args.vocab_size
    vqvae = VQVAE(vocab_size=vqvae_vocab, z_channels=args.z_channels, ch=args.ch, test_mode=True,
                  share_quant_resi=4, v_patch_nums=args.v_patch_nums,).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path, map_location=torch.device('cpu')))
    vqvae.eval()
    vqvae.to(device)

    # Optional: load a skip-conditioned decoder checkpoint (trained via train_decoder_refiner.py)
    decoder_refiner_ckpt = None
    if getattr(args, "decoder_refiner_checkpoint", None):
        decoder_refiner_ckpt = torch.load(args.decoder_refiner_checkpoint, map_location="cpu")
        if not isinstance(decoder_refiner_ckpt, dict) or "decoder_state_dict" not in decoder_refiner_ckpt:
            raise ValueError(
                f"--decoder_refiner_checkpoint must be a dict with key 'decoder_state_dict', got type={type(decoder_refiner_ckpt)}"
            )
        dec_sd = decoder_refiner_ckpt["decoder_state_dict"]
        skip_base = decoder_refiner_ckpt.get("skip_base_resolutions", None)
        skip_ch = decoder_refiner_ckpt.get("skip_in_channels", None)
        if skip_base is None or skip_ch is None:
            raise ValueError(
                "decoder_refiner_checkpoint is missing skip metadata (skip_base_resolutions / skip_in_channels)."
            )
        vq_cfg = decoder_refiner_ckpt.get("vqvae", {}) if isinstance(decoder_refiner_ckpt.get("vqvae", {}), dict) else {}
        ch_mult = tuple(vq_cfg.get("ch_mult", (1, 1, 2, 2, 4)))
        num_res_blocks = int(vq_cfg.get("num_res_blocks", 2))
        dec_ch = int(vq_cfg.get("ch", args.ch))
        dec_z = int(vq_cfg.get("z_channels", args.z_channels))

        # Use args.json next to the decoder refiner checkpoint to decide decoder architecture (e.g., output channels).
        refiner_args = {}
        refiner_out_ch = None
        try:
            ref_dir = os.path.dirname(os.path.abspath(args.decoder_refiner_checkpoint))
            ref_args_path = os.path.join(ref_dir, "args.json")
            if os.path.exists(ref_args_path):
                with open(ref_args_path, "r", encoding="utf-8") as f:
                    refiner_args = json.load(f) or {}
                print(f"[Inference] Loaded decoder refiner args.json: {ref_args_path}")
                if isinstance(refiner_args, dict) and ("decoder_out_channels" in refiner_args):
                    refiner_out_ch = int(refiner_args.get("decoder_out_channels", 3))
                # Carry over preferred thresholds for metrics if present.
                if isinstance(refiner_args, dict):
                    if "val_pred_thr_01" in refiner_args:
                        setattr(args, "val_pred_thr_01", float(refiner_args.get("val_pred_thr_01", 0.5)))
                    if "val_gt_thr_01" in refiner_args:
                        setattr(args, "val_gt_thr_01", float(refiner_args.get("val_gt_thr_01", 0.1)))
        except Exception as e:
            print(f"[Inference] WARNING: failed to load decoder refiner args.json: {e}")
            refiner_args = {}
            refiner_out_ch = None

        # Fallback: infer output channels from the checkpoint conv_out weight shape.
        try:
            w = dec_sd.get("conv_out.weight", None)
            if isinstance(w, torch.Tensor):
                inferred_out = int(w.shape[0])
                if refiner_out_ch is None:
                    refiner_out_ch = inferred_out
                elif int(refiner_out_ch) != inferred_out:
                    print(
                        f"[Inference] WARNING: args.json decoder_out_channels={refiner_out_ch} but checkpoint conv_out.weight[0]={inferred_out}; "
                        f"using inferred={inferred_out}."
                    )
                    refiner_out_ch = inferred_out
        except Exception:
            pass
        if refiner_out_ch is None:
            refiner_out_ch = 3

        conditioned = ConditionedDecoder(
            ch=dec_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=0.0,
            in_channels=int(refiner_out_ch),
            z_channels=dec_z,
            using_sa=True,
            using_mid_sa=True,
            skip_base_resolutions=skip_base,
            skip_in_channels=skip_ch,
            skip_fuse_extra_depth=int(refiner_args.get("decoder_skip_fuse_extra_depth", 0) or 0),
            skip_fuse_extra_width_mult=float(refiner_args.get("decoder_skip_fuse_extra_width_mult", 1.0) or 1.0),
        )
        missing, unexpected = conditioned.load_state_dict(dec_sd, strict=False)
        if missing:
            print(f"[Inference] decoder_refiner_checkpoint missing keys: {len(missing)} (decoder will use default init for them)")
        if unexpected:
            print(f"[Inference] decoder_refiner_checkpoint unexpected keys: {len(unexpected)} (ignored)")
        vqvae.decoder = conditioned.to(device)
        # Optionally load a finetuned post_quant_conv if present.
        pqc_sd = decoder_refiner_ckpt.get("post_quant_conv_state_dict", None) if isinstance(decoder_refiner_ckpt, dict) else None
        if isinstance(pqc_sd, dict):
            missing_pqc, unexpected_pqc = vqvae.post_quant_conv.load_state_dict(pqc_sd, strict=False)
            if missing_pqc:
                print(f"[Inference] decoder_refiner_checkpoint post_quant_conv missing keys: {len(missing_pqc)}")
            if unexpected_pqc:
                print(f"[Inference] decoder_refiner_checkpoint post_quant_conv unexpected keys: {len(unexpected_pqc)}")
        setattr(args, "_decoder_refiner_out_channels", int(refiner_out_ch))
        vqvae.eval()

    var = None
    if not args.vqvae_only:
        var = build_remote_var(vae=vqvae, depth=args.depth, patch_nums=args.v_patch_nums, mask_type=args.mask_type,
                             cond_drop_rate=args.cond_drop_rate, bidirectional=args.bidirectional,
                             separate_decoding=args.separate_decoding, separator=args.separator, type_pos=args.type_pos,
                             indep=args.indep, multi_cond=args.multi_cond,
                             disable_cross_attention=args.disable_cross_attention,
                             enable_current_scale_tokens=False,
                             image_size=args.image_size,
                             use_high_res_context_levels=args.use_high_res_context_levels,
                             fusion_downsample_ratios=args.fusion_downsample_ratios,
                             fusion_num_heads=getattr(args, "fusion_num_heads", 8),
                             fusion_num_layers=getattr(args, "fusion_num_layers", 1),
                             fusion_cross_inner_dim=getattr(args, "fusion_cross_inner_dim", None),
                             fusion_use_feature_rectify=getattr(args, "fusion_use_feature_rectify", False),
                             fusion_downsample_first=getattr(args, "fusion_downsample_first", False),
                             drop_path_rate=args.drop_path_rate,
                             cross_attn_inner_dim=getattr(args, "cross_attn_inner_dim", 1024))


    if not args.vqvae_only:
        # Load checkpoint based on file extension
        if args.checkpoint.endswith('.safetensors'):
            state_dict = load_file(args.checkpoint, device='cpu')
        else:
            state_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))

        if 'model_state_dict' in state_dict.keys():
            var_state_dict = state_dict['model_state_dict']
        else:
            var_state_dict = state_dict

        # Use strict=False to allow loading checkpoints with different context dimensions
        missing_keys, unexpected_keys = var.load_state_dict(var_state_dict, strict=False)
        if missing_keys:
            print(f"[Inference] Missing keys in checkpoint (will be randomly initialized): {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"  - {key}")
        if unexpected_keys:
            print(f"[Inference] Unexpected keys in checkpoint (ignored): {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"  - {key}")
        var.to(device)

        # If the decoder refiner run also fine-tuned fusion modules, load them for decoder-skips ONLY (do NOT affect VAR context by default).
        # Reason: train_decoder_refiner can update fusion modules without re-training VAR transformer blocks; using those updated fusion
        # modules for cross-attention context can degrade token generation. Keeping context fusion frozen preserves baseline VAR behavior.
        try:
            if decoder_refiner_ckpt is not None and isinstance(decoder_refiner_ckpt.get("fusion_modules_state_dict", None), dict):
                if hasattr(var, "fusion_modules") and var.fusion_modules is not None:
                    var.fusion_modules_for_skips = copy.deepcopy(var.fusion_modules)
                    missing_fm, unexpected_fm = var.fusion_modules_for_skips.load_state_dict(
                        decoder_refiner_ckpt["fusion_modules_state_dict"], strict=False
                    )
                    var.fusion_modules_for_skips.to(device)
                    var.fusion_modules_for_skips.eval()
                    if missing_fm:
                        print(f"[Inference] decoder_refiner_checkpoint fusion_modules_for_skips missing keys: {len(missing_fm)}")
                    if unexpected_fm:
                        print(f"[Inference] decoder_refiner_checkpoint fusion_modules_for_skips unexpected keys: {len(unexpected_fm)}")
        except Exception as e:
            print(f"[Inference] WARNING: failed to load fusion_modules from decoder_refiner_checkpoint: {e}")
        var.eval()

    # Always build the TEST set from --test_dataset_name (default: whu_cd), regardless of what the run config used.
    cd_dataset = create_dataset(args.test_dataset_name, args, split="test")

    cd_dataloader = DataLoader(cd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Save outputs to conditional_inference subfolder in the *most relevant* checkpoint directory.
    # If a decoder refiner checkpoint is provided, save under that run directory (so results live with the refiner).
    # Otherwise, fall back to the RemoteVAR checkpoint directory.
    if getattr(args, "decoder_refiner_checkpoint", None):
        model_dir = os.path.dirname(os.path.abspath(args.decoder_refiner_checkpoint))
    elif args.checkpoint:
        model_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    else:
        model_dir = os.path.abspath(args.output_dir or "experiments")
    conditional_inference_dir = os.path.join(model_dir, "conditional_inference")
    if args.vqvae_only:
        conditional_inference_dir = os.path.join(conditional_inference_dir, "vqvae_only")
    os.makedirs(conditional_inference_dir, exist_ok=True)

    # Optional: save *all* predictions to a user-provided directory (one PNG per sample)
    save_predictions_dir = None
    if getattr(args, "save_predictions_png_dir", None):
        save_predictions_dir = os.path.abspath(str(args.save_predictions_png_dir))
        os.makedirs(save_predictions_dir, exist_ok=True)
        print(f"[Inference] Will save all prediction PNGs to: {save_predictions_dir}")

    # Aggregate segmentation metrics across processed batches
    # Keep accumulators on the same device as model outputs to avoid CPU/GPU mismatch.
    hist_total = torch.zeros((2, 2), dtype=torch.float32, device=device)
    labeled_total = torch.zeros((), dtype=torch.float32, device=device)
    correct_total = torch.zeros((), dtype=torch.float32, device=device)

    # Optional: VQVAE-only threshold sweep accumulators (one pass over test set)
    vqvae_thr_values = None
    vqvae_sweep_hist = None
    vqvae_sweep_labeled = None
    vqvae_sweep_correct = None

    var_thr_values = None
    var_sweep_hist = None
    var_sweep_labeled = None
    var_sweep_correct = None

    sweep_gt_thr = float(getattr(args, "vqvae_thr_sweep_gt_thr", 0.1))

    def _make_thr_values(_tmin: float, _tmax: float, _tstep: float) -> list:
        if _tstep <= 0:
            raise ValueError("--vqvae_thr_sweep_step must be > 0")
        out = []
        t = float(_tmin)
        while t <= float(_tmax) + 1e-12:
            out.append(float(t))
            t += float(_tstep)
        if len(out) == 0:
            raise ValueError("No thresholds generated for sweep; check min/max/step.")
        if len(out) > 1000:
            raise ValueError(f"Too many thresholds ({len(out)}). Increase step size.")
        return out

    # Use the same threshold range args for both sweeps (per request).
    tmin = float(getattr(args, "vqvae_thr_sweep_min", 0.0))
    tmax = float(getattr(args, "vqvae_thr_sweep_max", 0.5))
    tstep = float(getattr(args, "vqvae_thr_sweep_step", 0.01))

    if bool(getattr(args, "vqvae_only", False)) and bool(getattr(args, "vqvae_thr_sweep", False)):
        vqvae_thr_values = _make_thr_values(tmin, tmax, tstep)
        vqvae_sweep_hist = torch.zeros((len(vqvae_thr_values), 2, 2), dtype=torch.float32, device=device)
        vqvae_sweep_labeled = torch.zeros((len(vqvae_thr_values),), dtype=torch.float32, device=device)
        vqvae_sweep_correct = torch.zeros((len(vqvae_thr_values),), dtype=torch.float32, device=device)
        print(f"[Inference] VQVAE threshold sweep enabled: {len(vqvae_thr_values)} thresholds in [{tmin}, {tmax}] step={tstep}")

    if (not bool(getattr(args, "vqvae_only", False))) and bool(getattr(args, "var_thr_sweep", False)):
        var_thr_values = _make_thr_values(tmin, tmax, tstep)
        var_sweep_hist = torch.zeros((len(var_thr_values), 2, 2), dtype=torch.float32, device=device)
        var_sweep_labeled = torch.zeros((len(var_thr_values),), dtype=torch.float32, device=device)
        var_sweep_correct = torch.zeros((len(var_thr_values),), dtype=torch.float32, device=device)
        max_sweep_samples = int(getattr(args, "var_thr_sweep_max_samples", 20))
        print(
            f"[Inference] VAR threshold sweep enabled: {len(var_thr_values)} thresholds in [{tmin}, {tmax}] step={tstep}; "
            f"evaluating first {max_sweep_samples} samples only."
        )

    total_seen = 0
    num_save_samples = 50
    saved_samples = 0

    # Process only the first N batches (or all if max_batches < 0)
    max_batches = int(getattr(args, "max_batches", 5))
    for batch_idx, batch in tqdm(enumerate(cd_dataloader), total=len(cd_dataloader)):
        if max_batches >= 0 and batch_idx >= max_batches:
            break

        images_pre, images_post, masks, conditions, cond_type = (
            batch['images_pre'],
            batch['images_post'],
            batch['mask'],
            batch['cls'],
            batch['type'],
        )

        B = int(images_pre.shape[0])
        global_base_idx = int(total_seen)
        if args.vqvae_only:
            # Reconstruct the GT mask through VQVAE (last scale reconstruction)
            with torch.no_grad():
                recon = vqvae.img_to_recon(masks.to(device), v_patch_nums=args.v_patch_nums, last_one=True)
            images, conf_maps = recon, None
            # If decoder refiner outputs 1ch logits, convert recon to probability and expand to 3ch for visualization/metrics.
            if int(getattr(args, "_decoder_refiner_out_channels", 3)) == 1:
                images = torch.sigmoid(images.float())
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)
        else:
            # Only request intermediate reconstructions for the samples we actually save (keeps inference fast).
            want_intermediate = bool(getattr(args, "visualize_intermediate", False)) and (saved_samples < num_save_samples)
            want_entropy_all = bool(getattr(args, "visualize_entropy_all_scales", False)) and (saved_samples < num_save_samples)
            # Only request confidence maps when we will save visualizations (keeps inference fast).
            want_confidence = (saved_samples < num_save_samples) or want_entropy_all
            out = pix_cond_inference(
                images_pre,
                images_post,
                masks,
                conditions,
                cond_type,
                device,
                B,
                var,
                vqvae,
                False,
                False,
                guidance_scale=args.guidance_scale,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
                args=args,
                deterministic=args.deterministic,
                return_confidence=bool(want_confidence),
                return_confidence_all=bool(want_entropy_all),
                confidence_agg=str(getattr(args, "entropy_agg_mode", "mean")),
                return_intermediate=want_intermediate,
            )
            intermediate_masks = None
            conf_maps_agg = None
            conf_maps_per_stage = None
            if isinstance(out, tuple):
                if len(out) == 5:
                    images, conf_maps, conf_maps_agg, conf_maps_per_stage, intermediate_masks = out
                elif len(out) == 4:
                    images, conf_maps, conf_maps_agg, conf_maps_per_stage = out
                elif len(out) == 3:
                    images, conf_maps, intermediate_masks = out
                elif len(out) == 2:
                    images, conf_maps = out
                else:
                    images = out[0]
                    conf_maps = out[1] if len(out) > 1 else None
                    intermediate_masks = out[2] if len(out) > 2 else None
            else:
                images, conf_maps, intermediate_masks = out, None, None
                conf_maps_agg, conf_maps_per_stage = None, None

            # If decoder refiner outputs 1ch masks, keep outputs in [0,1] (RemoteVAR already applies the
            # correct post-processing for mask stream when decoder_out_channels==1) and expand to 3ch for
            # visualization/metrics compatibility.
            if int(getattr(args, "_decoder_refiner_out_channels", 3)) == 1:
                if images.shape[1] == 1:
                    images = images.repeat(1, 3, 1, 1)

        # Optional: save every prediction as a PNG (does not affect metrics / visualization saving)
        if save_predictions_dir is not None:
            try:
                # Use the exact same default threshold as metrics.
                pred_thr_01 = float(
                    getattr(
                        args,
                        "val_pred_thr_01",
                        (0.5 if int(getattr(args, "_decoder_refiner_out_channels", 3)) == 1 else 0.1),
                    )
                )
                save_prediction_pngs(
                    pred_images=images,
                    output_dir=save_predictions_dir,
                    image_size=int(args.image_size),
                    pred_thr_01=pred_thr_01,
                    fns=batch.get("fn", None) if isinstance(batch, dict) else None,
                    global_offset=global_base_idx,
                    prefix=("vqvae_only" if args.vqvae_only else "pred"),
                )
            except Exception as e:
                print(f"[Inference] WARNING: failed to save prediction PNGs for batch {batch_idx}: {e}")

        # Save ONLY a few visualization samples (but continue evaluating metrics for full test set).
        if saved_samples < num_save_samples:
            remaining = num_save_samples - saved_samples
            k = min(int(B), int(remaining))
            # Save each sample as a separate PNG for easier inspection.
            for i in range(k):
                sample_id = int(saved_samples)
                if args.vqvae_only:
                    im = create_vqvae_mask_recon_image(
                        masks[i : i + 1],
                        images[i : i + 1],
                        image_size=int(args.image_size),
                    )
                else:
                    im = create_comparison_image(
                        images_pre[i : i + 1],
                        images_post[i : i + 1],
                        images[i : i + 1],
                        masks[i : i + 1],
                        1,
                        args.image_size,
                        confidence_maps=(conf_maps[i : i + 1] if conf_maps is not None else None),
                        pred_thr_01=float(getattr(args, "val_pred_thr_01", 0.1)),
                        gt_thr_01=float(getattr(args, "val_gt_thr_01", 0.1)),
                        entropy_roi=str(getattr(args, "entropy_roi", "none")),
                        entropy_roi_norm=str(getattr(args, "entropy_roi_norm", "quantile")),
                        entropy_roi_q_low=float(getattr(args, "entropy_roi_q_low", 0.05)),
                        entropy_roi_q_high=float(getattr(args, "entropy_roi_q_high", 0.95)),
                        entropy_roi_gamma=float(getattr(args, "entropy_roi_gamma", 1.0)),
                    )
                im.save(
                    os.path.join(
                        conditional_inference_dir,
                        f"{'vqvae_only' if args.vqvae_only else 'deterministic'}_{args.deterministic}_sample{sample_id:03d}.png",
                    )
                )

                # Optional: save entropy visualizations (per-scale + aggregated)
                if (not args.vqvae_only) and bool(getattr(args, "visualize_entropy_all_scales", False)):
                    try:
                        if conf_maps_agg is not None:
                            im_agg = create_comparison_image(
                                images_pre[i : i + 1],
                                images_post[i : i + 1],
                                images[i : i + 1],
                                masks[i : i + 1],
                                1,
                                args.image_size,
                                confidence_maps=conf_maps_agg[i : i + 1],
                                pred_thr_01=float(getattr(args, "val_pred_thr_01", 0.1)),
                                gt_thr_01=float(getattr(args, "val_gt_thr_01", 0.1)),
                                entropy_roi=str(getattr(args, "entropy_roi", "none")),
                                entropy_roi_norm=str(getattr(args, "entropy_roi_norm", "quantile")),
                                entropy_roi_q_low=float(getattr(args, "entropy_roi_q_low", 0.05)),
                                entropy_roi_q_high=float(getattr(args, "entropy_roi_q_high", 0.95)),
                                entropy_roi_gamma=float(getattr(args, "entropy_roi_gamma", 1.0)),
                            )
                            im_agg.save(
                                os.path.join(
                                    conditional_inference_dir,
                                    f"entropyAgg_{getattr(args, 'entropy_agg_mode', 'mean')}_{args.deterministic}_sample{sample_id:03d}.png",
                                )
                            )

                        if isinstance(conf_maps_per_stage, list) and len(conf_maps_per_stage) > 0:
                            per_stage_i = []
                            for t in conf_maps_per_stage:
                                if isinstance(t, torch.Tensor):
                                    per_stage_i.append(t[i : i + 1])
                                else:
                                    per_stage_i.append(None)
                            ent_im = create_entropy_scales_image(
                                generated_change_maps=images[i : i + 1],
                                image_size=int(args.image_size),
                                confidence_maps_per_stage=per_stage_i,
                                confidence_map_agg=(conf_maps_agg[i : i + 1] if isinstance(conf_maps_agg, torch.Tensor) else None),
                                ground_truth_masks=masks[i : i + 1],
                                patch_nums=list(getattr(args, "v_patch_nums", [])),
                                pred_thr_01=float(getattr(args, "val_pred_thr_01", 0.1)),
                                gt_thr_01=float(getattr(args, "val_gt_thr_01", 0.1)),
                                entropy_roi=str(getattr(args, "entropy_roi", "none")),
                                entropy_roi_norm=str(getattr(args, "entropy_roi_norm", "quantile")),
                                entropy_roi_q_low=float(getattr(args, "entropy_roi_q_low", 0.05)),
                                entropy_roi_q_high=float(getattr(args, "entropy_roi_q_high", 0.95)),
                                entropy_roi_gamma=float(getattr(args, "entropy_roi_gamma", 1.0)),
                            )
                            ent_im.save(
                                os.path.join(
                                    conditional_inference_dir,
                                    f"entropy_scales_{getattr(args, 'entropy_agg_mode', 'mean')}_{args.deterministic}_sample{sample_id:03d}.png",
                                )
                            )
                    except Exception as e:
                        print(f"[Inference] WARNING: failed to save entropy visualizations for sample{sample_id:03d}: {e}")

                # Optional: save intermediate per-stage mask reconstructions (autoregressive only).
                if (not args.vqvae_only) and bool(getattr(args, "visualize_intermediate", False)):
                    try:
                        if isinstance(intermediate_masks, list) and len(intermediate_masks) > 0:
                            inter_i = [t[i : i + 1] for t in intermediate_masks if isinstance(t, torch.Tensor)]
                            # Patch scale labels
                            pn = None
                            try:
                                model_unwrapped = var.module if hasattr(var, "module") else var
                                pn = list(getattr(model_unwrapped, "patch_nums", []))
                            except Exception:
                                pn = list(getattr(args, "v_patch_nums", []))
                            inter_im = create_intermediate_recon_image(
                                pre_images=images_pre[i : i + 1],
                                post_images=images_post[i : i + 1],
                                intermediate_masks=inter_i,
                                ground_truth_masks=masks[i : i + 1],
                                image_size=int(args.image_size),
                                patch_nums=pn,
                                pred_thr_01=float(getattr(args, "val_pred_thr_01", 0.1)),
                                gt_thr_01=float(getattr(args, "val_gt_thr_01", 0.1)),
                            )
                            inter_im.save(
                                os.path.join(
                                    conditional_inference_dir,
                                    f"intermediate_{args.deterministic}_sample{sample_id:03d}.png",
                                )
                            )

                            # If masks are meaningful RGB (3ch), also save the raw RGB intermediate masks.
                            try:
                                has_rgb = False
                                for t in inter_i:
                                    if isinstance(t, torch.Tensor) and t.dim() == 4 and int(t.shape[1]) >= 3:
                                        has_rgb = True
                                        break
                                if (not has_rgb) and isinstance(masks, torch.Tensor) and masks.dim() == 4 and int(masks.shape[1]) >= 3:
                                    has_rgb = True
                                if has_rgb:
                                    inter_rgb = create_intermediate_recon_image_rgb(
                                        pre_images=images_pre[i : i + 1],
                                        post_images=images_post[i : i + 1],
                                        intermediate_masks=inter_i,
                                        ground_truth_masks=masks[i : i + 1],
                                        image_size=int(args.image_size),
                                        patch_nums=pn,
                                    )
                                    inter_rgb.save(
                                        os.path.join(
                                            conditional_inference_dir,
                                            f"intermediate_RGB_{args.deterministic}_sample{sample_id:03d}.png",
                                        )
                                    )
                            except Exception as e:
                                print(f"[Inference] WARNING: failed to save intermediate RGB visualization for sample{sample_id:03d}: {e}")
                    except Exception as e:
                        print(f"[Inference] WARNING: failed to save intermediate visualization for sample{sample_id:03d}: {e}")

                saved_samples += 1
                if saved_samples >= num_save_samples:
                    break

        # Metrics: deterministic generation + automatic pred binarization (Otsu)
        try:
            masks_dev = masks.to(device)
            if args.vqvae_only and vqvae_thr_values is not None and vqvae_sweep_hist is not None:
                # Sweep: compute GT bin once (near-zero threshold), then compute recon bin for each threshold.
                gt01 = to_01(masks_dev)
                gt_bin = (gt01.max(dim=1).values > sweep_gt_thr).to(dtype=torch.long)  # (B,H,W)

                rec01 = to_01(images)
                rec_max = rec01.max(dim=1).values  # (B,H,W) float in [0,1]

                n_cl = 2
                k = (gt_bin >= 0) & (gt_bin < n_cl)
                labeled = k.sum().to(dtype=torch.float32)
                for ti, thr in enumerate(vqvae_thr_values):
                    pred_bin = (rec_max > float(thr)).to(dtype=torch.long)
                    correct = (pred_bin[k] == gt_bin[k]).sum().to(dtype=torch.float32)
                    idx = (n_cl * gt_bin[k] + pred_bin[k]).to(dtype=torch.int64)
                    hist = torch.bincount(idx, minlength=n_cl**2).reshape(n_cl, n_cl).to(dtype=torch.float32)
                    vqvae_sweep_hist[ti] += hist
                    vqvae_sweep_labeled[ti] += labeled
                    vqvae_sweep_correct[ti] += correct

                # Also keep "single metrics" accumulators for convenience (use thr=0.1)
                h, lab, cor = confusion_from_pred_and_gt(
                    pred_images=images,
                    gt_masks=masks_dev,
                    image_size=int(args.image_size),
                    pred_thr_01=float(getattr(args, "val_pred_thr_01", (0.5 if int(getattr(args, "_decoder_refiner_out_channels", 3)) == 1 else 0.1))),
                    gt_thr_01=float(getattr(args, "val_gt_thr_01", 0.1)),
                )
                hist_total += h
                labeled_total += lab
                correct_total += cor
            elif (not args.vqvae_only) and (var_thr_values is not None) and (var_sweep_hist is not None):
                # VAR sweep: binarize predicted change map at multiple thresholds.
                gt01 = to_01(masks_dev)
                gt_bin = (gt01.max(dim=1).values > sweep_gt_thr).to(dtype=torch.long)  # (B,H,W)

                pred_change = extract_change_map(images, image_size=int(args.image_size))
                pred01 = to_01(pred_change)
                pred_max = pred01.max(dim=1).values  # (B,H,W) float in [0,1]

                n_cl = 2
                k = (gt_bin >= 0) & (gt_bin < n_cl)
                labeled = k.sum().to(dtype=torch.float32)
                for ti, thr in enumerate(var_thr_values):
                    pred_bin = (pred_max > float(thr)).to(dtype=torch.long)
                    correct = (pred_bin[k] == gt_bin[k]).sum().to(dtype=torch.float32)
                    idx = (n_cl * gt_bin[k] + pred_bin[k]).to(dtype=torch.int64)
                    hist = torch.bincount(idx, minlength=n_cl**2).reshape(n_cl, n_cl).to(dtype=torch.float32)
                    var_sweep_hist[ti] += hist
                    var_sweep_labeled[ti] += labeled
                    var_sweep_correct[ti] += correct

                # Also keep "single metrics" accumulators for convenience (use thr=0.1)
                h, lab, cor = confusion_from_pred_and_gt(
                    pred_images=images,
                    gt_masks=masks_dev,
                    image_size=int(args.image_size),
                    pred_thr_01=float(getattr(args, "val_pred_thr_01", (0.5 if int(getattr(args, "_decoder_refiner_out_channels", 3)) == 1 else 0.1))),
                    gt_thr_01=float(getattr(args, "val_gt_thr_01", 0.1)),
                )
                hist_total += h
                labeled_total += lab
                correct_total += cor
            else:
                h, lab, cor = confusion_from_pred_and_gt(
                    pred_images=images,
                    gt_masks=masks_dev,
                    image_size=int(args.image_size),
                    pred_thr_01=float(getattr(args, "val_pred_thr_01", (0.5 if int(getattr(args, "_decoder_refiner_out_channels", 3)) == 1 else 0.1))),
                    gt_thr_01=float(getattr(args, "val_gt_thr_01", 0.1)),
                )
                hist_total += h
                labeled_total += lab
                correct_total += cor
        except Exception as e:
            print(f"[Inference] Metric computation failed on batch {batch_idx}: {e}")
        total_seen += int(B)

        # If VAR sweep is enabled, stop early (first N samples only) to keep it fast.
        if (not args.vqvae_only) and bool(getattr(args, "var_thr_sweep", False)):
            max_sweep_samples = int(getattr(args, "var_thr_sweep_max_samples", 20))
            if total_seen >= max_sweep_samples:
                break

    # Save aggregated metrics (guard empty eval)
    if float(labeled_total.detach().cpu().item()) <= 0:
        print("[Inference] WARNING: No labeled pixels were evaluated (labeled_total=0). Skipping metric computation.")
        metrics = {
            "iou_bg": float("nan"),
            "iou_fg": float("nan"),
            "mean_iou": float("nan"),
            "mean_pixel_acc": float("nan"),
            "pixel_acc": float("nan"),
            "precision_fg": float("nan"),
            "recall_fg": float("nan"),
            "freq_iou": float("nan"),
            "mean_iou_no_back": float("nan"),
            "labeled": 0.0,
            "correct": float(correct_total.detach().cpu().item()),
        }
    else:
        metrics = scores_from_confusion(hist=hist_total, labeled=labeled_total, correct=correct_total)
    dataset_len = len(cd_dataset) if hasattr(cd_dataset, "__len__") else None
    print(
        "[Inference] Summary: "
        f"mode={'vqvae_only' if args.vqvae_only else 'var'}, "
        f"test_dataset='{args.test_dataset_name}', "
        f"test_samples={dataset_len if dataset_len is not None else 'unknown'}, "
        f"evaluated_samples={total_seen} | "
        f"mean_iou={metrics.get('mean_iou', float('nan')):.4f}, "
        f"iou_fg={metrics.get('iou_fg', float('nan')):.4f}, "
        f"pixel_acc={metrics.get('pixel_acc', float('nan')):.4f}, "
        f"precision_fg={metrics.get('precision_fg', float('nan')):.4f}, "
        f"recall_fg={metrics.get('recall_fg', float('nan')):.4f}"
    )
    try:
        payload = {
            "test_dataset_name": args.test_dataset_name,
            "mode": "vqvae_only" if args.vqvae_only else "var",
            "test_samples": int(dataset_len) if dataset_len is not None else None,
            "evaluated_samples": int(total_seen),
            "saved_visualizations": int(saved_samples),
            **metrics,
        }

        def _append_metrics(metrics_path: str, entry: dict) -> None:
            """
            Append metrics into a single JSON file for all datasets.
            The file format is a registry:
              {
                "schema_version": 2,
                "datasets": {
                  "<dataset_name>": [ { ...entry... }, ... ]
                }
              }
            Backward compatible with the legacy single-payload format (one dict with "test_dataset_name", ...).
            """
            registry = {"schema_version": 2, "datasets": {}}
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    # New format
                    if isinstance(existing, dict) and "datasets" in existing and isinstance(existing["datasets"], dict):
                        registry = existing
                        if "schema_version" not in registry:
                            registry["schema_version"] = 2
                    # Legacy single payload format
                    elif isinstance(existing, dict) and ("test_dataset_name" in existing):
                        ds = str(existing.get("test_dataset_name", "unknown"))
                        registry = {"schema_version": 2, "datasets": {ds: [existing]}}
                    # Legacy list-of-payloads (rare but safe)
                    elif isinstance(existing, list):
                        tmp = {}
                        for it in existing:
                            if not isinstance(it, dict):
                                continue
                            ds = str(it.get("test_dataset_name", "unknown"))
                            tmp.setdefault(ds, []).append(it)
                        registry = {"schema_version": 2, "datasets": tmp}
                except Exception as e:
                    print(f"[Inference] WARNING: failed to read existing metrics registry at {metrics_path}: {e}")
                    registry = {"schema_version": 2, "datasets": {}}

            ds_name = str(entry.get("test_dataset_name", "unknown"))
            registry.setdefault("datasets", {})
            if ds_name not in registry["datasets"] or not isinstance(registry["datasets"][ds_name], list):
                registry["datasets"][ds_name] = []

            # Add a bit of provenance for debugging/compare across runs.
            entry = dict(entry)
            entry.setdefault("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
            entry.setdefault("checkpoint", os.path.abspath(getattr(args, "checkpoint", "")) if getattr(args, "checkpoint", None) else None)
            entry.setdefault(
                "decoder_refiner_checkpoint",
                os.path.abspath(getattr(args, "decoder_refiner_checkpoint", "")) if getattr(args, "decoder_refiner_checkpoint", None) else None,
            )
            entry.setdefault("deterministic", bool(getattr(args, "deterministic", False)))
            entry.setdefault("val_pred_thr_01", float(getattr(args, "val_pred_thr_01", 0.5 if int(getattr(args, "_decoder_refiner_out_channels", 3)) == 1 else 0.1)))
            entry.setdefault("val_gt_thr_01", float(getattr(args, "val_gt_thr_01", 0.1)))

            registry["datasets"][ds_name].append(entry)

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)

        metrics_path = os.path.join(conditional_inference_dir, "metrics.json")
        _append_metrics(metrics_path, payload)
        print(f"[Inference] Saved metrics to: {metrics_path}")

        # If VQVAE sweep enabled, write sweep metrics and best threshold.
        if args.vqvae_only and vqvae_thr_values is not None and vqvae_sweep_hist is not None:
            sweep_out = []
            best_idx = None
            best_key = None
            for ti, thr in enumerate(vqvae_thr_values):
                m = scores_from_confusion(hist=vqvae_sweep_hist[ti], labeled=vqvae_sweep_labeled[ti], correct=vqvae_sweep_correct[ti])
                row = {"thr": float(thr), **m}
                sweep_out.append(row)
                key = (float(m.get("mean_iou", float("nan"))), float(m.get("pixel_acc", float("nan"))))
                if best_key is None or (key[0] > best_key[0]) or (key[0] == best_key[0] and key[1] > best_key[1]):
                    best_key = key
                    best_idx = ti
            sweep_payload = {
                "mode": "vqvae_only_threshold_sweep",
                "test_dataset_name": args.test_dataset_name,
                "test_samples": int(dataset_len) if dataset_len is not None else None,
                "evaluated_samples": int(total_seen),
                "gt_thr_01": float(sweep_gt_thr),
                "thresholds": sweep_out,
                "best": sweep_out[best_idx] if best_idx is not None else None,
            }
            sweep_path = os.path.join(conditional_inference_dir, "sweep_metrics.json")
            with open(sweep_path, "w") as f:
                json.dump(sweep_payload, f, indent=2)
            if sweep_payload["best"] is not None:
                b = sweep_payload["best"]
                print(
                    "[Inference] VQVAE sweep best: "
                    f"thr={b['thr']:.4f} mean_iou={b.get('mean_iou', float('nan')):.4f} "
                    f"pixel_acc={b.get('pixel_acc', float('nan')):.4f}"
                )
            print(f"[Inference] Saved sweep metrics to: {sweep_path}")

        # If VAR sweep enabled, write sweep metrics and best threshold (first N samples only).
        if (not args.vqvae_only) and (var_thr_values is not None) and (var_sweep_hist is not None):
            sweep_out = []
            best_idx = None
            best_key = None
            for ti, thr in enumerate(var_thr_values):
                m = scores_from_confusion(hist=var_sweep_hist[ti], labeled=var_sweep_labeled[ti], correct=var_sweep_correct[ti])
                row = {"thr": float(thr), **m}
                sweep_out.append(row)
                key = (float(m.get("mean_iou", float("nan"))), float(m.get("pixel_acc", float("nan"))))
                if best_key is None or (key[0] > best_key[0]) or (key[0] == best_key[0] and key[1] > best_key[1]):
                    best_key = key
                    best_idx = ti
            sweep_payload = {
                "mode": "var_threshold_sweep",
                "test_dataset_name": args.test_dataset_name,
                "test_samples": int(dataset_len) if dataset_len is not None else None,
                "evaluated_samples": int(total_seen),
                "gt_thr_01": float(sweep_gt_thr),
                "thr_range": {"min": float(tmin), "max": float(tmax), "step": float(tstep)},
                "thresholds": sweep_out,
                "best": sweep_out[best_idx] if best_idx is not None else None,
            }
            sweep_path = os.path.join(conditional_inference_dir, "sweep_metrics_var.json")
            with open(sweep_path, "w") as f:
                json.dump(sweep_payload, f, indent=2)
            if sweep_payload["best"] is not None:
                b = sweep_payload["best"]
                print(
                    "[Inference] VAR sweep best: "
                    f"thr={b['thr']:.4f} mean_iou={b.get('mean_iou', float('nan')):.4f} "
                    f"pixel_acc={b.get('pixel_acc', float('nan')):.4f}"
                )
            print(f"[Inference] Saved VAR sweep metrics to: {sweep_path}")
    except Exception as e:
        print(f"[Inference] Failed to write metrics.json: {e}")
