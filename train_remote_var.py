"""Adapted from ``train_control_var.py`` in the ControlVAR repository."""

import os
import argparse
import logging
for handler in logging.root.handlers:
    handler.setFormatter(logging.Formatter(
        '%(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    ))
import math
import random
import copy
import json
from bisect import bisect_right
from typing import List, Tuple, Optional
from collections import OrderedDict
import numpy as np
from itertools import chain
from time import time
from datetime import datetime
from tqdm.auto import tqdm
import wandb
from PIL import Image 

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.utils import make_grid

from hf_datasets_compat import ensure_huggingface_datasets

ensure_huggingface_datasets()

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, set_seed

from remotevar_datasets import create_dataset
from models import VQVAE, build_remote_var
from losses import FocalLoss
from utils.wandb import CustomWandbTracker
from ruamel.yaml import YAML

# Import inference functions for validation visualization
from inference import pix_cond_inference, create_comparison_image

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Validation segmentation metrics (binary change masks)
from utils.mask_metrics import confusion_from_pred_and_gt, scores_from_confusion

# Import custom LR/WD scheduling utilities
try:
    from utils import lr_wd_annealing, filter_params
    HAS_CUSTOM_SCHEDULER = True
except ImportError:
    HAS_CUSTOM_SCHEDULER = False
    filter_params = None

logger = get_logger(__name__)

def _build_token_loss_fn(args, device):
    """
    Build per-token loss function with `reduction='none'` for token prediction.

    Notes:
    - We keep `reduction='none'` because training applies its own masking/weighting reductions.
    - `disable_masking_loss` preserves previous behavior: no class weights, no ignore-mask reduction.
    """
    loss_type = str(getattr(args, "loss_type", "ce")).lower()

    # Only apply precomputed class weights when masking-loss is enabled and weighting isn't globally disabled.
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
        gamma = float(getattr(args, "focal_gamma", 2.0))
        # Class balancing should be handled via your existing class_weights (weight_type).
        return FocalLoss(gamma=gamma, alpha=None, weight=class_weights, reduction="none")

    raise ValueError(f"Unknown loss_type='{loss_type}'. Expected: 'ce' or 'focal'.")


def _infer_roots_for_run(args) -> Optional[List[str]]:
    """Match datasets/build.py defaults to build a stable dataset_id for caching."""
    dd = getattr(args, "data_dirs", None)
    if isinstance(dd, (list, tuple)) and len(dd) > 0:
        return list(dd)
    root = getattr(args, "dataset_root", None) or os.environ.get("DATASET_ROOT") or getattr(args, "data_dir", None)
    root = str(root) if root else ""
    if root:
        if args.dataset_name in {"whu_cd", "change_dataset"}:
            return [os.path.join(root, "whu_cd")]
        if args.dataset_name == "cd_union":
            cd_datasets = getattr(args, "cd_union_datasets", ["whu_cd", "levircd", "levircdplus", "s2looking"])
            return [os.path.join(root, str(ds)) for ds in cd_datasets]
        if args.dataset_name == "levircd_union":
            return [os.path.join(root, "levircd"), os.path.join(root, "levircdplus")]
        if args.dataset_name == "levircd":
            return [os.path.join(root, "levircd")]
        if args.dataset_name == "levircdplus":
            return [os.path.join(root, "levircdplus")]
        if args.dataset_name == "s2looking":
            return [os.path.join(root, "s2looking")]
    return None


def _dataset_id_for_run(args) -> str:
    import hashlib
    roots = _infer_roots_for_run(args)
    if roots:
        # Use a short hash instead of full paths to avoid "File name too long" errors
        roots_str = ",".join(sorted([str(r) for r in roots]))
        roots_hash = hashlib.md5(roots_str.encode()).hexdigest()[:12]
        roots_part = f"hash{roots_hash}"
    else:
        roots_part = "default"
    return (
        f"{args.dataset_name}__roots={roots_part}"
        f"__rgb={int(getattr(args,'mask_rgb_by_location', False))}"
        f"__grid={getattr(args,'mask_rgb_grid_size', None)}"
        f"__mode={getattr(args,'mask_rgb_index_mode', None)}"
    )


def _select_viz_indices(
    dataset,
    *,
    k: int,
    pixel_thr_01: float = 0.2,
    area_thr: float = 0.2,
    target_ratio: float = 0.5,
    fallback_target_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[List[int], List[float]]:
    """
    Deterministically pick `k` indices with the largest foreground ratios.
    Foreground ratio is computed from GT masks only (fast path if dataset supports it).

    Rules:
    - Prefer indices with fg_ratio > area_thr and closest to `target_ratio` (default ~0.5).
    - If not enough, fill with indices closest to `fallback_target_ratio` (default ~0.2).
    - If still not enough, fill with the closest-to-`target_ratio` overall.
    """
    n = len(dataset)
    if n == 0:
        return [], []

    def _foreground_ratio(ds, index: int) -> float:
        """
        Read only the GT mask when possible, including through dataset wrappers.

        ``cd_union`` is a ConcatDataset, so checking only the outer object would
        miss ChangeDataset.foreground_ratio() and materialize A/B/GT for every
        sample.
        """
        if isinstance(ds, Subset):
            return _foreground_ratio(ds.dataset, int(ds.indices[index]))

        if isinstance(ds, ConcatDataset):
            child_idx = bisect_right(ds.cumulative_sizes, index)
            child_start = 0 if child_idx == 0 else ds.cumulative_sizes[child_idx - 1]
            return _foreground_ratio(ds.datasets[child_idx], index - child_start)

        if hasattr(ds, "foreground_ratio"):
            return float(ds.foreground_ratio(index, pixel_thr=0))

        # Generic fallback for unknown dataset implementations.
        sample = ds[index]
        m = sample["mask"]
        m01 = (m + 1) / 2
        return float((m01.max(dim=0).values > float(pixel_thr_01)).float().mean().item())

    # Compute fg_ratio for every index.
    ratios = []
    for i in range(n):
        ratios.append(_foreground_ratio(dataset, i))

    # Candidate ordering
    idxs = list(range(n))
    # deterministic tie-breaking: shuffle first with seed
    rnd = random.Random(seed)
    rnd.shuffle(idxs)

    # 1) best near target among those above area_thr
    above = [i for i in idxs if ratios[i] > area_thr]
    above.sort(key=lambda i: abs(ratios[i] - target_ratio))
    chosen = list(above[:k])

    # 2) if not enough, fill with those above area_thr but closest to fallback_target_ratio
    if len(chosen) < k:
        remaining = [i for i in above if i not in set(chosen)]
        remaining.sort(key=lambda i: abs(ratios[i] - fallback_target_ratio))
        need = k - len(chosen)
        chosen.extend(remaining[:need])

    # 3) if still not enough (e.g., dataset has very small changes), fill with closest-to-target overall
    if len(chosen) < k:
        remaining_all = [i for i in idxs if i not in set(chosen)]
        remaining_all.sort(key=lambda i: abs(ratios[i] - target_ratio))
        need = k - len(chosen)
        chosen.extend(remaining_all[:need])

    chosen = chosen[:k]
    chosen_ratios = [ratios[i] for i in chosen]
    return chosen, chosen_ratios


def _load_or_create_viz_indices(
    *,
    args,
    dataset,
    split_name: str,
    cache_dir: str,
    k: int = 4,
    pixel_thr_01: float = 0.2,
    area_thr: float = 0.2,
    seed: int = 0,
    target_ratio: float = 0.5,
    fallback_target_ratio: float = 0.2,
) -> List[int]:
    os.makedirs(cache_dir, exist_ok=True)
    dataset_id = _dataset_id_for_run(args)
    path = os.path.join(
        cache_dir,
        f"{dataset_id}__split={split_name}__k={k}__pix={pixel_thr_01}__area={area_thr}"
        f"__t={target_ratio}__fb={fallback_target_ratio}.json",
    )
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            idxs = [int(x) for x in data.get("indices", [])]
            # sanity
            idxs = [i for i in idxs if 0 <= i < len(dataset)]
            if len(idxs) > 0:
                print(f"[viz_indices] Loaded cached {split_name} indices from: {os.path.basename(path)}")
                return idxs[:k]
        except Exception:
            pass

    print(f"[viz_indices] Generating NEW {split_name} indices (dataset_len={len(dataset)})...")
    idxs, ratios = _select_viz_indices(
        dataset,
        k=k,
        pixel_thr_01=pixel_thr_01,
        area_thr=area_thr,
        target_ratio=target_ratio,
        fallback_target_ratio=fallback_target_ratio,
        seed=seed,
    )
    payload = {
        "dataset_id": dataset_id,
        "dataset_name": args.dataset_name,
        "split": split_name,
        "k": k,
        "pixel_thr_01": pixel_thr_01,
        "area_thr": area_thr,
        "target_ratio": target_ratio,
        "fallback_target_ratio": fallback_target_ratio,
        "seed": seed,
        "dataset_len": int(len(dataset)),
        "indices": idxs,
        "fg_ratios": ratios,
    }
    print(f"[viz_indices] Saved NEW {split_name} cache to: {os.path.basename(path)}")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return idxs


def parse_args():
    parser = argparse.ArgumentParser()

    # config file
    parser.add_argument("--config", type=str, default='configs/change_detection.yaml', help="config file used to specify parameters")

    # data
    parser.add_argument("--data", type=str, default=None, help="data")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.environ.get("DATASET_ROOT", "data"),
        help="Root folder containing change-detection datasets (default: $DATASET_ROOT or ./data).",
    )
    # Backward compatibility (deprecated): treat --data_dir as dataset_root if dataset_root is unset.
    parser.add_argument("--data_dir", type=str, default=None, help="DEPRECATED alias for --dataset_root")
    parser.add_argument("--dataset_name", type=str, default="whu_cd", help="dataset name")
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
    parser.add_argument("--lr_scheduler", type=str, default='lin0', help='lr scheduler (cos, lin, lin0, lin00, exp)')
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        help=(
            "Absolute learning-rate floor. The custom lr_wd_annealing scheduler will clamp each optimizer "
            "param-group LR to be >= min_lr, regardless of lr_scheduler type."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=500, help='log interval for steps')
    parser.add_argument("--val_interval", type=int, default=1, help='validation interval for epochs')
    parser.add_argument(
        "--train_inference_interval_steps",
        type=int,
        default=100,
        help=(
            "Run expensive autoregressive inference visualization during training every N steps. "
            "Set <=0 to disable. Default 100 (matches previous hardcoded behavior)."
        ),
    )
    parser.add_argument("--save_interval", type=str, default='10', help='save interval: number for every N epochs, "epoch" for every epoch')
    parser.add_argument("--mixed_precision", type=str, default='bf16', help='mixed precision', choices=['no', 'fp16', 'bf16', 'fp8'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation steps')
    parser.add_argument("--lora", type=bool, default=False, help='use lora to train linear layers only')
    parser.add_argument("--clip", type=float, default=2., help='gradient clip, set to -1 if not used')
    parser.add_argument("--wp0", type=float, default=0.005, help='initial lr ratio at the begging of lr warm up')
    parser.add_argument("--wpe", type=float, default=0.01, help='final lr ratio at the end of training')
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--weight_decay_end", type=float, default=0.0, help='final weight decay at the end of training')
    parser.add_argument("--resume", type=bool, default=False, help='resume')
    # accelerate resume (directory produced by `accelerator.save_state`)
    # Example: experiments/<run>/epoch_99  (or point directly to model.safetensors inside that dir)
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Path to an Accelerate save_state directory (or a file inside it)",
    )
    # LR override to apply *after* resuming (since optimizer/scheduler state restores LR)
    parser.add_argument(
        "--resume_lr_scale",
        type=float,
        default=1.0,
        help="Multiply resumed LR by this factor (e.g., 0.1 to 10x smaller)",
    )
    parser.add_argument(
        "--resume_lr",
        type=float,
        default=None,
        help="Set LR to this absolute value after resume (overrides resume_lr_scale)",
    )
    parser.add_argument(
        "--resume_allow_mismatch",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If full Accelerate resume fails due to state_dict mismatch, fall back to loading model weights non-strictly from model*.safetensors in resume_dir",
    )
    parser.add_argument(
        "--resume_model_strict",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Strictness for the fallback weights-only load (default: non-strict)",
    )
    # vqvae
    parser.add_argument("--vocab_size", type=int, default=4096, nargs='+', help="codebook size")
    parser.add_argument("--z_channels", type=int, default=32, help="latent size of vqvae")
    parser.add_argument("--ch", type=int, default=160, help="channel size of vqvae")
    parser.add_argument("--vqvae_pretrained_path", type=str, default='pretrained/vae_ch160v4096z32.pth', help="vqvae pretrained path")
   # parser.add_argument("--var_pretrained_path", type=str, default='pretrained/d16.pth', help="var pretrained path")
    parser.add_argument("--var_pretrained_path", type=str, default=None, help="var pretrained path")
    parser.add_argument("--deterministic", type=bool, default=False, help="deterministic inference")
    # vpq model
    parser.add_argument("--cross_attn_inner_dim", type=int, default=1024, help="inner dimension of cross-attention")
    parser.add_argument("--v_patch_nums", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], help="number of patch numbers of each scale")
    parser.add_argument("--v_patch_layers", type=int, default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16], help="index of layers for predicting each scale")
    parser.add_argument("--depth", type=int, default=16, help="depth of vpq model")
    parser.add_argument("--embed_dim", type=int, default=1024, help="embedding dimension of vpq model")
    parser.add_argument("--num_heads", type=int, default=16, help="number of heads of vpq model")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="mlp ratio of vpq model")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="drop rate of vpq model")
    parser.add_argument("--attn_drop_rate", type=float, default=0.0, help="attn drop rate of vpq model")
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="drop path rate of vpq model")
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
    parser.add_argument("--disable_cross_attention", action="store_true", default=False, help="disable cross-attention layers (context will be set to None)")
    parser.add_argument(
        "--finetune_cross_and_fusion",
        action="store_true",
        default=False,
        help="freeze ALL VAR parameters except cross-attention layers and fusion modules (requires cross-attention enabled)",
    )
    parser.add_argument(
        "--allow_trainable_encoder",
        action="store_true",
        default=False,
        help=(
            "If set, create a trainable *copy* of the VQVAE encoder inside RemoteVAR and use it for fusion context. "
            "The original VQVAE remains frozen for GT tokenization (img_to_idxBl)."
        ),
    )
    parser.add_argument("--enable_current_scale_tokens", action="store_true", default=False, help="inject current-scale pre/post token embeddings so mask generation can attend to them at the same stage")
    parser.add_argument(
        "--noisy_tf_mask_prob",
        type=float,
        default=0.0,
        help=(
            "Exposure-bias reduction: when building teacher-forcing inputs for the MASK stream, "
            "randomly corrupt GT mask token IDs with probability p per token. "
            "This perturbs the cumulative f_hat used for later scales (scheduled-sampling-like). "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--noisy_tf_mask_mode",
        type=str,
        default="random",
        choices=["random", "shuffle"],
        help=(
            "How to corrupt GT mask token IDs for noisy teacher forcing. "
            "'random' replaces with uniform random vocab IDs; 'shuffle' replaces with a random permutation of tokens within the same tensor."
        ),
    )
    parser.add_argument(
        "--fusion_downsample_ratios",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Per-context-level fusion downsample ratios applied INSIDE fusion modules after high-res fusion "
            "(stride-2 Conv2D stacks). Example for 256x256 inputs: "
            "--fusion_downsample_ratios 4 2 1 1 1 1 (downsample fused 256->64 and 128->64; keep others). "
            "Set to all-ones to disable. If omitted, model default is used."
        ),
    )
    parser.add_argument(
        "--fusion_num_heads",
        type=int,
        default=8,
        help="Number of attention heads inside each fusion module (scalar; can also be overridden per-level via YAML list).",
    )
    parser.add_argument(
        "--fusion_num_layers",
        type=int,
        default=1,
        help="Number of CrossPath layers stacked inside each fusion module (scalar; can also be overridden per-level via YAML list).",
    )
    parser.add_argument(
        "--fusion_cross_inner_dim",
        type=int,
        default=None,
        help="Optional CrossPath inner dim for fusion modules (defaults to dim; can also be overridden per-level via YAML list).",
    )
    parser.add_argument(
        "--fusion_use_feature_rectify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable FeatureRectifyModule inside fusion modules (scalar; can also be overridden per-level via YAML list).",
    )
    parser.add_argument(
        "--fusion_downsample_first",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, apply fusion_downsample_ratios before token mixing (recommended when using high-res context levels).",
    )
    parser.add_argument(
        "--fusion_lr_scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for fusion modules (applies to any parameter name containing 'fusion_modules').",
    )
    parser.add_argument(
        "--fusion_wd_scale",
        type=float,
        default=1.0,
        help=(
            "Weight-decay multiplier for fusion modules (applies to any parameter name containing 'fusion_modules'). "
            "Tip: if you set fusion_lr_scale>1, consider fusion_wd_scale=1/fusion_lr_scale to keep lr*wd roughly constant."
        ),
    )
    parser.add_argument(
        "--use_high_res_context_levels",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, use all encoder context levels (e.g., 256/128/64/32/16 + 16(post-middle)). "
            "If false, use the legacy 4-level context only (64/32/16 + 16(post-middle))."
        ),
    )
    parser.add_argument("--disable_masking_loss", action="store_true", default=False, help="compute loss on all tokens instead of just mask tokens")
    # Loss choice (token prediction loss for mask tokens)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Token prediction loss: 'ce' (cross entropy) or 'focal'.",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma (only used if loss_type='focal').")

    # Validation mask metrics (binary change detection)
    parser.add_argument(
        "--compute_val_metrics",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, run conditional generation on the entire validation set and compute IoU/pixel-acc metrics.",
    )
    parser.add_argument(
        "--val_metrics_every",
        type=int,
        default=1,
        help="Compute expensive validation IoU metrics every N epochs (only applies when --compute_val_metrics is true). Default 1 (every validation).",
    )
    parser.add_argument(
        "--val_metrics_max_batches",
        type=int,
        default=-1,
        help="If >0, compute validation IoU metrics on only the first N validation batches (loss is still computed on the full val set). Default -1 (all batches).",
    )
    # Note: Validation metrics are ALWAYS computed deterministically (argmax, no CFG) and
    # binarization uses an automatic threshold (Otsu) for predictions to avoid user-tuned thresholds.
    # condition model
    parser.add_argument("--condition_model", type=str, default="class_embedder", help="condition model")
    parser.add_argument("--num_classes", type=int, default=1000, help="number of classes for condition model")
    parser.add_argument("--cond_drop_rate", type=float, default=0.1, help="drop rate of condition model")
    
    # diversity loss weighting
    parser.add_argument("--diversity_alpha", type=float, default=2.0, help="diversity push strength for inverse frequency weighting (1.0=standard, 2.0=strong, 3.0=very strong, <0=disabled)")
    parser.add_argument("--max_weight_ratio", type=float, default=100.0, help="maximum ratio between highest and lowest weight for stability")
    parser.add_argument("--diversity_warmup_steps", type=int, default=0, help="number of steps to warmup diversity alpha from 1.0 to target value")
    parser.add_argument("--use_ema_token_freq", type=bool, default=False, help="use exponential moving average for token frequencies")
    
    # precomputed class weights (more stable alternative)
    parser.add_argument("--use_precomputed_weights", type=bool, default=False, help="use precomputed class weights from token_freq_path")
    parser.add_argument(
        "--token_freq_path",
        type=str,
        default=None,
        help="path to precomputed token frequencies JSON (if omitted, auto-select from token_frequencies/ based on dataset+mask settings)",
    )
    parser.add_argument("--weight_type", type=str, default="alpha", choices=["inv", "alpha", "effective"], help="which weight type to use: inv, alpha, or effective")
    parser.add_argument("--disable_all_weighting", action="store_true", default=False, help="disable ALL frequency weighting (both precomputed and dynamic), use only ignore_mask")
    
    # Dataset filtering
    parser.add_argument("--filter_empty_masks", type=bool, default=False, help="filter out training samples where mask is entirely black (no changes)")
    parser.add_argument("--empty_mask_threshold", type=float, default=0.001, help="threshold for considering a mask as empty (ratio of non-black pixels)")

    # Image normalization (pre/post only; mask unaffected). Always clamp to [-1,1] before VQVAE.
    parser.add_argument(
        "--image_normalization",
        type=str,
        default="m11",
        choices=["m11", "dataset"],
        help="image normalization for change-detection datasets: 'm11' uses (x-0.5)*2; 'dataset' uses (x-mean)/std per dataset root",
    )
    parser.add_argument("--dataset_stats_max_samples", type=int, default=500, help="max samples used to estimate per-dataset RGB mean/std (uses both A and B images)")
    parser.add_argument("--dataset_stats_seed", type=int, default=0, help="seed for sampling images when estimating dataset mean/std")

    # Mask RGB location coding colormap resolution:
    # If mask_rgb_by_location is enabled, we auto-pick the minimum channel-level count L such that (L^3 - 1) >= (gy*gx)
    # unless you override with --mask_rgb_levels.
    parser.add_argument(
        "--mask_rgb_levels",
        type=int,
        default=0,
        help="number of channel levels for location-coded RGB mask colormap; 0 means auto-minimum for the chosen grid size",
    )

    # Change-detection augmentations (used by datasets/build.py for whu_cd/change_dataset)
    # Use BooleanOptionalAction so you can pass --no-enable_random_crop etc.
    parser.add_argument(
        "--enable_random_crop",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="enable random scale+crop augmentation for (pre, post, mask) during training",
    )
    parser.add_argument(
        "--enable_random_flip",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="enable random horizontal/vertical flips during training",
    )
    parser.add_argument(
        "--enable_random_rotation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="enable random small-angle rotation during training",
    )
    parser.add_argument(
        "--enable_gaussian_blur",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="enable gaussian blur on images (not mask) during training",
    )
    parser.add_argument(
        "--enable_color_jitter",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="enable per-image ColorJitter on pre/post (independently), not applied to mask",
    )
    parser.add_argument("--color_jitter_probability", type=float, default=0.8, help="probability to apply ColorJitter to each of pre/post images")
    parser.add_argument("--color_jitter_brightness", type=float, default=0.2, help="ColorJitter brightness factor (0 disables)")
    parser.add_argument("--color_jitter_contrast", type=float, default=0.2, help="ColorJitter contrast factor (0 disables)")
    parser.add_argument("--color_jitter_saturation", type=float, default=0.2, help="ColorJitter saturation factor (0 disables)")
    parser.add_argument("--color_jitter_hue", type=float, default=0.05, help="ColorJitter hue factor (0 disables)")
    parser.add_argument(
        "--crop_scale_range",
        type=float,
        nargs=2,
        default=(0.8, 1.0),
        metavar=("MIN", "MAX"),
        help="random crop scale range as two floats: MIN MAX (e.g. 0.8 1.0)",
    )
    parser.add_argument(
        "--min_crop_size",
        type=int,
        default=64,
        help="minimum crop size (in pixels) before resizing to image_size",
    )
    parser.add_argument(
        "--max_crop_size",
        type=int,
        default=256,
        help="maximum crop size (in pixels) before resizing to image_size",
    )
    parser.add_argument(
        "--rotation_angle",
        type=float,
        default=10.0,
        help="max absolute rotation angle in degrees",
    )
    parser.add_argument(
        "--blur_probability",
        type=float,
        default=0.5,
        help="probability of applying gaussian blur",
    )
    parser.add_argument(
        "--blur_kernel_sizes",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="list of odd kernel sizes for gaussian blur (e.g. 3 5 7)",
    )
    parser.add_argument(
        "--flip_probability",
        type=float,
        default=0.5,
        help="probability for each flip direction (horizontal and vertical are sampled independently)",
    )
    
    # Wandb logging
    parser.add_argument(
        "--use_wandb",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument("--wandb_project", type=str, default="RemoteVAR", help="wandb project name")
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")

    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # Visualization index selection (fixed viz set)
    parser.add_argument("--viz_target_ratio", type=float, default=0.5, help="target GT foreground ratio for viz_indices selection")
    parser.add_argument("--viz_fallback_ratio", type=float, default=0.2, help="fallback GT foreground ratio if not enough samples near target")

    # fFirst parse of command-line args to check for config file
    args = parser.parse_args()

    # If a config file is specified, load it and set defaults
    if args.config is not None:
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml = YAML(typ='safe')
            with open(args.config, 'r', encoding='utf-8') as file:
                config_args = yaml.load(file)
            parser.set_defaults(**config_args)

    # re-parse command-line args to overwrite with any command-line inputs
    args = parser.parse_args()

    return args


def _apply_resume_lr_override(optimizer, lr_scheduler, *, resume_lr=None, resume_lr_scale=1.0, logger=None):
    """
    When resuming with Accelerate, optimizer/scheduler state restores the LR.
    If we want to change LR at resume time, we must do it AFTER `accelerator.load_state()`.
    Also scale scheduler.base_lrs to avoid the next scheduler.step() snapping back.
    """
    if optimizer is None:
        return

    old_lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]

    if resume_lr is not None:
        new_lrs = []
        for pg in optimizer.param_groups:
            pg["lr"] = float(resume_lr)
            new_lrs.append(float(pg["lr"]))
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = list(new_lrs)
        if lr_scheduler is not None and hasattr(lr_scheduler, "_last_lr"):
            lr_scheduler._last_lr = list(new_lrs)
    else:
        scale = float(resume_lr_scale) if resume_lr_scale is not None else 1.0
        if scale == 1.0:
            return
        new_lrs = []
        for pg in optimizer.param_groups:
            pg["lr"] = float(pg.get("lr", 0.0)) * scale
            new_lrs.append(float(pg["lr"]))
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = [float(x) * scale for x in lr_scheduler.base_lrs]
        if lr_scheduler is not None and hasattr(lr_scheduler, "_last_lr"):
            lr_scheduler._last_lr = [float(x) * scale for x in lr_scheduler._last_lr]

    if logger is not None:
        logger.info(
            f"Applied resume LR override. "
            f"old_lrs={old_lrs} -> new_lrs={[float(pg.get('lr', 0.0)) for pg in optimizer.param_groups]}"
        )


def _infer_resume_dir(path: str) -> str:
    # Allow passing either the directory or a file inside it (e.g., model.safetensors)
    if path is None:
        return None
    p = str(path)
    if os.path.isdir(p):
        return p
    return os.path.dirname(p)


def _infer_starting_epoch_from_resume_dir(resume_dir: str):
    """
    If resume_dir is .../epoch_99, return 100 (next epoch index to run).
    If parsing fails, return None.
    """
    try:
        base = os.path.basename(os.path.normpath(resume_dir))
        if base.startswith("epoch_"):
            ep = int(base.split("_", 1)[1])
            return ep + 1
    except Exception:
        pass
    return None


def _infer_run_name_from_resume_dir(resume_dir: str) -> str:
    """
    Given an Accelerate state dir like:
      experiments/<timestamp>-<run_name>/epoch_99
    infer and return <run_name>.

    Falls back to the parent directory name if parsing fails.
    """
    try:
        exp_dir_name = os.path.basename(os.path.dirname(os.path.normpath(resume_dir)))
        parts = exp_dir_name.split("-")
        # timestamp is YYYY-MM-DD-HH-MM-SS => 6 hyphen-separated parts
        if len(parts) > 6:
            return "-".join(parts[6:])
        return exp_dir_name
    except Exception:
        return os.path.basename(os.path.dirname(os.path.normpath(resume_dir)))


def _strip_prefix_from_state_dict(state_dict, prefix: str):
    try:
        keys = list(state_dict.keys())
        if len(keys) == 0:
            return state_dict
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
        return state_dict
    except Exception:
        return state_dict


def _load_safetensors_state(path: str):
    try:
        from safetensors.torch import load_file
    except Exception as e:
        raise RuntimeError(
            "Missing dependency for loading .safetensors. Please ensure `safetensors` is installed."
        ) from e
    return load_file(path)


def _fallback_load_models_from_accelerate_dir(
    resume_dir: str,
    *,
    accelerator: "Accelerator",
    var,
    cond_model,
    vqvae,
    strict: bool = False,
    logger=None,
):
    """
    Fallback for sequential fine-tuning when architecture changes:
    - Load model weights non-strictly from model*.safetensors in `resume_dir`.
    - Does NOT load optimizer state (param groups may differ).
    - Optionally the caller can load scheduler.bin separately.
    """
    # Figure out file paths (accelerate uses model.safetensors, model_1.safetensors, model_2.safetensors, ...)
    model_paths = []
    for name in ["model.safetensors", "model_1.safetensors", "model_2.safetensors"]:
        p = os.path.join(resume_dir, name)
        if os.path.exists(p):
            model_paths.append(p)

    if len(model_paths) == 0:
        raise FileNotFoundError(f"No model*.safetensors found under resume_dir={resume_dir}")

    # Map: 0->var, 1->cond_model (if exists), 2->vqvae (if exists)
    # Use unwrap_model so we load into the underlying modules even if wrapped.
    targets = [accelerator.unwrap_model(var)]
    if cond_model is not None:
        targets.append(accelerator.unwrap_model(cond_model))
    if vqvae is not None:
        targets.append(accelerator.unwrap_model(vqvae))

    # Load as many as we can match; warn if counts differ.
    n = min(len(model_paths), len(targets))
    if logger is not None:
        logger.warning(
            f"Falling back to weights-only load from {resume_dir} (strict={strict}). "
            f"Found {len(model_paths)} model files, have {len(targets)} model objects; loading {n}."
        )

    for i in range(n):
        sd = _load_safetensors_state(model_paths[i])
        sd = _strip_prefix_from_state_dict(sd, "module.")
        missing, unexpected = targets[i].load_state_dict(sd, strict=bool(strict))
        if logger is not None:
            logger.warning(
                f"Loaded {os.path.basename(model_paths[i])} into model[{i}] "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )


def train_epoch(accelerator, var, vqvae, cond_model, dataloader, optimizer, progress_bar, args):

    var.train()
    if cond_model is not None:
        cond_model.train()
    # Track epoch-level training loss (weighted by batch size)
    epoch_loss_sum = 0.0
    epoch_sample_count = 0
    
    # Log masking loss configuration
    if accelerator.is_main_process and not hasattr(args, '_logged_masking_config'):
        if args.disable_masking_loss:
            logger.info("MASKING LOSS DISABLED - computing loss on ALL tokens equally (no weighting, no ignore_mask)")
        else:
            logger.info(f"MASKING LOSS ENABLED - computing loss only on mask tokens (mask_type={args.mask_type})")
        args._logged_masking_config = True

    # Log noisy teacher forcing configuration (mask stream)
    if accelerator.is_main_process and not hasattr(args, "_logged_noisy_tf"):
        p = float(getattr(args, "noisy_tf_mask_prob", 0.0))
        mode = str(getattr(args, "noisy_tf_mask_mode", "random"))
        if p > 0:
            logger.info(f"Noisy teacher forcing enabled for MASK stream: noisy_tf_mask_prob={p}, mode={mode}")
        else:
            logger.info("Noisy teacher forcing disabled for MASK stream (noisy_tf_mask_prob=0.0)")
        args._logged_noisy_tf = True
    
    # Setup loss function (per-token, reduction='none'); masking-loss logic still controls reduction/weighting later.
    loss_fn = _build_token_loss_fn(args, accelerator.device)
    if accelerator.is_main_process and not hasattr(args, "_logged_loss_type"):
        logger.info(f"Using token loss: {getattr(args, 'loss_type', 'ce')}")
        if str(getattr(args, "loss_type", "ce")).lower() == "focal":
            logger.info(f"  focal_gamma={float(getattr(args, 'focal_gamma', 2.0))}")
        args._logged_loss_type = True
    if accelerator.is_main_process and not hasattr(args, "_logged_loss_weighting"):
        if bool(getattr(args, "disable_masking_loss", False)):
            logger.info("  Masking loss disabled: averaging loss over ALL tokens (no ignore_mask reduction, no class weighting).")
        elif bool(getattr(args, "disable_all_weighting", False)):
            logger.info("  ALL frequency weighting DISABLED - using only ignore_mask.")
        elif bool(getattr(args, "use_precomputed_weights", False)) and hasattr(args, "class_weights"):
            cw = args.class_weights.to(accelerator.device)
            logger.info(f"  Using precomputed class weights from {getattr(args, 'token_freq_path', None)} (type={getattr(args, 'weight_type', None)})")
            logger.info(f"    Weight stats - Min: {cw.min():.3f}, Max: {cw.max():.3f}, Mean: {cw.mean():.3f}, Ratio: {(cw.max()/cw.min()):.1f}x")
        args._logged_loss_weighting = True
    
    # Initialize EMA token frequency buffer for dynamic weighting (if not using precomputed or disabled)
    if not args.disable_masking_loss and not args.disable_all_weighting and not args.use_precomputed_weights and not hasattr(args, 'ema_token_counts'):
        vocab_size_val = args.vocab_size if isinstance(args.vocab_size, int) else args.vocab_size[0]
        args.ema_token_counts = None  # Will be initialized on first batch
        args.ema_momentum = 0.95  # Momentum for exponential moving average

    for _, batch in enumerate(dataloader):

        with accelerator.accumulate(var):
            images_pre,images_post, masks, conditions, cond_type = batch['images_pre'], batch['images_post'], batch['mask'], batch['cls'], batch['type']
            
            # Store the batch for potential inference visualization
            args.last_batch = batch

            # forward to get input ids
            with torch.no_grad():
                mask_labels_list = vqvae.img_to_idxBl(masks, v_patch_nums=args.v_patch_nums)
                # Optionally corrupt GT mask token IDs when building teacher-forcing inputs for later scales.
                # IMPORTANT: keep labels (mask_labels_list) clean; only the teacher-forcing INPUT is noisy.
                noisy_p = float(getattr(args, "noisy_tf_mask_prob", 0.0))
                if noisy_p < 0 or noisy_p > 1:
                    raise ValueError(f"noisy_tf_mask_prob must be in [0,1], got {noisy_p}")
                noisy_mode = str(getattr(args, "noisy_tf_mask_mode", "random")).lower()

                mask_labels_for_tf = mask_labels_list
                if noisy_p > 0:
                    vocab_size_val = int(getattr(vqvae, "vocab_size", args.vocab_size if isinstance(args.vocab_size, int) else args.vocab_size[0]))
                    mask_labels_for_tf = []
                    for idx_Bl in mask_labels_list:
                        if idx_Bl is None:
                            mask_labels_for_tf.append(idx_Bl)
                            continue
                        if noisy_mode == "random":
                            noise_mask = torch.rand(idx_Bl.shape, device=idx_Bl.device) < noisy_p
                            noise_idx = torch.randint(0, vocab_size_val, size=idx_Bl.shape, device=idx_Bl.device, dtype=idx_Bl.dtype)
                            idx_noisy = torch.where(noise_mask, noise_idx, idx_Bl)
                        elif noisy_mode == "shuffle":
                            flat = idx_Bl.reshape(-1)
                            perm = torch.randperm(flat.numel(), device=idx_Bl.device)
                            shuffled = flat[perm].view_as(idx_Bl)
                            noise_mask = torch.rand(idx_Bl.shape, device=idx_Bl.device) < noisy_p
                            idx_noisy = torch.where(noise_mask, shuffled, idx_Bl)
                        else:
                            raise ValueError(f"Unknown noisy_tf_mask_mode={noisy_mode} (expected 'random' or 'shuffle')")
                        mask_labels_for_tf.append(idx_noisy)

                # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                mask_input_h_list = vqvae.idxBl_to_h(mask_labels_for_tf, include_next_scale=False)

                # labels_list: List[(B, 1), (B, 4), (B, 9)]
                labels_list_pre = vqvae.img_to_idxBl(images_pre, v_patch_nums=args.v_patch_nums)
                labels_list_post = vqvae.img_to_idxBl(images_post, v_patch_nums=args.v_patch_nums)
                # from labels get inputs fhat list: List[(B, 2**2, 32), (B, 3**2, 32))]
                # If enabled, include the *current-scale* GT residual contribution in the PRE/POST streams
                # (safe because pre/post are known condition tokens). Keep MASK stream autoregressive.
                use_current_scale = bool(getattr(args, "enable_current_scale_tokens", False)) and args.mask_type == "change_append"
                input_h_list_pre = vqvae.idxBl_to_h(labels_list_pre, include_next_scale=use_current_scale)
                input_h_list_post = vqvae.idxBl_to_h(labels_list_post, include_next_scale=use_current_scale)

            # IMPORTANT:
            # - VQVAE tokenization above should stay in fp32 (deterministic/stable token IDs).
            # - Context computation must be OUTSIDE torch.no_grad() so fusion modules receive gradients!
            # - For fp16/bf16, run the VAR forward under autocast for speed/memory.
            if args.disable_cross_attention:
                context = None
            else:
                with accelerator.autocast():
                    # Use trainable fusion modules to fuse pre and post image contexts
                    # Handle DDP wrapper: access underlying module if wrapped
                    model = var.module if hasattr(var, 'module') else var
                    context = model.encode_context_with_fusion([images_pre, images_post])

            # Temporary: Check VQVAE reconstructions (remove after verification)
            if args.completed_steps == 100 and accelerator.is_main_process:
                with torch.no_grad():
                        # Reconstruct images from indices
                        recon_pre = vqvae.idxBl_to_img(labels_list_pre, same_shape=True, last_one=True)
                        recon_post = vqvae.idxBl_to_img(labels_list_post, same_shape=True, last_one=True)
                        recon_mask = vqvae.idxBl_to_img(mask_labels_list, same_shape=True, last_one=True)
                        
                        # Normalize to [0, 1] range for visualization
                        recon_pre_vis = ((recon_pre + 1) / 2).clamp(0, 1)
                        recon_post_vis = ((recon_post + 1) / 2).clamp(0, 1)
                        recon_mask_vis = ((recon_mask + 1) / 2).clamp(0, 1)
                        
                        orig_pre_vis = ((images_pre + 1) / 2).clamp(0, 1)
                        orig_post_vis = ((images_post + 1) / 2).clamp(0, 1)
                        orig_mask_vis = ((masks + 1) / 2).clamp(0, 1)
                        
                        # Analyze per-sample token statistics for masks
                        mask_token_stats = []
                        for i in range(min(4, images_pre.shape[0])):
                            # Get tokens for this mask across all scales
                            sample_tokens = []
                            per_scale_unique = []
                            for scale_idx, scale_tokens in enumerate(mask_labels_list):
                                if i < scale_tokens.shape[0]:
                                    sample_tokens.append(scale_tokens[i])
                                    # Track unique tokens per scale
                                    unique_in_scale = len(torch.unique(scale_tokens[i]))
                                    per_scale_unique.append((args.v_patch_nums[scale_idx], unique_in_scale))
                            all_tokens_sample = torch.cat(sample_tokens)
                            
                            # Count unique tokens
                            unique_tokens_sample = len(torch.unique(all_tokens_sample))
                            total_tokens_sample = all_tokens_sample.numel()
                            
                            # Find most common tokens
                            token_counts = torch.bincount(all_tokens_sample)
                            top_5_tokens = torch.topk(token_counts, k=min(5, len(token_counts))).indices.tolist()
                            top_5_counts = torch.topk(token_counts, k=min(5, len(token_counts))).values.tolist()
                            
                            # Pixel statistics
                            mask_pixels = orig_mask_vis[i]
                            total_pixels = mask_pixels.numel()
                            zero_pixels = (mask_pixels == 0.0).sum().item()
                            unique_pixel_values = len(torch.unique(mask_pixels))
                            zero_ratio = zero_pixels / total_pixels
                            
                            mask_token_stats.append({
                                'unique_tokens': unique_tokens_sample,
                                'total_tokens': total_tokens_sample,
                                'zero_pixels': zero_pixels,
                                'total_pixels': total_pixels,
                                'zero_ratio': zero_ratio,
                                'unique_pixel_values': unique_pixel_values,
                                'per_scale_unique': per_scale_unique,
                                'top_tokens': top_5_tokens,
                                'top_counts': top_5_counts
                            })
                        
                        # Create grid: [orig_pre, recon_pre, orig_post, recon_post, orig_mask, recon_mask]
                        batch_to_show = min(4, images_pre.shape[0])
                        comparison_images = []
                        for i in range(batch_to_show):
                            comparison_images.extend([
                                orig_pre_vis[i], recon_pre_vis[i],
                                orig_post_vis[i], recon_post_vis[i],
                                orig_mask_vis[i], recon_mask_vis[i]
                            ])
                        
                        grid = make_grid(comparison_images, nrow=6, padding=5, pad_value=1.0)
                        
                        # Save to file
                        vqvae_check_dir = os.path.join(args.project_dir, "vqvae_checks")
                        os.makedirs(vqvae_check_dir, exist_ok=True)
                        grid_np = grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                        grid_img = Image.fromarray(grid_np)
                        
                        # Add labels and statistics
                        from PIL import ImageDraw, ImageFont
                        draw = ImageDraw.Draw(grid_img)
                        try:
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
                            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
                        except:
                            font = None
                            font_small = None
                        
                        labels = ["Orig Pre", "Recon Pre", "Orig Post", "Recon Post", "Orig Mask", "Recon Mask"]
                        img_w = args.image_size
                        for j, label in enumerate(labels):
                            x_pos = j * (img_w + 5) + img_w // 2
                            if font:
                                bbox = draw.textbbox((0, 0), label, font=font)
                                text_width = bbox[2] - bbox[0]
                                draw.text((x_pos - text_width//2, 5), label, fill=(255, 0, 0), font=font)
                            else:
                                draw.text((x_pos - 30, 5), label, fill=(255, 0, 0))
                        
                        # Add per-sample statistics on the right side of each row
                        for i, stats in enumerate(mask_token_stats):
                            y_base = 25 + i * (img_w + 5)
                            stats_text = (
                                f"Sample {i+1}:\n"
                                f"Tokens: {stats['unique_tokens']}/{stats['total_tokens']}\n"
                                f"Zero pixels: {stats['zero_ratio']:.1%}\n"
                                f"Unique px vals: {stats['unique_pixel_values']}"
                            )
                            if font_small:
                                draw.text((grid_img.width - 180, y_base), stats_text, fill=(0, 0, 255), font=font_small)
                            else:
                                draw.text((grid_img.width - 180, y_base), stats_text, fill=(0, 0, 255))
                        
                        grid_img.save(os.path.join(vqvae_check_dir, f"vqvae_check_step{args.completed_steps}.png"))
                        
                        # Log detailed statistics
                        logger.info(f"Saved VQVAE reconstruction check at step {args.completed_steps}")
                        for i, stats in enumerate(mask_token_stats):
                            logger.info(f"  Sample {i+1}: {stats['unique_tokens']} unique tokens / {stats['total_tokens']} total, "
                                       f"{stats['zero_ratio']:.1%} zero pixels, {stats['unique_pixel_values']} unique pixel values")
                            # Log per-scale breakdown
                            scale_breakdown = ", ".join([f"s{pn}:{unique}" for pn, unique in stats['per_scale_unique']])
                            logger.info(f"    Per-scale unique tokens: {scale_breakdown}")
                            # Log most common tokens
                            top_tokens_str = ", ".join([f"{tok}({cnt})" for tok, cnt in zip(stats['top_tokens'], stats['top_counts'])])
                            logger.info(f"    Top tokens (id:count): {top_tokens_str}")

            # handle mask
            if args.mask_type == 'replace':
                # Image: r1, r2, r3, Mask: m1, m2, m3
                # New: r1, m2, r3
                # Note that image goes first
                for i in range(len(input_h_list)):
                    if i % 2 == 0:
                        labels_list[i] = mask_labels_list[i]
                        input_h_list[i] = mask_input_h_list[i]
                mask_first = False
            elif args.mask_type == 'change_append':
                # For each scale: Pre: p1, p2, ..., Post: o1, o2, ..., Mask: m1, m2, ...
                # Creates sequence: [pre1, post1, mask1, pre2, post2, mask2, ...]
                # Structure: first pn tokens are pre, next pn are post, last pn are mask per scale
                labels_list_ = list(chain.from_iterable(zip(labels_list_pre, labels_list_post, mask_labels_list)))
                input_h_list_ = list(chain.from_iterable(zip(input_h_list_pre, input_h_list_post, mask_input_h_list)))
                mask_first = False  # SOS token order: [pre_sos, post_sos, mask_cond_token]
   
                labels_list, input_h_list = labels_list_, input_h_list_
            else:
                raise NotImplementedError
            
            x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)

            # forward through model (autocast for fp16/bf16 friendliness)
            with accelerator.autocast():
                logits = var(conditions, x_BLCv_wo_first_l, context=context, mask_first=mask_first, cond_type=cond_type)  # BLC, C=vocab size
            labels = torch.cat(labels_list, dim=1)  # (B, L)
            Bsz, L = labels.shape

            logits = logits.view(-1, logits.size(-1))  # (B*L, V)
            labels_1d = labels.reshape(-1)  # (B*L,)

            # IMPORTANT (memory): avoid materializing a fp32 copy of `logits` here.
            # `logits` can be huge (B*L*V); doing logits.float() can double peak memory and cause OOM in fp16 runs.
            # Let PyTorch handle internal accumulation/precision in the CE kernel.
            loss = loss_fn(logits, labels_1d)

            # If disable_masking_loss is True, compute simple mean loss on all tokens with NO weighting
            if args.disable_masking_loss:
                # Simple average loss over all tokens - no ignore_mask, no frequency weighting
                loss = loss.mean()
                
                # For logging purposes only, track all positions
                mask_token_positions = torch.arange(labels_1d.numel(), device=labels_1d.device)
                mask_token_indices = []
            else:
                # IMPORTANT: build ignore_mask PER SAMPLE (shape BxL) then flatten.
                # The old code built a 1D mask for a single sequence and applied it to labels.view(-1),
                # which effectively masked only the first sample in the batch.
                ignore_mask_2d = torch.zeros((Bsz, L), dtype=torch.float, device=loss.device)

                # Calculate positions based on mask_type (within a single sequence of length L)
                current_pos = 0
                if args.mask_type == 'change_append':
                    for pn in args.v_patch_nums:
                        tokens_per_scale = pn * pn
                        tokens_per_group = tokens_per_scale * 3  # pre + post + mask per scale
                        mask_start = current_pos + 2 * tokens_per_scale
                        mask_end = current_pos + 3 * tokens_per_scale
                        ignore_mask_2d[:, mask_start:mask_end] = 1.0
                        current_pos += tokens_per_group
                elif args.mask_type == 'interleave_append':
                    for pn in args.v_patch_nums:
                        tokens_per_scale = pn * pn
                        # alternating sequence length = 2*tokens_per_scale (mask + img)
                        if mask_first:
                            ignore_mask_2d[:, current_pos:current_pos + 2 * tokens_per_scale:2] = 1.0
                        else:
                            ignore_mask_2d[:, current_pos + 1:current_pos + 2 * tokens_per_scale:2] = 1.0
                        current_pos += 2 * tokens_per_scale
                elif args.mask_type == 'replace':
                    for pn in args.v_patch_nums:
                        tokens_per_scale = pn * pn
                        ignore_mask_2d[:, current_pos:current_pos + tokens_per_scale] = 1.0
                        current_pos += tokens_per_scale
                else:
                    raise NotImplementedError

                ignore_mask = ignore_mask_2d.reshape(-1)  # (B*L,)
                mask_token_positions = torch.where(ignore_mask > 0.5)[0]  # 1D indices into logits/labels_1d
                mask_token_indices = []  # kept for compatibility below

            # Log token diversity statistics periodically (main process only; avoids expensive argmax/bincount on every rank)
            if accelerator.is_main_process and args.completed_steps % args.log_interval == 0:
                # Calculate pixel-level background ratio in original mask images
                with torch.no_grad():
                    # Masks are normalized to [-1, 1], convert back to [0, 1]
                    masks_01 = (masks + 1) / 2
                    
                    # Debug: Check for completely empty masks in batch
                    batch_size = masks_01.shape[0]
                    for b_idx in range(batch_size):
                        sample_mask = masks_01[b_idx]
                        sample_nonzero = (sample_mask > 0.001).float().sum().item()
                        sample_total = sample_mask.numel()
                        sample_ratio = sample_nonzero / sample_total
                        if sample_ratio <= args.empty_mask_threshold:
                            logger.warning(f"EMPTY MASK DETECTED in batch! Sample {b_idx}: "
                                         f"min={sample_mask.min():.3f}, max={sample_mask.max():.3f}, "
                                         f"mean={sample_mask.mean():.3f}, non-zero ratio={sample_ratio:.6f}")
                    
                    # Count exactly zero pixels (no threshold)
                    zero_pixels = (masks_01 == 0.0).sum().item()
                    total_pixels = masks_01.numel()
                    pixel_zero_ratio = zero_pixels / total_pixels if total_pixels > 0 else 0
                    
                    # Also check near-zero pixels for comparison (< 0.01)
                    near_zero_pixels = (masks_01 < 0.01).sum().item()
                    pixel_near_zero_ratio = near_zero_pixels / total_pixels if total_pixels > 0 else 0
                
                # Ground truth tokens
                all_mask_tokens_gt = labels_1d[mask_token_positions]
                unique_tokens_gt = torch.unique(all_mask_tokens_gt)
                # Handle vocab_size as either int or list
                vocab_size_val = args.vocab_size if isinstance(args.vocab_size, int) else args.vocab_size[0]
                token_counts_gt = torch.bincount(all_mask_tokens_gt, minlength=vocab_size_val)
                
                # Find most common (background) token in GT
                most_common_token_gt = torch.argmax(token_counts_gt).item()
                most_common_count_gt = token_counts_gt[most_common_token_gt].item()
                total_mask_tokens = all_mask_tokens_gt.numel()
                background_ratio_gt = most_common_count_gt / total_mask_tokens if total_mask_tokens > 0 else 0
                
                # Predicted tokens
                with torch.no_grad():
                    # Only compute argmax for mask-token positions (much cheaper than argmax over all tokens).
                    if mask_token_positions.numel() > 0:
                        pred_mask_tokens = torch.argmax(logits[mask_token_positions], dim=-1)
                    else:
                        pred_mask_tokens = torch.empty((0,), device=labels_1d.device, dtype=torch.long)
                    unique_tokens_pred = torch.unique(pred_mask_tokens)
                    token_counts_pred = torch.bincount(pred_mask_tokens, minlength=vocab_size_val)
                    
                    # Find most common predicted token
                    most_common_token_pred = torch.argmax(token_counts_pred).item()
                    most_common_count_pred = token_counts_pred[most_common_token_pred].item()
                    background_ratio_pred = most_common_count_pred / total_mask_tokens if total_mask_tokens > 0 else 0
                
                logger.info(f"Step {args.completed_steps} - Mask statistics:\n"
                           f"  PIXEL  - Exact zero: {pixel_zero_ratio:.2%}, Near-zero (<0.01): {pixel_near_zero_ratio:.2%}\n"
                           f"  GT     - Unique tokens: {len(unique_tokens_gt)}/{vocab_size_val}, "
                           f"Most common: {most_common_token_gt} ({background_ratio_gt:.2%})\n"
                           f"  PRED   - Unique tokens: {len(unique_tokens_pred)}/{vocab_size_val}, "
                           f"Most common: {most_common_token_pred} ({background_ratio_pred:.2%})\n"
                           f"  GAP    - Token diversity gap: {len(unique_tokens_gt) - len(unique_tokens_pred)}, "
                           f"Pixel-token ratio gap: {(pixel_near_zero_ratio - background_ratio_gt):.2%}")
                
                accelerator.log({
                    "mask_pixels/exact_zero_ratio": pixel_zero_ratio,
                    "mask_pixels/near_zero_ratio": pixel_near_zero_ratio,
                    "mask_tokens/gt_unique_count": len(unique_tokens_gt),
                    "mask_tokens/gt_background_ratio": background_ratio_gt,
                    "mask_tokens/gt_most_common_token": most_common_token_gt,
                    "mask_tokens/pred_unique_count": len(unique_tokens_pred),
                    "mask_tokens/pred_background_ratio": background_ratio_pred,
                    "mask_tokens/pred_most_common_token": most_common_token_pred,
                    "mask_tokens/diversity_gap": len(unique_tokens_gt) - len(unique_tokens_pred),
                    "mask_tokens/pixel_vs_token_ratio_gap": pixel_near_zero_ratio - background_ratio_gt,
                }, step=args.completed_steps)
            
            # Apply weighting and reduce loss (only when disable_masking_loss=False)
            if not args.disable_masking_loss:
                # Apply diversity-based weighting to combat trivial solution
                # Give higher weight to rare tokens (non-background)
                # Set diversity_alpha < 0 to disable diversity weighting entirely
                # Skip dynamic weighting if using precomputed class weights (already in loss_fn) or if all weighting is disabled
                if not args.disable_all_weighting and not args.use_precomputed_weights and args.diversity_alpha >= 0 and mask_token_positions.numel() > 0:
                    all_mask_tokens_for_weight = labels_1d[mask_token_positions]
                    
                    # Calculate token frequencies across the batch
                    vocab_size_val = args.vocab_size if isinstance(args.vocab_size, int) else args.vocab_size[0]
                    batch_token_counts = torch.bincount(all_mask_tokens_for_weight, minlength=vocab_size_val).float()
                    
                    # Use EMA token frequencies for stability (reduces batch-to-batch variance)
                    if args.use_ema_token_freq:
                        if args.ema_token_counts is None:
                            # Initialize EMA with first batch
                            args.ema_token_counts = batch_token_counts.clone()
                        else:
                            # Update EMA: ema = momentum * ema + (1 - momentum) * new
                            args.ema_token_counts = args.ema_momentum * args.ema_token_counts + \
                                                   (1 - args.ema_momentum) * batch_token_counts
                        token_counts = args.ema_token_counts.clone()
                    else:
                        token_counts = batch_token_counts
                    
                    token_counts = token_counts + 1  # Add smoothing to avoid division by zero
                    
                    # Warmup for diversity alpha (gradual increase from 1.0 to target)
                    if args.diversity_warmup_steps > 0:
                        warmup_progress = min(1.0, args.completed_steps / args.diversity_warmup_steps)
                        current_alpha = 1.0 + (args.diversity_alpha - 1.0) * warmup_progress
                    else:
                        current_alpha = args.diversity_alpha
                    
                    # Inverse frequency weighting with exponent for stronger diversity push
                    token_weights = (1.0 / token_counts) ** current_alpha
                    # Normalize weights so the EXPECTED weight under current token distribution is 1.0
                    # (keeps loss magnitude comparable to unweighted case)
                    expected_w = (token_weights * token_counts).sum() / token_counts.sum().clamp_min(1.0)
                    token_weights = token_weights / expected_w.clamp_min(1e-8)
                    
                    # STABILITY: Clamp weights to prevent extreme values
                    # max_weight_ratio controls the maximum difference between weights
                    if args.max_weight_ratio > 0:
                        active_weights_mask = token_counts > 1
                        if active_weights_mask.any():
                            min_weight = token_weights[active_weights_mask].min()
                            max_allowed_weight = min_weight * args.max_weight_ratio
                            token_weights = torch.clamp(token_weights, max=max_allowed_weight)
                            # Re-normalize after clamping (expected=1)
                            expected_w = (token_weights * token_counts).sum() / token_counts.sum().clamp_min(1.0)
                            token_weights = token_weights / expected_w.clamp_min(1e-8)
                    
                    # Log weight statistics periodically
                    if args.completed_steps % args.log_interval == 0:
                        active_weights = token_weights[token_counts > 1]  # Exclude padding
                        logger.info(f"  Diversity weights (alpha={current_alpha:.2f}) - "
                                   f"Min: {active_weights.min():.3f}, "
                                   f"Max: {active_weights.max():.3f}, "
                                   f"Mean: {active_weights.mean():.3f}, "
                                   f"Ratio (max/min): {(active_weights.max() / active_weights.min()):.1f}x")
                    
                    # Apply weights to loss
                    sample_weights = token_weights[labels_1d]
                    # Combine with ignore_mask (only apply weights to mask positions)
                    final_weights = ignore_mask * sample_weights
                else:
                    # Using precomputed weights, all weighting disabled, diversity disabled, or no mask tokens
                    # When using precomputed weights, class weights are already in loss_fn
                    final_weights = ignore_mask
                    
                    if args.completed_steps % args.log_interval == 0:
                        if args.disable_all_weighting:
                            logger.info(f"  ALL frequency weighting DISABLED - using only ignore_mask")
                        elif args.use_precomputed_weights:
                            logger.info(f"  Using precomputed class weights (type={args.weight_type})")
                        elif args.diversity_alpha < 0:
                            logger.info(f"  Diversity weighting DISABLED (alpha={args.diversity_alpha:.2f})")
                
                # Apply final weighted loss computation
                loss = (loss * final_weights).sum() / (final_weights.sum() + 1e-6)
            
            # STABILITY: Check for NaN/Inf before backward
            # Ensure loss is a scalar at this point
            if loss.numel() != 1:
                logger.error(f"Loss is not a scalar! Shape: {loss.shape}, numel: {loss.numel()}")
                logger.error(f"disable_masking_loss={args.disable_masking_loss}")
                raise RuntimeError("Loss must be a scalar tensor at this point")
            
            if torch.isnan(loss).item() or torch.isinf(loss).item():
                logger.warning(f"NaN/Inf loss detected at step {args.completed_steps}! Skipping this batch.")
                if not args.disable_masking_loss and 'final_weights' in locals():
                    logger.warning(f"  final_weights sum: {final_weights.sum():.3f}, "
                                  f"max: {final_weights.max():.3f}, "
                                  f"contains nan: {torch.isnan(final_weights).any()}")
                continue  # Skip this batch
            
            # STABILITY: Apply gradient clipping after backward pass
            accelerator.backward(loss)
            if args.clip > 0:
                grad_norm = accelerator.clip_grad_norm_(var.parameters(), args.clip)
                # Log gradient norm occasionally
                if args.completed_steps % (args.log_interval * 10) == 0:
                    logger.info(f"  Gradient norm: {grad_norm:.3f} (clip: {args.clip})")
            
            # Apply custom LR/WD scheduling
            if not HAS_CUSTOM_SCHEDULER:
                raise ImportError("utils.lr_wd_annealing not available. Check utils/__init__.py")
            # Calculate warmup steps if not already done
            if not hasattr(args, 'num_warmup_steps_for_custom'):
                args.num_warmup_steps_for_custom = int(args.wp0 * args.max_train_steps)
            
            min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(
                args.lr_scheduler,
                optimizer,
                peak_lr=args.scaled_lr,  # Use scaled LR (batch-size-aware)
                wd=args.weight_decay,
                wd_end=args.weight_decay_end,
                cur_it=args.completed_steps,
                wp_it=args.num_warmup_steps_for_custom,
                max_it=args.max_train_steps,
                wp0=args.wp0,
                wpe=args.wpe,
                min_lr=float(getattr(args, "min_lr", 0.0)),
            )
            
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate epoch loss on every (micro)batch
            bsz = int(images_pre.shape[0]) if isinstance(images_pre, torch.Tensor) else int(masks.shape[0])
            epoch_loss_sum += float(loss.detach().item()) * bsz
            epoch_sample_count += bsz

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            args.completed_steps += 1

        # Log metrics
        if args.completed_steps % args.log_interval == 0:
            log_dict = {
                    "train/loss": loss.item(),
                    "step": args.completed_steps,
                    "epoch": args.epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                    "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0)
            }
            
            # Add stability metrics
            if args.clip > 0 and 'grad_norm' in locals():
                log_dict["train/grad_norm"] = grad_norm
            if 'current_alpha' in locals():
                log_dict["train/diversity_alpha"] = current_alpha
            if 'active_weights' in locals():
                log_dict["train/weight_max"] = active_weights.max().item()
                log_dict["train/weight_min"] = active_weights.min().item()
                log_dict["train/weight_ratio"] = (active_weights.max() / active_weights.min()).item()
            
            accelerator.log(log_dict, step=args.completed_steps)

        # Save model - step-based saving disabled, using epoch-based saving instead
        # if isinstance(args.save_interval, int):
        #     if args.completed_steps % args.save_interval == 0:
        #         save_dir = os.path.join(args.project_dir, f"step_{args.completed_steps}")
        #         os.makedirs(save_dir, exist_ok=True)
        #         accelerator.save_state(save_dir)

        # Optional expensive visualization inference during training.
        train_inf_every = int(getattr(args, "train_inference_interval_steps", 100))
        if accelerator.is_main_process and train_inf_every > 0 and args.completed_steps % train_inf_every == 0 and hasattr(args, 'last_batch'):
            # Use FIXED visualization indices/batch if available
            if hasattr(args, "viz_train_batch"):
                vb = args.viz_train_batch
                inference(
                    accelerator, var, vqvae, cond_model,
                    vb["images_pre"], vb["images_post"], vb["mask"],
                    vb["cls"], vb["type"],
                    fns=vb.get("fn", None),
                    guidance_scale=4.0, top_k=900, top_p=0.95, seed=42, tag="train_step",
                    select_samples=False,
                    deterministic=getattr(args, "deterministic", False),
                )
            else:
                # fallback to current batch
                current_batch = args.last_batch
                inference(
                    accelerator, var, vqvae, cond_model,
                    current_batch['images_pre'], current_batch['images_post'], current_batch['mask'],
                    current_batch['cls'], current_batch['type'],
                    fns=current_batch.get('fn', None),
                    guidance_scale=4.0, top_k=900, top_p=0.95, seed=42, tag="train_step",
                    deterministic=getattr(args, "deterministic", False),
                )

    # Reduce epoch loss across all processes and return average
    loss_sum_t = torch.tensor(epoch_loss_sum, device=accelerator.device, dtype=torch.float64)
    count_t = torch.tensor(epoch_sample_count, device=accelerator.device, dtype=torch.float64)
    loss_sum_t = accelerator.reduce(loss_sum_t, reduction="sum")
    count_t = accelerator.reduce(count_t, reduction="sum")
    avg_epoch_loss = (loss_sum_t / count_t.clamp_min(1.0)).item()
    return avg_epoch_loss

@torch.no_grad()
def _select_nonempty_samples(pre_images, post_images, masks, conditions, cond_type, fns=None, k=4):
    """
    Prefer samples whose GT mask has substantial foreground area.
    Masks are expected to be in [-1, 1] after dataset normalization.
    A sample is considered "non-black" if the fraction of pixels with mask>0.2 (in [0,1] space)
    is greater than 0.2 (i.e., >20% of the image).
    Returns sliced tensors (or original type for conditions/cond_type if not tensor).
    """
    try:
        # masks: (B, C, H, W)
        B = masks.shape[0]
        want = min(int(k), int(B))
        # Convert to [0,1] then compute foreground ratio using max channel (works for RGB-coded masks too)
        masks_01 = (masks + 1) / 2
        fg = (masks_01.max(dim=1).values > 0.2)  # (B,H,W) boolean
        fg_ratio = fg.view(B, -1).float().mean(dim=1)  # (B,)
        area_thr = 0.2
        non_empty_idx = torch.where(fg_ratio > area_thr)[0]

        # Always return `want` samples (unless B < want), to avoid logging single-image grids.
        if non_empty_idx.numel() >= want:
            perm = torch.randperm(non_empty_idx.numel(), device=non_empty_idx.device)
            idx = non_empty_idx[perm][:want]
            kept = int(want)
        else:
            # Not enough >area_thr masks in this batch (common for CD datasets).
            # Fill with the best remaining masks by fg_ratio to avoid showing all-black masks.
            kept = int(non_empty_idx.numel())
            need = want - kept

            # Candidates excluding already selected
            all_idx = torch.arange(B, device=masks.device)
            if kept > 0:
                mask_sel = torch.ones(B, dtype=torch.bool, device=masks.device)
                mask_sel[non_empty_idx] = False
                remaining = all_idx[mask_sel]
            else:
                remaining = all_idx

            # Take top fg_ratio among remaining
            if remaining.numel() > 0 and need > 0:
                vals = fg_ratio[remaining]
                topk = min(int(need), int(remaining.numel()))
                _, rel = torch.topk(vals, k=topk, largest=True)
                fill = remaining[rel]
                idx = torch.cat([non_empty_idx, fill], dim=0) if kept > 0 else fill
            else:
                idx = non_empty_idx
            # If still empty (degenerate), fallback to first `want`
            if idx.numel() == 0:
                idx = torch.arange(want, device=masks.device)

        def _maybe_index(x):
            if isinstance(x, torch.Tensor) and x.shape[0] == masks.shape[0]:
                return x[idx]
            return x
        fns_sel = None
        if isinstance(fns, (list, tuple)) and len(fns) == masks.shape[0]:
            fns_sel = [fns[i] for i in idx.tolist()]
        # Attach fg_ratio for debugging via returned kept/total; caller can compute if needed.
        return pre_images[idx], post_images[idx], masks[idx], _maybe_index(conditions), _maybe_index(cond_type), fns_sel, kept, int(masks.shape[0])
    except Exception:
        # safest fallback
        sel = slice(0, min(k, masks.shape[0]))
        fns_sel = None
        if isinstance(fns, (list, tuple)) and len(fns) == masks.shape[0]:
            fns_sel = list(fns[sel])
        return pre_images[sel], post_images[sel], masks[sel], conditions, cond_type, fns_sel, 0, min(k, masks.shape[0])


@torch.no_grad()
def inference(accelerator, var, vqvae, cond_model, pre_images, post_images, masks, conditions, cond_type,
              guidance_scale=4.0, top_k=900, top_p=0.95, seed=42, tag="train", fns=None, select_samples: bool = True, deterministic: bool = False):

    # IMPORTANT: visualization inference should run only on the main process.
    # Otherwise every GPU does a heavy autoregressive decode and can OOM.
    if not accelerator.is_main_process:
        return

    var.eval()
    if cond_model is not None:
        cond_model.eval()

    if select_samples:
        # Prefer substantial GT masks for logging so the grid is informative
        pre_images, post_images, masks, conditions, cond_type, fns_sel, kept, total = _select_nonempty_samples(
            pre_images, post_images, masks, conditions, cond_type, fns=fns, k=min(4, len(pre_images))
        )
    else:
        fns_sel = fns
        kept, total = 0, int(pre_images.shape[0]) if isinstance(pre_images, torch.Tensor) else 0

    # Compute fg ratios for debugging in caption
    try:
        m01 = (masks + 1) / 2
        fg = (m01.max(dim=1).values > 0.2).float().mean(dim=(1, 2))
        fg_str = ", ".join([f"{x:.2f}" for x in fg.tolist()])
    except Exception:
        fg_str = "n/a"

    # Use the inference method from inference.py for change detection
    # Reduce fragmentation risk before the heavy autoregressive decode.
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    out = pix_cond_inference(
        pre_images,
        post_images,
        masks,
        conditions,
        cond_type,
        accelerator.device,
        len(pre_images),
        var,
        vqvae,
        False,
        False,
        guidance_scale=guidance_scale,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        args=type('Args', (), {
                                   'v_patch_nums': [1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
                                   'image_size': 256
        })(),
        deterministic=deterministic,
        return_confidence=True,
    )
    if isinstance(out, tuple):
        images, conf_maps = out
    else:
        images, conf_maps = out, None

    # Create comprehensive comparison visualization (with optional confidence map)
    comparison_image = create_comparison_image(
        pre_images,
        post_images,
        images,
        masks,
        len(pre_images),
        256,
        confidence_maps=conf_maps,
    )
    caption = f"Inference ({tag}): Pre-Post-Predicted-GT | non_black(>20%)={kept}/{total} | fg_ratio={fg_str}"
    if fns_sel:
        caption += f" | fn={', '.join(map(str, fns_sel))}"
    accelerator.log({f"inference_comparison/{tag}": [wandb.Image(comparison_image, caption=caption)]})

    var.train()
    if cond_model is not None:
        cond_model.train()

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


@torch.no_grad()
def validate(accelerator, var, vqvae, cond_model, val_dataloader, args):
    """
    Calculate validation loss on the validation set.
    Optionally compute binary mask metrics (IoU/pixel-acc) over the entire validation set.
    Returns: (avg_loss, avg_loss_filtered, metrics_dict_or_None)
      - avg_loss: standard validation loss on all samples
      - avg_loss_filtered: validation loss only on samples with sufficient foreground (matching training filter)
    """
    var.eval()
    if cond_model is not None:
        cond_model.eval()
    
    # Use same token loss as training (per-token, reduction='none')
    loss_fn = _build_token_loss_fn(args, accelerator.device)
    
    total_loss = 0.0
    total_samples = 0
    
    # Filtered validation: only samples with sufficient foreground (matching training filter)
    total_loss_filtered = 0.0
    total_samples_filtered = 0
    filter_thr = float(getattr(args, 'empty_mask_threshold', 0.01))

    # Confusion matrix accumulation for binary mask metrics
    do_metrics = bool(getattr(args, "compute_val_metrics", True))
    # Optionally compute expensive IoU metrics only every N epochs to speed up training.
    try:
        every = int(getattr(args, "val_metrics_every", 1))
    except Exception:
        every = 1
    if every > 1:
        do_metrics = do_metrics and (int(getattr(args, "epoch", 0)) % every == 0)
    max_metric_batches = int(getattr(args, "val_metrics_max_batches", -1))
    hist_total = torch.zeros((2, 2), dtype=torch.float32, device=accelerator.device)
    labeled_total = torch.zeros((), dtype=torch.float32, device=accelerator.device)
    correct_total = torch.zeros((), dtype=torch.float32, device=accelerator.device)
    
    logger.info("Running validation...")
    for batch_idx, batch in enumerate(val_dataloader):
        images_pre, images_post, masks, conditions, cond_type = batch['images_pre'], batch['images_post'], batch['mask'], batch['cls'], batch['type']
        
        # Forward to get input ids
        mask_labels_list = vqvae.img_to_idxBl(masks, v_patch_nums=args.v_patch_nums)
        mask_input_h_list = vqvae.idxBl_to_h(mask_labels_list, include_next_scale=False)
        
        labels_list_pre = vqvae.img_to_idxBl(images_pre, v_patch_nums=args.v_patch_nums)
        labels_list_post = vqvae.img_to_idxBl(images_post, v_patch_nums=args.v_patch_nums)
        use_current_scale = bool(getattr(args, "enable_current_scale_tokens", False)) and args.mask_type == "change_append"
        input_h_list_pre = vqvae.idxBl_to_h(labels_list_pre, include_next_scale=use_current_scale)
        input_h_list_post = vqvae.idxBl_to_h(labels_list_post, include_next_scale=use_current_scale)
        
        # Compute context (autocast for fp16/bf16 friendliness)
        if args.disable_cross_attention:
            context = None
        else:
            with accelerator.autocast():
                model = var.module if hasattr(var, 'module') else var
                context = model.encode_context_with_fusion([images_pre, images_post])
        
        # Handle mask type
        if args.mask_type == 'change_append':
            labels_list_ = list(chain.from_iterable(zip(labels_list_pre, labels_list_post, mask_labels_list)))
            input_h_list_ = list(chain.from_iterable(zip(input_h_list_pre, input_h_list_post, mask_input_h_list)))
            mask_first = False
            labels_list, input_h_list = labels_list_, input_h_list_
        else:
            raise NotImplementedError
        
        x_BLCv_wo_first_l = torch.concat(input_h_list, dim=1)
        
        # Forward through model (autocast for fp16/bf16 friendliness)
        with accelerator.autocast():
            logits = var(conditions, x_BLCv_wo_first_l, context=context, mask_first=mask_first, cond_type=cond_type)
        labels = torch.cat(labels_list, dim=1)  # (B, L)
        Bsz, L = labels.shape
        logits = logits.view(-1, logits.size(-1))
        labels_1d = labels.reshape(-1)
        
        # IMPORTANT (memory): avoid materializing a fp32 copy of `logits` here (can OOM).
        loss = loss_fn(logits, labels_1d)
        
        # If disable_masking_loss is True, compute simple mean loss on all tokens with NO weighting
        if args.disable_masking_loss:
            batch_loss = loss.mean()
        else:
            # Create ignore_mask PER SAMPLE then flatten (same as training)
            ignore_mask_2d = torch.zeros((Bsz, L), dtype=torch.float, device=loss.device)
            current_pos = 0
            if args.mask_type == 'change_append':
                for pn in args.v_patch_nums:
                    tokens_per_scale = pn * pn
                    tokens_per_group = tokens_per_scale * 3
                    mask_start = current_pos + 2 * tokens_per_scale
                    mask_end = current_pos + 3 * tokens_per_scale
                    ignore_mask_2d[:, mask_start:mask_end] = 1.0
                    current_pos += tokens_per_group
            elif args.mask_type == 'interleave_append':
                for pn in args.v_patch_nums:
                    tokens_per_scale = pn * pn
                    if mask_first:
                        ignore_mask_2d[:, current_pos:current_pos + 2 * tokens_per_scale:2] = 1.0
                    else:
                        ignore_mask_2d[:, current_pos + 1:current_pos + 2 * tokens_per_scale:2] = 1.0
                    current_pos += 2 * tokens_per_scale
            elif args.mask_type == 'replace':
                for pn in args.v_patch_nums:
                    tokens_per_scale = pn * pn
                    ignore_mask_2d[:, current_pos:current_pos + tokens_per_scale] = 1.0
                    current_pos += tokens_per_scale
            else:
                raise NotImplementedError

            ignore_mask = ignore_mask_2d.reshape(-1)
            batch_loss = (loss * ignore_mask).sum() / (ignore_mask.sum() + 1e-6)
        
        # Skip batch if loss is NaN/Inf
        if torch.isnan(batch_loss) or torch.isinf(batch_loss):
            logger.warning(f"NaN/Inf loss detected in validation batch {batch_idx}! Skipping.")
            continue
        
        # Accumulate total loss (all samples)
        batch_size = images_pre.size(0)
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
        
        # Compute per-sample foreground ratio from GT masks for filtered validation
        # masks are in [-1,1] format from the dataloader
        masks_01 = (masks + 1) / 2  # convert to [0,1]
        fg_ratios = masks_01.flatten(1).max(dim=1)[0].mean(dim=-1)  # (B,) max across channels, then mean
        fg_mask = fg_ratios > filter_thr  # (B,) bool mask
        
        if fg_mask.any():
            total_loss_filtered += batch_loss.item() * fg_mask.sum().item()
            total_samples_filtered += fg_mask.sum().item()

        # Compute segmentation metrics on generated masks vs GT masks.
        # Note: this runs conditional generation on the full validation set (expensive but requested).
        if do_metrics and (max_metric_batches < 0 or batch_idx < max_metric_batches):
            try:
                B = int(images_pre.shape[0])
                pred_images = pix_cond_inference(
                    images_pre,
                    images_post,
                    masks,
                    conditions,
                    cond_type,
                    accelerator.device,
                    B,
                    var,
                    vqvae,
                    False,
                    False,
                    guidance_scale=1.0,
                    top_k=1,
                    top_p=0.0,
                    seed=0,
                    args=args,
                    deterministic=True,
                    return_confidence=False,
                    context=context,
                    c_img_pre_idxBl=labels_list_pre,
                    c_img_post_idxBl=labels_list_post,
                )
                if isinstance(pred_images, tuple):
                    pred_images = pred_images[0]

                h, lab, cor = confusion_from_pred_and_gt(
                    pred_images=pred_images,
                    gt_masks=masks,
                    image_size=int(getattr(args, "image_size", 256)),
                    pred_thr_01=0.1,
                    gt_thr_01=0.1,
                )
                hist_total += h.to(device=hist_total.device, dtype=hist_total.dtype)
                labeled_total += lab.to(device=labeled_total.device, dtype=labeled_total.dtype)
                correct_total += cor.to(device=correct_total.device, dtype=correct_total.dtype)
            except Exception as e:
                if accelerator.is_main_process:
                    logger.warning(f"Validation metric computation failed on batch {batch_idx}: {e}")
    
    # Calculate average losses
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_loss_filtered = total_loss_filtered / total_samples_filtered if total_samples_filtered > 0 else 0.0

    metrics = None
    if do_metrics:
        # Reduce across processes (val_dataloader is sharded by accelerate.prepare()).
        hist_sum = accelerator.reduce(hist_total, reduction="sum")
        labeled_sum = accelerator.reduce(labeled_total, reduction="sum")
        correct_sum = accelerator.reduce(correct_total, reduction="sum")
        metrics = scores_from_confusion(hist=hist_sum, labeled=labeled_sum, correct=correct_sum)
    
    # Set models back to train mode
    var.train()
    if cond_model is not None:
        cond_model.train()
    
    return avg_loss, avg_loss_filtered, metrics


def main():

    args = parse_args()

    # If resuming from a saved state dir, force run_name to match the resumed run,
    # but with a "CONTINUE" prefix so the new run is clearly separated.
    if args.resume_dir is not None:
        resume_state_dir = _infer_resume_dir(args.resume_dir)
        inferred_run_name = _infer_run_name_from_resume_dir(resume_state_dir)
        continue_name = f"CONTINUE_{inferred_run_name}"
        if args.run_name != continue_name:
            print(
                f"[resume] Overriding run_name to '{continue_name}' "
                f"(inferred from resume_dir='{resume_state_dir}')."
            )
        args.run_name = continue_name

    # seed (safe pre-Accelerate)
    # NOTE: accelerate's `set_seed(..., device_specific=True)` requires `AcceleratorState()` to be initialized
    # (i.e., `accelerator = Accelerator(...)` must already exist). We seed here in a process-agnostic way so any
    # pre-accelerator work is deterministic, then re-seed (device-specific) after Accelerator init below.
    set_seed(args.seed, device_specific=False)
    
    # Load precomputed class weights if specified
    if args.use_precomputed_weights:
        from types import SimpleNamespace

        def _infer_roots_for_token_freq(_args):
            """
            Return a deterministic list of dataset roots for building a stable dataset_id.
            This matches `datasets/build.py` defaults (WHU is fixed; unions are fixed unless data_dirs is explicitly set).
            """
            dd = getattr(_args, "data_dirs", None)
            if isinstance(dd, (list, tuple)) and len(dd) > 0:
                return list(dd)
            root = getattr(_args, "dataset_root", None) or os.environ.get("DATASET_ROOT") or getattr(_args, "data_dir", None)
            root = str(root) if root else ""
            if root:
                if _args.dataset_name in {"whu_cd", "change_dataset"}:
                    return [os.path.join(root, "whu_cd")]
                if _args.dataset_name == "cd_union":
                    cd_datasets = getattr(_args, "cd_union_datasets", ["whu_cd", "levircd", "levircdplus", "s2looking"])
                    return [os.path.join(root, str(ds)) for ds in cd_datasets]
                if _args.dataset_name == "levircd_union":
                    return [os.path.join(root, "levircd"), os.path.join(root, "levircdplus")]
                if _args.dataset_name == "levircd":
                    return [os.path.join(root, "levircd")]
                if _args.dataset_name == "levircdplus":
                    return [os.path.join(root, "levircdplus")]
                if _args.dataset_name == "s2looking":
                    return [os.path.join(root, "s2looking")]
            return None

        def _dataset_id_for_token_freq(_args):
            import hashlib
            roots = _infer_roots_for_token_freq(_args)
            if roots:
                # Use a short hash instead of full paths to avoid "File name too long" errors
                roots_str = ",".join(sorted([str(r) for r in roots]))
                roots_hash = hashlib.md5(roots_str.encode()).hexdigest()[:12]
                roots_part = f"hash{roots_hash}"
            else:
                roots_part = "default"
            return (
                f"{_args.dataset_name}__roots={roots_part}"
                f"__rgb={int(getattr(_args,'mask_rgb_by_location', False))}"
                f"__grid={getattr(_args,'mask_rgb_grid_size', None)}"
                f"__mode={getattr(_args,'mask_rgb_index_mode', None)}"
            )

        # Auto-select a frequency file if not explicitly provided
        if args.token_freq_path is None:
            dataset_id = _dataset_id_for_token_freq(args)
            args.token_freq_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token_frequencies", dataset_id + ".json")

        # If missing, auto-compute token frequencies for this dataset + mask representation.
        if not os.path.exists(args.token_freq_path):
            print(f"Token frequency file not found: {args.token_freq_path}")
            print("Auto-generating token frequencies for current training dataset/settings...")
            try:
                from calculate_token_frequencies import calculate_token_frequencies as _calc_freq
                os.makedirs(os.path.dirname(args.token_freq_path), exist_ok=True)

                vocab_size_val = args.vocab_size if isinstance(args.vocab_size, int) else args.vocab_size[0]
                freq_args = SimpleNamespace(
                    # dataset + roots
                    dataset_name=args.dataset_name,
                    data_dirs=_infer_roots_for_token_freq(args),
                    dataset_root=getattr(args, "dataset_root", None),
                    cd_union_datasets=getattr(args, "cd_union_datasets", None),
                    # vqvae params
                    vocab_size=vocab_size_val,
                    z_channels=args.z_channels,
                    ch=args.ch,
                    vqvae_pretrained_path=args.vqvae_pretrained_path,
                    v_patch_nums=args.v_patch_nums,
                    # output
                    output_path=args.token_freq_path,
                    # speed knobs
                    batch_size=min(int(getattr(args, "batch_size", 8)), 16),
                    num_workers=int(getattr(args, "num_workers", 8)),
                    diversity_alpha=float(getattr(args, "diversity_alpha", 2.0)),
                    # mask representation
                    mask_rgb_by_location=bool(getattr(args, "mask_rgb_by_location", False)),
                    mask_rgb_grid_size=getattr(args, "mask_rgb_grid_size", None),
                    mask_rgb_index_mode=getattr(args, "mask_rgb_index_mode", None),
                    # filtering consistent with training
                    filter_empty_masks=bool(getattr(args, "filter_empty_masks", False)),
                    empty_mask_threshold=float(getattr(args, "empty_mask_threshold", 0.001)),
                    # disable random aug during frequency calculation
                    enable_random_crop=False,
                    enable_random_flip=False,
                    enable_random_rotation=False,
                    enable_gaussian_blur=False,
                    enable_color_jitter=False,
                    min_crop_size=int(getattr(args, "min_crop_size", 64)),
                    max_crop_size=int(getattr(args, "max_crop_size", 256)),
                    crop_scale_range=getattr(args, "crop_scale_range", (1.0, 1.0)),
                    image_size=int(getattr(args, "image_size", 256)),
                )
                _calc_freq(freq_args)
            except Exception as e:
                raise RuntimeError(f"Failed to auto-generate token frequencies at {args.token_freq_path}: {e}") from e

        print(f"Loading precomputed token frequencies from: {args.token_freq_path}")
        try:
            with open(args.token_freq_path, 'r') as f:
                token_data = json.load(f)

            # Safety check: ensure the frequency file matches the current dataset + mask representation.
            expected_id = _dataset_id_for_token_freq(args)

            file_id = token_data.get("dataset_id", None)
            if expected_id is not None and file_id is not None and file_id != expected_id:
                raise ValueError(
                    f"Token frequency file mismatch.\n"
                    f"  expected dataset_id: {expected_id}\n"
                    f"  file dataset_id:     {file_id}\n"
                    f"Recompute frequencies for this dataset/mask setting or pass the correct --token_freq_path."
                )
            
            # Select weight type
            weight_key = f'class_weights_{args.weight_type}'
            if weight_key not in token_data:
                raise ValueError(f"Weight type '{args.weight_type}' not found in {args.token_freq_path}. "
                               f"Available types: inv, alpha, effective")
            
            class_weights = torch.tensor(token_data[weight_key], dtype=torch.float32)

            # Normalize weights so the expected weight under the (mask-token) training distribution is 1.0.
            # This keeps the overall loss scale comparable to the unweighted case.
            # Prefer token_counts_all if present (counts across ALL mask tokens).
            if 'token_counts_all' in token_data:
                counts = torch.tensor(token_data['token_counts_all'], dtype=torch.float32)
                denom = counts.sum().clamp_min(1.0)
                expected_w = (class_weights * counts).sum() / denom
                if torch.isfinite(expected_w) and expected_w > 0:
                    class_weights = class_weights / expected_w

                # STABILITY: clamp extreme weights to keep loss scale stable (mirrors dynamic weighting path).
                # Use counts>0 to avoid padding/unseen tokens.
                if args.max_weight_ratio > 0:
                    active = counts > 0
                    if active.any():
                        w_active = class_weights[active]
                        w_min = w_active.min().clamp_min(1e-8)
                        w_max_allowed = w_min * float(args.max_weight_ratio)
                        class_weights = torch.clamp(class_weights, max=w_max_allowed)
                        # Re-normalize expected weight to 1.0 after clamping
                        expected_w2 = (class_weights * counts).sum() / denom
                        if torch.isfinite(expected_w2) and expected_w2 > 0:
                            class_weights = class_weights / expected_w2

            args.class_weights = class_weights
            
            print(f"  Loaded {args.weight_type} weights")
            print(f"  Vocab size: {token_data['vocab_size']}")
            print(f"  Unique tokens in training: {token_data['unique_tokens_used']}")
            print(f"  Total tokens: {token_data['total_tokens']:,}")
            print(f"  Weight stats - Min: {class_weights.min():.3f}, Max: {class_weights.max():.3f}, "
                  f"Mean: {class_weights.mean():.3f}")
            # Extra debug: stable signature so we can confirm weight_type changes really changed the vector.
            try:
                import hashlib as _hashlib
                w_np = class_weights.detach().cpu().to(dtype=torch.float32).numpy()
                sig = _hashlib.md5(w_np.tobytes()).hexdigest()[:12]
                topv, topi = torch.topk(class_weights.detach().cpu(), k=5)
                botv, boti = torch.topk((-class_weights.detach().cpu()), k=5)
                botv = -botv
                print(f"  Weight signature(md5[:12]): {sig}")
                print("  Top-5 weights: " + ", ".join([f"{int(i)}:{float(v):.3f}" for i, v in zip(topi.tolist(), topv.tolist())]))
                print("  Bot-5 weights: " + ", ".join([f"{int(i)}:{float(v):.3f}" for i, v in zip(boti.tolist(), botv.tolist())]))
            except Exception:
                pass
            
            # Override diversity_alpha to indicate using precomputed weights
            print(f"  Note: Dynamic diversity weighting disabled (using precomputed weights)")
            
        except FileNotFoundError:
            raise RuntimeError(f"Token frequency file not found even after auto-generation: {args.token_freq_path}")
        except Exception as e:
            print(f"ERROR loading token frequencies: {e}")
            exit(1)


    # Setup accelerator:
    if args.run_name is None:
        model_name = f'vqvae_ch{args.ch}v{args.vocab_size}z{args.z_channels}_vpa_d{args.depth}e{args.embed_dim}h{args.num_heads}_{args.dataset_name}_ep{args.num_epochs}_bs{args.batch_size}'
        print(f"No run_name specified, using auto-generated name: {model_name}")
    else:
        model_name = args.run_name
        print(f"Using custom run name: {model_name}")
    args.model_name = model_name
    print(f"Wandb project: {args.wandb_project}")
    timestamp = datetime.fromtimestamp(time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.project_dir = f"{args.output_dir}/{timestamp}-{model_name}"  # Create an experiment folder
    os.makedirs(args.project_dir, exist_ok=True)
    
    # Save config and source files for reproducibility
    import shutil
    try:
        # Create codes subfolder
        codes_dir = os.path.join(args.project_dir, "codes")
        os.makedirs(codes_dir, exist_ok=True)
        
        # Save config YAML
        if args.config and os.path.exists(args.config):
            config_dst = os.path.join(codes_dir, os.path.basename(args.config))
            shutil.copy2(args.config, config_dst)
            print(f"Saved config to: {config_dst}")
        
        # Save training script
        train_script_src = os.path.abspath(__file__)
        train_script_dst = os.path.join(codes_dir, "train_remote_var.py")
        shutil.copy2(train_script_src, train_script_dst)
        print(f"Saved training script to: {train_script_dst}")
        
        # Save dataset script
        dataset_script_src = os.path.join(
            os.path.dirname(train_script_src),
            "remotevar_datasets",
            "change_dataset_simple.py",
        )
        if os.path.exists(dataset_script_src):
            dataset_script_dst = os.path.join(codes_dir, "change_dataset_simple.py")
            shutil.copy2(dataset_script_src, dataset_script_dst)
            print(f"Saved dataset script to: {dataset_script_dst}")
        
        # Save entire models folder
        models_src = os.path.join(os.path.dirname(train_script_src), "models")
        if os.path.exists(models_src) and os.path.isdir(models_src):
            models_dst = os.path.join(codes_dir, "models")
            shutil.copytree(models_src, models_dst, ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '*.pyo'))
            print(f"Saved models folder to: {models_dst}")
    except Exception as e:
        print(f"Warning: Failed to save config/source files for reproducibility: {e}")
    
    save_interval = args.save_interval
    if save_interval is not None and save_interval.isdigit():
        save_interval = int(save_interval)
        args.save_interval = save_interval

    tracker = None
    if args.use_wandb:
        tracker = CustomWandbTracker(
            model_name,
            project=args.wandb_project,
            mode=args.wandb_mode,
        )

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=tracker,
        project_dir=args.project_dir)

    # seed (post-Accelerate)
    # Now that AcceleratorState exists, use device_specific=True so each rank gets a distinct (but reproducible) RNG stream.
    set_seed(args.seed, device_specific=True)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Log run configuration after accelerator is initialized
    if accelerator.is_main_process:
        logger.info(f"Run name: {model_name}")
        logger.info(f"Wandb project: {args.wandb_project}")
        logger.info(f"Project directory: {args.project_dir}")


    # create dataset (filtering is now done inside the dataset class if filter_empty_masks=True)
    logger.info("Creating training dataset")
    if args.dataset_name == "cd_union":
        cd_datasets = getattr(args, "cd_union_datasets", ["whu_cd", "levircd", "levircdplus", "s2looking"])
        logger.info(f"  cd_union includes: {cd_datasets}")
    dataset = create_dataset(args.dataset_name, args, split='train')
    
    # create validation dataset (no filtering for validation - keep all samples)
    logger.info("Creating validation dataset")
    val_dataset = create_dataset(args.dataset_name, args, split='val')
    
    # Synchronize all ranks after dataset creation (prevents deadlock from rank-skew during filtering/init)
    if accelerator.num_processes > 1:
        logger.info("Synchronizing all ranks after dataset creation...")
        torch.distributed.barrier()
        logger.info("All ranks synchronized, creating dataloaders...")
    
    # create dataloader
    # Speed: keep workers alive across epochs when using >0 workers.
    dl_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if int(getattr(args, "num_workers", 0)) > 0:
        dl_kwargs["persistent_workers"] = True
        # default is 2, but be explicit (and only valid when num_workers>0)
        dl_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(dataset, shuffle=True, **dl_kwargs)
    # Keep validation batch size consistent with training (avoid smaller last batch).
    val_dataloader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
    
    # Calculate total batch size
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    args.total_batch_size = total_batch_size

    # Create VQVAE Model
    logger.info("Creating VQVAE model")
    vqvae = VQVAE(vocab_size=args.vocab_size, z_channels=args.z_channels, ch=args.ch, test_mode=True, share_quant_resi=4, v_patch_nums=args.v_patch_nums)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    if args.vqvae_pretrained_path is not None:
        vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path, map_location=torch.device('cpu')))

    # Create VPA Model
    logger.info("Creating VAR model")

    if bool(getattr(args, "allow_trainable_encoder", False)) and bool(getattr(args, "disable_cross_attention", False)):
        raise ValueError(
            "--allow_trainable_encoder requires cross-attention to be enabled (do not set --disable_cross_attention)."
        )

    var = build_remote_var(vae=vqvae, depth=args.depth, patch_nums=args.v_patch_nums, mask_type=args.mask_type,
                         cond_drop_rate=args.cond_drop_rate, bidirectional=args.bidirectional,
                         separate_decoding=args.separate_decoding, separator=args.separator, multi_cond=args.multi_cond,
                         disable_cross_attention=args.disable_cross_attention,
                         enable_current_scale_tokens=args.enable_current_scale_tokens,
                         image_size=args.image_size,
                         use_high_res_context_levels=args.use_high_res_context_levels,
                         fusion_downsample_ratios=args.fusion_downsample_ratios,
                         fusion_num_heads=getattr(args, "fusion_num_heads", 8),
                         fusion_num_layers=getattr(args, "fusion_num_layers", 1),
                         fusion_cross_inner_dim=getattr(args, "fusion_cross_inner_dim", None),
                         fusion_use_feature_rectify=getattr(args, "fusion_use_feature_rectify", False),
                         fusion_downsample_first=getattr(args, "fusion_downsample_first", False),
                         allow_trainable_encoder=bool(getattr(args, "allow_trainable_encoder", False)),
                         drop_path_rate=args.drop_path_rate,
                         cross_attn_inner_dim=args.cross_attn_inner_dim)

    if args.var_pretrained_path is not None:
        # Check if the file is a safetensors file
        if args.var_pretrained_path.endswith('.safetensors'):
            state_dict = _load_safetensors_state(args.var_pretrained_path)
            if 'model_state_dict' in state_dict.keys():
                var_state_dict = state_dict['model_state_dict']
            else:
                var_state_dict = state_dict
            
            missing_keys, unexpected_keys = var.load_state_dict(var_state_dict, strict=False)
            if accelerator.is_main_process:
                if unexpected_keys:
                    logger.warning(f"Unexpected keys in pretrained model: {unexpected_keys}")
                if missing_keys:
                    logger.warning(f"Missing keys in pretrained model: {missing_keys}")
        else:
            var_state_dict = torch.load(args.var_pretrained_path, map_location=torch.device('cpu'))
            init_std = math.sqrt(1 / args.embed_dim / 3)
            if args.mask_type == 'change_append':
                for key in ['lvl_1L', 'pos_start', 'attn_bias_for_masking', 'pos_1LC']:
                    if key in var_state_dict:
                        del var_state_dict[key]  # will be handled in the init
            var.load_state_dict(var_state_dict, strict=False)

    if args.lora:
        lora_params = []
        for name, _ in var.named_modules():
            if ('attn.' in name and 'attn.proj_drop' not in name) or 'ffn.fc' in name or 'ada_lin.1' in name:
                lora_params.append(name)
        # Define LoRA Config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=lora_params,
            lora_dropout=0.05,
            bias="none",
        )
        # add LoRA adaptor
        # var = prepare_model_for_kbit_training(var)
        var = get_peft_model(var, lora_config)
        var.print_trainable_parameters()

    # Optionally freeze ALL VAR params except cross-attention + fusion modules
    # Must happen BEFORE optimizer creation and before accelerator.prepare()
    if args.finetune_cross_and_fusion:
        if args.disable_cross_attention:
            raise ValueError("--finetune_cross_and_fusion requires cross-attention to be enabled (do not set --disable_cross_attention).")
        # If PEFT/LoRA is enabled, it can re-mark params as trainable; disallow for clarity.
        if args.lora:
            raise ValueError("--finetune_cross_and_fusion is not compatible with --lora (choose one fine-tuning strategy).")
        if hasattr(var, "freeze_all_except_cross_and_fusion"):
            var.freeze_all_except_cross_and_fusion()
        else:
            # Fallback: best-effort freezing by name
            for p in var.parameters():
                p.requires_grad_(False)
            # Fusion modules
            if hasattr(var, "fusion_modules"):
                for p in var.fusion_modules.parameters():
                    p.requires_grad_(True)
            # Optional trainable encoder copy (if present)
            if bool(getattr(args, "allow_trainable_encoder", False)) and hasattr(var, "trainable_encoder") and var.trainable_encoder is not None:
                for p in var.trainable_encoder.parameters():
                    p.requires_grad_(True)
            # Cross-attention modules
            if hasattr(var, "blocks"):
                for b in var.blocks:
                    if hasattr(b, "cross_attn") and b.cross_attn is not None:
                        for p in b.cross_attn.parameters():
                            p.requires_grad_(True)
                        # Zero-init cross-attn output projection to mimic "zero conv" finetuning.
                        if hasattr(b.cross_attn, "proj") and isinstance(b.cross_attn.proj, torch.nn.Linear):
                            with torch.no_grad():
                                b.cross_attn.proj.weight.zero_()
                                if b.cross_attn.proj.bias is not None:
                                    b.cross_attn.proj.bias.zero_()

    var.train()
    
    # Print trainability status to verify VQVAE is frozen and fusion modules are trainable
    if accelerator.is_main_process:
        # Handle DDP wrapper: access underlying module if wrapped
        model = var.module if hasattr(var, 'module') else var
        model.print_trainability_status()

    # Create Condition Model
    logger.info("Creating conditional model")
    if args.condition_model is None:
        cond_model = None
    elif args.condition_model == 'class_embedder':
        from models.class_embedder import ClassEmbedder
        cond_model = ClassEmbedder(num_classes=args.num_classes, embed_dim=args.embed_dim, cond_drop_rate=args.cond_drop_rate)
    else:
        raise NotImplementedError(f"Condition model {args.condition_model} is not implemented")

    # Create Optimizer with proper parameter grouping (no WD for biases/norms/embeddings)
    logger.info("Creating optimizer")
    
    if not HAS_CUSTOM_SCHEDULER or filter_params is None:
        raise ImportError("utils.filter_params not available. Check utils/__init__.py")
    
    # Scale learning rate based on total batch size (linear scaling rule)
    # Reference batch size is 512 (common baseline)
    args.scaled_lr = args.learning_rate * total_batch_size / 32
    if accelerator.is_main_process:
        logger.info(f"  Base learning rate: {args.learning_rate}")
        logger.info(f"  Scaled learning rate: {args.scaled_lr:.6f} (batch_size={total_batch_size}, ref=512)")
    
    # Filter VAR parameters into decay/no-decay groups
    nowd_keys = {
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    }
    allow_trainable_encoder = bool(getattr(args, "allow_trainable_encoder", False))
    allow_frozen_for_opt = bool(args.finetune_cross_and_fusion) or allow_trainable_encoder

    # Safety: in the default "train everything" mode, we expect *no frozen params* except
    # the intentionally-frozen latent head inside the trainable encoder copy (unused by forward_context()).
    if allow_trainable_encoder and not bool(args.finetune_cross_and_fusion):
        allowed_frozen = {
            "trainable_encoder.norm_out.weight",
            "trainable_encoder.norm_out.bias",
            "trainable_encoder.conv_out.weight",
            "trainable_encoder.conv_out.bias",
        }
        frozen_names = []
        for n, p in var.named_parameters():
            n = n.replace("_fsdp_wrapped_module.", "")
            if not p.requires_grad:
                frozen_names.append(n)
        extra = sorted(set(frozen_names) - allowed_frozen)
        if len(extra) > 0:
            raise ValueError(
                "Unexpected frozen parameters with allow_trainable_encoder=True. "
                f"Expected only {sorted(allowed_frozen)}, got extra frozen: {extra}"
            )

    names, paras, para_groups = filter_params(
        var,
        nowd_keys=nowd_keys,
        allow_frozen=allow_frozen_for_opt,
    )
    
    # Add condition model params (if exists) to the appropriate groups
    if cond_model is not None:
        for name, para in cond_model.named_parameters():
            if not para.requires_grad:
                continue
            # Determine if this param should have weight decay
            if para.ndim == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
                group_name = 'ND'
            else:
                group_name = 'D'
            # Find or create the group
            group = next((g for g in para_groups if g.get('wd_sc') == (0. if group_name == 'ND' else 1.)), None)
            if group is None:
                # Create new group if it doesn't exist
                wd_sc = 0. if group_name == 'ND' else 1.
                para_groups.append({'params': [], 'wd_sc': wd_sc, 'lr_sc': 1.})
                group = para_groups[-1]
            group['params'].append(para)

    # Optionally: use a different LR multiplier for fusion modules (freshly initialized).
    # This works with the custom lr_wd_annealing scheduler via per-group 'lr_sc'.
    fusion_lr_scale = float(getattr(args, "fusion_lr_scale", 1.0))
    fusion_wd_scale = float(getattr(args, "fusion_wd_scale", 1.0))
    if fusion_lr_scale <= 0:
        raise ValueError(f"--fusion_lr_scale must be > 0, got {fusion_lr_scale}")
    if fusion_wd_scale < 0:
        raise ValueError(f"--fusion_wd_scale must be >= 0, got {fusion_wd_scale}")
    if abs(fusion_lr_scale - 1.0) > 1e-12 or abs(fusion_wd_scale - 1.0) > 1e-12:
        if accelerator.is_main_process:
            logger.info(
                f"Applying fusion multipliers: fusion_lr_scale={fusion_lr_scale}, fusion_wd_scale={fusion_wd_scale}"
            )

        # Map VAR parameter objects -> their names (from filter_params output)
        param_id_to_name = {id(p): n for n, p in zip(names, paras)}
        new_groups = []
        moved = 0

        for g in para_groups:
            params = list(g.get("params", []))
            if not params:
                continue
            fusion_params = []
            other_params = []
            for p in params:
                pname = param_id_to_name.get(id(p), "")
                if "fusion_modules" in pname:
                    fusion_params.append(p)
                else:
                    other_params.append(p)

            if not fusion_params:
                continue

            moved += len(fusion_params)
            if not other_params:
                # Group contains only fusion params: just scale its lr_sc
                g["lr_sc"] = float(g.get("lr_sc", 1.0)) * fusion_lr_scale
                g["wd_sc"] = float(g.get("wd_sc", 1.0)) * fusion_wd_scale
            else:
                # Split group into non-fusion + fusion subgroups
                g["params"] = other_params
                new_groups.append(
                    {
                        "params": fusion_params,
                        "wd_sc": float(g.get("wd_sc", 1.0)) * fusion_wd_scale,
                        "lr_sc": float(g.get("lr_sc", 1.0)) * fusion_lr_scale,
                    }
                )

        para_groups.extend(new_groups)
        if accelerator.is_main_process:
            logger.info(f"  Fusion LR scale applied to {moved} param tensors across {len(new_groups)} new group(s)")
    
    # Log parameter group statistics
    if accelerator.is_main_process:
        total_params = sum(len(g['params']) for g in para_groups)
        decay_params = sum(len(g['params']) for g in para_groups if g.get('wd_sc', 1.0) > 0)
        no_decay_params = sum(len(g['params']) for g in para_groups if g.get('wd_sc', 1.0) == 0)
        logger.info(f"Parameter groups: {len(para_groups)} groups, {total_params} total params")
        logger.info(f"  Decay group: {decay_params} params (WD will be applied)")
        logger.info(f"  No-decay group: {no_decay_params} params (bias, norms, embeddings)")
    
    optimizer = torch.optim.AdamW(para_groups, lr=args.scaled_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    # Compute max_train_steps
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_epochs * num_update_steps_per_epoch // accelerator.num_processes

    # Note: We use custom lr_wd_annealing scheduler, so we don't need HF's get_scheduler
    # The scheduling is done manually in the training loop via lr_wd_annealing()
    logger.info("Using custom lr_wd_annealing scheduler with weight decay scheduling")
    
    # Send to accelerator (no lr_scheduler needed for custom scheduling)
    var, cond_model, vqvae, optimizer, dataloader, val_dataloader = accelerator.prepare(var, cond_model, vqvae, optimizer, dataloader, val_dataloader)

    # Start tracker
    experiment_config = vars(args)
    if args.use_wandb:
        accelerator.init_trackers(model_name, config=experiment_config)

    # Start training
    if accelerator.is_main_process:
        logger.info("***** Training arguments *****")
        logger.info(args)
        logger.info("***** Running training *****")
        logger.info(f"  Num training examples = {len(dataset)}")
        logger.info(f"  Num validation examples = {len(val_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Total optimization steps per epoch {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        logger.info(f"  Base learning rate = {args.learning_rate}")
        logger.info(f"  Scaled learning rate = {args.scaled_lr:.6f} (batch_size={total_batch_size}, ref=512)")
        logger.info(f"  Weight decay schedule: {args.weight_decay} -> {args.weight_decay_end}")
        logger.info(f"  LR schedule: {args.lr_scheduler} (wp0={args.wp0}, wpe={args.wpe}, min_lr={getattr(args, 'min_lr', 0.0)})")
        logger.info(f"  Validation interval = every {args.val_interval} epoch(s)")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    args.completed_steps = 0
    args.starting_epoch = 0

    # Resume from an Accelerate save_state directory (restores model/optimizer/scheduler).
    # IMPORTANT: if you want to change LR on resume, do it AFTER load_state().
    if args.resume_dir is not None:
        resume_dir = _infer_resume_dir(args.resume_dir)
        if accelerator.is_main_process:
            logger.info(f"Resuming from Accelerate state dir: {resume_dir}")

        did_full_resume = False
        try:
            accelerator.load_state(resume_dir)
            did_full_resume = True
        except Exception as e:
            if not args.resume_allow_mismatch:
                raise
            # Fallback: sequential fine-tuning where architecture differs (e.g., cross-attn enabled now)
            if accelerator.is_main_process:
                logger.warning(
                    f"Full accelerate.load_state() failed (likely due to state_dict mismatch): {e}\n"
                    f"Falling back to weights-only load from {resume_dir} (strict={args.resume_model_strict})."
                )
            _fallback_load_models_from_accelerate_dir(
                resume_dir,
                accelerator=accelerator,
                var=var,
                cond_model=cond_model,
                vqvae=vqvae,
                strict=bool(args.resume_model_strict),
                logger=logger if accelerator.is_main_process else None,
            )
            # Note: With custom lr_wd_annealing, we don't use HF scheduler state.
            # Step count is inferred from checkpoint metadata or folder name below.

        # Best-effort epoch restore from folder name (epoch_99 -> next epoch 100)
        inferred_start_ep = _infer_starting_epoch_from_resume_dir(resume_dir)
        if inferred_start_ep is not None:
            args.starting_epoch = inferred_start_ep

        # Bring progress bar up to date
        try:
            progress_bar.update(args.completed_steps)
        except Exception:
            pass

        # Apply LR override if requested (scale relative to the resumed LR)
        # Note: with custom scheduler, lr_scheduler is None
        _apply_resume_lr_override(
            optimizer,
            None,  # no HF scheduler when using custom lr_wd_annealing
            resume_lr=args.resume_lr,
            resume_lr_scale=args.resume_lr_scale,
            logger=logger if accelerator.is_main_process else None,
        )

        if accelerator.is_main_process:
            try:
                cur_lr = optimizer.param_groups[0]["lr"] if optimizer is not None else None
            except Exception:
                cur_lr = None
            logger.info(
                f"Resume summary: starting_epoch={args.starting_epoch}, "
                f"completed_steps={args.completed_steps}, current_lr={cur_lr}"
            )
    # Build fixed visualization indices/batches once on rank 0. Every other rank
    # waits instead of independently scanning the complete train and val datasets.
    train_viz_idxs = None
    val_viz_idxs = None
    if accelerator.is_main_process:
        try:
            # Use a no-augmentation clone of args for stable visualization frames.
            viz_args = copy.deepcopy(args)
            viz_args.enable_random_crop = False
            viz_args.enable_random_flip = False
            viz_args.enable_random_rotation = False
            viz_args.enable_gaussian_blur = False
            viz_args.enable_color_jitter = False

            viz_train_dataset = create_dataset(viz_args.dataset_name, viz_args, split="train")
            viz_val_dataset = val_dataset  # build.py already disables aug for val/test

            viz_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz_indices")
            viz_k = 4
            viz_pix_thr = 0.2
            viz_area_thr = 0.2
            # Prefer mid-sized change for visualization (easier to see differences than near-100% masks).
            viz_target_ratio = float(getattr(args, "viz_target_ratio", 0.5))
            viz_fallback_ratio = float(getattr(args, "viz_fallback_ratio", 0.2))

            # Ensure viz_args has the same data_dirs as main args for consistent cache keys.
            if hasattr(args, "data_dirs"):
                viz_args.data_dirs = args.data_dirs

            logger.info(f"Selecting fixed train visualization indices for dataset_id: {_dataset_id_for_run(viz_args)}")
            train_viz_idxs = _load_or_create_viz_indices(
                args=viz_args,
                dataset=viz_train_dataset,
                split_name="train",
                cache_dir=viz_cache_dir,
                k=viz_k,
                pixel_thr_01=viz_pix_thr,
                area_thr=viz_area_thr,
                seed=int(getattr(args, "seed", 0)),
                target_ratio=viz_target_ratio,
                fallback_target_ratio=viz_fallback_ratio,
            )
            logger.info(f"Selecting fixed val visualization indices for dataset_id: {_dataset_id_for_run(args)}")
            val_viz_idxs = _load_or_create_viz_indices(
                args=args,
                dataset=viz_val_dataset,
                split_name="val",
                cache_dir=viz_cache_dir,
                k=viz_k,
                pixel_thr_01=viz_pix_thr,
                area_thr=viz_area_thr,
                seed=int(getattr(args, "seed", 0)) + 1,
                target_ratio=viz_target_ratio,
                fallback_target_ratio=viz_fallback_ratio,
            )

            viz_train_loader = DataLoader(
                Subset(viz_train_dataset, train_viz_idxs),
                batch_size=len(train_viz_idxs),
                shuffle=False,
                num_workers=0,
            )
            viz_val_loader = DataLoader(
                Subset(viz_val_dataset, val_viz_idxs),
                batch_size=len(val_viz_idxs),
                shuffle=False,
                num_workers=0,
            )
            viz_train_batch = next(iter(viz_train_loader))
            viz_val_batch = next(iter(viz_val_loader))

            args.viz_train_batch = viz_train_batch
            args.viz_val_batch = viz_val_batch
            args.viz_train_fns = viz_train_batch.get("fn", None)
            args.viz_val_fns = viz_val_batch.get("fn", None)

            logger.info(f"Initial inference on FIXED TRAIN indices: {train_viz_idxs}")
            inference(
                accelerator, var, vqvae, cond_model,
                viz_train_batch["images_pre"], viz_train_batch["images_post"], viz_train_batch["mask"],
                viz_train_batch["cls"], viz_train_batch["type"],
                fns=viz_train_batch.get("fn", None),
                guidance_scale=4.0, top_k=900, top_p=0.95, seed=42, tag="train_init",
                select_samples=False,
                deterministic=getattr(args, "deterministic", False),
            )
        except Exception as e:
            logger.warning(f"Fixed visualization index selection failed; falling back to batch-based logging. Error: {e}")
            sample_batch = next(iter(dataloader))
            inference(
                accelerator, var, vqvae, cond_model,
                sample_batch["images_pre"], sample_batch["images_post"], sample_batch["mask"],
                sample_batch["cls"], sample_batch["type"],
                fns=sample_batch.get("fn", None),
                guidance_scale=4.0, top_k=900, top_p=0.95, seed=42, tag="train_init",
                deterministic=getattr(args, "deterministic", False),
            )

    # Propagate the selected indices for consistent rank state/debugging. The
    # visualization tensors stay on rank 0 because only rank 0 performs inference/logging.
    viz_indices_payload = [train_viz_idxs, val_viz_idxs]
    broadcast_object_list(viz_indices_payload, from_process=0)
    args.viz_train_indices, args.viz_val_indices = viz_indices_payload
    accelerator.wait_for_everyone()

    # Training
    # Track best checkpoint based on validation segmentation metrics (mean IoU primary, pixel acc tie-break).
    best_mean_iou = -1.0
    best_pixel_acc = -1.0
    best_epoch = -1
    best_step = -1
    best_dir = os.path.join(args.project_dir, "best")
    best_meta_path = os.path.join(best_dir, "best_metrics.json")
    if accelerator.is_main_process and os.path.exists(best_meta_path):
        try:
            with open(best_meta_path, "r") as f:
                _bm = json.load(f)
            best_mean_iou = float(_bm.get("mean_iou", best_mean_iou))
            best_pixel_acc = float(_bm.get("pixel_acc", best_pixel_acc))
            best_epoch = int(_bm.get("epoch", best_epoch))
            best_step = int(_bm.get("step", best_step))
            logger.info(
                f"[best] Loaded existing best metrics: mean_iou={best_mean_iou:.4f}, "
                f"pixel_acc={best_pixel_acc:.4f} (epoch={best_epoch}, step={best_step})"
            )
        except Exception:
            pass

    for epoch in range(args.starting_epoch, args.num_epochs):

        args.epoch = epoch
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1}/{args.num_epochs}")

        # train epoch
        train_epoch_loss = train_epoch(accelerator, var, vqvae, cond_model, dataloader, optimizer, progress_bar, args)
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Train Loss: {train_epoch_loss:.4f}")
            accelerator.log({
                "train/epoch_loss": train_epoch_loss,
                "epoch": epoch,
                "step": args.completed_steps,
            }, step=args.completed_steps)

        if epoch % args.val_interval == 0:
            # Calculate validation loss (both total and filtered)
            val_loss, val_loss_filtered, val_metrics = validate(accelerator, var, vqvae, cond_model, val_dataloader, args)
            
            # Log validation losses
            if accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch+1}/{args.num_epochs} - "
                    f"Val Loss: {val_loss:.4f} (all) | {val_loss_filtered:.4f} (filtered, thr={getattr(args, 'empty_mask_threshold', 0.01)})"
                )
                accelerator.log({
                    "val/loss": val_loss,
                    "val/loss_filtered": val_loss_filtered,
                    "epoch": epoch,
                    "step": args.completed_steps
                }, step=args.completed_steps)

                # Log segmentation metrics (mean over full validation set)
                if val_metrics is not None:
                    logger.info(
                        f"Val metrics - mean_iou={val_metrics.get('mean_iou', float('nan')):.4f}, "
                        f"iou_fg={val_metrics.get('iou_fg', float('nan')):.4f}, "
                        f"pixel_acc={val_metrics.get('pixel_acc', float('nan')):.4f}, "
                        f"precision_fg={val_metrics.get('precision_fg', float('nan')):.4f}, "
                        f"recall_fg={val_metrics.get('recall_fg', float('nan')):.4f}"
                    )
                    accelerator.log(
                        {
                            "val/mean_iou": val_metrics.get("mean_iou"),
                            "val/iou_fg": val_metrics.get("iou_fg"),
                            "val/iou_bg": val_metrics.get("iou_bg"),
                            "val/pixel_acc": val_metrics.get("pixel_acc"),
                            "val/mean_pixel_acc": val_metrics.get("mean_pixel_acc"),
                            "val/precision_fg": val_metrics.get("precision_fg"),
                            "val/recall_fg": val_metrics.get("recall_fg"),
                            "val/freq_iou": val_metrics.get("freq_iou"),
                        },
                        step=args.completed_steps,
                    )

                    # Save best checkpoint based on metrics.
                    cur_mean_iou = float(val_metrics.get("mean_iou", -1.0))
                    cur_pixel_acc = float(val_metrics.get("pixel_acc", -1.0))
                    eps = 1e-12
                    improved = (cur_mean_iou > best_mean_iou + eps) or (
                        abs(cur_mean_iou - best_mean_iou) <= eps and cur_pixel_acc > best_pixel_acc + eps
                    )
                    if improved:
                        os.makedirs(best_dir, exist_ok=True)
                        accelerator.save_state(best_dir)
                        best_mean_iou = cur_mean_iou
                        best_pixel_acc = cur_pixel_acc
                        best_epoch = int(epoch)
                        best_step = int(args.completed_steps)
                        payload = {
                            "epoch": best_epoch,
                            "step": best_step,
                            **{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in val_metrics.items()},
                            "val_metric_deterministic": True,
                            "val_metric_pred_binarize": "thr0.1",
                            "val_metric_gt_thr_01": 0.1,
                        }
                        try:
                            with open(best_meta_path, "w") as f:
                                json.dump(payload, f, indent=2)
                        except Exception as e:
                            logger.warning(f"[best] Failed to write best_metrics.json: {e}")
                        logger.info(
                            f"[best] Saved new best checkpoint to '{best_dir}' "
                            f"(mean_iou={best_mean_iou:.4f}, pixel_acc={best_pixel_acc:.4f})"
                        )
            
            # Visualization inference and its fallback DataLoader read belong to rank 0 only.
            if accelerator.is_main_process:
                try:
                    if hasattr(args, "viz_val_batch"):
                        vb = args.viz_val_batch
                        logger.info(f"Epoch {epoch+1}: Running inference on FIXED VALIDATION indices")
                        # Inference decode can be memory-heavy; run multiple 1-sample decodes sequentially
                        # to keep peak VRAM low while still visualizing multiple examples.
                        k_viz = min(4, int(vb["images_pre"].shape[0]))
                        for i in range(k_viz):
                            inference(
                                accelerator, var, vqvae, cond_model,
                                vb["images_pre"][i:i+1], vb["images_post"][i:i+1], vb["mask"][i:i+1],
                                vb["cls"][i:i+1], vb["type"][i:i+1],
                                fns=vb.get("fn", None),
                                guidance_scale=4.0, top_k=900, top_p=0.95, seed=42, tag=f"val_{i}",
                                select_samples=False,
                                deterministic=args.deterministic,
                            )
                    else:
                        val_batch = next(iter(val_dataloader))
                        logger.info(f"Epoch {epoch+1}: Running inference on VALIDATION batch (fallback)")
                        inference(
                            accelerator, var, vqvae, cond_model,
                            val_batch["images_pre"], val_batch["images_post"], val_batch["mask"],
                            val_batch["cls"], val_batch["type"],
                            fns=val_batch.get("fn", None),
                            guidance_scale=4.0, top_k=900, top_p=0.95, seed=42, tag="val",
                            deterministic=getattr(args, "deterministic", False),
                        )
                except Exception as e:
                    logger.warning(f"Validation visualization failed: {e}")


        # Save every N epochs or at every epoch
        should_save = False
        if args.save_interval == 'epoch':
            should_save = True
        elif isinstance(args.save_interval, int):
            should_save = (args.epoch + 1) % args.save_interval == 0

        if should_save:
            save_dir = os.path.join(args.project_dir, f"epoch_{args.epoch}")
            os.makedirs(save_dir, exist_ok=True)
            accelerator.save_state(save_dir)
    
    # end training
    accelerator.end_training()

    
if __name__ == '__main__':
    main()