import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hf_datasets_compat import ensure_huggingface_datasets

ensure_huggingface_datasets()

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from ruamel.yaml import YAML
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from remotevar_datasets import create_dataset
from remotevar_datasets.utils import auto_color_levels_for_required_colors, create_entity_like_color_map
from models import build_remote_var
from models.vae_modules import ConditionedDecoder, Decoder
from models.fusion import FeatureFusionModule
from models.vqvae import VQVAE
from utils.lr_control import lr_wd_annealing, filter_params
from utils.mask_metrics import confusion_from_pred_and_gt, scores_from_confusion


logger = get_logger(__name__)


def _as_int(x):
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


def _to_01(x: torch.Tensor) -> torch.Tensor:
    # dataset masks are in [-1,1]; keep this stable for safety.
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


@torch.no_grad()
def _encode_context_with_fusion_frozen(
    *,
    var_model,
    images_pre: torch.Tensor,
    images_post: torch.Tensor,
) -> list:
    """
    Encode cross-attention context using the *frozen* VQVAE encoder + the *frozen* fusion modules.

    Why: when `train_fusion=true`, we may be training a separate fusion copy for decoder skips
    (e.g., var_model.fusion_modules_for_skips). The RemoteVAR transformer blocks are frozen in this
    refiner, so feeding them a drifting context representation can degrade mask-token prediction.
    """
    if getattr(var_model, "disable_cross_attention", False):
        return [None] * int(len(getattr(var_model, "fusion_modules", [])))
    vae = var_model.vae_proxy[0]
    # Force frozen encoder context (never use trainable_encoder for cross-attn context here).
    pre_contexts = vae.encoder.forward_context(images_pre.to(dtype=torch.float32), return_all_levels=var_model.use_high_res_context_levels)
    post_contexts = vae.encoder.forward_context(images_post.to(dtype=torch.float32), return_all_levels=var_model.use_high_res_context_levels)
    fused_contexts = []
    for i, (pre_ctx, post_ctx) in enumerate(zip(pre_contexts, post_contexts)):
        B, L, C = pre_ctx.shape
        H = W = int(L ** 0.5)
        pre_2d = pre_ctx.transpose(1, 2).reshape(B, C, H, W)
        post_2d = post_ctx.transpose(1, 2).reshape(B, C, H, W)
        fused_2d = var_model.fusion_modules[i](pre_2d, post_2d)
        fused_blc = fused_2d.flatten(2).transpose(1, 2)
        fused_contexts.append(fused_blc)
    return fused_contexts


@torch.no_grad()
def _pred_mask_fhat_from_teacher_forcing_forward(
    *,
    vqvae: VQVAE,
    var_model: nn.Module,
    images_pre: torch.Tensor,
    images_post: torch.Tensor,
    gt_mask: torch.Tensor,
    conditions: torch.Tensor,
    cond_type: torch.Tensor,
    context: Optional[Sequence[torch.Tensor]],
    v_patch_nums: Sequence[int],
    mask_type: str,
    noisy_tf_mask_prob: float = 0.0,
    noisy_tf_mask_mode: str = "random",
    enable_current_scale_tokens: bool = False,
) -> torch.Tensor:
    """
    Compute predicted mask f_hat (B, Cvae, pn, pn) ON-THE-FLY using RemoteVAR teacher-forcing forward (single pass),
    without running autoregressive sampling.

    This mirrors `train_remote_var.py` teacher-forcing input construction, but instead of computing a CE loss,
    we take argmax token predictions for the mask stream and reconstruct the cumulative f_hat using VQVAE quantizer.
    """
    if str(mask_type) != "change_append":
        raise NotImplementedError(
            f"--use_teacher_forcing_forward is currently implemented only for mask_type='change_append', got '{mask_type}'."
        )

    # Ensure stable tokenization (fp32, no autocast)
    images_pre = images_pre.to(dtype=torch.float32)
    images_post = images_post.to(dtype=torch.float32)
    gt_mask = gt_mask.to(dtype=torch.float32)

    with torch.autocast(device_type=images_pre.device.type, enabled=False):
        # GT token IDs
        labels_list_pre = vqvae.img_to_idxBl(images_pre, v_patch_nums=v_patch_nums)
        labels_list_post = vqvae.img_to_idxBl(images_post, v_patch_nums=v_patch_nums)
        mask_labels_list = vqvae.img_to_idxBl(gt_mask, v_patch_nums=v_patch_nums)

        # Optional: noisy teacher forcing for MASK stream (input only; labels stay clean).
        noisy_p = float(noisy_tf_mask_prob or 0.0)
        noisy_mode = str(noisy_tf_mask_mode or "random").lower()
        mask_labels_for_tf = mask_labels_list
        if noisy_p > 0:
            vocab_size_val = int(getattr(vqvae, "vocab_size", vqvae.V if hasattr(vqvae, "V") else 4096))
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

        # Teacher-forcing inputs (cumulative f_hat up to previous scale; per-scale list, excluding the first stage)
        use_current_scale = bool(enable_current_scale_tokens) and str(mask_type) == "change_append"
        input_h_list_pre = vqvae.idxBl_to_h(labels_list_pre, include_next_scale=use_current_scale)
        input_h_list_post = vqvae.idxBl_to_h(labels_list_post, include_next_scale=use_current_scale)
        mask_input_h_list = vqvae.idxBl_to_h(mask_labels_for_tf, include_next_scale=False)

        # change_append layout: [pre1, post1, mask1, pre2, post2, mask2, ...]
        input_h_list = list(chain.from_iterable(zip(input_h_list_pre, input_h_list_post, mask_input_h_list)))
        x_BLCv_wo_first_l = torch.cat(input_h_list, dim=1) if len(input_h_list) > 0 else None

    if x_BLCv_wo_first_l is None:
        raise RuntimeError("Unexpected empty teacher-forcing input; check v_patch_nums.")

    # Run single-pass forward (teacher-forcing) to get logits over vocab for all tokens.
    # NOTE: `context` can be None only if disable_cross_attention=True (not supported in this refiner).
    logits = var_model(
        conditions,
        x_BLCv_wo_first_l,
        context=context,
        mask_first=False,
        cond_type=cond_type,
    )  # (B, L, V)
    vocab = int(getattr(vqvae, "vocab_size", 4096))
    pred_ids = torch.argmax(logits[:, :, :vocab], dim=-1).to(dtype=torch.long)  # (B, L)

    # Extract mask-token IDs per scale from the change_append layout.
    ms_idx_mask = []
    cur = 0
    for pn in list(v_patch_nums):
        pn = int(pn)
        t = pn * pn
        mask_start = cur + 2 * t
        mask_end = cur + 3 * t
        ms_idx_mask.append(pred_ids[:, mask_start:mask_end])
        cur += 3 * t

    # Convert ms mask token IDs -> embeddings -> cumulative f_hat at max scale.
    ms_h_mask = []
    Cvae = int(getattr(vqvae, "Cvae", 32))
    for idx_Bl, pn in zip(ms_idx_mask, list(v_patch_nums)):
        pn = int(pn)
        h = vqvae.quantize.embedding(idx_Bl)  # (B, pn^2, Cvae)
        h = h.transpose(1, 2).contiguous().view(h.shape[0], Cvae, pn, pn)  # (B, Cvae, pn, pn)
        ms_h_mask.append(h)
    f_hat = vqvae.quantize.embed_to_fhat(ms_h_mask, all_to_max_scale=True, last_one=True)  # (B,Cvae,pn,pn)
    return f_hat.to(device=images_pre.device, dtype=torch.float32)


def _infer_roots_for_run(args) -> Optional[list]:
    """Match datasets/build.py defaults to build a stable dataset_id for caching (same as train_remote_var.py)."""
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
    roots = _infer_roots_for_run(args)
    if roots:
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


def _resolve_precomputed_predictions_source(args) -> tuple:
    """
    Determine which cached prediction files to load.

    By default, we load:
      <predictions_dir>/<dataset_name>_train_mask_fhat.pt
      <predictions_dir>/<dataset_name>_val_mask_fhat.pt

    If you changed `dataset_name` (e.g., training on whu_cd only) but you *already* have cached
    predictions for a larger union (e.g., cd_union), you can reuse those caches by setting:

      --precomputed_predictions_dataset_name cd_union
      --precomputed_predictions_cd_union_datasets whu_cd levircd levircdplus s2looking
      --precomputed_predictions_allow_subset_from_union
    """
    src_name = getattr(args, "precomputed_predictions_dataset_name", None) or getattr(args, "dataset_name", None)
    src_name = str(src_name)
    src_union_list = getattr(args, "precomputed_predictions_cd_union_datasets", None)
    if src_union_list is not None:
        src_union_list = [str(x) for x in list(src_union_list)]
    return src_name, src_union_list


def _slice_precomputed_predictions_from_cd_union(
    pred: torch.Tensor,
    *,
    split: str,
    args,
    source_cd_union_datasets: Sequence[str],
    target_dataset_name: str,
    target_cd_union_datasets: Optional[Sequence[str]],
) -> torch.Tensor:
    """
    Slice/reorder a precomputed prediction tensor generated for cd_union so it matches the current dataset.

    Assumption (matches datasets/build.py for cd_union):
    - cd_union is built as ConcatDataset([ds_0, ds_1, ...]) in the order of cd_union_datasets.
    - Cached predictions were saved in that same dataset order.

    Supported target modes:
    - target_dataset_name in {"whu_cd","levircd","levircdplus","s2looking",...}: returns the corresponding contiguous slice.
    - target_dataset_name == "cd_union": returns concatenated slices for `target_cd_union_datasets` order.
    """
    # Compute per-dataset lengths for this split using the SAME cd_union filtering logic as the cache generator:
    # - For split != "train", cd_union drops roots that do not have <split>.txt (it does not auto-create them).
    # - This matters for levircdplus / s2looking which often have train only.
    lengths: Dict[str, int] = {}
    for ds_name in list(source_cd_union_datasets):
        ds_name = str(ds_name)
        try:
            tmp = copy.copy(args)
            # Force cd_union construction for exactly one component dataset.
            setattr(tmp, "cd_union_datasets", [ds_name])
            # Ensure we do not override roots via data_dirs unless the caller explicitly wants that behavior.
            # (Most runs rely on the fixed defaults inside datasets/build.py.)
            if hasattr(tmp, "data_dirs"):
                setattr(tmp, "data_dirs", None)
            ds = create_dataset("cd_union", tmp, split=split)
            lengths[ds_name] = int(len(ds))
        except FileNotFoundError:
            # cd_union for this split had no valid roots for this component (e.g., no val.txt) -> treat as absent.
            lengths[ds_name] = 0
        except ValueError as e:
            # Unknown cd_union component dataset name -> propagate with clear context.
            raise ValueError(f"Unknown entry in precomputed_predictions_cd_union_datasets: '{ds_name}'. Error: {e}")

    # Build offsets in the SOURCE union order
    offsets: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for ds_name in list(source_cd_union_datasets):
        n = int(lengths.get(str(ds_name), 0))
        offsets[str(ds_name)] = (cur, cur + n)
        cur += n

    # Determine target dataset list
    target_name = str(target_dataset_name)
    if target_name == "cd_union":
        if not target_cd_union_datasets:
            raise ValueError(
                "Target dataset_name is cd_union but target_cd_union_datasets is empty. "
                "Set --cd_union_datasets (or in config) to define the target union order."
            )
        wanted = [str(x) for x in list(target_cd_union_datasets)]
    else:
        wanted = [target_name]

    # Slice and concatenate in target order
    chunks = []
    pred_len = int(pred.shape[0])
    for ds_name in wanted:
        if ds_name not in offsets:
            raise ValueError(
                f"Cannot reuse cd_union precomputed predictions for target dataset '{ds_name}' "
                f"because it is not present in source_cd_union_datasets={list(source_cd_union_datasets)}."
            )
        a, b = offsets[ds_name]
        if b <= a:
            # missing for this split -> skip
            continue
        if b > pred_len:
            raise ValueError(
                f"Precomputed predictions are too short to slice dataset '{ds_name}' for split='{split}'. "
                f"Need slice [{a}:{b}) but pred_len={pred_len}. "
                f"This usually means the cached predictions were generated with a different cd_union split composition "
                f"(e.g., some datasets were skipped for this split), or precomputed_predictions_cd_union_datasets order doesn't match."
            )
        chunks.append(pred[a:b])

    if len(chunks) == 0:
        raise ValueError(
            f"After slicing, no samples remained for target_dataset_name='{target_name}' split='{split}'. "
            f"Check that the dataset exists for this split and that source_cd_union_datasets is correct."
        )
    return torch.cat(chunks, dim=0)


def _select_viz_indices(
    dataset,
    *,
    k: int,
    pixel_thr_01: float = 0.2,
    area_thr: float = 0.2,
    target_ratio: float = 0.5,
    fallback_target_ratio: float = 0.2,
    seed: int = 0,
) -> Tuple[list, list]:
    """
    Deterministically pick `k` indices with GT foreground ratios close to `target_ratio`,
    preferring those with ratio > area_thr. Same logic as train_remote_var.py.
    """
    n = len(dataset)
    if n == 0:
        return [], []

    ratios = []
    for i in range(n):
        r = None
        if hasattr(dataset, "foreground_ratio"):
            r = float(dataset.foreground_ratio(i, pixel_thr=0))
        else:
            sample = dataset[i]
            m = sample["mask"]
            m01 = (m + 1) / 2
            fg = (m01.max(dim=0).values > float(pixel_thr_01)).float().mean().item()
            r = float(fg)
        ratios.append(r)

    idxs = list(range(n))
    rnd = random.Random(int(seed))
    rnd.shuffle(idxs)

    above = [i for i in idxs if ratios[i] > float(area_thr)]
    above.sort(key=lambda i: abs(ratios[i] - float(target_ratio)))
    chosen = list(above[:k])

    if len(chosen) < k:
        remaining = [i for i in above if i not in set(chosen)]
        remaining.sort(key=lambda i: abs(ratios[i] - float(fallback_target_ratio)))
        need = k - len(chosen)
        chosen.extend(remaining[:need])

    if len(chosen) < k:
        remaining_all = [i for i in idxs if i not in set(chosen)]
        remaining_all.sort(key=lambda i: abs(ratios[i] - float(target_ratio)))
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
) -> list:
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


class _WithIdx(Dataset):
    """Wrap a dataset so each item dict includes a stable integer `idx` for joining with cached predictions."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        if isinstance(item, dict):
            out = dict(item)
            out["idx"] = int(idx)
            return out
        return item


def _load_state_dict(path: str) -> Dict[str, Any]:
    if path is None:
        raise ValueError("Checkpoint path is None.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if path.endswith(".safetensors"):
        obj = load_file(path, device="cpu")
    else:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict checkpoint at {path}, got {type(obj)}")
    return obj


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/decoder_refiner.yaml", help="YAML config file")

    # data
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=os.environ.get("DATASET_ROOT", "data"),
        help="Root folder containing change-detection datasets (default: $DATASET_ROOT or ./data).",
    )
    # Backward compatibility (deprecated): treat --data_dir as dataset_root if dataset_root is unset.
    parser.add_argument("--data_dir", type=str, default=None, help="DEPRECATED alias for --dataset_root")
    parser.add_argument("--dataset_name", type=str, default="cd_union")
    parser.add_argument("--cd_union_datasets", type=str, nargs="+", default=["whu_cd", "levircd", "levircdplus", "s2looking"])
    parser.add_argument(
        "--data_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Explicit dataset roots in cd_union order; overrides canonical dataset_root subfolders.",
    )
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=8)

    # optimization
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decay_end", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="cos", choices=["cos", "lin", "lin0", "lin00", "exp"])
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--wp0", type=float, default=0.005)
    parser.add_argument("--wpe", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16", "fp8"])

    # logging / io
    parser.add_argument("--output_dir", type=str, default="experiments")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100, help="Log train metrics every N optimizer steps.")

    # wandb (via Accelerate tracker)
    parser.add_argument("--use_wandb", default=True, action=argparse.BooleanOptionalAction, help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="RemoteVAR", help="W&B project name.")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["offline", "online"],
        help="W&B mode. Use 'offline' to avoid API key/login requirements in no-tty environments.",
    )

    # validation metrics + qualitative visualization
    parser.add_argument(
        "--compute_val_metrics",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, compute IoU / pixel-acc / precision / recall on the validation set.",
    )
    parser.add_argument(
        "--val_metrics_max_batches",
        type=int,
        default=-1,
        help="If >0, compute validation metrics only on the first N batches (useful for quick debugging).",
    )
    parser.add_argument("--val_pred_thr_01", type=float, default=0.1, help="Prediction threshold in [0,1] for binarization.")
    parser.add_argument("--val_gt_thr_01", type=float, default=0.1, help="GT threshold in [0,1] for binarization.")
    parser.add_argument(
        "--save_val_images",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, save a few validation comparison images per epoch.",
    )
    parser.add_argument("--val_num_save_samples", type=int, default=10, help="How many validation samples to visualize per epoch.")
    parser.add_argument("--val_viz_every", type=int, default=1, help="Save/log validation images every N epochs (1 = every epoch).")
    parser.add_argument("--viz_target_ratio", type=float, default=0.5, help="Target GT foreground ratio for fixed val visualization indices.")
    parser.add_argument("--viz_fallback_ratio", type=float, default=0.2, help="Fallback GT foreground ratio for fixed val visualization indices.")

    # VQVAE
    parser.add_argument("--vocab_size", type=int, nargs="+", default=4096)
    parser.add_argument("--z_channels", type=int, default=32)
    parser.add_argument("--ch", type=int, default=160)
    parser.add_argument("--vqvae_pretrained_path", type=str, default="pretrained/vae_ch160v4096z32.pth")
    parser.add_argument(
        "--train_post_quant_conv",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, also finetune VQVAE.post_quant_conv together with the conditioned decoder.",
    )

    # RemoteVAR (used ONLY as frozen fusion feature extractor)
    parser.add_argument("--remotevar_checkpoint", type=str, default=None, help="Pretrained RemoteVAR checkpoint")
    parser.add_argument("--v_patch_nums", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 13, 16])
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--mask_type", type=str, default="change_append", choices=["replace", "interleave_append", "change_append"])
    # These must match the RemoteVAR checkpoint architecture when using teacher-forcing forward.
    parser.add_argument("--bidirectional", default=False, action=argparse.BooleanOptionalAction, help="Shuffle mask and image order in each stage.")
    parser.add_argument("--separate_decoding", default=False, action=argparse.BooleanOptionalAction, help="Separate decode mask and image in each stage.")
    parser.add_argument("--separator", default=False, action=argparse.BooleanOptionalAction, help="Use special tokens as separator.")
    parser.add_argument("--type_pos", default=False, action=argparse.BooleanOptionalAction, help="Use type positional embedding.")
    parser.add_argument("--indep", default=False, action=argparse.BooleanOptionalAction, help="Independent separate decoding.")
    parser.add_argument("--multi_cond", default=True, action=argparse.BooleanOptionalAction, help="Multi-type conditions (required for change_append).")
    parser.add_argument(
        "--enable_current_scale_tokens",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Inject current-scale PRE/POST token embeddings so mask generation can attend to them at the same stage.",
    )
    parser.add_argument("--cond_drop_rate", type=float, default=0.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--cross_attn_inner_dim", type=int, default=1024)
    parser.add_argument("--use_high_res_context_levels", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--fusion_downsample_ratios", type=int, nargs="+", default=[1, 1, 1, 1])
    parser.add_argument("--fusion_num_heads", type=int, default=8)
    parser.add_argument("--fusion_num_layers", type=int, default=1)
    parser.add_argument("--fusion_cross_inner_dim", type=int, default=None)
    parser.add_argument("--fusion_use_feature_rectify", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--fusion_downsample_first", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--disable_cross_attention", action="store_true", default=False)

    # Optional: train fusion modules (and optionally a trainable encoder copy) jointly with the decoder refiner.
    parser.add_argument(
        "--train_fusion",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, finetune RemoteVAR fusion_modules (UNet-style skip extractor) jointly with the decoder.",
    )
    parser.add_argument(
        "--allow_trainable_encoder",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, create a trainable *copy* of the VQVAE encoder inside RemoteVAR for fusion-context extraction. "
            "Only used when --train_fusion is enabled. The original VQVAE stays frozen for tokenization."
        ),
    )
    parser.add_argument("--fusion_lr_scale", type=float, default=1.0, help="LR multiplier for fusion params when train_fusion=True.")
    parser.add_argument("--fusion_wd_scale", type=float, default=1.0, help="WD multiplier for fusion params when train_fusion=True.")

    # Extra (high-res) fusion modules for decoder skips ONLY (not part of pretrained RemoteVAR).
    # Example: [256, 128] will add two new fusion modules initialized from scratch.
    parser.add_argument(
        "--decoder_extra_fusion_resolutions",
        type=int,
        nargs="*",
        default=[],
        help="Extra encoder resolutions to fuse (e.g., 256 128) and pass into the conditioned decoder as additional skips.",
    )
    parser.add_argument("--decoder_extra_fusion_num_heads", type=int, default=8, help="Heads for extra fusion modules.")
    parser.add_argument("--decoder_extra_fusion_num_layers", type=int, default=1, help="CrossPath layers for extra fusion modules.")
    parser.add_argument("--decoder_extra_fusion_cross_inner_dim", type=int, default=None, help="Inner dim for extra fusion modules (optional).")
    parser.add_argument(
        "--decoder_extra_fusion_use_feature_rectify",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable FeatureRectifyModule inside extra fusion modules.",
    )
    parser.add_argument(
        "--decoder_extra_fusion_downsample_first",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If true, downsample inputs before token mixing inside extra fusion modules (not recommended when you want same-res skips).",
    )

    # cached predictions (mask f_hat) from pretrained RemoteVAR
    parser.add_argument(
        "--use_precomputed_predictions",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, load cached predicted mask f_hat tensors from predictions_dir.",
    )
    parser.add_argument("--predictions_dir", type=str, default="predictions")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--use_teacher_forcing_forward",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, compute mask_fhat on-the-fly using RemoteVAR teacher-forcing forward pass (single forward), "
            "instead of loading fixed cached predictions. This allows using geometric augmentations again."
        ),
    )
    parser.add_argument(
        "--val_use_precomputed_predictions",
        default=None,
        action=argparse.BooleanOptionalAction,
        help=(
            "Validation-only override for predicted mask_fhat source. "
            "If set: use cached predictions for val even when training uses --use_teacher_forcing_forward. "
            "If omitted: defaults to using cached predictions when --use_precomputed_predictions is true; "
            "and in teacher-forcing-forward mode, it will AUTO-use cached val predictions if the cache file exists."
        ),
    )
    parser.add_argument(
        "--noisy_tf_mask_prob",
        type=float,
        default=0.0,
        help="Optional: corrupt GT mask token IDs when building teacher-forcing inputs for later scales (mask stream only).",
    )
    parser.add_argument(
        "--noisy_tf_mask_mode",
        type=str,
        default="random",
        choices=["random", "shuffle"],
        help="Noisy teacher forcing mode for mask stream.",
    )

    # Precomputed prediction source override (advanced)
    parser.add_argument(
        "--precomputed_predictions_dataset_name",
        type=str,
        default=None,
        help=(
            "Override the dataset_name used to LOAD cached predictions from <predictions_dir>. "
            "Useful when training on a subset dataset but reusing a larger union cache (e.g., load cd_union caches while training whu_cd)."
        ),
    )
    parser.add_argument(
        "--precomputed_predictions_cd_union_datasets",
        type=str,
        nargs="*",
        default=None,
        help=(
            "If precomputed_predictions_dataset_name=cd_union, specify the cd_union_datasets ORDER that was used "
            "when generating the cached predictions. This lets us slice out per-dataset segments without regenerating caches."
        ),
    )
    parser.add_argument(
        "--precomputed_predictions_allow_subset_from_union",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, and cached prediction lengths don't match the current dataset, try to slice/reorder a cd_union cache "
            "to match the current dataset (e.g., reuse whu_cd segment from cd_union)."
        ),
    )

    # Decoder output channels
    # - 3: legacy RGB mask output (location-coded RGB)
    # - 1: binary change-mask logits (recommended for BCE-only training)
    parser.add_argument("--decoder_out_channels", type=int, default=3, choices=[1, 3])

    # Decoder-refiner scaling knobs (for the *new* skip-fusion modules inside ConditionedDecoder).
    # These do NOT change the pretrained VQVAE decoder architecture; they only add extra capacity
    # when fusing external skips.
    parser.add_argument(
        "--decoder_skip_fuse_extra_depth",
        type=int,
        default=0,
        help=(
            "Number of extra ResnetBlocks in an additional residual adapter per decoder level to fuse skips. "
            "0 disables (default / backward-compatible)."
        ),
    )
    parser.add_argument(
        "--decoder_skip_fuse_extra_width_mult",
        type=float,
        default=1.0,
        help=(
            "Width multiplier for the extra skip-fuse adapter hidden channels (relative to the decoder level channels). "
            "Hidden channels are rounded up to a multiple of 32 for GroupNorm stability."
        ),
    )

    # Optional: add a tiny conv head to map decoder RGB output -> 1-channel mask logits for BCE training.
    parser.add_argument(
        "--use_rgb_to_mask_head",
        default=False,
        action=argparse.BooleanOptionalAction,
        help=(
            "If true, append a 1x1 conv head after the decoder to produce 1-channel logits. "
            "This avoids max/argmax reductions on RGB and lets BCE gradients flow through all channels."
        ),
    )
    parser.add_argument(
        "--rgb_to_mask_head_init",
        type=str,
        default="sum_rgb",
        choices=["sum_rgb", "zero"],
        help="Initialization for the RGB->mask head. 'sum_rgb' sets weights to [1,1,1] and bias=0.",
    )

    # When using cached predictions, geometric augs must be disabled to keep alignment.
    # Photometric augs *can* be enabled safely (they don't change geometry), but for change detection
    # we strongly recommend paired color jitter (same jitter on pre/post) to avoid fake changes.
    parser.add_argument(
        "--allow_color_jitter_with_cached_predictions",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Allow ColorJitter even when use_precomputed_predictions=True (recommended only with color_jitter_pairwise=True).",
    )
    parser.add_argument(
        "--allow_gaussian_blur_with_cached_predictions",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Allow Gaussian blur even when use_precomputed_predictions=True (blur is applied equally to pre/post in our aug).",
    )
    parser.add_argument(
        "--color_jitter_pairwise",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="If true, apply the SAME sampled ColorJitter transform to both pre/post images (reduces spurious change cues).",
    )

    # NEW: cached-prediction augmentation (mask_fhat noise)
    # Motivation: cached predictions can be "too clean" vs actual AR inference latents.
    # Add Gaussian noise to pred_mask_fhat during TRAINING only to improve robustness.
    parser.add_argument(
        "--cached_pred_fhat_noise_strength",
        type=float,
        default=0.0,
        help=(
            "Std/scale of Gaussian noise added to cached pred_mask_fhat during training (0 disables). "
            "If mode='relative', this is a multiplier on per-sample latent std; "
            "if mode='absolute', this is the raw sigma in latent units."
        ),
    )
    parser.add_argument(
        "--cached_pred_fhat_noise_mode",
        type=str,
        default="relative",
        choices=["relative", "absolute"],
        help="How to interpret cached_pred_fhat_noise_strength: relative-to-sample-std or absolute sigma.",
    )
    parser.add_argument(
        "--cached_pred_fhat_noise_clip",
        type=float,
        default=3.0,
        help="If >0, clip the sampled noise to +/- clip * sigma (helps avoid rare extreme perturbations).",
    )

    # Training loss knobs (L2 RGB is default; add BCE/Dice to directly optimize fg IoU)
    parser.add_argument("--loss_l2_rgb_weight", type=float, default=1.0, help="Weight for L2 loss on RGB mask in [0,1].")
    parser.add_argument("--loss_bce_weight", type=float, default=0.0, help="Weight for BCE on grayscale mask (max-channel).")
    parser.add_argument("--loss_dice_weight", type=float, default=0.0, help="Weight for soft Dice loss on grayscale mask (max-channel).")
    parser.add_argument("--loss_focal_weight", type=float, default=0.0, help="Weight for sigmoid focal loss (binary foreground/background).")
    parser.add_argument("--loss_focal_gamma", type=float, default=2.0, help="Focal loss gamma (>=0).")
    parser.add_argument("--loss_focal_alpha", type=float, default=0.25, help="Focal loss alpha in [0,1] for positive class.")
    parser.add_argument(
        "--loss_gt_thr_01",
        type=float,
        default=0.1,
        help="GT binarization threshold in [0,1] for BCE/Dice losses (foreground if max-channel > thr).",
    )
    parser.add_argument(
        "--loss_bce_pos_weight",
        type=float,
        default=1.0,
        help="Positive-class weight for BCE (>=1 emphasizes foreground). Implemented as weighted BCE on probabilities.",
    )
    parser.add_argument(
        "--loss_bce_neg_weight",
        type=float,
        default=1.0,
        help="Negative-class weight for BCE (>=1 penalizes false positives more; can improve precision).",
    )
    parser.add_argument(
        "--loss_fp_weight",
        type=float,
        default=0.0,
        help="Extra penalty on predicted foreground probability over GT background: mean(pred_gray * (1-gt_bin)).",
    )
    parser.add_argument(
        "--loss_fg_reduce",
        type=str,
        default="max",
        choices=["max", "mean", "soft_or", "l2norm"],
        help="How to convert RGB mask (in [0,1]) to a 1-channel foreground probability for BCE/Dice/FP losses.",
    )
    parser.add_argument(
        "--loss_palette_ce_weight",
        type=float,
        default=0.0,
        help="Optional: palette cross-entropy against location-coded RGB mask classes (black + colormap).",
    )
    parser.add_argument(
        "--loss_palette_ce_temp",
        type=float,
        default=0.05,
        help="Temperature for palette CE soft assignment: logits = -||pred_rgb - color||^2 / temp.",
    )

    # First parse (so --config can override)
    args = parser.parse_args()

    # Load YAML config (if exists) and set defaults
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            yaml = YAML(typ="safe")
            cfg = yaml.load(f) or {}
        parser.set_defaults(**cfg)
        args = parser.parse_args()

    return args


def _weighted_bce_prob(
    pred01: torch.Tensor,
    tgt01: torch.Tensor,
    *,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Weighted BCE on probabilities in [0,1] (not logits). This avoids needing logits while still
    upweighting positives similarly to BCEWithLogitsLoss(pos_weight=...).
    """
    p = pred01.clamp(eps, 1.0 - eps)
    y = tgt01
    pw = float(pos_weight)
    nw = float(neg_weight)
    # -[ pw*y*log(p) + nw*(1-y)*log(1-p) ]
    return -(pw * y * torch.log(p) + nw * (1.0 - y) * torch.log(1.0 - p)).mean()


def _sigmoid_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: Optional[float] = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Binary focal loss computed from logits for numerical stability.

    FL = - alpha * y * (1-p)^gamma * log(p) - (1-alpha) * (1-y) * p^gamma * log(1-p)
    where p = sigmoid(logits).
    """
    p = torch.sigmoid(logits)
    y = targets.to(dtype=p.dtype)

    # Safe logs in mixed precision
    log_p = torch.log(p.clamp(eps, 1.0))
    log_1mp = torch.log((1.0 - p).clamp(eps, 1.0))

    g = float(gamma)
    if g < 0:
        raise ValueError(f"loss_focal_gamma must be >= 0, got {g}")

    # p_t from focal loss paper
    pt = y * p + (1.0 - y) * (1.0 - p)

    if alpha is None:
        alpha_t = 1.0
    else:
        a = float(alpha)
        if not (0.0 <= a <= 1.0):
            raise ValueError(f"loss_focal_alpha must be in [0,1], got {a}")
        alpha_t = y * a + (1.0 - y) * (1.0 - a)

    loss = -alpha_t * (1.0 - pt).pow(g) * (y * log_p + (1.0 - y) * log_1mp)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unknown focal reduction='{reduction}'. Use 'none', 'mean', or 'sum'.")


def _dice_loss(pred01: torch.Tensor, tgt01: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice loss for probabilities in [0,1] against binary targets {0,1}.
    """
    p = pred01
    y = tgt01
    # (B,1,H,W) -> (B, N)
    p = p.reshape(p.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    inter = (p * y).sum(dim=1)
    denom = p.sum(dim=1) + y.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def _rgb_to_fg_prob(pred01_rgb: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Convert predicted RGB mask in [0,1] to a single-channel foreground probability.

    Why this exists:
    - Using max() is a hard selection: per-pixel gradients flow only through the argmax channel.
    - Some alternatives (e.g., soft_or) let gradients flow through all channels, which can be
      noticeably better when your supervision is binary foreground/background.
    """
    m = str(mode or "max").lower()
    if m == "max":
        return pred01_rgb.max(dim=1, keepdim=True).values
    if m == "mean":
        return pred01_rgb.mean(dim=1, keepdim=True)
    if m == "soft_or":
        # p_fg = 1 - Π_c (1 - p_c)  (differentiable OR)
        return 1.0 - torch.prod(1.0 - pred01_rgb.clamp(0.0, 1.0), dim=1, keepdim=True)
    if m == "l2norm":
        # Normalize to [0,1] (approximately) by dividing by sqrt(C)
        c = max(1, int(pred01_rgb.shape[1]))
        return torch.linalg.norm(pred01_rgb, dim=1, keepdim=True) / float(math.sqrt(c))
    raise ValueError(f"Unknown loss_fg_reduce='{mode}'. Choose from: max, mean, soft_or, l2norm.")


def _maybe_add_cached_fhat_noise(
    pred_mask_fhat: torch.Tensor,
    *,
    strength: float,
    mode: str = "relative",
    clip: float = 3.0,
) -> torch.Tensor:
    """
    Add Gaussian noise to cached pred_mask_fhat (B,C,H,W).

    - mode='relative': sigma = strength * std(pred_mask_fhat) per-sample (over C,H,W)
    - mode='absolute': sigma = strength (same for all samples/channels)

    Variance is sigma^2.
    """
    s = float(strength or 0.0)
    if s <= 0:
        return pred_mask_fhat
    m = str(mode or "relative").lower()
    if m not in {"relative", "absolute"}:
        raise ValueError(f"Unknown cached_pred_fhat_noise_mode={mode}. Choose 'relative' or 'absolute'.")

    x = pred_mask_fhat
    # Make sure we're in floating point for noise.
    if not x.is_floating_point():
        x = x.float()

    if m == "absolute":
        sigma = torch.full((x.shape[0], 1, 1, 1), s, device=x.device, dtype=x.dtype)
    else:
        # Per-sample std over (C,H,W). Keep it stable when std is near zero.
        sigma = x.flatten(1).std(dim=1, keepdim=True).clamp_min(1e-6).view(-1, 1, 1, 1) * s

    eps = torch.randn_like(x)
    c = float(clip or 0.0)
    if c > 0:
        eps = eps.clamp(-c, c)
    return x + eps * sigma


def _build_mask_rgb_palette_and_pos_map(args) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build:
    - palette01: (K,3) float in [0,1], where palette01[0]=black and palette01[1:]=colormap colors.
    - pos_class_map: (H,W) long with values in [1..K-1] giving the expected foreground class id per pixel location.

    This matches `datasets/utils.py:binary_mask_to_rgb_by_location()` behavior.
    """
    if not bool(getattr(args, "mask_rgb_by_location", False)):
        raise ValueError("Palette CE requires mask_rgb_by_location=true (GT must be location-coded RGB).")

    img_sz = int(getattr(args, "image_size", 256))
    grid_size = getattr(args, "mask_rgb_grid_size", 11)
    index_mode = str(getattr(args, "mask_rgb_index_mode", "grid"))

    if isinstance(grid_size, (list, tuple)):
        gy, gx = int(grid_size[0]), int(grid_size[1])
    else:
        gy = gx = int(grid_size)
    gy = max(1, gy)
    gx = max(1, gx)

    # Determine channel levels exactly like ChangeDataset
    required = gy * gx
    levels_cfg = getattr(args, "mask_rgb_levels", None)
    if levels_cfg is None or levels_cfg == 0:
        levels = auto_color_levels_for_required_colors(required, drop_black=True)
    elif isinstance(levels_cfg, int):
        L = max(2, int(levels_cfg))
        import numpy as _np

        lv = _np.round(_np.linspace(0, 255, L)).astype(_np.int32)
        lv[0] = 0
        lv[-1] = 255
        levels = tuple(int(x) for x in _np.unique(lv).tolist())
    else:
        levels = tuple(int(x) for x in levels_cfg)

    cmap = create_entity_like_color_map(levels=levels, drop_black=True)  # (N,3) uint8
    if cmap.shape[0] < 1:
        raise ValueError("Invalid colormap (no colors).")

    import numpy as np

    palette_u8 = np.concatenate([np.zeros((1, 3), dtype=np.uint8), cmap], axis=0)  # (K,3)
    palette01 = torch.from_numpy(palette_u8).float().div_(255.0)  # (K,3)

    # Build per-pixel foreground class id map, matching idx = y_bin*gx + x_bin (or mul) and modulo cmap size.
    H = W = img_sz
    xs = torch.arange(W, dtype=torch.long)
    ys = torch.arange(H, dtype=torch.long)
    x_bin = (xs * gx) // max(1, W)
    y_bin = (ys * gy) // max(1, H)
    if index_mode == "mul":
        idx = x_bin[None, :] * y_bin[:, None]
    elif index_mode == "grid":
        idx = y_bin[:, None] * gx + x_bin[None, :]
    else:
        raise ValueError(f"Unknown mask_rgb_index_mode={index_mode}. Use 'grid' or 'mul'.")
    idx = idx % int(cmap.shape[0])
    pos_class_map = idx + 1  # background is 0

    return palette01, pos_class_map


class MaskFhatToRgbDecoder(nn.Module):
    """
    Tiny module to make multi-GPU (DDP/FSDP) safe.

    We avoid calling `vqvae.post_quant_conv` / `vqvae.decoder` as attributes on a DDP wrapper.
    Instead, we wrap the trainable pieces into a single Module with a clean `forward()`.
    """

    def __init__(
        self,
        post_quant_conv: nn.Module,
        decoder: nn.Module,
        *,
        use_rgb_to_mask_head: bool = False,
        rgb_to_mask_head_init: str = "sum_rgb",
    ):
        super().__init__()
        self.post_quant_conv = post_quant_conv
        self.decoder = decoder
        self.rgb_to_mask: Optional[nn.Module] = None

        if bool(use_rgb_to_mask_head):
            # Tiny head: 3 -> 1 logits. Use kernel=1 so it's purely per-pixel channel mixing.
            head = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=True)
            init = str(rgb_to_mask_head_init or "sum_rgb").lower()
            with torch.no_grad():
                if init == "sum_rgb":
                    # [1,1,1] so logits ~ sum(rgb). If decoder outputs ~[-1,1], background (~-1,-1,-1) -> -3 (sigmoid ~ 0.05).
                    head.weight.fill_(1.0)
                    if head.bias is not None:
                        head.bias.zero_()
                elif init == "zero":
                    head.weight.zero_()
                    if head.bias is not None:
                        head.bias.zero_()
                else:
                    raise ValueError(f"Unknown rgb_to_mask_head_init={rgb_to_mask_head_init}")
            self.rgb_to_mask = head

    def forward(self, pred_mask_fhat: torch.Tensor, decoder_skips: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        z = self.post_quant_conv(pred_mask_fhat)
        # ConditionedDecoder expects `skips=...` kwarg; legacy Decoder ignores it.
        return self.decoder(z, skips=decoder_skips)

    def forward_mask_logits(self, pred_mask_fhat: torch.Tensor, decoder_skips: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        if self.rgb_to_mask is None:
            raise RuntimeError("rgb_to_mask head is not enabled. Set --use_rgb_to_mask_head.")
        rgb = self.forward(pred_mask_fhat, decoder_skips)
        return self.rgb_to_mask(rgb)


class FusionSkipExtractor(nn.Module):
    """
    Wrap RemoteVAR skip extraction into a `forward()` so Accelerate/DDP can wrap it.

    IMPORTANT: We cannot call `encode_context_with_fusion_2d` on a DDP wrapper (method isn't exposed),
    so we encapsulate it here and always call this module in training.
    """

    def __init__(
        self,
        *,
        var_model: nn.Module,
        extra_resolutions: Sequence[int] = (),
        extra_num_heads: int = 8,
        extra_num_layers: int = 1,
        extra_cross_inner_dim: Optional[int] = None,
        extra_use_feature_rectify: bool = False,
        extra_downsample_first: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        self.var = var_model
        self.extra_resolutions = [int(x) for x in (extra_resolutions or [])]
        self.extra_num_heads = int(extra_num_heads)
        self.extra_num_layers = int(extra_num_layers)
        self.extra_cross_inner_dim = None if extra_cross_inner_dim is None else int(extra_cross_inner_dim)
        self.extra_use_feature_rectify = bool(extra_use_feature_rectify)
        self.extra_downsample_first = bool(extra_downsample_first)

        # Build extra fusion modules (BCHW -> BCHW) for the requested encoder resolutions.
        self.extra_fusion_modules = nn.ModuleList()
        self.extra_dims: list = []
        if len(self.extra_resolutions) > 0:
            # Infer per-resolution channel dims from the VQVAE encoder config.
            enc = self._get_encoder_for_dim_inference()
            base_hw = int(getattr(self.var, "image_size", 256))
            down_res = [base_hw // (2**i) for i in range(int(getattr(enc, "num_resolutions", 0)))]
            down_dims = [int(getattr(enc, "ch", 0) * m) for m in getattr(enc, "ch_mult", ())]
            res_to_dim = {int(r): int(d) for r, d in zip(down_res, down_dims)}

            for r in self.extra_resolutions:
                if r not in res_to_dim:
                    raise ValueError(
                        f"decoder_extra_fusion_resolutions includes {r}, but encoder down resolutions are {down_res} "
                        f"(image_size={base_hw}, num_resolutions={getattr(enc, 'num_resolutions', None)})."
                    )
                dim = int(res_to_dim[r])
                self.extra_dims.append(dim)
                self.extra_fusion_modules.append(
                    FeatureFusionModule(
                        dim=dim,
                        reduction=1,
                        num_heads=int(self.extra_num_heads),
                        num_groups=32,
                        downsample_ratio=1,  # keep SAME resolution for decoder concatenation
                        downsample_first=bool(self.extra_downsample_first),
                        num_cross_layers=int(self.extra_num_layers),
                        cross_inner_dim=self.extra_cross_inner_dim,
                        use_feature_rectify=bool(self.extra_use_feature_rectify),
                    )
                )

        # Expose the decoder-skip metadata to keep decoder + extractor consistent.
        base_res = list(getattr(self.var, "encoder_spatial_resolutions", []))
        base_dim = list(getattr(self.var, "context_dims_per_level", []))
        self.skip_base_resolutions = [*self.extra_resolutions, *base_res]
        self.skip_in_channels = [*self.extra_dims, *base_dim]

        # Avoid DDP unused-parameter issues when not training fusion.
        if not bool(trainable):
            for p in self.extra_fusion_modules.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        """
        Override default `.train()` propagation.

        Why:
        - `FusionSkipExtractor` contains the full `RemoteVAR` as a child module (`self.var`).
        - When `train_fusion=true`, we want to train ONLY:
            - var.fusion_modules_for_skips (decoder skips)
            - optional var.trainable_encoder (for skip features)
            - optional extra_fusion_modules (decoder-only high-res skips)
          while keeping the frozen RemoteVAR transformer blocks and the original context fusion (`var.fusion_modules`)
          in eval mode. Otherwise, dropout / drop-path / cond-drop in the frozen transformer can change `mask_fhat`
          generation and make teacher-forcing training unstable or mismatch inference.
        """
        self.training = mode

        # Extra decoder-only fusion modules (train only if they are trainable)
        for m in self.extra_fusion_modules:
            if any(p.requires_grad for p in m.parameters()):
                m.train(mode)
            else:
                m.eval()

        # Always keep the main RemoteVAR (transformer + original context fusion) in eval mode.
        if hasattr(self, "var") and isinstance(self.var, nn.Module):
            self.var.eval()

            # But if we have a separate skip-fusion copy, let it follow `mode`.
            fm_skips = getattr(self.var, "fusion_modules_for_skips", None)
            if fm_skips is not None and isinstance(fm_skips, nn.Module):
                if any(p.requires_grad for p in fm_skips.parameters()):
                    fm_skips.train(mode)
                else:
                    fm_skips.eval()

            # Optional trainable encoder copy (used for skip fusion features only)
            enc = getattr(self.var, "trainable_encoder", None)
            if enc is not None and isinstance(enc, nn.Module):
                if any(p.requires_grad for p in enc.parameters()):
                    enc.train(mode)
                else:
                    enc.eval()

        return self

    def forward(self, images_pre_fp32: torch.Tensor, images_post_fp32: torch.Tensor):
        # images_* must be float32 to avoid bf16 conv bias mismatches in the frozen VQVAE encoder.
        base_skips = self._encode_skips_2d(images_pre_fp32, images_post_fp32)

        if len(self.extra_fusion_modules) == 0:
            return base_skips

        pre_all, post_all, down_res = self._encode_all_levels(images_pre_fp32, images_post_fp32)
        res_to_idx = {int(r): int(i) for i, r in enumerate(down_res)}

        extra_skips = []
        for mi, r in enumerate(self.extra_resolutions):
            idx = res_to_idx[int(r)]
            pre_ctx = pre_all[idx]
            post_ctx = post_all[idx]
            # BLC -> BCHW
            B, L, C = pre_ctx.shape
            H = W = int(L ** 0.5)
            if H * W != int(L):
                raise ValueError(f"Encoder context for res={r} is not square: L={L} (H={H}, W={W}).")
            pre_2d = pre_ctx.transpose(1, 2).reshape(B, C, H, W)
            post_2d = post_ctx.transpose(1, 2).reshape(B, C, H, W)
            fused = self.extra_fusion_modules[mi](pre_2d, post_2d)
            extra_skips.append(fused)

        return extra_skips + list(base_skips)

    def _encode_skips_2d(self, images_pre_fp32: torch.Tensor, images_post_fp32: torch.Tensor):
        """
        Like RemoteVAR.encode_context_with_fusion_2d(), but allows using a separate trainable fusion copy
        for decoder skips (var.fusion_modules_for_skips) while keeping var.fusion_modules frozen for transformer context.
        """
        var = self.var
        if getattr(var, "disable_cross_attention", False):
            raise RuntimeError("FusionSkipExtractor requires fusion modules (disable_cross_attention=False).")
        vae = var.vae_proxy[0]

        # Choose fusion module set for SKIPS.
        fm = getattr(var, "fusion_modules_for_skips", None)
        if fm is None:
            fm = getattr(var, "fusion_modules", None)
        if fm is None or len(fm) == 0:
            raise RuntimeError("Fusion modules are not initialized for skip extraction.")

        # Encoder contexts (BLC) -> BCHW and fuse
        if getattr(var, "allow_trainable_encoder", False) and getattr(var, "trainable_encoder", None) is not None:
            pre_contexts = var.trainable_encoder.forward_context(images_pre_fp32, return_all_levels=var.use_high_res_context_levels)
            post_contexts = var.trainable_encoder.forward_context(images_post_fp32, return_all_levels=var.use_high_res_context_levels)
        else:
            with torch.no_grad():
                pre_contexts = vae.encoder.forward_context(images_pre_fp32, return_all_levels=var.use_high_res_context_levels)
                post_contexts = vae.encoder.forward_context(images_post_fp32, return_all_levels=var.use_high_res_context_levels)
        if len(pre_contexts) != len(fm):
            raise RuntimeError(f"Expected {len(fm)} context levels from encoder, got {len(pre_contexts)}.")

        fused_contexts_2d = []
        for i, (pre_ctx, post_ctx) in enumerate(zip(pre_contexts, post_contexts)):
            B, L, C = pre_ctx.shape
            H = W = int(L ** 0.5)
            pre_2d = pre_ctx.transpose(1, 2).reshape(B, C, H, W)
            post_2d = post_ctx.transpose(1, 2).reshape(B, C, H, W)
            fused = fm[i](pre_2d, post_2d)
            fused_contexts_2d.append(fused)
        return fused_contexts_2d

    def _get_encoder_for_dim_inference(self) -> nn.Module:
        # Prefer trainable encoder copy if present, else use frozen VQVAE encoder.
        if getattr(self.var, "trainable_encoder", None) is not None:
            return self.var.trainable_encoder
        vae = self.var.vae_proxy[0]
        return vae.encoder

    def _encode_all_levels(self, images_pre_fp32: torch.Tensor, images_post_fp32: torch.Tensor):
        """
        Return (pre_ctx_all, post_ctx_all, down_resolutions) for ALL down levels (no post-middle).
        We use only down levels for the extra high-res fusion modules.
        """
        enc = self._get_encoder_for_dim_inference()
        base_hw = int(images_pre_fp32.shape[-1])
        num_down = int(getattr(enc, "num_resolutions", 0))
        down_res = [base_hw // (2**i) for i in range(num_down)]  # e.g., [256,128,64,32,16]

        # Encoder contexts come as [down_levels..., post_middle]. We drop the final post-middle entry.
        if getattr(self.var, "trainable_encoder", None) is not None:
            pre_all = enc.forward_context(images_pre_fp32, return_all_levels=True)
            post_all = enc.forward_context(images_post_fp32, return_all_levels=True)
        else:
            with torch.no_grad():
                pre_all = enc.forward_context(images_pre_fp32, return_all_levels=True)
                post_all = enc.forward_context(images_post_fp32, return_all_levels=True)

        if len(pre_all) < len(down_res):
            raise RuntimeError(f"Expected at least {len(down_res)} encoder levels, got {len(pre_all)}")
        pre_all = list(pre_all[: len(down_res)])
        post_all = list(post_all[: len(down_res)])
        return pre_all, post_all, down_res


def main():
    args = parse_args()

    if args.disable_cross_attention:
        raise ValueError("This refiner requires fusion modules; do not set --disable_cross_attention.")
    if args.remotevar_checkpoint is None:
        raise ValueError("--remotevar_checkpoint is required (pretrained RemoteVAR).")

    # Create run name early so we can attach W&B tracker in Accelerator ctor.
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = args.run_name or f"decoder_refiner_fusion_skips__{args.dataset_name}"
    full_run_name = f"{ts}-{model_name}"

    log_with = None
    if bool(getattr(args, "use_wandb", False)):
        try:
            from utils.wandb import CustomWandbTracker

            log_with = CustomWandbTracker(
                full_run_name,
                project=str(getattr(args, "wandb_project", "RemoteVAR")),
                mode=str(getattr(args, "wandb_mode", "offline")),
            )
        except Exception as e:
            # Robust fallback: don't hard-fail training just because W&B isn't configured (e.g., no WANDB_API_KEY).
            # Users can set WANDB_API_KEY or pass --wandb_mode offline / --no-use_wandb.
            print(f"[DecoderRefiner] Warning: W&B init failed, disabling wandb logging. Error: {e}")
            log_with = None
            args.use_wandb = False

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Make each rank's RNG distinct but reproducible (matches train_remote_var.py behavior).
    set_seed(int(args.seed), device_specific=True)

    # Teacher-forcing forward mode computes predictions on-the-fly, so cached predictions MUST be disabled.
    use_tf_forward = bool(getattr(args, "use_teacher_forcing_forward", False))
    if use_tf_forward:
        if bool(getattr(args, "use_precomputed_predictions", False)) and accelerator.is_main_process:
            logger.info("[DecoderRefiner] --use_teacher_forcing_forward enabled: ignoring --use_precomputed_predictions and cached prediction files.")
        setattr(args, "use_precomputed_predictions", False)
        if str(getattr(args, "mask_type", "")) == "change_append" and (not bool(getattr(args, "multi_cond", True))):
            raise ValueError("mask_type='change_append' requires multi_cond=true (RemoteVAR uses cond_embed for this layout).")

    # If we rely on cached predicted f_hat, disable train-time random augmentations to keep alignment.
    if bool(getattr(args, "use_precomputed_predictions", False)):
        # Always disable geometric augs (they would desync cached f_hat vs GT/mask)
        for k in ["enable_random_crop", "enable_random_flip", "enable_random_rotation"]:
            setattr(args, k, False)

        # Photometric augs can be allowed explicitly (safe w.r.t. geometry).
        allow_cj = bool(getattr(args, "allow_color_jitter_with_cached_predictions", False))
        allow_blur = bool(getattr(args, "allow_gaussian_blur_with_cached_predictions", False))

        setattr(args, "enable_color_jitter", bool(getattr(args, "enable_color_jitter", False)) and allow_cj)
        setattr(args, "enable_gaussian_blur", bool(getattr(args, "enable_gaussian_blur", False)) and allow_blur)

        # For change detection, keep jitter paired by default to avoid introducing fake inter-image changes.
        if getattr(args, "enable_color_jitter", False):
            setattr(args, "color_jitter_pairwise", bool(getattr(args, "color_jitter_pairwise", True)))

    # Datasets
    base_train_dataset = create_dataset(args.dataset_name, args, split="train")
    base_val_dataset = create_dataset(args.dataset_name, args, split="val")
    train_dataset = _WithIdx(base_train_dataset)
    val_dataset = _WithIdx(base_val_dataset)

    # Synchronize all ranks after dataset creation (helps avoid deadlocks from rank-skew during init).
    try:
        if accelerator.num_processes > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass

    # Cached predictions (mask f_hat)
    pred_train = pred_val = None
    pred_dir = str(getattr(args, "predictions_dir", "predictions"))
    src_pred_name, src_pred_union_list = _resolve_precomputed_predictions_source(args)
    train_path = os.path.join(pred_dir, f"{src_pred_name}_train_mask_fhat.pt")
    val_path = os.path.join(pred_dir, f"{src_pred_name}_val_mask_fhat.pt")

    want_train_cached = bool(getattr(args, "use_precomputed_predictions", False))

    # Validation can optionally use cached predictions even when training uses teacher-forcing forward.
    vflag = getattr(args, "val_use_precomputed_predictions", None)
    if want_train_cached:
        want_val_cached = True
    elif vflag is None:
        # AUTO: in teacher-forcing-forward mode, prefer cached val preds if the cache file exists (for autoregressive realism).
        # Otherwise, follow train behavior.
        if bool(getattr(args, "use_teacher_forcing_forward", False)):
            want_val_cached = bool(os.path.exists(val_path))
        else:
            want_val_cached = bool(getattr(args, "use_precomputed_predictions", False))
    else:
        want_val_cached = bool(vflag)

    # Explicit request for cached val preds -> error if missing.
    if bool(vflag) and (not os.path.exists(val_path)):
        raise FileNotFoundError(
            f"--val_use_precomputed_predictions is set but val cache file not found: {val_path}. "
            "Generate it with generate_refiner_predictions.py (dataset_name should match the cache)."
        )
    if want_train_cached and (not os.path.exists(train_path)):
        raise FileNotFoundError(f"Train cache file not found: {train_path}")
    if want_val_cached and (not os.path.exists(val_path)):
        raise FileNotFoundError(f"Val cache file not found: {val_path}")

    if accelerator.is_main_process and (want_train_cached or want_val_cached):
        logger.info(
            f"[DecoderRefiner] Loading cached predictions:"
            f" train={'yes' if want_train_cached else 'no'} ({train_path}),"
            f" val={'yes' if want_val_cached else 'no'} ({val_path})"
        )

    if want_train_cached:
        train_obj = torch.load(train_path, map_location="cpu")
        pred_train = train_obj["mask_fhat"] if isinstance(train_obj, dict) and "mask_fhat" in train_obj else train_obj
    if want_val_cached:
        val_obj = torch.load(val_path, map_location="cpu")
        pred_val = val_obj["mask_fhat"] if isinstance(val_obj, dict) and "mask_fhat" in val_obj else val_obj

    # Optional: allow reusing a cd_union cache when training/validating on an individual dataset / subset union.
    allow_subset = bool(getattr(args, "precomputed_predictions_allow_subset_from_union", False))
    if allow_subset and str(src_pred_name) == "cd_union" and src_pred_union_list is None and (pred_train is not None or pred_val is not None):
        raise ValueError(
            "precomputed_predictions_allow_subset_from_union=true, but precomputed_predictions_cd_union_datasets is not set. "
            "Set it to the dataset order used when generating cd_union caches, e.g.:\n"
            "  precomputed_predictions_dataset_name: cd_union\n"
            "  precomputed_predictions_cd_union_datasets: [whu_cd, levircd, levircdplus, s2looking]\n"
        )

    if pred_train is not None and pred_train.shape[0] != len(train_dataset):
        if allow_subset and str(src_pred_name) == "cd_union":
            pred_train = _slice_precomputed_predictions_from_cd_union(
                pred_train,
                split="train",
                args=args,
                source_cd_union_datasets=src_pred_union_list,
                target_dataset_name=str(args.dataset_name),
                target_cd_union_datasets=getattr(args, "cd_union_datasets", None),
            )
    if pred_train is not None and pred_train.shape[0] != len(train_dataset):
        raise ValueError(
            f"Train pred len mismatch: {pred_train.shape[0]} vs {len(train_dataset)}. "
            f"This usually means your cached predictions were generated for a different dataset union/config."
        )

    if pred_val is not None and pred_val.shape[0] != len(val_dataset):
        if allow_subset and str(src_pred_name) == "cd_union":
            pred_val = _slice_precomputed_predictions_from_cd_union(
                pred_val,
                split="val",
                args=args,
                source_cd_union_datasets=src_pred_union_list,
                target_dataset_name=str(args.dataset_name),
                target_cd_union_datasets=getattr(args, "cd_union_datasets", None),
            )


    # Dataloaders
    dl_kwargs = dict(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    if int(getattr(args, "num_workers", 0)) > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)

    # Optional: precompute palette + per-pixel class map for palette CE loss.
    palette01 = None
    pos_class_map = None
    if float(getattr(args, "loss_palette_ce_weight", 0.0)) > 0:
        try:
            palette01, pos_class_map = _build_mask_rgb_palette_and_pos_map(args)
        except Exception as e:
            raise RuntimeError(f"Failed to build palette CE structures. Error: {e}")

    # Models
    vocab_size = _as_int(args.vocab_size)
    vqvae = VQVAE(
        vocab_size=vocab_size,
        z_channels=int(args.z_channels),
        ch=int(args.ch),
        test_mode=False,
        share_quant_resi=4,
        v_patch_nums=tuple(int(x) for x in args.v_patch_nums),
    )
    vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path, map_location="cpu"))
    # IMPORTANT: Some checkpoints may store bf16 weights. Normalize to fp32 to avoid conv dtype/bias mismatches under autocast.
    vqvae = vqvae.float()

    # Keep a frozen copy for visualization (pretrained/original decoder behavior).
    original_post_quant_conv = copy.deepcopy(vqvae.post_quant_conv).float().eval()
    original_decoder = copy.deepcopy(vqvae.decoder).float().eval()
    for p in original_post_quant_conv.parameters():
        p.requires_grad_(False)
    for p in original_decoder.parameters():
        p.requires_grad_(False)
    original_decoder_model = MaskFhatToRgbDecoder(post_quant_conv=original_post_quant_conv, decoder=original_decoder)

    # Build RemoteVAR with fusion modules; keep it frozen.
    var = build_remote_var(
        vae=vqvae,
        depth=int(args.depth),
        patch_nums=tuple(int(x) for x in args.v_patch_nums),
        mask_type=str(args.mask_type),
        cond_drop_rate=float(args.cond_drop_rate),
        bidirectional=bool(getattr(args, "bidirectional", False)),
        separate_decoding=bool(getattr(args, "separate_decoding", False)),
        separator=bool(getattr(args, "separator", False)),
        type_pos=bool(getattr(args, "type_pos", False)),
        indep=bool(getattr(args, "indep", False)),
        multi_cond=bool(getattr(args, "multi_cond", True)),
        drop_path_rate=float(args.drop_path_rate),
        disable_cross_attention=bool(getattr(args, "disable_cross_attention", False)),
        enable_current_scale_tokens=bool(getattr(args, "enable_current_scale_tokens", False)),
        cross_attn_inner_dim=int(getattr(args, "cross_attn_inner_dim", 1024)),
        image_size=int(args.image_size),
        use_high_res_context_levels=bool(getattr(args, "use_high_res_context_levels", False)),
        fusion_downsample_ratios=tuple(int(x) for x in getattr(args, "fusion_downsample_ratios", [1, 1, 1, 1])),
        fusion_num_heads=getattr(args, "fusion_num_heads", 8),
        fusion_num_layers=getattr(args, "fusion_num_layers", 1),
        fusion_cross_inner_dim=getattr(args, "fusion_cross_inner_dim", None),
        fusion_use_feature_rectify=getattr(args, "fusion_use_feature_rectify", False),
        fusion_downsample_first=getattr(args, "fusion_downsample_first", False),
        allow_trainable_encoder=bool(getattr(args, "allow_trainable_encoder", False)),
    )
    missing, unexpected = var.load_state_dict(_load_state_dict(args.remotevar_checkpoint), strict=False)
    if accelerator.is_main_process:
        logger.info(f"[DecoderRefiner] Loaded RemoteVAR checkpoint strict=False (missing={len(missing)} unexpected={len(unexpected)})")

    # Only fusion modules are used for skip extraction; keep them in fp32 for numerical stability + dtype safety.
    if hasattr(var, "fusion_modules") and var.fusion_modules is not None:
        var.fusion_modules = var.fusion_modules.float()

    # Trainability: by default, keep the whole RemoteVAR frozen. Optionally unfreeze fusion modules.
    train_fusion = bool(getattr(args, "train_fusion", False))
    if train_fusion and bool(getattr(args, "disable_cross_attention", False)):
        raise ValueError("--train_fusion requires fusion modules; do not set --disable_cross_attention.")
    for p in var.parameters():
        p.requires_grad_(False)
    # Keep the frozen transformer blocks deterministic (no dropout / drop-path).
    # We'll selectively put skip-fusion modules (and optional trainable encoder copy) back into train mode below.
    var.eval()
    if train_fusion:
        # IMPORTANT:
        # - Keep `var.fusion_modules` frozen as the ORIGINAL context provider for the (frozen) transformer.
        # - Train a SEPARATE copy `var.fusion_modules_for_skips` used ONLY for decoder skips.
        if getattr(var, "fusion_modules_for_skips", None) is None:
            var.fusion_modules_for_skips = copy.deepcopy(var.fusion_modules).float()
        for p in var.fusion_modules_for_skips.parameters():
            p.requires_grad_(True)
        # Optional trainable encoder copy (already constructed in RemoteVAR when allow_trainable_encoder=True)
        if bool(getattr(args, "allow_trainable_encoder", False)) and getattr(var, "trainable_encoder", None) is not None:
            for p in var.trainable_encoder.parameters():
                p.requires_grad_(True)
            # Keep unused latent head frozen (see RemoteVAR __init__ note).
            if hasattr(var.trainable_encoder, "norm_out") and var.trainable_encoder.norm_out is not None:
                for p in var.trainable_encoder.norm_out.parameters():
                    p.requires_grad_(False)
            if hasattr(var.trainable_encoder, "conv_out") and var.trainable_encoder.conv_out is not None:
                for p in var.trainable_encoder.conv_out.parameters():
                    p.requires_grad_(False)
        # Put skip-fusion modules in train mode (context fusion stays eval/frozen)
        if getattr(var, "fusion_modules_for_skips", None) is not None:
            var.fusion_modules_for_skips.train()
        if getattr(var, "trainable_encoder", None) is not None:
            var.trainable_encoder.train()
    # else: var stays eval()

    # Build skip extractor (includes optional extra high-res fusion modules for decoder skips).
    skip_extractor = FusionSkipExtractor(
        var_model=var,
        extra_resolutions=getattr(args, "decoder_extra_fusion_resolutions", []),
        extra_num_heads=int(getattr(args, "decoder_extra_fusion_num_heads", 8)),
        extra_num_layers=int(getattr(args, "decoder_extra_fusion_num_layers", 1)),
        extra_cross_inner_dim=getattr(args, "decoder_extra_fusion_cross_inner_dim", None),
        extra_use_feature_rectify=bool(getattr(args, "decoder_extra_fusion_use_feature_rectify", False)),
        extra_downsample_first=bool(getattr(args, "decoder_extra_fusion_downsample_first", False)),
        trainable=bool(train_fusion),
    )
    if len(getattr(args, "decoder_extra_fusion_resolutions", [])) > 0 and (not train_fusion) and accelerator.is_main_process:
        logger.warning(
            "decoder_extra_fusion_resolutions is set but train_fusion=False; extra fusion modules will stay frozen at random init."
        )

    # Replace VQVAE decoder with a skip-conditioned decoder, initialized from pretrained decoder weights.
    base_decoder_sd = vqvae.decoder.state_dict()
    out_ch = int(getattr(args, "decoder_out_channels", 3))
    if out_ch != 3 and bool(getattr(args, "use_rgb_to_mask_head", False)):
        raise ValueError("--use_rgb_to_mask_head requires decoder_out_channels=3 (head expects RGB input).")
    cond_dec = ConditionedDecoder(
        ch=int(args.ch),
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        dropout=args.dropout,
        in_channels=int(out_ch),
        z_channels=int(args.z_channels),
        using_sa=True,
        using_mid_sa=True,
        skip_base_resolutions=tuple(int(x) for x in getattr(skip_extractor, "skip_base_resolutions", getattr(var, "encoder_spatial_resolutions", [64, 32, 16, 16]))),
        skip_in_channels=tuple(int(x) for x in getattr(skip_extractor, "skip_in_channels", getattr(var, "context_dims_per_level", [320, 320, 640, 640]))),
        skip_fuse_extra_depth=int(getattr(args, "decoder_skip_fuse_extra_depth", 0)),
        skip_fuse_extra_width_mult=float(getattr(args, "decoder_skip_fuse_extra_width_mult", 1.0)),
    )
    # If switching to 1ch output, avoid conv_out shape mismatch and initialize it from pretrained RGB conv_out by summing weights.
    if int(out_ch) == 1 and ("conv_out.weight" in base_decoder_sd) and (base_decoder_sd["conv_out.weight"].shape[0] == 3):
        sd_wo_out = {k: v for k, v in base_decoder_sd.items() if not k.startswith("conv_out.")}
        cond_dec.load_state_dict(sd_wo_out, strict=False)
        with torch.no_grad():
            w = base_decoder_sd["conv_out.weight"]  # (3, Cin, 3,3)
            cond_dec.conv_out.weight.copy_(w.sum(dim=0, keepdim=True))
            if cond_dec.conv_out.bias is not None:
                b = base_decoder_sd.get("conv_out.bias", None)
                if b is not None:
                    cond_dec.conv_out.bias.copy_(b.sum().view(1))
                else:
                    cond_dec.conv_out.bias.zero_()
    else:
        cond_dec.load_state_dict(base_decoder_sd, strict=False)
    vqvae.decoder = cond_dec

    # Freeze everything except decoder (and optionally post_quant_conv).
    for p in vqvae.parameters():
        p.requires_grad_(False)
    for p in vqvae.decoder.parameters():
        p.requires_grad_(True)
    if bool(getattr(args, "train_post_quant_conv", False)):
        for p in vqvae.post_quant_conv.parameters():
            p.requires_grad_(True)

    # Build the trainable decoder module (DDP-safe forward).
    decoder_model = MaskFhatToRgbDecoder(
        post_quant_conv=vqvae.post_quant_conv,
        decoder=vqvae.decoder,
        use_rgb_to_mask_head=bool(getattr(args, "use_rgb_to_mask_head", False)),
        rgb_to_mask_head_init=str(getattr(args, "rgb_to_mask_head_init", "sum_rgb")),
    )

    # Optimizer + schedule (use repo helper to create proper WD/noWD param groups)
    _names, _paras, param_groups = filter_params(decoder_model, allow_frozen=True)
    if train_fusion:
        _, _, fusion_groups = filter_params(skip_extractor, allow_frozen=True)
        lr_sc = float(getattr(args, "fusion_lr_scale", 1.0))
        wd_sc = float(getattr(args, "fusion_wd_scale", 1.0))
        for g in fusion_groups:
            g["lr_sc"] = float(g.get("lr_sc", 1.0)) * lr_sc
            g["wd_sc"] = float(g.get("wd_sc", 1.0)) * wd_sc
        param_groups.extend(fusion_groups)
    optimizer = torch.optim.AdamW(param_groups, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    # Prepare trainable module + optimizer + loaders. Keep frozen VAR unfrozen/unwrapped to avoid DDP attr issues.
    decoder_model, skip_extractor, optimizer, train_loader, val_loader = accelerator.prepare(
        decoder_model, skip_extractor, optimizer, train_loader, val_loader
    )
    # Ensure the *frozen* VQVAE encoder used by RemoteVAR fusion lives on the right device (it is not registered inside var).
    try:
        # `vqvae.encoder` is referenced via RemoteVAR.vae_proxy (tuple, not registered), so move explicitly.
        enc = vqvae.encoder
        enc.to(accelerator.device)
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        # If we compute teacher-forcing predictions on-the-fly, we also need quant_conv + quantizer on device.
        if bool(getattr(args, "use_teacher_forcing_forward", False)):
            qc = vqvae.quant_conv
            qc.to(accelerator.device)
            qc.eval()
            for p in qc.parameters():
                p.requires_grad_(False)
            q = vqvae.quantize
            q.to(accelerator.device)
            q.eval()
            for p in q.parameters():
                p.requires_grad_(False)
    except Exception:
        pass

    # Start tracker (W&B) after accelerator exists (consistent with train_remote_var.py)
    if accelerator.is_main_process and bool(getattr(args, "use_wandb", False)):
        accelerator.init_trackers(full_run_name, config=vars(args))

    # Run dir
    if accelerator.is_main_process:
        run_dir = os.path.join(str(args.output_dir), full_run_name)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            import json

            json.dump(vars(args), f, indent=2)
    else:
        run_dir = None

    # Fixed validation visualization indices (same cache/heuristics as train_remote_var.py).
    # Only main process computes them; others wait.
    viz_val_idxs = None
    viz_val_batch = None
    if accelerator.is_main_process and bool(getattr(args, "save_val_images", True)):
        try:
            viz_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz_indices")
            k_viz = int(getattr(args, "val_num_save_samples", 4))
            k_viz = max(1, min(k_viz, len(base_val_dataset)))
            viz_val_idxs = _load_or_create_viz_indices(
                args=args,
                dataset=base_val_dataset,
                split_name="val",
                cache_dir=viz_cache_dir,
                k=k_viz,
                pixel_thr_01=0.2,
                area_thr=0.2,
                seed=int(getattr(args, "seed", 0)) + 1,
                target_ratio=float(getattr(args, "viz_target_ratio", 0.5)),
                fallback_target_ratio=float(getattr(args, "viz_fallback_ratio", 0.2)),
            )
            viz_loader = DataLoader(Subset(base_val_dataset, viz_val_idxs), batch_size=len(viz_val_idxs), shuffle=False, num_workers=0)
            viz_val_batch = next(iter(viz_loader))
            if accelerator.is_main_process:
                logger.info(f"[viz_indices] Fixed val indices: {viz_val_idxs}")
        except Exception as e:
            logger.warning(f"[viz_indices] Failed to build fixed val viz batch: {e}")
            viz_val_idxs = None
            viz_val_batch = None

    # Sync so all ranks start training together (especially if viz index selection was slow).
    try:
        if accelerator.num_processes > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception:
        pass

    # Train loop
    total_steps = int(args.num_epochs) * (len(train_loader) // max(1, int(args.gradient_accumulation_steps)))
    warmup_steps = max(1, int(float(args.wp0) * max(1, total_steps)))
    best_val = None
    global_step = 0

    progress = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(int(args.num_epochs)):
        decoder_model.train()
        if train_fusion:
            skip_extractor.train()
        else:
            skip_extractor.eval()
        epoch_loss = 0.0
        epoch_n = 0

        for batch in train_loader:
            with accelerator.accumulate(decoder_model, skip_extractor):
                images_pre = batch["images_pre"]
                images_post = batch["images_post"]
                gt_mask = batch["mask"]
                idx_B = batch.get("idx", None)
                conditions = batch.get("cls", None)
                cond_type = batch.get("type", None)

                # Compute fusion skips under no_grad (RemoteVAR is frozen).
                ipre = images_pre.to(dtype=torch.float32)
                ipost = images_post.to(dtype=torch.float32)
                if train_fusion:
                    decoder_skips = skip_extractor(ipre, ipost)
                else:
                    with torch.no_grad():
                        decoder_skips = skip_extractor(ipre, ipost)

                # Predicted mask f_hat: either cached (fixed) or computed on-the-fly via teacher-forcing forward.
                if bool(getattr(args, "use_teacher_forcing_forward", False)):
                    if conditions is None or cond_type is None:
                        raise KeyError("Batch missing 'cls'/'type' required for --use_teacher_forcing_forward.")
                    se = skip_extractor.module if hasattr(skip_extractor, "module") else skip_extractor
                    var_tf = getattr(se, "var", None)
                    if var_tf is None:
                        raise RuntimeError("Could not access RemoteVAR as skip_extractor.var for teacher-forcing forward.")
                    # IMPORTANT: keep cross-attention context computed with ORIGINAL (frozen) fusion modules
                    # so the frozen transformer does not see a drifting context when we train skip-fusion modules.
                    context = _encode_context_with_fusion_frozen(var_model=var_tf, images_pre=images_pre, images_post=images_post)
                    with accelerator.autocast():
                        pred_mask_fhat = _pred_mask_fhat_from_teacher_forcing_forward(
                            vqvae=vqvae,
                            var_model=var_tf,
                            images_pre=images_pre,
                            images_post=images_post,
                            gt_mask=gt_mask,
                            conditions=conditions,
                            cond_type=cond_type,
                            context=context,
                            v_patch_nums=tuple(int(x) for x in args.v_patch_nums),
                            mask_type=str(getattr(args, "mask_type", "change_append")),
                            noisy_tf_mask_prob=float(getattr(args, "noisy_tf_mask_prob", 0.0)),
                            noisy_tf_mask_mode=str(getattr(args, "noisy_tf_mask_mode", "random")),
                            enable_current_scale_tokens=bool(getattr(args, "enable_current_scale_tokens", False)),
                        )
                elif pred_train is not None:
                    if idx_B is None:
                        raise KeyError("Dataset batch missing 'idx' required for cached predictions.")
                    idx_cpu = idx_B.detach().to(device="cpu", dtype=torch.long)
                    pred_mask_fhat = pred_train[idx_cpu].to(device=images_pre.device)
                    # NEW: optional Gaussian noise augmentation on cached predictions (TRAIN ONLY).
                    pred_mask_fhat = _maybe_add_cached_fhat_noise(
                        pred_mask_fhat,
                        strength=float(getattr(args, "cached_pred_fhat_noise_strength", 0.0)),
                        mode=str(getattr(args, "cached_pred_fhat_noise_mode", "relative")),
                        clip=float(getattr(args, "cached_pred_fhat_noise_clip", 3.0)),
                    )
                else:
                    raise RuntimeError(
                        "No source of predicted mask_fhat available. "
                        "Set --use_precomputed_predictions to load cached predictions, "
                        "or set --use_teacher_forcing_forward to compute them on-the-fly."
                    )

                with accelerator.autocast():
                    pred_out = decoder_model(pred_mask_fhat, decoder_skips)
                    gt01 = _to_01(gt_mask.float())
                    out_ch = int(pred_out.shape[1])
                    use_head = bool(getattr(args, "use_rgb_to_mask_head", False))

                    loss = 0.0
                    w_l2 = float(getattr(args, "loss_l2_rgb_weight", 1.0))
                    w_bce = float(getattr(args, "loss_bce_weight", 0.0))
                    w_dice = float(getattr(args, "loss_dice_weight", 0.0))
                    w_focal = float(getattr(args, "loss_focal_weight", 0.0))
                    w_pce = float(getattr(args, "loss_palette_ce_weight", 0.0))

                    # 1ch mode: decoder outputs logits directly (recommended for BCE-only)
                    if out_ch == 1:
                        if w_l2 > 0 or w_pce > 0:
                            raise RuntimeError(
                                "decoder_out_channels=1 only supports BCE/Dice/Focal/FP losses "
                                "(set loss_l2_rgb_weight=0 and loss_palette_ce_weight=0)."
                            )
                        pred_logits = pred_out
                        pred_prob = torch.sigmoid(pred_logits.float())
                        gt_bin = (gt01.max(dim=1, keepdim=True).values > float(getattr(args, "loss_gt_thr_01", 0.1))).to(dtype=pred_prob.dtype)
                        # BCE on logits with per-pixel pos/neg weights
                        if (w_bce > 0) or (w_dice > 0) or (w_focal > 0):
                            bce_map = F.binary_cross_entropy_with_logits(pred_logits.float(), gt_bin, reduction="none")
                            pw = float(getattr(args, "loss_bce_pos_weight", 1.0))
                            nw = float(getattr(args, "loss_bce_neg_weight", 1.0))
                            weight_map = gt_bin * pw + (1.0 - gt_bin) * nw
                            if w_bce > 0:
                                loss = loss + w_bce * (bce_map * weight_map).mean()
                            if w_focal > 0:
                                loss = loss + w_focal * _sigmoid_focal_loss_with_logits(
                                    pred_logits.float(),
                                    gt_bin,
                                    alpha=float(getattr(args, "loss_focal_alpha", 0.25)),
                                    gamma=float(getattr(args, "loss_focal_gamma", 2.0)),
                                    reduction="mean",
                                )
                            if w_dice > 0:
                                loss = loss + w_dice * _dice_loss(pred_prob, gt_bin)
                            w_fp = float(getattr(args, "loss_fp_weight", 0.0))
                            if w_fp > 0:
                                loss = loss + w_fp * (pred_prob * (1.0 - gt_bin)).mean()
                        # Skip the RGB/palette losses
                    else:
                        pred_rgb = pred_out
                        pred01 = _to_01(pred_rgb.float())
                        if w_l2 > 0:
                            loss = loss + w_l2 * F.mse_loss(pred01, gt01)

                    if (w_bce > 0) or (w_dice > 0) or (w_focal > 0):
                        if out_ch != 1:
                            gt_bin = (gt01.max(dim=1, keepdim=True).values > float(getattr(args, "loss_gt_thr_01", 0.1))).to(dtype=pred01.dtype)
                            if use_head:
                                # BCE on logits (preferred): gradients flow through ALL RGB channels via the 1x1 conv head.
                                pred_logits = decoder_model.forward_mask_logits(pred_mask_fhat, decoder_skips)
                                pred_prob = torch.sigmoid(pred_logits.float())
                                bce_map = F.binary_cross_entropy_with_logits(pred_logits.float(), gt_bin, reduction="none")
                                pw = float(getattr(args, "loss_bce_pos_weight", 1.0))
                                nw = float(getattr(args, "loss_bce_neg_weight", 1.0))
                                weight_map = gt_bin * pw + (1.0 - gt_bin) * nw
                                if w_bce > 0:
                                    loss = loss + w_bce * (bce_map * weight_map).mean()
                                if w_focal > 0:
                                    loss = loss + w_focal * _sigmoid_focal_loss_with_logits(
                                        pred_logits.float(),
                                        gt_bin,
                                        alpha=float(getattr(args, "loss_focal_alpha", 0.25)),
                                        gamma=float(getattr(args, "loss_focal_gamma", 2.0)),
                                        reduction="mean",
                                    )
                                if w_dice > 0:
                                    loss = loss + w_dice * _dice_loss(pred_prob, gt_bin)
                                w_fp = float(getattr(args, "loss_fp_weight", 0.0))
                                if w_fp > 0:
                                    loss = loss + w_fp * (pred_prob * (1.0 - gt_bin)).mean()
                            else:
                                pred_gray = _rgb_to_fg_prob(pred01, getattr(args, "loss_fg_reduce", "max"))  # (B,1,H,W) in [0,1]
                                gt_bin = gt_bin.to(dtype=pred_gray.dtype)
                                if w_bce > 0:
                                    loss = loss + w_bce * _weighted_bce_prob(
                                        pred_gray,
                                        gt_bin,
                                        pos_weight=float(getattr(args, "loss_bce_pos_weight", 1.0)),
                                        neg_weight=float(getattr(args, "loss_bce_neg_weight", 1.0)),
                                    )
                                if w_focal > 0:
                                    pred_logits = torch.logit(pred_gray.clamp(1e-6, 1.0 - 1e-6))
                                    loss = loss + w_focal * _sigmoid_focal_loss_with_logits(
                                        pred_logits,
                                        gt_bin,
                                        alpha=float(getattr(args, "loss_focal_alpha", 0.25)),
                                        gamma=float(getattr(args, "loss_focal_gamma", 2.0)),
                                        reduction="mean",
                                    )
                                if w_dice > 0:
                                    loss = loss + w_dice * _dice_loss(pred_gray, gt_bin)
                                w_fp = float(getattr(args, "loss_fp_weight", 0.0))
                                if w_fp > 0:
                                    loss = loss + w_fp * (pred_gray * (1.0 - gt_bin)).mean()

                    # Optional: palette CE to encourage discrete RGB outputs (black + location-coded colormap).
                    # This avoids hard argmax/max gating and can strongly suppress "almost-black" background noise.
                    if w_pce > 0:
                        if out_ch == 1:
                            raise RuntimeError("loss_palette_ce_weight is only supported for decoder_out_channels=3.")
                        if palette01 is None or pos_class_map is None:
                            raise RuntimeError("loss_palette_ce_weight>0 but palette01/pos_class_map not initialized.")
                        B, _C, H, W = pred01.shape
                        pal = palette01.to(device=pred01.device, dtype=pred01.dtype)  # (K,3)
                        K = int(pal.shape[0])
                        # Distances: (B,K,H,W) = sum_c (pred - pal_k)^2
                        dist = (pred01[:, None, :, :, :] - pal[None, :, :, None, None]).pow(2).sum(dim=2)
                        temp = float(getattr(args, "loss_palette_ce_temp", 0.05))
                        temp = max(temp, 1e-6)
                        logits = -dist / temp

                        # Targets: background=0, foreground=pos_class_map (1..K-1)
                        gt_non_black = gt01.max(dim=1).values > float(getattr(args, "loss_gt_thr_01", 0.1))  # (B,H,W) bool
                        pos_map = pos_class_map.to(device=pred01.device).long()  # (H,W)
                        pos_map = pos_map.unsqueeze(0).expand(B, H, W)
                        target = torch.where(gt_non_black, pos_map, torch.zeros_like(pos_map))
                        # Safety: clamp just in case
                        target = target.clamp_(0, K - 1)
                        loss = loss + w_pce * F.cross_entropy(logits, target)

                accelerator.backward(loss)
                # Always accumulate epoch stats (per microbatch)
                epoch_loss += float(loss.detach().item()) * int(images_pre.shape[0])
                epoch_n += int(images_pre.shape[0])

                # Only step optimizer when gradients are synchronized (i.e., on the last accumulation step)
                if accelerator.sync_gradients:
                    if float(args.clip) > 0:
                        accelerator.clip_grad_norm_(decoder_model.parameters(), float(args.clip))
                        if train_fusion:
                            accelerator.clip_grad_norm_(skip_extractor.parameters(), float(args.clip))

                    lr_wd_annealing(
                        str(args.lr_scheduler),
                        optimizer,
                        peak_lr=float(args.learning_rate),
                        wd=float(args.weight_decay),
                        wd_end=float(args.weight_decay_end),
                        cur_it=int(global_step),
                        wp_it=int(warmup_steps),
                        max_it=int(total_steps),
                        wp0=float(args.wp0),
                        wpe=float(args.wpe),
                        min_lr=float(getattr(args, "min_lr", 0.0)),
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    progress.update(1)
                    global_step += 1

                    if accelerator.is_main_process and (global_step % int(getattr(args, "log_interval", 100)) == 0):
                        try:
                            lr = float(optimizer.param_groups[0].get("lr", 0.0))
                        except Exception:
                            lr = 0.0
                        accelerator.log(
                            {
                                "train/loss_total": float(loss.detach().item()),
                                "train/lr": lr,
                                "train/epoch": int(epoch),
                            },
                            step=int(global_step),
                        )

        # Validation
        decoder_model.eval()
        skip_extractor.eval()
        val_loss_sum = 0.0
        val_n = 0

        do_metrics = bool(getattr(args, "compute_val_metrics", True))
        hist_total = torch.zeros((2, 2), dtype=torch.float32, device=accelerator.device)
        labeled_total = torch.zeros((), dtype=torch.float32, device=accelerator.device)
        correct_total = torch.zeros((), dtype=torch.float32, device=accelerator.device)

        save_viz = (
            bool(getattr(args, "save_val_images", True))
            and (viz_val_batch is not None)
            and (viz_val_idxs is not None)
            and int(getattr(args, "val_viz_every", 1)) > 0
            and (epoch % int(getattr(args, "val_viz_every", 1)) == 0)
        )
        viz_dir = None
        if accelerator.is_main_process and save_viz and run_dir is not None:
            viz_dir = os.path.join(run_dir, "val_viz", f"epoch_{epoch:04d}")
            os.makedirs(viz_dir, exist_ok=True)

        # Lazy imports for visualization/W&B images
        create_comparison_image = None
        wandb = None
        if accelerator.is_main_process and save_viz:
            try:
                from inference import create_comparison_image as _cci

                create_comparison_image = _cci
            except Exception:
                create_comparison_image = None
            if bool(getattr(args, "use_wandb", False)):
                try:
                    import wandb as _wandb

                    wandb = _wandb
                except Exception:
                    wandb = None

        with torch.no_grad():
            val_batch_idx = 0
            max_batches = int(getattr(args, "val_metrics_max_batches", -1))
            for batch in val_loader:
                images_pre = batch["images_pre"]
                images_post = batch["images_post"]
                gt_mask = batch["mask"]
                idx_B = batch.get("idx", None)
                conditions = batch.get("cls", None)
                cond_type = batch.get("type", None)

                # Keep skip extraction in fp32 (avoid bf16 conv bias mismatches in frozen encoder).
                ipre = images_pre.to(dtype=torch.float32)
                ipost = images_post.to(dtype=torch.float32)
                decoder_skips = skip_extractor(ipre, ipost)

                # Predicted mask f_hat for validation:
                # Prefer cached (autoregressive) predictions if available; otherwise fall back to teacher-forcing forward.
                if pred_val is not None:
                    if idx_B is None:
                        raise KeyError("Dataset batch missing 'idx' required for cached predictions.")
                    idx_cpu = idx_B.detach().to(device="cpu", dtype=torch.long)
                    pred_mask_fhat = pred_val[idx_cpu].to(device=images_pre.device)
                elif bool(getattr(args, "use_teacher_forcing_forward", False)):
                    if conditions is None or cond_type is None:
                        raise KeyError("Batch missing 'cls'/'type' required for --use_teacher_forcing_forward.")
                    se = skip_extractor.module if hasattr(skip_extractor, "module") else skip_extractor
                    var_tf = getattr(se, "var", None)
                    if var_tf is None:
                        raise RuntimeError("Could not access RemoteVAR as skip_extractor.var for teacher-forcing forward.")
                    context = _encode_context_with_fusion_frozen(var_model=var_tf, images_pre=images_pre, images_post=images_post)
                    with accelerator.autocast():
                        pred_mask_fhat = _pred_mask_fhat_from_teacher_forcing_forward(
                            vqvae=vqvae,
                            var_model=var_tf,
                            images_pre=images_pre,
                            images_post=images_post,
                            gt_mask=gt_mask,
                            conditions=conditions,
                            cond_type=cond_type,
                            context=context,
                            v_patch_nums=tuple(int(x) for x in args.v_patch_nums),
                            mask_type=str(getattr(args, "mask_type", "change_append")),
                            noisy_tf_mask_prob=float(getattr(args, "noisy_tf_mask_prob", 0.0)),
                            noisy_tf_mask_mode=str(getattr(args, "noisy_tf_mask_mode", "random")),
                            enable_current_scale_tokens=bool(getattr(args, "enable_current_scale_tokens", False)),
                        )
                else:
                    raise RuntimeError(
                        "Validation has no source of predicted mask_fhat available. "
                        "Set --use_precomputed_predictions or --use_teacher_forcing_forward."
                    )

                with accelerator.autocast():
                    pred_out = decoder_model(pred_mask_fhat, decoder_skips)
                    gt01 = _to_01(gt_mask.float())
                    out_ch = int(pred_out.shape[1])
                    use_head = bool(getattr(args, "use_rgb_to_mask_head", False))

                    loss_v = 0.0
                    w_l2 = float(getattr(args, "loss_l2_rgb_weight", 1.0))
                    w_bce = float(getattr(args, "loss_bce_weight", 0.0))
                    w_dice = float(getattr(args, "loss_dice_weight", 0.0))
                    w_focal = float(getattr(args, "loss_focal_weight", 0.0))
                    w_pce = float(getattr(args, "loss_palette_ce_weight", 0.0))

                    if out_ch == 1:
                        if w_l2 > 0 or w_pce > 0:
                            raise RuntimeError(
                                "decoder_out_channels=1 only supports BCE/Dice/Focal/FP losses "
                                "(set loss_l2_rgb_weight=0 and loss_palette_ce_weight=0)."
                            )
                        pred_logits = pred_out
                        pred_prob = torch.sigmoid(pred_logits.float())
                        gt_bin = (gt01.max(dim=1, keepdim=True).values > float(getattr(args, "loss_gt_thr_01", 0.1))).to(dtype=pred_prob.dtype)
                        if (w_bce > 0) or (w_dice > 0) or (w_focal > 0):
                            bce_map = F.binary_cross_entropy_with_logits(pred_logits.float(), gt_bin, reduction="none")
                            pw = float(getattr(args, "loss_bce_pos_weight", 1.0))
                            nw = float(getattr(args, "loss_bce_neg_weight", 1.0))
                            weight_map = gt_bin * pw + (1.0 - gt_bin) * nw
                            if w_bce > 0:
                                loss_v = loss_v + w_bce * (bce_map * weight_map).mean()
                            if w_focal > 0:
                                loss_v = loss_v + w_focal * _sigmoid_focal_loss_with_logits(
                                    pred_logits.float(),
                                    gt_bin,
                                    alpha=float(getattr(args, "loss_focal_alpha", 0.25)),
                                    gamma=float(getattr(args, "loss_focal_gamma", 2.0)),
                                    reduction="mean",
                                )
                            if w_dice > 0:
                                loss_v = loss_v + w_dice * _dice_loss(pred_prob, gt_bin)
                            w_fp = float(getattr(args, "loss_fp_weight", 0.0))
                            if w_fp > 0:
                                loss_v = loss_v + w_fp * (pred_prob * (1.0 - gt_bin)).mean()
                    else:
                        pred_rgb = pred_out
                        pred01 = _to_01(pred_rgb.float())
                        if w_l2 > 0:
                            loss_v = loss_v + w_l2 * F.mse_loss(pred01, gt01)

                        if (w_bce > 0) or (w_dice > 0) or (w_focal > 0):
                            gt_bin = (gt01.max(dim=1, keepdim=True).values > float(getattr(args, "loss_gt_thr_01", 0.1))).to(dtype=pred01.dtype)
                            if use_head:
                                pred_logits = decoder_model.forward_mask_logits(pred_mask_fhat, decoder_skips)
                                pred_prob = torch.sigmoid(pred_logits.float())
                                bce_map = F.binary_cross_entropy_with_logits(pred_logits.float(), gt_bin, reduction="none")
                                pw = float(getattr(args, "loss_bce_pos_weight", 1.0))
                                nw = float(getattr(args, "loss_bce_neg_weight", 1.0))
                                weight_map = gt_bin * pw + (1.0 - gt_bin) * nw
                                if w_bce > 0:
                                    loss_v = loss_v + w_bce * (bce_map * weight_map).mean()
                                if w_focal > 0:
                                    loss_v = loss_v + w_focal * _sigmoid_focal_loss_with_logits(
                                        pred_logits.float(),
                                        gt_bin,
                                        alpha=float(getattr(args, "loss_focal_alpha", 0.25)),
                                        gamma=float(getattr(args, "loss_focal_gamma", 2.0)),
                                        reduction="mean",
                                    )
                                if w_dice > 0:
                                    loss_v = loss_v + w_dice * _dice_loss(pred_prob, gt_bin)
                                w_fp = float(getattr(args, "loss_fp_weight", 0.0))
                                if w_fp > 0:
                                    loss_v = loss_v + w_fp * (pred_prob * (1.0 - gt_bin)).mean()
                            else:
                                pred_gray = _rgb_to_fg_prob(pred01, getattr(args, "loss_fg_reduce", "max"))
                                gt_bin = gt_bin.to(dtype=pred_gray.dtype)
                                if w_bce > 0:
                                    loss_v = loss_v + w_bce * _weighted_bce_prob(
                                        pred_gray,
                                        gt_bin,
                                        pos_weight=float(getattr(args, "loss_bce_pos_weight", 1.0)),
                                        neg_weight=float(getattr(args, "loss_bce_neg_weight", 1.0)),
                                    )
                                if w_focal > 0:
                                    pred_logits = torch.logit(pred_gray.clamp(1e-6, 1.0 - 1e-6))
                                    loss_v = loss_v + w_focal * _sigmoid_focal_loss_with_logits(
                                        pred_logits,
                                        gt_bin,
                                        alpha=float(getattr(args, "loss_focal_alpha", 0.25)),
                                        gamma=float(getattr(args, "loss_focal_gamma", 2.0)),
                                        reduction="mean",
                                    )
                                if w_dice > 0:
                                    loss_v = loss_v + w_dice * _dice_loss(pred_gray, gt_bin)
                                w_fp = float(getattr(args, "loss_fp_weight", 0.0))
                                if w_fp > 0:
                                    loss_v = loss_v + w_fp * (pred_gray * (1.0 - gt_bin)).mean()

                        if w_pce > 0:
                            if palette01 is None or pos_class_map is None:
                                raise RuntimeError("loss_palette_ce_weight>0 but palette01/pos_class_map not initialized.")
                            B, _C, H, W = pred01.shape
                            pal = palette01.to(device=pred01.device, dtype=pred01.dtype)  # (K,3)
                            K = int(pal.shape[0])
                            dist = (pred01[:, None, :, :, :] - pal[None, :, :, None, None]).pow(2).sum(dim=2)
                            temp = float(getattr(args, "loss_palette_ce_temp", 0.05))
                            temp = max(temp, 1e-6)
                            logits_p = -dist / temp
                            gt_non_black = gt01.max(dim=1).values > float(getattr(args, "loss_gt_thr_01", 0.1))
                            pos_map = pos_class_map.to(device=pred01.device).long().unsqueeze(0).expand(B, H, W)
                            target = torch.where(gt_non_black, pos_map, torch.zeros_like(pos_map)).clamp_(0, K - 1)
                            loss_v = loss_v + w_pce * F.cross_entropy(logits_p, target)
                val_loss_sum += float(loss_v.detach().item()) * int(images_pre.shape[0])
                val_n += int(images_pre.shape[0])

                # Metrics
                if do_metrics:
                    # For 1ch logits output: compute metrics on sigmoid probabilities.
                    if int(pred_out.shape[1]) == 1:
                        pred_prob01 = torch.sigmoid(pred_out.float())
                        pred_for_metrics = pred_prob01.repeat(1, 3, 1, 1)
                    else:
                        pred_for_metrics = pred_out.float()
                    h, lab, cor = confusion_from_pred_and_gt(
                        pred_images=pred_for_metrics,
                        gt_masks=gt_mask.float(),
                        image_size=int(args.image_size),
                        pred_thr_01=float(getattr(args, "val_pred_thr_01", 0.1)),
                        gt_thr_01=float(getattr(args, "val_gt_thr_01", 0.1)),
                    )
                    hist_total += h.to(device=hist_total.device)
                    labeled_total += lab.to(device=labeled_total.device)
                    correct_total += cor.to(device=correct_total.device)

                val_batch_idx += 1
                if max_batches > 0 and val_batch_idx >= max_batches:
                    break

        # Fixed-index visualization (main process only): compare ORIGINAL vs FINETUNED decoder on the same samples.
        if accelerator.is_main_process and save_viz and (create_comparison_image is not None) and (viz_dir is not None):
            try:
                vb = viz_val_batch
                pre = vb["images_pre"].to(device=accelerator.device)
                post = vb["images_post"].to(device=accelerator.device)
                gt = vb["mask"].to(device=accelerator.device)
                vb_cls = vb.get("cls", None)
                vb_type = vb.get("type", None)
                idxs = [int(i) for i in viz_val_idxs]

                ipre = pre.to(dtype=torch.float32)
                ipost = post.to(dtype=torch.float32)
                with torch.no_grad():
                    decoder_skips = skip_extractor(ipre, ipost)

                # Determine predicted mask_fhat for visualization samples.
                # Prefer cached (autoregressive) predictions if available; otherwise fall back to teacher-forcing forward.
                if pred_val is not None:
                    pred_mask_fhat = pred_val[torch.tensor(idxs, dtype=torch.long, device="cpu")].to(
                        device=accelerator.device, dtype=torch.float32
                    )
                elif bool(getattr(args, "use_teacher_forcing_forward", False)):
                    if vb_cls is None or vb_type is None:
                        raise KeyError("viz_val_batch missing 'cls'/'type' required for --use_teacher_forcing_forward.")
                    conds = vb_cls.to(device=accelerator.device)
                    ctypes = vb_type.to(device=accelerator.device)
                    se = skip_extractor.module if hasattr(skip_extractor, "module") else skip_extractor
                    var_tf = getattr(se, "var", None)
                    if var_tf is None:
                        raise RuntimeError("Could not access RemoteVAR as skip_extractor.var for teacher-forcing forward.")
                    context = _encode_context_with_fusion_frozen(var_model=var_tf, images_pre=pre, images_post=post)
                    with accelerator.autocast():
                        pred_mask_fhat = _pred_mask_fhat_from_teacher_forcing_forward(
                            vqvae=vqvae,
                            var_model=var_tf,
                            images_pre=pre,
                            images_post=post,
                            gt_mask=gt,
                            conditions=conds,
                            cond_type=ctypes,
                            context=context,
                            v_patch_nums=tuple(int(x) for x in args.v_patch_nums),
                            mask_type=str(getattr(args, "mask_type", "change_append")),
                            noisy_tf_mask_prob=float(getattr(args, "noisy_tf_mask_prob", 0.0)),
                            noisy_tf_mask_mode=str(getattr(args, "noisy_tf_mask_mode", "random")),
                            enable_current_scale_tokens=bool(getattr(args, "enable_current_scale_tokens", False)),
                        )
                else:
                    raise RuntimeError("save_val_images requires cached predictions or --use_teacher_forcing_forward.")

                with accelerator.autocast():
                    pred_ft_out = decoder_model(pred_mask_fhat, decoder_skips).float()

                # Baseline/original decoder (ignores skips internally)
                original_decoder_model.to(accelerator.device)
                with torch.no_grad():
                    pred_orig_rgb = original_decoder_model(pred_mask_fhat, decoder_skips).float()

                # If finetuned decoder outputs 1ch logits, visualize probability mask; otherwise visualize RGB directly.
                if int(pred_ft_out.shape[1]) == 1:
                    ft_prob = torch.sigmoid(pred_ft_out).clamp(0.0, 1.0)
                    pred_ft = (ft_prob * 2.0 - 1.0).repeat(1, 3, 1, 1)
                else:
                    pred_ft = pred_ft_out
                pred_orig = pred_orig_rgb

                k = int(pre.shape[0])
                im_orig = create_comparison_image(
                    pre.detach().cpu().float(),
                    post.detach().cpu().float(),
                    pred_orig.detach().cpu().float(),
                    gt.detach().cpu().float(),
                    k,
                    int(args.image_size),
                    confidence_maps=None,
                    samples_per_row=1,
                )
                im_ft = create_comparison_image(
                    pre.detach().cpu().float(),
                    post.detach().cpu().float(),
                    pred_ft.detach().cpu().float(),
                    gt.detach().cpu().float(),
                    k,
                    int(args.image_size),
                    confidence_maps=None,
                    samples_per_row=1,
                )

                # Concatenate side-by-side for easy diff inspection
                from PIL import Image, ImageDraw, ImageFont

                W = im_orig.width + im_ft.width
                H = max(im_orig.height, im_ft.height)
                combo = Image.new("RGB", (W, H), color=(128, 128, 128))
                combo.paste(im_orig, (0, 0))
                combo.paste(im_ft, (im_orig.width, 0))
                draw = ImageDraw.Draw(combo)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
                except Exception:
                    font = None
                left_title = "ORIGINAL decoder"
                right_title = "FINETUNED decoder"
                if font is not None:
                    draw.text((20, 5), left_title, fill=(255, 255, 255), font=font, stroke_fill=(0, 0, 0), stroke_width=2)
                    draw.text((im_orig.width + 20, 5), right_title, fill=(255, 255, 255), font=font, stroke_fill=(0, 0, 0), stroke_width=2)

                out_path = os.path.join(viz_dir, "val_grid_orig_vs_finetuned.png")
                combo.save(out_path)
                if bool(getattr(args, "use_wandb", False)) and wandb is not None:
                    accelerator.log(
                        {"val_viz/orig_vs_finetuned": [wandb.Image(combo, caption=f"epoch_{epoch:04d}_idxs={idxs}")]}
                        , step=int(global_step)
                    )
            except Exception as e:
                logger.warning(f"[DecoderRefiner] Failed to write fixed val viz image: {e}")

        # Reduce metrics across processes
        loss_t = torch.tensor([epoch_loss, epoch_n, val_loss_sum, val_n], device=accelerator.device, dtype=torch.float64)
        loss_t = accelerator.reduce(loss_t, reduction="sum")
        train_loss = (loss_t[0] / loss_t[1].clamp_min(1.0)).item()
        val_loss = (loss_t[2] / loss_t[3].clamp_min(1.0)).item()

        if accelerator.is_main_process:
            logger.info(f"[DecoderRefiner] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if bool(getattr(args, "use_wandb", False)):
                accelerator.log(
                    {
                        "epoch/train_loss": float(train_loss),
                        "epoch/val_loss": float(val_loss),
                        "epoch": int(epoch),
                    },
                    step=int(global_step),
                )

        # Reduce + log CD metrics
        if do_metrics:
            hist_sum = accelerator.reduce(hist_total, reduction="sum")
            labeled_sum = accelerator.reduce(labeled_total, reduction="sum")
            correct_sum = accelerator.reduce(correct_total, reduction="sum")
            metrics = scores_from_confusion(hist=hist_sum, labeled=labeled_sum, correct=correct_sum)
            if accelerator.is_main_process:
                logger.info(
                    "[DecoderRefiner][val-metrics] "
                    f"mean_iou={metrics.get('mean_iou', float('nan')):.4f} "
                    f"iou_fg={metrics.get('iou_fg', float('nan')):.4f} "
                    f"pixel_acc={metrics.get('pixel_acc', float('nan')):.4f} "
                    f"precision_fg={metrics.get('precision_fg', float('nan')):.4f} "
                    f"recall_fg={metrics.get('recall_fg', float('nan')):.4f}"
                )
                if bool(getattr(args, "use_wandb", False)):
                    accelerator.log(
                        {f"val/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                        step=int(global_step),
                    )
                # Save JSON per epoch for easy offline inspection
                if run_dir is not None:
                    try:
                        import json

                        with open(os.path.join(run_dir, f"val_metrics_epoch_{epoch:04d}.json"), "w") as f:
                            json.dump(
                                {
                                    "epoch": int(epoch),
                                    "global_step": int(global_step),
                                    "train_loss": float(train_loss),
                                    "val_loss": float(val_loss),
                                    "metrics": metrics,
                                },
                                f,
                                indent=2,
                            )
                    except Exception:
                        pass

        # Save best
        if accelerator.is_main_process:
            if best_val is None or val_loss < best_val:
                best_val = float(val_loss)
                # NOTE: Do NOT call `accelerator.unwrap_model()` here.
                # Some Accelerate versions try to import DeepSpeed inside unwrap, and environments where
                # `deepspeed` is installed but CUDA toolchain vars (CUDA_HOME) are not configured will crash.
                # Our training uses regular DDP via Accelerate, so `.module` is sufficient.
                unwrapped = decoder_model.module if hasattr(decoder_model, "module") else decoder_model
                unwrapped_skip = skip_extractor.module if hasattr(skip_extractor, "module") else skip_extractor
                ckpt = {
                    "decoder_state_dict": unwrapped.decoder.state_dict(),
                    "post_quant_conv_state_dict": (unwrapped.post_quant_conv.state_dict() if bool(getattr(args, "train_post_quant_conv", False)) else None),
                    "rgb_to_mask_head_state_dict": (unwrapped.rgb_to_mask.state_dict() if getattr(unwrapped, "rgb_to_mask", None) is not None else None),
                    "use_rgb_to_mask_head": bool(getattr(args, "use_rgb_to_mask_head", False)),
                    "rgb_to_mask_head_init": str(getattr(args, "rgb_to_mask_head_init", "sum_rgb")),
                    # Decoder skip metadata (must match how the conditioned decoder was constructed)
                    "skip_base_resolutions": list(getattr(unwrapped_skip, "skip_base_resolutions", [])),
                    "skip_in_channels": list(getattr(unwrapped_skip, "skip_in_channels", [])),
                    # Optional: save fusion weights if they were trained (so inference can reproduce them).
                    # IMPORTANT: we train fusion_modules_for_skips (decoder skips) and keep fusion_modules frozen (context).
                    "fusion_modules_state_dict": (
                        (unwrapped_skip.var.fusion_modules_for_skips.state_dict() if getattr(unwrapped_skip.var, "fusion_modules_for_skips", None) is not None else unwrapped_skip.var.fusion_modules.state_dict())
                        if bool(getattr(args, "train_fusion", False))
                        else None
                    ),
                    "decoder_extra_fusion": {
                        "resolutions": list(getattr(unwrapped_skip, "extra_resolutions", [])),
                        "num_heads": int(getattr(unwrapped_skip, "extra_num_heads", 8)),
                        "num_layers": int(getattr(unwrapped_skip, "extra_num_layers", 1)),
                        "cross_inner_dim": (None if getattr(unwrapped_skip, "extra_cross_inner_dim", None) is None else int(getattr(unwrapped_skip, "extra_cross_inner_dim"))),
                        "use_feature_rectify": bool(getattr(unwrapped_skip, "extra_use_feature_rectify", False)),
                        "downsample_first": bool(getattr(unwrapped_skip, "extra_downsample_first", False)),
                        "state_dict": unwrapped_skip.extra_fusion_modules.state_dict() if hasattr(unwrapped_skip, "extra_fusion_modules") else None,
                    },
                    "vqvae": {
                        "vocab_size": int(vocab_size),
                        "z_channels": int(args.z_channels),
                        "ch": int(args.ch),
                        "ch_mult": (1, 1, 2, 2, 4),
                        "num_res_blocks": 2,
                    },
                    "meta": {
                        "dataset_name": str(args.dataset_name),
                        "image_size": int(args.image_size),
                        "best_val_loss": float(best_val),
                        "epoch": int(epoch),
                        "time": time.time(),
                    },
                }
                out_path = os.path.join(run_dir, "best_decoder_refiner.pth")
                torch.save(ckpt, out_path)
                logger.info(f"[DecoderRefiner] Saved best checkpoint to: {out_path}")

    if accelerator.is_main_process:
        logger.info("[DecoderRefiner] Done.")
    try:
        accelerator.end_training()
    except Exception:
        pass


if __name__ == "__main__":
    main()


