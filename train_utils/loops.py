import copy
import json
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .loss import build_token_loss_fn


def select_nonempty_samples(pre_images, post_images, masks, conditions, cond_type, fns=None, k=4):
    """
    Helper for visualization/inference: pick up to k samples with non-empty masks.
    """
    if pre_images is None:
        return None
    B = int(pre_images.shape[0])
    k = int(min(max(1, k), B))

    # masks are in [-1,1]; foreground if max-channel > 0.1 in [0,1]
    m01 = (masks + 1) / 2
    fg = (m01.max(dim=1).values > 0.1).flatten(1).float().mean(dim=1)  # (B,)
    idxs = torch.argsort(fg, descending=True)[:k]

    def _sel(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            return [t[idxs] if torch.is_tensor(t) else t for t in x]
        return x[idxs]

    out = {
        "images_pre": _sel(pre_images),
        "images_post": _sel(post_images),
        "masks": _sel(masks),
        "conditions": _sel(conditions),
        "cond_type": _sel(cond_type),
        "fns": None if fns is None else [fns[i] for i in idxs.tolist()],
    }
    return out


def train_epoch(accelerator, var, vqvae, cond_model, dataloader, optimizer, progress_bar, args):
    """
    Training loop for one epoch.
    This is moved out of the entry script for readability.
    """
    logger = accelerator.logger if hasattr(accelerator, "logger") else None

    var.train()
    if cond_model is not None:
        cond_model.train()

    epoch_loss_sum = 0.0
    epoch_sample_count = 0

    loss_fn = build_token_loss_fn(args, accelerator.device)

    for _, batch in enumerate(dataloader):
        with accelerator.accumulate(var):
            images_pre = batch["images_pre"]
            images_post = batch["images_post"]
            masks = batch["mask"]
            conditions = batch["cls"]
            cond_type = batch["type"]

            # store last batch for visualization
            args.last_batch = batch

            with torch.no_grad():
                mask_labels_list = vqvae.img_to_idxBl(masks, v_patch_nums=args.v_patch_nums)

                # optional noisy teacher forcing for mask stream
                noisy_p = float(getattr(args, "noisy_tf_mask_prob", 0.0))
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

                input_h_list_pre = vqvae.img_to_h(images_pre, v_patch_nums=args.v_patch_nums)
                input_h_list_post = vqvae.img_to_h(images_post, v_patch_nums=args.v_patch_nums)
                mask_input_h_list = vqvae.idxBl_to_h(mask_labels_for_tf)
                input_h_list = list(sum(zip(input_h_list_pre, input_h_list_post, mask_input_h_list), ()))
                x_BLCv_wo_first_l = torch.cat(input_h_list, dim=1)

            # conditional embeddings
            if cond_model is not None:
                cond = cond_model(conditions)
            else:
                cond = conditions

            logits = var(cond, x_BLCv_wo_first_l, mask_first=False, cond_type=cond_type)  # (B,L,V)

            # build labels stream (same layout as teacher forcing)
            labels_list_pre = vqvae.img_to_idxBl(images_pre, v_patch_nums=args.v_patch_nums)
            labels_list_post = vqvae.img_to_idxBl(images_post, v_patch_nums=args.v_patch_nums)
            labels_list = list(sum(zip(labels_list_pre, labels_list_post, mask_labels_list), ()))
            labels = torch.cat(labels_list, dim=1)

            # token loss (per-token), then reduce
            vocab_size_val = int(getattr(args, "vocab_size", labels.max().item() + 1))
            logits = logits[:, :, :vocab_size_val]
            loss_per_token = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).view(labels.shape)
            loss = loss_per_token.mean()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                if float(getattr(args, "clip", 0.0)) > 0:
                    accelerator.clip_grad_norm_(var.parameters(), float(args.clip))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                progress_bar.update(1)

            epoch_loss_sum += float(loss.detach().item()) * int(images_pre.shape[0])
            epoch_sample_count += int(images_pre.shape[0])

    avg_loss = epoch_loss_sum / max(1, epoch_sample_count)
    return avg_loss


def inference(
    accelerator,
    *,
    pix_cond_inference_fn,
    create_comparison_image_fn,
    var,
    vqvae,
    cond_model,
    pre_images,
    post_images,
    masks,
    conditions,
    cond_type,
    args,
    context=None,
    labels_list_pre=None,
    labels_list_post=None,
):
    """
    Thin wrapper around the repo's `pix_cond_inference` for training-time visualization.
    """
    B = int(pre_images.shape[0])
    pred_images = pix_cond_inference_fn(
        pre_images,
        post_images,
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

    try:
        im = create_comparison_image_fn(
            pre_images.detach().cpu().float(),
            post_images.detach().cpu().float(),
            pred_images.detach().cpu().float(),
            masks.detach().cpu().float(),
            B,
            int(getattr(args, "image_size", 256)),
            confidence_maps=None,
            samples_per_row=1,
        )
    except Exception:
        im = None
    return pred_images, im


def validate(accelerator, var, vqvae, cond_model, val_dataloader, args, *, pix_cond_inference_fn):
    """
    Validation loop (loss + optional metrics). Kept as a helper to keep entry script smaller.
    """
    from utils.mask_metrics import confusion_from_pred_and_gt, scores_from_confusion

    var.eval()
    if cond_model is not None:
        cond_model.eval()

    total_loss = 0.0
    total_samples = 0

    do_metrics = bool(getattr(args, "compute_val_metrics", False))
    max_metric_batches = int(getattr(args, "val_metrics_max_batches", -1))
    hist_total = torch.zeros((2, 2), dtype=torch.float32, device=accelerator.device)
    labeled_total = torch.zeros((), dtype=torch.float32, device=accelerator.device)
    correct_total = torch.zeros((), dtype=torch.float32, device=accelerator.device)

    loss_fn = build_token_loss_fn(args, accelerator.device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            images_pre = batch["images_pre"]
            images_post = batch["images_post"]
            masks = batch["mask"]
            conditions = batch["cls"]
            cond_type = batch["type"]

            mask_labels_list = vqvae.img_to_idxBl(masks, v_patch_nums=args.v_patch_nums)
            input_h_list_pre = vqvae.img_to_h(images_pre, v_patch_nums=args.v_patch_nums)
            input_h_list_post = vqvae.img_to_h(images_post, v_patch_nums=args.v_patch_nums)
            mask_input_h_list = vqvae.idxBl_to_h(mask_labels_list)
            input_h_list = list(sum(zip(input_h_list_pre, input_h_list_post, mask_input_h_list), ()))
            x_BLCv_wo_first_l = torch.cat(input_h_list, dim=1)

            if cond_model is not None:
                cond = cond_model(conditions)
            else:
                cond = conditions

            logits = var(cond, x_BLCv_wo_first_l, mask_first=False, cond_type=cond_type)
            labels_list_pre = vqvae.img_to_idxBl(images_pre, v_patch_nums=args.v_patch_nums)
            labels_list_post = vqvae.img_to_idxBl(images_post, v_patch_nums=args.v_patch_nums)
            labels_list = list(sum(zip(labels_list_pre, labels_list_post, mask_labels_list), ()))
            labels = torch.cat(labels_list, dim=1)

            vocab_size_val = int(getattr(args, "vocab_size", labels.max().item() + 1))
            logits = logits[:, :, :vocab_size_val]
            loss_per_token = loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).view(labels.shape)
            batch_loss = loss_per_token.mean()

            bs = int(images_pre.shape[0])
            total_loss += float(batch_loss.detach().item()) * bs
            total_samples += bs

            if do_metrics and (max_metric_batches < 0 or batch_idx < max_metric_batches):
                try:
                    B = int(images_pre.shape[0])
                    pred_images = pix_cond_inference_fn(
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
                        context=None,
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
                except Exception:
                    pass

    avg_loss = total_loss / max(1, total_samples)
    metrics = None
    if do_metrics:
        hist_sum = accelerator.reduce(hist_total, reduction="sum")
        labeled_sum = accelerator.reduce(labeled_total, reduction="sum")
        correct_sum = accelerator.reduce(correct_total, reduction="sum")
        metrics = scores_from_confusion(hist=hist_sum, labeled=labeled_sum, correct=correct_sum)

    var.train()
    if cond_model is not None:
        cond_model.train()
    return avg_loss, metrics

