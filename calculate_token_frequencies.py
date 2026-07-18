"""Calculate mask-token frequencies and class weights for RemoteVAR training."""

import argparse
import json
import os
from typing import Any, Dict

import torch
from ruamel.yaml import YAML
from tqdm import tqdm

from models import VQVAE
from remotevar_datasets import create_dataset
from train_utils.dataset_id import dataset_id_for_run


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/change_detection.yaml")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--cd_union_datasets", nargs="+", default=None)
    parser.add_argument("--data_dirs", nargs="*", default=None)
    parser.add_argument("--vqvae_pretrained_path", default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--z_channels", type=int, default=None)
    parser.add_argument("--ch", type=int, default=None)
    parser.add_argument("--v_patch_nums", type=int, nargs="+", default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--diversity_alpha", type=float, default=2.0)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--mask_rgb_by_location",
        default=None,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--mask_rgb_grid_size", type=int, default=None)
    parser.add_argument("--mask_rgb_index_mode", choices=["grid", "mul"], default=None)
    cli = parser.parse_args()

    config: Dict[str, Any] = {}
    if cli.config and os.path.isfile(cli.config):
        with open(cli.config, "r", encoding="utf-8") as file:
            config = YAML(typ="safe").load(file) or {}

    for key, value in vars(cli).items():
        if value is not None:
            config[key] = value

    defaults = {
        "dataset_name": "cd_union",
        "dataset_root": "data",
        "cd_union_datasets": ["whu_cd", "levircd", "levircdplus", "s2looking"],
        "vqvae_pretrained_path": "pretrained/vae_ch160v4096z32.pth",
        "vocab_size": 4096,
        "z_channels": 32,
        "ch": 160,
        "v_patch_nums": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
        "image_size": 256,
        "mask_rgb_by_location": True,
        "mask_rgb_grid_size": 11,
        "mask_rgb_index_mode": "grid",
        "mask_rgb_levels": 0,
        "filter_empty_masks": False,
        "empty_mask_threshold": 0.1,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    for key in (
        "enable_random_crop",
        "enable_random_flip",
        "enable_random_rotation",
        "enable_gaussian_blur",
        "enable_color_jitter",
    ):
        config[key] = False
    return argparse.Namespace(**config)


def _normalized_change_weights(
    counts: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    power: float,
) -> torch.Tensor:
    result = torch.full((counts.numel(),), 0.1, dtype=torch.float32)
    indices = torch.where(token_mask)[0]
    if indices.numel() == 0:
        return result
    frequencies = counts.float() / counts.sum().clamp_min(1)
    selected = (1.0 / (frequencies[indices] * counts.numel()).clamp_min(1e-8)).pow(power)
    result[indices] = selected / selected.mean().clamp_min(1e-8)
    return result


def calculate_token_frequencies(args):
    """Calculate and save token frequencies for ``args``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = args.vocab_size[0] if isinstance(args.vocab_size, (list, tuple)) else args.vocab_size
    vocab_size = int(vocab_size)

    vqvae = VQVAE(
        vocab_size=vocab_size,
        z_channels=int(args.z_channels),
        ch=int(args.ch),
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=args.v_patch_nums,
    ).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path, map_location="cpu"))
    vqvae.eval()
    vqvae.requires_grad_(False)

    dataset = create_dataset(args.dataset_name, args, split="train")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    counts_all = torch.zeros(vocab_size, dtype=torch.int64)
    counts_background = torch.zeros(vocab_size, dtype=torch.int64)
    counts_change = torch.zeros(vocab_size, dtype=torch.int64)
    all_black_masks = 0
    mixed_masks = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Calculating token frequencies"):
            masks = batch["mask"].to(device, non_blocking=True)
            mask_tokens = vqvae.img_to_idxBl(masks, v_patch_nums=args.v_patch_nums)
            masks_01 = (masks + 1.0) / 2.0

            for sample_index in range(masks.shape[0]):
                sample_tokens = torch.cat(
                    [scale[sample_index].detach().to(device="cpu", dtype=torch.long) for scale in mask_tokens]
                )
                sample_counts = torch.bincount(sample_tokens, minlength=vocab_size)
                foreground_ratio = (
                    masks_01[sample_index].amax(dim=0).gt(0.05).float().mean().item()
                )
                if foreground_ratio < 0.01:
                    counts_background += sample_counts
                    all_black_masks += 1
                else:
                    counts_change += sample_counts
                    mixed_masks += 1
                counts_all += sample_counts

    change_tokens = counts_change > 0
    background_only_tokens = (counts_background > 0) & ~change_tokens
    class_weights_inv = _normalized_change_weights(counts_change, change_tokens, power=1.0)
    class_weights_alpha = _normalized_change_weights(
        counts_change,
        change_tokens,
        power=float(args.diversity_alpha),
    )

    class_weights_effective = torch.full((vocab_size,), 0.1, dtype=torch.float32)
    change_indices = torch.where(change_tokens)[0]
    if change_indices.numel() > 0:
        beta = 0.9999
        effective = 1.0 - torch.pow(beta, counts_change[change_indices].float())
        selected = (1.0 - beta) / effective.clamp_min(1e-8)
        class_weights_effective[change_indices] = selected / selected.mean().clamp_min(1e-8)

    dataset_id = dataset_id_for_run(args)
    output_path = args.output_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "token_frequencies",
        f"{dataset_id}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    output = {
        "dataset_id": dataset_id,
        "dataset_name": args.dataset_name,
        "vocab_size": vocab_size,
        "total_tokens": int(counts_all.sum().item()),
        "unique_tokens_used": int((counts_all > 0).sum().item()),
        "num_all_black_masks": all_black_masks,
        "num_mixed_masks": mixed_masks,
        "total_background_tokens": int(counts_background.sum().item()),
        "total_change_tokens": int(counts_change.sum().item()),
        "num_background_only_tokens": int(background_only_tokens.sum().item()),
        "num_change_tokens": int(change_tokens.sum().item()),
        "background_only_token_ids": torch.where(background_only_tokens)[0].tolist(),
        "change_token_ids": change_indices.tolist(),
        "token_counts_all": counts_all.tolist(),
        "token_counts_background": counts_background.tolist(),
        "token_counts_change": counts_change.tolist(),
        "class_weights_inv": class_weights_inv.tolist(),
        "class_weights_alpha": class_weights_alpha.tolist(),
        "class_weights_effective": class_weights_effective.tolist(),
        "diversity_alpha": float(args.diversity_alpha),
        "vqvae_path": args.vqvae_pretrained_path,
        "mask_rgb_by_location": bool(args.mask_rgb_by_location),
        "mask_rgb_grid_size": args.mask_rgb_grid_size,
        "mask_rgb_index_mode": args.mask_rgb_index_mode,
        "data_dirs": getattr(args, "data_dirs", None),
        "dataset_root": getattr(args, "dataset_root", None),
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"Wrote token frequencies: {output_path}")
    return output


if __name__ == "__main__":
    calculate_token_frequencies(parse_args())
