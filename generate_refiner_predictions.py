"""Generate deterministic RemoteVAR mask-latent caches for decoder-refiner training."""

import argparse
import os
from typing import Any, Dict

import torch
import torch.distributed as torch_dist
from ruamel.yaml import YAML
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from models import build_remote_var
from models.vqvae import VQVAE
from remotevar_datasets import create_dataset


def _load_yaml(path: str) -> Dict[str, Any]:
    yaml = YAML(typ="safe")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file) or {}


def _to_namespace(values: Dict[str, Any]):
    return argparse.Namespace(**values)


def _load_safetensors_state(path: str) -> Dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate mask_fhat caches used by train_decoder_refiner.py.",
    )
    parser.add_argument("--config", default="configs/change_detection.yaml", help="RemoteVAR YAML configuration.")
    parser.add_argument("--checkpoint", default=None, help="RemoteVAR checkpoint; overrides var_pretrained_path in YAML.")
    parser.add_argument("--dataset_root", default=None, help="Dataset root; overrides dataset_root in YAML.")
    parser.add_argument("--dataset_name", default=None, help="Dataset name; overrides dataset_name in YAML.")
    parser.add_argument(
        "--cd_union_datasets",
        nargs="+",
        default=None,
        help="Components and order for cd_union; overrides the YAML value.",
    )
    parser.add_argument(
        "--data_dirs",
        nargs="+",
        default=None,
        help="Explicit dataset roots in cd_union order; overrides canonical dataset_root subfolders.",
    )
    parser.add_argument("--device", default=None, help="Device for a single-process run.")
    parser.add_argument("--batch_size", type=int, default=1, help="Autoregressive batch size per GPU.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional per-split sample limit for smoke tests.",
    )
    parser.add_argument("--out_dir", default="predictions", help="Destination for generated cache files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite completed cache files.")
    parser.add_argument(
        "--cleanup_shards",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Remove distributed per-rank shards after rank 0 merges them.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
    )
    parser.add_argument(
        "--deterministic",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use no CFG and argmax sampling.",
    )
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--top_k", type=int, default=900)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--vqvae_pretrained_path",
        default=None,
        help="VQ-VAE checkpoint; overrides vqvae_pretrained_path in YAML.",
    )
    return parser.parse_args()


def _disable_random_augmentations(config: Dict[str, Any]) -> None:
    # Cached latents must remain index-aligned with the unaugmented dataset.
    for key in (
        "enable_random_crop",
        "enable_random_flip",
        "enable_random_rotation",
        "enable_gaussian_blur",
        "enable_color_jitter",
    ):
        config[key] = False


class _WithIndex(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        if not isinstance(item, dict):
            raise TypeError("RemoteVAR datasets must return dictionaries.")
        result = dict(item)
        result["idx"] = int(index)
        return result


def _distributed_info() -> Dict[str, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and torch_dist.is_available() and not torch_dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch_dist.init_process_group(backend=backend)
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def _rank():
    return torch_dist.get_rank() if torch_dist.is_initialized() else 0


def _world_size():
    return torch_dist.get_world_size() if torch_dist.is_initialized() else 1


@torch.no_grad()
def _generate_split(
    *,
    split: str,
    args,
    device: torch.device,
    vqvae: VQVAE,
    var,
    output_path: str,
):
    base_dataset = create_dataset(args.dataset_name, args, split=split)
    if args.max_samples is not None and int(args.max_samples) > 0:
        limit = min(int(args.max_samples), len(base_dataset))
        base_dataset = Subset(base_dataset, range(limit))
    dataset = _WithIndex(base_dataset)
    rank = _rank()
    world_size = _world_size()

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    sample_count = len(dataset)
    latent_hw = int(args.v_patch_nums[-1])
    cvae = int(getattr(vqvae, "Cvae", 0))
    if cvae <= 0:
        raise ValueError("VQVAE must expose a positive Cvae value.")

    save_dtype = torch.float16 if args.save_dtype == "fp16" else torch.float32
    bank = None
    if world_size == 1:
        bank = torch.empty((sample_count, cvae, latent_hw, latent_hw), dtype=save_dtype)
    shard_indices = []
    shard_latents = []

    if args.deterministic:
        cfg_weights = [0.0, 0.0, 0.0]
        top_k = 1
        top_p = 0.0
    else:
        cfg_weights = [float(args.guidance_scale)] * 3
        top_k = int(args.top_k)
        top_p = float(args.top_p)

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"[generate_refiner_predictions][rank{rank}] {split}",
        disable=rank != 0,
    )
    for batch in progress:
        images_pre = batch["images_pre"].to(device, non_blocking=True)
        images_post = batch["images_post"].to(device, non_blocking=True)
        conditions = batch["cls"].to(device, non_blocking=True)
        condition_types = batch["type"].to(device, non_blocking=True)
        original_indices = batch["idx"]
        batch_size = int(images_pre.shape[0])

        pre_tokens = vqvae.img_to_idxBl(images_pre, v_patch_nums=args.v_patch_nums)
        post_tokens = vqvae.img_to_idxBl(images_post, v_patch_nums=args.v_patch_nums)
        context = var.encode_context_with_fusion([images_pre, images_post])

        _, mask_fhat = var.conditional_infer_cfg(
            B=batch_size,
            label_B=conditions,
            cfg=cfg_weights,
            top_k=top_k,
            top_p=top_p,
            g_seed=int(args.seed),
            cond_type=condition_types,
            c_img_pre=pre_tokens,
            c_img_post=post_tokens,
            context=context,
            return_confidence=False,
            return_mask_fhat=True,
        )

        index_cpu = original_indices.to(dtype=torch.long, device="cpu")
        latent_cpu = mask_fhat.detach().to(device="cpu", dtype=save_dtype)
        if world_size == 1:
            bank[index_cpu] = latent_cpu
        else:
            shard_indices.append(index_cpu)
            shard_latents.append(latent_cpu)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    metadata = {
        "split": split,
        "dataset_name": str(args.dataset_name),
        "len": sample_count,
        "latent_hw": latent_hw,
        "cvae": cvae,
        "deterministic": bool(args.deterministic),
        "guidance_scale": float(args.guidance_scale),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "save_dtype": str(args.save_dtype),
        "world_size": world_size,
        "checkpoint": str(args.var_pretrained_path),
        "cd_union_datasets": list(getattr(args, "cd_union_datasets", [])),
    }

    if world_size == 1:
        torch.save({"mask_fhat": bank, "meta": metadata}, output_path)
        print(
            f"[generate_refiner_predictions] Wrote {output_path} "
            f"(shape={tuple(bank.shape)}, dtype={bank.dtype})"
        )
        return

    shard_path = output_path.replace(".pt", f".rank{rank}.pt")
    indices = (
        torch.cat(shard_indices, dim=0)
        if shard_indices
        else torch.empty((0,), dtype=torch.long)
    )
    latents = (
        torch.cat(shard_latents, dim=0)
        if shard_latents
        else torch.empty((0, cvae, latent_hw, latent_hw), dtype=save_dtype)
    )
    torch.save({"idx": indices, "mask_fhat": latents, "meta": metadata}, shard_path)
    torch_dist.barrier()

    if rank == 0:
        merged = torch.empty((sample_count, cvae, latent_hw, latent_hw), dtype=save_dtype)
        seen = torch.zeros(sample_count, dtype=torch.bool)
        for shard_rank in range(world_size):
            path = output_path.replace(".pt", f".rank{shard_rank}.pt")
            shard = torch.load(path, map_location="cpu")
            shard_idx = shard["idx"].long()
            if shard_idx.numel() == 0:
                continue
            merged[shard_idx] = shard["mask_fhat"]
            seen[shard_idx] = True

        if not bool(seen.all()):
            missing = int((~seen).sum().item())
            raise RuntimeError(f"Distributed cache merge missed {missing} dataset indices.")

        torch.save({"mask_fhat": merged, "meta": metadata}, output_path)
        print(
            f"[generate_refiner_predictions] Wrote {output_path} "
            f"(shape={tuple(merged.shape)}, dtype={merged.dtype})"
        )

        if args.cleanup_shards:
            for shard_rank in range(world_size):
                path = output_path.replace(".pt", f".rank{shard_rank}.pt")
                try:
                    os.remove(path)
                except OSError:
                    pass

    torch_dist.barrier()


def _load_remotevar_checkpoint(var, checkpoint: str):
    if checkpoint.endswith(".safetensors"):
        state = _load_safetensors_state(checkpoint)
    else:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if checkpoint.endswith(".pth"):
            for key in ("lvl_1L", "pos_start", "attn_bias_for_masking", "pos_1LC"):
                state.pop(key, None)

    missing, unexpected = var.load_state_dict(state, strict=False)
    if _rank() == 0:
        print(
            "[generate_refiner_predictions] Loaded RemoteVAR checkpoint "
            f"(missing={len(missing)}, unexpected={len(unexpected)}): {checkpoint}"
        )


def main():
    cli = parse_args()
    distributed = _distributed_info()
    config = _load_yaml(cli.config)
    _disable_random_augmentations(config)

    defaults = {
        "vocab_size": 4096,
        "z_channels": 32,
        "ch": 160,
        "v_patch_nums": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16],
        "depth": 16,
        "mask_type": "change_append",
        "cond_drop_rate": 0.0,
        "bidirectional": False,
        "separate_decoding": False,
        "separator": False,
        "multi_cond": True,
        "disable_cross_attention": False,
        "image_size": 256,
        "use_high_res_context_levels": False,
        "fusion_downsample_ratios": None,
        "fusion_num_heads": 8,
        "fusion_num_layers": 1,
        "fusion_cross_inner_dim": None,
        "fusion_use_feature_rectify": False,
        "fusion_downsample_first": False,
        "drop_path_rate": 0.0,
        "cross_attn_inner_dim": 1024,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    if cli.checkpoint is not None:
        config["var_pretrained_path"] = cli.checkpoint
    if cli.dataset_root is not None:
        config["dataset_root"] = cli.dataset_root
    if cli.dataset_name is not None:
        config["dataset_name"] = cli.dataset_name
    if cli.cd_union_datasets is not None:
        config["cd_union_datasets"] = list(cli.cd_union_datasets)
    if cli.data_dirs is not None:
        config["data_dirs"] = list(cli.data_dirs)
    if cli.vqvae_pretrained_path is not None:
        config["vqvae_pretrained_path"] = cli.vqvae_pretrained_path

    config.update(
        {
            "batch_size": cli.batch_size,
            "num_workers": cli.num_workers,
            "max_samples": cli.max_samples,
            "deterministic": cli.deterministic,
            "guidance_scale": cli.guidance_scale,
            "top_k": cli.top_k,
            "top_p": cli.top_p,
            "seed": cli.seed,
            "save_dtype": cli.save_dtype,
            "cleanup_shards": cli.cleanup_shards,
        }
    )
    args = _to_namespace(config)

    if not getattr(args, "dataset_root", None):
        raise ValueError("Set --dataset_root or dataset_root in the YAML config.")
    if not getattr(args, "var_pretrained_path", None):
        raise ValueError("Set --checkpoint or var_pretrained_path in the YAML config.")
    if not getattr(args, "vqvae_pretrained_path", None):
        raise ValueError("Set --vqvae_pretrained_path or its YAML equivalent.")

    if torch.cuda.is_available() and distributed["world_size"] > 1:
        torch.cuda.set_device(distributed["local_rank"])
        device = torch.device(f"cuda:{distributed['local_rank']}")
    else:
        device = torch.device(cli.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if _rank() == 0:
        print(f"[generate_refiner_predictions] device={device}, world_size={_world_size()}")

    vocab_size = args.vocab_size[0] if isinstance(args.vocab_size, (list, tuple)) else args.vocab_size
    vqvae = VQVAE(
        vocab_size=int(vocab_size),
        z_channels=int(args.z_channels),
        ch=int(args.ch),
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=args.v_patch_nums,
    ).to(device)
    vqvae.load_state_dict(torch.load(args.vqvae_pretrained_path, map_location="cpu"))
    vqvae.eval()
    vqvae.requires_grad_(False)

    var = build_remote_var(
        vae=vqvae,
        depth=args.depth,
        patch_nums=args.v_patch_nums,
        mask_type=args.mask_type,
        cond_drop_rate=args.cond_drop_rate,
        bidirectional=args.bidirectional,
        separate_decoding=args.separate_decoding,
        separator=args.separator,
        multi_cond=args.multi_cond,
        disable_cross_attention=args.disable_cross_attention,
        enable_current_scale_tokens=getattr(args, "enable_current_scale_tokens", False),
        image_size=args.image_size,
        use_high_res_context_levels=args.use_high_res_context_levels,
        fusion_downsample_ratios=args.fusion_downsample_ratios,
        fusion_num_heads=args.fusion_num_heads,
        fusion_num_layers=args.fusion_num_layers,
        fusion_cross_inner_dim=args.fusion_cross_inner_dim,
        fusion_use_feature_rectify=args.fusion_use_feature_rectify,
        fusion_downsample_first=args.fusion_downsample_first,
        allow_trainable_encoder=False,
        drop_path_rate=args.drop_path_rate,
        cross_attn_inner_dim=args.cross_attn_inner_dim,
    ).to(device)
    _load_remotevar_checkpoint(var, str(args.var_pretrained_path))
    var.eval()
    var.requires_grad_(False)

    os.makedirs(cli.out_dir, exist_ok=True)
    for split in cli.splits:
        output_path = os.path.join(cli.out_dir, f"{args.dataset_name}_{split}_mask_fhat.pt")
        if os.path.exists(output_path) and not cli.overwrite:
            if _rank() == 0:
                print(f"[generate_refiner_predictions] Skip existing: {output_path}")
            continue
        _generate_split(
            split=split,
            args=args,
            device=device,
            vqvae=vqvae,
            var=var,
            output_path=output_path,
        )

    if torch_dist.is_initialized():
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
