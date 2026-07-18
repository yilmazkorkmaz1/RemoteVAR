import math
import random
import copy
from functools import partial
from typing import Optional, Tuple, Union
from itertools import chain

import torch
import torch.nn as nn
from torch.nn import functional as F

import dist
from models.basic_var import AdaLNSABlock, SABlock
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
from models.vqvae import VQVAE, VectorQuantizer2
from models.fusion import FeatureFusionModule


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class RemoteVAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, norm_eps=1e-6, aln=1, aln_gamma_init=1e-3, shared_aln=False, cond_drop_rate=0.1,
        depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        layer_scale=-1., tau=4, cos_attn=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True, mask_factor=2, bidirectional=False, separate_decoding=False,
        separator=False, type_pos=False, indep=True, multi_cond=False, disable_cross_attention=False,
        enable_current_scale_tokens: bool = False,
        image_size: int = 256,
        # If False, use legacy 4-level context: [64, 32, 16(pre), 16(post-middle)] (no 256/128 contexts).
        # If True, use all encoder down levels + post-middle: e.g. [256, 128, 64, 32, 16, 16(post)] for 256px input.
        use_high_res_context_levels: bool = True,
        # Optional post-fusion downsampling inside fusion modules (stride Conv2D stacks).
        # This preserves high-res learning inside fusion, while limiting context token count for transformer cross-attn.
        # Example (256x256 input): downsample fused 256/128 contexts to 64x64, leaving 64/32/16/16mid unchanged.
        fusion_downsample_ratios: Optional[Tuple[int, ...]] = None,
        # Fusion-module scaling knobs (can be scalar or per-level tuples/lists via YAML).
        fusion_num_heads: Union[int, Tuple[int, ...]] = 8,
        fusion_num_layers: Union[int, Tuple[int, ...]] = 1,
        fusion_cross_inner_dim: Optional[Union[int, Tuple[int, ...]]] = None,
        fusion_use_feature_rectify: Union[bool, Tuple[bool, ...]] = False,
        fusion_downsample_first: Union[bool, Tuple[bool, ...]] = False,
        # If True, use a trainable *copy* of the VQVAE encoder for context features used by fusion modules.
        # The original VQVAE stays frozen for GT tokenization (img_to_idxBl) and should NEVER be trained.
        allow_trainable_encoder: bool = False,
        cross_attn_inner_dim=1024,
    ):
        super().__init__()
        # 0. hyperparameters
        cos_attn = True if depth == 30 else False
        if cos_attn:
            print(f'Rewrite cos_attn to True when depth={depth}')
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        self.using_aln, self.aln_init, self.aln_gamma_init, self.layer_scale = aln >= 0, aln, aln_gamma_init, layer_scale
        if self.using_aln and layer_scale != -1:
            print(f'**WARNING**: using AdaLNSABlock with {aln=:g}, {aln_gamma_init=:g}; the arg {layer_scale=:g} will be IGNORED because only SABlock cares about layer_scale', flush=True)

        self.separator = separator
        self.bidirectional = bidirectional
        self.separate_decoding = separate_decoding
        self.type_pos = type_pos
        self.indep = indep
        self.multi_cond = multi_cond
        self.disable_cross_attention = disable_cross_attention
        self.enable_current_scale_tokens = enable_current_scale_tokens
        self.image_size = int(image_size)
        self.use_high_res_context_levels = bool(use_high_res_context_levels)
        self.fusion_downsample_ratios = fusion_downsample_ratios
        self.fusion_num_heads = fusion_num_heads
        self.fusion_num_layers = fusion_num_layers
        self.fusion_cross_inner_dim = fusion_cross_inner_dim
        self.fusion_use_feature_rectify = fusion_use_feature_rectify
        self.fusion_downsample_first = fusion_downsample_first

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.mask_factor = mask_factor
        self.L = sum(pn ** 2 * mask_factor for pn in self.patch_nums)  # image mask pair
        if self.separator:
            self.L += (len(self.patch_nums) - 1) * mask_factor  # special tokens
        self.first_l = self.patch_nums[0] ** 2 * mask_factor
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            num_sp_tokens = 1 if i != 0 and self.separator else 0
            self.begin_ends.append((cur, cur+(pn ** 2 + num_sp_tokens) * mask_factor))
            cur += (pn ** 2 + num_sp_tokens) * mask_factor
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # Optional: trainable encoder copy for fusion-context features.
        # This MUST NOT touch the frozen VQVAE used for GT tokenization.
        self.allow_trainable_encoder = bool(allow_trainable_encoder)
        if self.allow_trainable_encoder and self.disable_cross_attention:
            raise ValueError(
                "allow_trainable_encoder=True requires cross-attention to be enabled "
                "(disable_cross_attention=False), otherwise the encoder copy would be unused."
            )
        if self.allow_trainable_encoder:
            # Start from the frozen encoder weights, but allow gradients for richer context features.
            self.trainable_encoder = copy.deepcopy(vae_local.encoder)
            for p in self.trainable_encoder.parameters():
                p.requires_grad_(True)
            # IMPORTANT: `Encoder.forward_context()` does NOT use the final latent head:
            #   - norm_out
            #   - conv_out
            # Leaving them trainable would trigger DDP "unused parameters" warnings/errors.
            if hasattr(self.trainable_encoder, "norm_out") and self.trainable_encoder.norm_out is not None:
                for p in self.trainable_encoder.norm_out.parameters():
                    p.requires_grad_(False)
            if hasattr(self.trainable_encoder, "conv_out") and self.trainable_encoder.conv_out is not None:
                for p in self.trainable_encoder.conv_out.parameters():
                    p.requires_grad_(False)
        else:
            self.trainable_encoder = None

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.selecting_idx = torch.full((1, num_classes), fill_value=1/num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)

        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            num_sp_tokens = 1 if i != 0 and self.separator else 0
            pe = torch.empty(1, (pn*pn + num_sp_tokens) * mask_factor, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        # TODO: test separate mask/image level embedding
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)  #  lvl_1L = mT[:, 0].contiguous()
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        if self.type_pos:
            self.type_embed = nn.Embedding(self.mask_factor, self.C)  # lvl_1L = mT[:, 0].contiguous()
            nn.init.trunc_normal_(self.type_embed.weight.data, mean=0, std=init_std)
            print('Creating type positional encoding')
            m, m_ = [], []
            for i, pn in enumerate(self.patch_nums):
                num_sp_tokens = 1 if (i != 0 and self.separator) else 0
                m.append(torch.full((pn*pn + num_sp_tokens,), 1))
                m.append(torch.full((pn * pn + num_sp_tokens,), 0))
                m_.append(torch.full((pn * pn + num_sp_tokens,), 0))
                m_.append(torch.full((pn * pn + num_sp_tokens,), 1))
            m = torch.cat(m).view(1, self.L, 1)
            m_ = torch.cat(m_).view(1, self.L, 1)
            mT = m.transpose(1, 2)  # dT: 11L
            mT_ = m_.transpose(1, 2)  # dT: 11L
            type_1L = mT[:, 0].contiguous()
            type_1L_ = mT_[:, 0].contiguous()
            self.register_buffer('type_1L', type_1L)
            self.register_buffer('type_1L_', type_1L_)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln and self.using_aln else nn.Identity()

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)

        # Context levels come from the frozen VQVAE encoder:
        # - all down-path levels (one per resolution)
        # - plus one extra "post-middle" feature at the final spatial resolution
        enc = vae_local.encoder
        num_down_levels = int(getattr(enc, "num_resolutions", 0))
        # Build expected encoder spatial resolutions for this run, derived from image_size and downsampling schedule.
        # For vq-f16 ch_mult length 5 and image_size=256:
        # all down levels: [256, 128, 64, 32, 16], post-middle: [16]
        all_down_res = [self.image_size // (2 ** i) for i in range(num_down_levels)]
        if len(all_down_res) == 0:
            raise RuntimeError("VQVAE encoder has num_resolutions=0; cannot build context levels.")

        if self.use_high_res_context_levels:
            # all down-path + post-middle
            encoder_spatial_resolutions = list(all_down_res) + [all_down_res[-1]]
        else:
            # legacy: last 3 down levels (64/32/16 for 256px) + post-middle (16)
            if len(all_down_res) < 3:
                raise RuntimeError(
                    f"Legacy 4-level context requires >=3 down levels, got {len(all_down_res)} "
                    f"(image_size={self.image_size}, num_resolutions={num_down_levels})."
                )
            encoder_spatial_resolutions = list(all_down_res[-3:]) + [all_down_res[-1]]

        num_encoder_levels = len(encoder_spatial_resolutions)

        # Compute per-level context channel dims from the encoder config (depends on VQVAE `ch`).
        # For vq-f16 (ch_mult=(1,1,2,2,4)) and ch=160, this yields:
        # down levels: [160,160,320,320,640] and post-middle: [640]
        enc_ch = getattr(enc, "ch", None)
        enc_ch_mult = getattr(enc, "ch_mult", None)
        if enc_ch is None or enc_ch_mult is None:
            raise RuntimeError(
                "VQVAE encoder must expose `ch` and `ch_mult` to build multi-resolution context dims. "
                "Please ensure models/vae_modules.py Encoder sets `self.ch_mult`."
            )
        down_dims = [int(enc_ch * m) for m in enc_ch_mult]
        post_mid_dim = int(enc_ch * enc_ch_mult[-1])
        
        if self.use_high_res_context_levels:
            # all down levels + post-middle
            context_dims_per_level = down_dims + [post_mid_dim]
        else:
            # legacy: last 3 down levels + post-middle
            context_dims_per_level = down_dims[-3:] + [post_mid_dim]

        assert len(context_dims_per_level) == num_encoder_levels, (len(context_dims_per_level), num_encoder_levels)

        # Default downsample ratios: cap effective context to 64x64 when possible (only affects >64 levels).
        if self.fusion_downsample_ratios is None:
            default = []
            for base_hw in encoder_spatial_resolutions:
                if base_hw > 64 and base_hw % 64 == 0:
                    ratio = base_hw // 64
                    # keep stride-conv stack power-of-two only
                    if ratio & (ratio - 1) == 0:
                        default.append(int(ratio))
                        continue
                default.append(1)
            self.fusion_downsample_ratios = tuple(default)

        if self.fusion_downsample_ratios is not None and len(self.fusion_downsample_ratios) != num_encoder_levels:
            raise ValueError(
                f"fusion_downsample_ratios must be length {num_encoder_levels} (or None), got {len(self.fusion_downsample_ratios)}."
            )

        # Precompute the context token length for each level AFTER optional post-fusion downsampling.
        # This is used for cross-attn 2D positional embeddings (initialized at construction).
        context_lens_per_level = []
        for i, base_hw in enumerate(encoder_spatial_resolutions):
            ratio = 1 if self.fusion_downsample_ratios is None else int(self.fusion_downsample_ratios[i])
            if ratio < 1:
                raise ValueError(f"fusion_downsample_ratios[{i}] must be >= 1, got {ratio}")
            if base_hw % ratio != 0:
                raise ValueError(
                    f"fusion_downsample_ratios[{i}]={ratio} incompatible with base_hw={base_hw} (not divisible). "
                    f"Check image_size={self.image_size} and your encoder schedule."
                )
            hw = base_hw // ratio
            context_lens_per_level.append(int(hw * hw))

        # Map each transformer block to a context level.
        # IMPORTANT: Assign COARSE contexts to EARLY blocks and FINE contexts to LATE blocks
        # (coarse -> fine across depth), which matches the original design intent.
        # We keep indices aligned with encoder context list ordering (0..num_encoder_levels-1),
        # but we fill blocks in a reversed (coarse->fine) order of those indices.
        def _allocate_counts(total: int, weights):
            ws = [float(w) for w in weights]
            s = sum(ws)
            raw = [w / s * total for w in ws]
            base = [int(math.floor(x)) for x in raw]
            rem = total - sum(base)
            fracs = sorted([(raw[i] - base[i], i) for i in range(len(ws))], reverse=True)
            for _, i in fracs[:rem]:
                base[i] += 1
            return base

        # Encoder context indices are ordered from fine->coarse in the down path, then post-middle:
        #   0:256, 1:128, 2:64, 3:32, 4:16(pre), 5:16(post-middle)  (for 256px input & 6 levels)
        # We want to allocate blocks from COARSE->FINE:
        #   5 -> 4 -> 3 -> 2 -> 1 -> 0
        coarse_to_fine_level_order = list(range(num_encoder_levels - 1, -1, -1))

        # Heuristic: bias towards mid-resolution contexts (32/64) while still giving capacity to extremes.
        # These weights are in COARSE->FINE order (i.e., aligned to coarse_to_fine_level_order).
        if num_encoder_levels == 6:
            counts_c2f = _allocate_counts(depth, weights=[1, 1, 2, 2, 1, 1])
        else:
            counts_c2f = _allocate_counts(depth, weights=[1] * num_encoder_levels)

        self.block_to_resolution_idx = []
        for pos, enc_level_idx in enumerate(coarse_to_fine_level_order):
            n = counts_c2f[pos]
            self.block_to_resolution_idx.extend([enc_level_idx] * n)
        assert len(self.block_to_resolution_idx) == depth, (len(self.block_to_resolution_idx), depth)

        # Print explicit per-block context assignment (which transformer blocks use which context level).
        # Keep this compact and grouped by level.
        blocks_by_level = {i: [] for i in range(num_encoder_levels)}
        for bidx, ridx in enumerate(self.block_to_resolution_idx):
            blocks_by_level[int(ridx)].append(int(bidx))
        # Effective HW after post-fusion downsampling (if enabled)
        eff_hw = []
        for i, base_hw in enumerate(encoder_spatial_resolutions):
            ratio = 1 if self.fusion_downsample_ratios is None else int(self.fusion_downsample_ratios[i])
            eff_hw.append(int(base_hw // max(1, ratio)))

        # Store context metadata for downstream modules (e.g., skip-conditioned decoders).
        self.num_encoder_levels = int(num_encoder_levels)
        self.encoder_spatial_resolutions = list(int(x) for x in encoder_spatial_resolutions)  # base H=W per level
        self.context_dims_per_level = list(int(x) for x in context_dims_per_level)            # channel dim per level
        self.context_lens_per_level = list(int(x) for x in context_lens_per_level)            # effective (post-ds) token len
        self.context_eff_hw = list(int(x) for x in eff_hw)                                    # effective (post-ds) H=W
        print("  Block -> context level assignment:")
        for i in range(num_encoder_levels):
            hw = eff_hw[i]
            print(
                f"    level {i}: {hw}x{hw} (len={context_lens_per_level[i]}, dim={context_dims_per_level[i]}) "
                f"<= blocks {blocks_by_level[i]}"
            )
        
        # Initialize trainable feature fusion modules for each resolution level.
        # IMPORTANT: if cross-attention is disabled, these modules would be unused
        # (context is always None) and would trigger DDP "unused parameters" errors.
        if not self.disable_cross_attention:
            def _as_per_level(v, name: str):
                if isinstance(v, (list, tuple)):
                    if len(v) != num_encoder_levels:
                        raise ValueError(f"{name} must be length {num_encoder_levels}, got {len(v)}")
                    return list(v)
                return [v] * num_encoder_levels

            fusion_heads = _as_per_level(self.fusion_num_heads, "fusion_num_heads")
            fusion_layers = _as_per_level(self.fusion_num_layers, "fusion_num_layers")
            fusion_inner_dims = _as_per_level(self.fusion_cross_inner_dim, "fusion_cross_inner_dim")
            fusion_rectify = _as_per_level(self.fusion_use_feature_rectify, "fusion_use_feature_rectify")
            fusion_ds_first = _as_per_level(self.fusion_downsample_first, "fusion_downsample_first")

            self.fusion_modules = nn.ModuleList([
                FeatureFusionModule(
                    dim=context_dims_per_level[i],
                    reduction=1,
                    num_heads=int(fusion_heads[i]) if fusion_heads[i] is not None else 8,
                    num_groups=32,
                    downsample_ratio=(1 if self.fusion_downsample_ratios is None else int(self.fusion_downsample_ratios[i])),
                    downsample_first=bool(fusion_ds_first[i]),
                    num_cross_layers=int(fusion_layers[i]),
                    cross_inner_dim=(None if fusion_inner_dims[i] is None else int(fusion_inner_dims[i])),
                    use_feature_rectify=bool(fusion_rectify[i]),
                )
                for i in range(num_encoder_levels)
            ])
        else:
            self.fusion_modules = nn.ModuleList([])
        
        print(f"\n[Multi-Resolution Context Mapping]")
        print(f"  Encoder levels: {num_encoder_levels}")
        print(f"  Spatial resolutions: {encoder_spatial_resolutions}")
        print(f"  Context lengths per level: {context_lens_per_level}")
        print(f"  Context dims per level: {context_dims_per_level}")
        counts_by_level = [0] * num_encoder_levels
        for pos, enc_level_idx in enumerate(coarse_to_fine_level_order):
            counts_by_level[enc_level_idx] = counts_c2f[pos]
        print(f"  Blocks per encoder level: {counts_by_level}  (assigned coarse->fine across depth)")
        print(f"  Block to resolution mapping: {self.block_to_resolution_idx}")
        print(f"  Fusion modules: {len(self.fusion_modules)} trainable modules for context fusion (disable_cross_attention={self.disable_cross_attention})")


        self.blocks = nn.ModuleList([
            AdaLNSABlock(
                cond_dim=self.D, shared_aln=shared_aln,
                context_dim=context_dims_per_level[self.block_to_resolution_idx[block_idx]],
                context_len=context_lens_per_level[self.block_to_resolution_idx[block_idx]],
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                tau=tau, cos_attn=cos_attn,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                disable_cross_attention=self.disable_cross_attention,
                cross_attn_inner_dim=cross_attn_inner_dim,
            ) if self.using_aln else SABlock(
                layer_scale=layer_scale,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                tau=tau, cos_attn=cos_attn,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
                cross_attn_inner_dim=cross_attn_inner_dim,
            )
            for block_idx in range(depth)
        ])

        if self.blocks[-1].fused_add_norm_fn is not None:
            self.gamma2_last = nn.Parameter(self.layer_scale * torch.ones(embed_dim), requires_grad=True) if self.layer_scale >= 0 else 1
        else:
            self.gamma2_last = None

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [vGPT config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d = []
        for i, pn in enumerate(self.patch_nums):
            num_sp_tokens = 1 if (i != 0 and self.separator) else 0
            d.append(torch.full(((pn*pn + num_sp_tokens) * mask_factor,), i))
        d: torch.Tensor = torch.cat(d).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L

        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)

        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)

        if separate_decoding:
            # ignore the upper right half in each stage
            d = []
            dT = []
            for i, pn in enumerate(self.patch_nums):
                num_sp_tokens = 1 if i != 0 and self.separator else 0
                d.extend([torch.full((pn*pn + num_sp_tokens,), 1 + 4 * i,), torch.full((pn*pn + num_sp_tokens,), 3 + 4 * i,)])
                dT.extend([torch.full((pn*pn + num_sp_tokens,), 1 + 4 * i, ), torch.full((pn*pn + num_sp_tokens,), 2 + 4 * i, )])
            d = torch.cat(d).view(1, self.L, 1)
            dT = torch.cat(dT).view(1, 1, self.L)
            attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)

            if self.indep:
                d = []
                dT = []
                for i, pn in enumerate(self.patch_nums):
                    num_sp_tokens = 1 if i != 0 and self.separator else 0
                    d.extend([torch.full((pn * pn + num_sp_tokens,), 3 + 4 * i, ), torch.full((pn * pn + num_sp_tokens,), 1 + 4 * i, )])
                    dT.extend([torch.full((pn * pn + num_sp_tokens,), 2 + 4 * i, ), torch.full((pn * pn + num_sp_tokens,), 0 + 4 * i, )])
                d = torch.cat(d).view(1, self.L, 1)
                dT = torch.cat(dT).view(1, 1, self.L)
                attn_bias_for_masking += torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)

                # attn_bias_for_masking_ = torch.where(attn_bias_for_masking == 0, 0., 255).reshape(1, 1, self.L, self.L)
                # import numpy as np
                # from PIL import Image
                # Image.fromarray(attn_bias_for_masking_.cpu().numpy().astype(np.uint8)[0, 0]).convert('L').save('mask_.png')

        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # 6. classifier head
        num_total_sp_tokens = self.num_stages_minus_1 * mask_factor if self.separator else 0
        if self.using_aln:
            self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
            self.head = nn.Linear(self.C, self.V + num_total_sp_tokens)
        else:
            self.head_nm = MultiInpIdentity()
            self.head = nn.Sequential(norm_layer(self.C), nn.Linear(self.C, self.V + num_total_sp_tokens))
        if self.separator:
            self.special_embed = nn.Embedding(self.num_stages_minus_1 * self.mask_factor, self.C)  # skip the first stage
            nn.init.trunc_normal_(self.special_embed.weight.data, mean=0, std=init_std)
        if self.multi_cond:
            self.cond_embed = nn.Embedding(5, self.C)
            nn.init.trunc_normal_(self.cond_embed.weight.data, mean=0, std=init_std)

    def encode_context_with_fusion(self, images: list) -> list:
        """
        Encode context from pre and post images and fuse them per-resolution.

        By default (allow_trainable_encoder=False), this uses the *frozen* VQVAE encoder under
        torch.no_grad() to avoid any chance of impacting GT tokenization.

        If allow_trainable_encoder=True, it uses a trainable *copy* of the VQVAE encoder
        (self.trainable_encoder) so gradients flow into that copy + fusion modules, while the
        original VQVAE remains frozen for img_to_idxBl().
        
        Args:
            images: List of [pre_image, post_image] tensors
            
        Returns:
            List of fused context tensors, one per encoder level (down-path levels + post-middle).
        """
        if self.disable_cross_attention:
            raise RuntimeError(
                "encode_context_with_fusion() was called while disable_cross_attention=True. "
                "This should not happen: pass context=None and skip context encoding when cross-attention is disabled."
            )
        if len(self.fusion_modules) == 0:
            raise RuntimeError(
                "Fusion modules are not initialized, but encode_context_with_fusion() was called. "
                "Did you disable cross-attention?"
            )
        vae = self.vae_proxy[0]
        pre_image = images[0]
        post_image = images[1]
        
        # Get multi-resolution contexts from encoder
        # Each context is in BLC format (Batch, Length, Channels)
        if self.allow_trainable_encoder:
            if self.trainable_encoder is None:
                raise RuntimeError("allow_trainable_encoder=True but trainable_encoder is None.")
            pre_contexts = self.trainable_encoder.forward_context(
                pre_image, return_all_levels=self.use_high_res_context_levels
            )
            post_contexts = self.trainable_encoder.forward_context(
                post_image, return_all_levels=self.use_high_res_context_levels
            )
        else:
            with torch.no_grad():
                pre_contexts = vae.encoder.forward_context(
                    pre_image, return_all_levels=self.use_high_res_context_levels
                )
                post_contexts = vae.encoder.forward_context(
                    post_image, return_all_levels=self.use_high_res_context_levels
                )
        if len(pre_contexts) != len(self.fusion_modules):
            raise RuntimeError(
                f"Expected {len(self.fusion_modules)} context levels from encoder, got {len(pre_contexts)}. "
                f"Did you change encoder context levels without updating RemoteVAR?"
            )
        
        # Apply trainable fusion modules to fuse pre and post contexts
        # Fusion modules expect BCHW format, so we need to reshape
        fused_contexts = []
        for i, (pre_ctx, post_ctx) in enumerate(zip(pre_contexts, post_contexts)):
            # pre_ctx and post_ctx are in BLC format (B, H*W, C)
            B, L, C = pre_ctx.shape
            H = W = int(L ** 0.5)  # Assume square spatial dimensions
            
            # Reshape to BCHW for fusion module
            pre_ctx_2d = pre_ctx.transpose(1, 2).reshape(B, C, H, W)
            post_ctx_2d = post_ctx.transpose(1, 2).reshape(B, C, H, W)
            
            # Apply fusion module (returns BCHW)
            fused = self.fusion_modules[i](pre_ctx_2d, post_ctx_2d)
            
            # Reshape back to BLC format for cross-attention
            fused_blc = fused.flatten(2).transpose(1, 2)  # B, H*W, C
            fused_contexts.append(fused_blc)
        
        return fused_contexts

    def encode_context_with_fusion_2d(self, images: list) -> list:
        """
        Same as `encode_context_with_fusion`, but returns fused context tensors in BCHW format
        (one per encoder level), suitable for UNet-style skip connections in a conditioned decoder.
        """
        if self.disable_cross_attention:
            raise RuntimeError(
                "encode_context_with_fusion_2d() was called while disable_cross_attention=True. "
                "Fusion modules are not initialized in this mode."
            )
        if len(self.fusion_modules) == 0:
            raise RuntimeError(
                "Fusion modules are not initialized, but encode_context_with_fusion_2d() was called. "
                "Did you disable cross-attention?"
            )
        vae = self.vae_proxy[0]
        pre_image = images[0]
        post_image = images[1]

        # Get multi-resolution contexts from encoder (BLC), then reshape to BCHW for fusion.
        if self.allow_trainable_encoder:
            if self.trainable_encoder is None:
                raise RuntimeError("allow_trainable_encoder=True but trainable_encoder is None.")
            pre_contexts = self.trainable_encoder.forward_context(
                pre_image, return_all_levels=self.use_high_res_context_levels
            )
            post_contexts = self.trainable_encoder.forward_context(
                post_image, return_all_levels=self.use_high_res_context_levels
            )
        else:
            with torch.no_grad():
                pre_contexts = vae.encoder.forward_context(
                    pre_image, return_all_levels=self.use_high_res_context_levels
                )
                post_contexts = vae.encoder.forward_context(
                    post_image, return_all_levels=self.use_high_res_context_levels
                )
        if len(pre_contexts) != len(self.fusion_modules):
            raise RuntimeError(
                f"Expected {len(self.fusion_modules)} context levels from encoder, got {len(pre_contexts)}. "
                f"Did you change encoder context levels without updating RemoteVAR?"
            )

        fused_contexts_2d = []
        for i, (pre_ctx, post_ctx) in enumerate(zip(pre_contexts, post_contexts)):
            B, L, C = pre_ctx.shape
            H = W = int(L ** 0.5)
            pre_ctx_2d = pre_ctx.transpose(1, 2).reshape(B, C, H, W)
            post_ctx_2d = post_ctx.transpose(1, 2).reshape(B, C, H, W)
            fused = self.fusion_modules[i](pre_ctx_2d, post_ctx_2d)  # BCHW (possibly downsampled)
            fused_contexts_2d.append(fused)
        return fused_contexts_2d
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # is h_and_residual, so fused_add_norm must be used, so self.gamma2_last is not None
            h = resi + self.gamma2_last * self.blocks[-1].drop_path(h)
        else:   # is h, so fused_add_norm is not used, and self.gamma2_last is None
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    @torch.no_grad()
    def conditional_infer_cfg(
            self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
            g_seed: Optional[int] = None, cfg=(1.5, 1.5, 1.5), top_k=0, top_p=0.0,
            more_smooth=False, cond_type=None, c_mask=None, c_img_pre=None, c_img_post=None, context: Optional[torch.Tensor] = None,
            return_confidence: bool = False,
            return_confidence_all: bool = False,
            confidence_agg: str = "mean",
            return_intermediate: bool = False,
            return_mask_fhat: bool = False,
            decoder_skips: Optional[list] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], list], Tuple[torch.Tensor, list]]:  # returns reconstructed image (B, C, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param return_mask_fhat: return the final change-mask latent with the reconstructed output
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC

        if label_B is None:
            label_B = torch.multinomial(self.selecting_idx, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B,
                                 device=self.lvl_1L.device)
        # Decide whether to run CFG branches. If all CFG weights are zero, skip batch replication.
        try:
            _cfg_vals = [float(x) for x in cfg]
        except Exception:
            _cfg_vals = [float(cfg)]
        use_cfg = any(abs(x) > 1e-12 for x in _cfg_vals)

        if use_cfg:
            empty_cls = torch.full_like(label_B, fill_value=self.num_classes)
            # p(c1|c2,C,I)p(c2|C,I)p(C|I)p(I)
            label_B = torch.cat((label_B, empty_cls, empty_cls, empty_cls), dim=0)
        sos = cond_BD = self.class_emb(label_B)

        empty_cond_type = torch.full((B,), fill_value=4, device=self.lvl_1L.device).long()
        if cond_type is None:
            cond_type = empty_cond_type
        if use_cfg:
            # p(c1|c2,C,I)p(c2|C,I)p(C|I)p(I)
            cond_type = torch.concat([cond_type, cond_type, empty_cond_type, empty_cond_type], dim=0)
        sos = sos.unsqueeze(1)
        cond_token = self.cond_embed(cond_type).unsqueeze(1)
        next_token_map = torch.concat([sos, sos, cond_token], dim=1)

        repeat_num = label_B.shape[0] // B
        next_token_map = next_token_map + self.pos_start.expand(repeat_num * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        # Expand context for CFG (repeat each context tensor repeat_num times)
        if context is not None and repeat_num > 1:
            context = [ctx.repeat(repeat_num, 1, 1) if ctx is not None else None for ctx in context]

        for b in self.blocks: b.attn.kv_caching(True)

        # Decoder/vae handle (used for final reconstruction and optional intermediate visualizations).
        vae = self.vae_proxy[0]
        out_ch = 3
        try:
            if hasattr(vae, "decoder") and hasattr(vae.decoder, "conv_out") and hasattr(vae.decoder.conv_out, "out_channels"):
                out_ch = int(vae.decoder.conv_out.out_channels)
        except Exception:
            out_ch = 3
        intermediate_mask_imgs: Optional[list] = [] if return_intermediate else None  # list[(B,C,H,W)] in [0,1], mask-stream only

        cur_L = 0
        num_sp_token = 0
        f_hat = sos.new_zeros(repeat_num * B, self.Cvae, self.patch_nums[-1] * self.mask_factor, self.patch_nums[-1])
        conf_mask_last_B1pp: Optional[torch.Tensor] = None  # (B,1,pn,pn) at final stage, mask tokens only (normalized predictive entropy)
        conf_masks_per_stage_B1pp = [] if bool(return_confidence_all) else None  # list[Optional[(B,1,pn,pn)]]
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            cur_L += (pn * pn + num_sp_token) * self.mask_factor
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            SABlock.forward
            x = next_token_map

            # If enabled, inject CURRENT-SCALE token embeddings for pre/post into the transformer input
            # so mask tokens at this stage can attend to them (instead of only previous-scale f_hat).
            # Implemented for mask_factor==3 (change_append layout: [pre, post, mask]).
            if self.enable_current_scale_tokens and self.mask_factor == 3 and (c_img_pre is not None or c_img_post is not None):
                stage_len = (pn * pn + num_sp_token) * self.mask_factor
                stage_start = cur_L - stage_len

                # Stage positional embedding that was already added into x
                if stage_start == 0:
                    # stage0 included pos_start as well
                    stage_pos = self.pos_start.expand(repeat_num * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
                else:
                    stage_pos = lvl_pos[:, stage_start:cur_L]  # (1, stage_len, C)

                x_wo_pos = x - stage_pos
                x_wo_pos = x_wo_pos.clone()

                # pre segment: [0 : pn^2], post segment: [pn^2 : 2*pn^2]
                if c_img_pre is not None:
                    pre_idx = c_img_pre[si]
                    # Add (not replace): match training where we use cumulative f_hat plus the *current-scale* GT residual.
                    q = self.vae_quant_proxy[0]  # VectorQuantizer2
                    SN = len(self.patch_nums)
                    HW = int(self.patch_nums[-1])
                    h = q.embedding(pre_idx).transpose(1, 2).view(B, self.Cvae, pn, pn)
                    h_up = torch.nn.functional.interpolate(h, size=(HW, HW), mode="bicubic")
                    h_res = q.quant_resi[si / (SN - 1)](h_up)
                    h_dn = torch.nn.functional.interpolate(h_res, size=(pn, pn), mode="area")
                    delta = h_dn.view(B, self.Cvae, -1).transpose(1, 2)  # (B, pn^2, Cvae)
                    delta = self.word_embed(delta)  # (B, pn^2, C)
                    delta = delta.repeat(repeat_num, 1, 1)
                    x_wo_pos[:, : pn * pn] = x_wo_pos[:, : pn * pn] + delta

                if c_img_post is not None:
                    post_idx = c_img_post[si]
                    q = self.vae_quant_proxy[0]
                    SN = len(self.patch_nums)
                    HW = int(self.patch_nums[-1])
                    h = q.embedding(post_idx).transpose(1, 2).view(B, self.Cvae, pn, pn)
                    h_up = torch.nn.functional.interpolate(h, size=(HW, HW), mode="bicubic")
                    h_res = q.quant_resi[si / (SN - 1)](h_up)
                    h_dn = torch.nn.functional.interpolate(h_res, size=(pn, pn), mode="area")
                    delta = h_dn.view(B, self.Cvae, -1).transpose(1, 2)
                    delta = self.word_embed(delta)
                    delta = delta.repeat(repeat_num, 1, 1)
                    x_wo_pos[:, pn * pn : 2 * pn * pn] = x_wo_pos[:, pn * pn : 2 * pn * pn] + delta

                x = x_wo_pos + stage_pos

            for block_idx, b in enumerate(self.blocks):
                res_idx = self.block_to_resolution_idx[block_idx]
                # Set context to None if cross-attention is disabled
                block_context = None if self.disable_cross_attention else context[res_idx]
                x = b(x=x, cond_BD=cond_BD_or_gss, context=block_context, attn_bias=None if not self.indep else
                self.attn_bias_for_masking[:, :, (cur_L - (pn * pn + num_sp_token) * self.mask_factor):cur_L, :cur_L])
            logits_BlV = self.get_logits(x, cond_BD)
            # class, cond_type, pixel_cond
            # [c1, c2, C], [x, c2, C], [x, x, C], [x, x, x]
            t1, t2, t3 = cfg[0] * ratio, cfg[1] * ratio, cfg[2] * ratio

            if repeat_num == 4:
                # logits_BlV = t1 * logits_BlV[:B] \
                #              + (t2 - t1) * logits_BlV[B:2 * B] \
                #              + (t3 - t2) * logits_BlV[2 * B:3 * B] \
                #              + (1 - t3) * logits_BlV[-B:]
                logits_BlV = (1 + t1) * logits_BlV[:B] \
                             + (t2 - t1) * logits_BlV[B:2 * B] \
                             + (t3 - t2) * logits_BlV[2 * B:3 * B] \
                             - t3 * logits_BlV[-B:]
            elif repeat_num == 3:
                # logits_BlV = t1 * logits_BlV[:B] \
                #              + (t2 - t1) * logits_BlV[B:2 * B] \
                #              + (1 - t2) * logits_BlV[-B:]
                logits_BlV = (1 + t1) * logits_BlV[:B] \
                             + (t2 - t1) * logits_BlV[B:2 * B] \
                             - t2 * logits_BlV[-B:]

            # Optional: predictive entropy map from the *unfiltered* distribution (before top-k/top-p).
            # Entropy is a property of the whole distribution and does NOT depend on the sampled idxs.
            # To make it visible for large vocabularies, we min-max normalize entropy per-sample over mask-token positions.
            if bool(return_confidence_all) or (return_confidence and si == self.num_stages_minus_1):
                conf_stage_B1pp: Optional[torch.Tensor] = None
                try:
                    logits_full = logits_BlV[:, :, :self.V].float()  # (B, stage_len, V), UNFILTERED
                    lse = torch.logsumexp(logits_full, dim=-1)        # (B, stage_len)
                    probs = torch.softmax(logits_full, dim=-1)        # (B, stage_len, V)
                    exp_logit = (probs * logits_full).sum(dim=-1)     # E_p[logit]
                    ent = (lse - exp_logit).clamp_min(0.0)            # H(p)

                    seg = pn * pn + int(num_sp_token)
                    if self.mask_factor == 3:
                        mask_start = 2 * seg
                    elif self.mask_factor == 2:
                        mask_start = 1 * seg
                    else:
                        mask_start = 0
                    mask_end = mask_start + pn * pn
                    if mask_end <= ent.shape[1]:
                        ent_mask = ent[:, mask_start:mask_end]  # (B, pn*pn)
                        ent_min = ent_mask.min(dim=1, keepdim=True).values
                        ent_max = ent_mask.max(dim=1, keepdim=True).values
                        ent_norm = (ent_mask - ent_min) / (ent_max - ent_min + 1e-8)
                        conf_stage_B1pp = ent_norm.contiguous().view(B, 1, pn, pn).clamp(0.0, 1.0)
                except Exception:
                    conf_stage_B1pp = None

                if conf_masks_per_stage_B1pp is not None:
                    conf_masks_per_stage_B1pp.append(conf_stage_B1pp)
                if si == self.num_stages_minus_1:
                    conf_mask_last_B1pp = conf_stage_B1pp

            logits_BlV = logits_BlV.repeat(repeat_num, 1, 1)
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

            # if c_mask is not None :  # Teaching force
            #     if repeat_num == 4:
            #         idx_Bl[:B, :pn * pn] = c_mask[si]
            #         idx_Bl[B:2 * B, :pn * pn] = c_mask[si]
            #         idx_Bl[2 * B:3 * B, :pn * pn] = c_mask[si]
            #     elif repeat_num == 3:
            #         idx_Bl[:B, :pn * pn] = c_mask[si]
            #         idx_Bl[B:2 * B, :pn * pn] = c_mask[si]
            if c_img_pre is not None:
                if repeat_num == 1:
                    idx_Bl[:, :pn * pn] = c_img_pre[si]
                elif repeat_num == 4:
                    idx_Bl[:B, :pn * pn] = c_img_pre[si]
                    idx_Bl[B:2 * B, :pn * pn] = c_img_pre[si]
                    idx_Bl[2 * B:3 * B, :pn * pn] = c_img_pre[si]
                elif repeat_num == 3:
                    idx_Bl[:B, :pn * pn] = c_img_pre[si]
                    idx_Bl[B:2 * B, :pn * pn] = c_img_pre[si]
            if c_img_post is not None:
                if repeat_num == 1:
                    idx_Bl[:, pn*pn: 2 * pn * pn] = c_img_post[si]
                elif repeat_num == 4:
                    idx_Bl[:B, pn*pn: 2 * pn * pn] = c_img_post[si]
                    idx_Bl[B:2 * B, pn*pn:2 * pn * pn] = c_img_post[si]
                    idx_Bl[2 * B:3 * B, pn*pn:2 * pn * pn] = c_img_post[si]
                elif repeat_num == 3:
                    idx_Bl[:B, pn*pn:2 * pn * pn] = c_img_post[si]
                    idx_Bl[B:2 * B, pn*pn:2 * pn * pn] = c_img_post[si]

            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1,
                                                 rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2)
            h_BChw_1 = h_BChw[:, :, :pn * pn].reshape(repeat_num * B, self.Cvae, pn, pn)  # first part
            h_BChw_2 = h_BChw[:, :, pn * pn: 2*pn * pn].reshape(repeat_num * B, self.Cvae, pn, pn)  # second part
            h_BChw_3 = h_BChw[:, :, 2*pn * pn:].reshape(repeat_num * B, self.Cvae, pn, pn)  # third part
            f_hat_1 = f_hat[:, :, :self.patch_nums[-1], :]
            f_hat_2 = f_hat[:, :, self.patch_nums[-1]: 2*self.patch_nums[-1], :]
            f_hat_3 = f_hat[:, :, 2*self.patch_nums[-1]:, :]
            f_hat_1, next_token_map_1 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_1, h_BChw_1)
            f_hat_2, next_token_map_2 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_2, h_BChw_2)
            f_hat_3, next_token_map_3 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_3, h_BChw_3)
            f_hat = torch.concat((f_hat_1, f_hat_2, f_hat_3), dim=2)  # [b, c, 2pn, pn]

            # Optional: decode intermediate mask reconstructions after each autoregressive stage.
            # This is purely for visualization/debugging and is disabled by default to avoid extra compute.
            if return_intermediate and intermediate_mask_imgs is not None:
                try:
                    f3_vis = f_hat_3[:B]  # only the conditional branch (skip CFG replicas)
                    if out_ch == 1:
                        logits3 = vae.decoder(vae.post_quant_conv(f3_vis), skips=decoder_skips)  # (B,1,H,W)
                        img3_si = torch.sigmoid(logits3.float())
                    else:
                        img3_si = vae.fhat_to_img(f3_vis, decoder_skips=decoder_skips).add_(1).mul_(0.5)  # (B,3,H,W) in [0,1]
                    intermediate_mask_imgs.append(img3_si)
                except Exception:
                    # Best-effort: intermediate visualization should never crash generation.
                    pass

            next_token_map_1 = next_token_map_1.view(repeat_num * B, self.Cvae, -1).transpose(1, 2)
            next_token_map_2 = next_token_map_2.view(repeat_num * B, self.Cvae, -1).transpose(1, 2)
            next_token_map_3 = next_token_map_3.view(repeat_num * B, self.Cvae, -1).transpose(1, 2)
            next_token_map = torch.concat((next_token_map_1, next_token_map_2, next_token_map_3), dim=1)  # [b, c, 2pn, pn]
            next_token_map = self.word_embed(next_token_map)
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map + lvl_pos[:, cur_L:cur_L + (self.patch_nums[si + 1] ** 2) * self.mask_factor]

        f_hat_1 = f_hat_1[:B]
        f_hat_2 = f_hat_2[:B]
        f_hat_3 = f_hat_3[:B]
        for b in self.blocks: b.attn.kv_caching(False)
        img1 = vae.fhat_to_img(f_hat_1).add_(1).mul_(0.5)
        img2 = vae.fhat_to_img(f_hat_2).add_(1).mul_(0.5)

        # Optionally use external decoder skips (e.g., from fusion modules) ONLY for the mask stream.
        #
        # Special case: if the decoder was refined to output 1 channel (binary mask logits),
        # decode WITHOUT clamp and apply sigmoid to produce a probability map in [0,1].
        # This matches how the refiner is trained (BCEWithLogits on raw decoder output).
        if out_ch == 1:
            logits3 = vae.decoder(vae.post_quant_conv(f_hat_3), skips=decoder_skips)  # (B,1,H,W)
            img3 = torch.sigmoid(logits3.float()).to(dtype=img1.dtype)
        else:
            img3 = vae.fhat_to_img(f_hat_3, decoder_skips=decoder_skips).add_(1).mul_(0.5)
        out = torch.concat([img1, img2, img3], dim=2)  # de-normalize, from [-1, 1] to [0, 1]

        mask_fhat = None
        if bool(return_mask_fhat):
            if int(getattr(self, "mask_factor", 0)) != 3:
                raise NotImplementedError(
                    "return_mask_fhat is implemented only for change_append (mask_factor=3)."
                )
            mask_fhat = f_hat_3.contiguous()

        if bool(return_confidence_all):
            # Per-scale entropy maps (and aggregated) for visualization/debugging.
            # This is optional and kept off by default to preserve original behavior/perf.
            conf_map_last = None
            if conf_mask_last_B1pp is not None:
                conf_map_last = F.interpolate(
                    conf_mask_last_B1pp.float(),
                    size=img3.shape[-2:],
                    mode="nearest",
                ).clamp(0, 1)

            conf_maps_per_stage = []
            if conf_masks_per_stage_B1pp is not None:
                for c in conf_masks_per_stage_B1pp:
                    if c is None:
                        conf_maps_per_stage.append(None)
                    else:
                        conf_maps_per_stage.append(
                            F.interpolate(c.float(), size=img3.shape[-2:], mode="nearest").clamp(0, 1)
                        )

            conf_map_agg = None
            valid = [m for m in conf_maps_per_stage if m is not None]
            if len(valid) > 0:
                stack = torch.stack(valid, dim=0)  # (S,B,1,H,W)
                mode = str(confidence_agg or "mean").strip().lower()
                if mode == "max":
                    conf_map_agg = stack.max(dim=0).values
                else:
                    conf_map_agg = stack.mean(dim=0)
                # normalize per sample for visibility (same spirit as per-scale min-max)
                vmin = conf_map_agg.amin(dim=(-2, -1), keepdim=True)
                vmax = conf_map_agg.amax(dim=(-2, -1), keepdim=True)
                conf_map_agg = ((conf_map_agg - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1)

            if return_intermediate:
                result = (
                    out,
                    conf_map_last,
                    conf_map_agg,
                    conf_maps_per_stage,
                    intermediate_mask_imgs if intermediate_mask_imgs is not None else [],
                )
            else:
                result = (out, conf_map_last, conf_map_agg, conf_maps_per_stage)
            return (*result, mask_fhat) if mask_fhat is not None else result

        if return_confidence:
            conf_map = None
            if conf_mask_last_B1pp is not None:
                # Upsample token-grid confidence (pn x pn) to pixel space (H x W) for visualization.
                conf_map = F.interpolate(
                    conf_mask_last_B1pp.float(),
                    size=img3.shape[-2:],
                    mode="nearest",
                ).clamp(0, 1)
            if return_intermediate:
                result = (
                    out,
                    conf_map,
                    intermediate_mask_imgs if intermediate_mask_imgs is not None else [],
                )
            else:
                result = (out, conf_map)
            return (*result, mask_fhat) if mask_fhat is not None else result
        if return_intermediate:
            result = (out, intermediate_mask_imgs if intermediate_mask_imgs is not None else [])
            return (*result, mask_fhat) if mask_fhat is not None else result
        if mask_fhat is not None:
            return out, mask_fhat
        return out

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, cond_type=None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.selecting_idx, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        mask_first = False
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC

        if self.multi_cond:
            if cond_type is None:
                if B == 4:
                    cond_type = torch.tensor([0, 1, 2, 3], device=dist.get_device())
                    uncond_type = torch.tensor([4, 4, 4, 4], device=dist.get_device())
                else:
                    cond_idx = torch.full((1, 4), fill_value=1 / 4, dtype=torch.float32, device=dist.get_device())
                    cond_type = torch.multinomial(cond_idx, num_samples=B, replacement=True, generator=rng).reshape(B)
                    uncond_type = torch.full((B,), fill_value=4, device=self.lvl_1L.device)
            elif isinstance(cond_type, int):
                assert cond_type <= 3 and cond_type > 0
                cond_type = torch.full((B,), fill_value=cond_type, device=self.lvl_1L.device)
                uncond_type = torch.full((B,), fill_value=4, device=self.lvl_1L.device)
            else:
                uncond_type = torch.full((B,), fill_value=4, device=self.lvl_1L.device).long()
            cond_type = torch.concat([cond_type, uncond_type], dim=0)  # copy for cfg
            sos = sos.unsqueeze(1)
            cond_token = self.cond_embed(cond_type).unsqueeze(1)
            mask_first = False
            if mask_first:
                next_token_map = torch.concat([cond_token, sos, sos], dim=1)
            else:
                next_token_map = torch.concat([sos, sos, cond_token], dim=1)


            next_token_map = next_token_map + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        # else:
        #     if self.mask_factor == 3:  # random shuffle mask and image sos
        #         ch_sign = sos.new_ones(2 * B, self.first_l // 2, 1)
        #         sign = random.choice([-1, 1])
        #         ch_sign = torch.cat([ch_sign * sign, -ch_sign * sign], dim=1)
        #         mask_first = False
        #         next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) * ch_sign + \
        #                          self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        #     else:
        #         next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + \
        #                          self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        if self.type_pos:
            type_pos = self.type_embed(self.type_1L.expand(B, -1)) if mask_first else self.type_embed(self.type_1L_.expand(B, -1))

        for b in self.blocks: b.attn.kv_caching(True)

        if self.separate_decoding and not self.indep:
            cur_L = 0
            next_token_map_1 = next_token_map[:, :self.patch_nums[0]]
            next_token_map_2 = next_token_map[:, self.patch_nums[0]:]
            f_hat_1 = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
            f_hat_2 = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
            iter_patch_nums = list(chain.from_iterable(zip(self.patch_nums, self.patch_nums)))
            num_sp_token = 0
            for si, pn in enumerate(iter_patch_nums):  # si: i-th segment
                ratio = (si // 2) / self.num_stages_minus_1
                cur_L += pn * pn + num_sp_token
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                SABlock.forward
                if si == 0:
                    x = next_token_map_1
                elif si == 1:
                    x = next_token_map_2
                else:
                    x = next_token_map
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.get_logits(x, cond_BD)
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                logits_BlV = logits_BlV[:, :, :self.V]  # ignore special tokens
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

                if si > 1 and self.separator:
                    idx_Bl = idx_Bl[:, :-1]  # discard special token if used
                    num_sp_token = 1
                if not more_smooth:
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1,
                                                     rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                if si % 2 == 0:
                    f_hat_1, _ = self.vae_quant_proxy[0].get_next_autoregressive_input(si//2, len(self.patch_nums), f_hat_1, h_BChw)
                    next_token_map = F.interpolate(f_hat_1, size=(iter_patch_nums[si+1], iter_patch_nums[si+1]), mode='area')
                else:
                    f_hat_2, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si//2, len(self.patch_nums), f_hat_2, h_BChw)

                if si != len(iter_patch_nums) - 1:  # prepare for next stage
                    next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                    if self.separator and si >= 1:
                        # teaching force
                        mapping = [i for i in range(18)] if mask_first else [i + 1 if i % 2 == 0 else i - 1 for i in range(18)]
                        special_token = self.special_embed(torch.full((B,), fill_value=mapping[si-1], device=sos.device, dtype=torch.long))
                        next_token_map = torch.concat((self.word_embed(next_token_map), special_token.unsqueeze(1)), dim=1)
                    else:
                        next_token_map = self.word_embed(next_token_map)
                    next_token_map = next_token_map + lvl_pos[:, cur_L:cur_L + iter_patch_nums[si + 1] ** 2 + num_sp_token]
                    if self.type_pos:
                        next_token_map = next_token_map + type_pos[:, cur_L:cur_L + (self.patch_nums[si + 1] ** 2 + num_sp_token) * self.mask_factor]
                    next_token_map = next_token_map.repeat(2, 1, 1)  # double the batch sizes due to CFG

        else:
            cur_L = 0
            num_sp_token = 0
            f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1] * self.mask_factor, self.patch_nums[-1])
            for si, pn in enumerate(self.patch_nums):   # si: i-th segment
                ratio = si / self.num_stages_minus_1
                cur_L += (pn*pn + num_sp_token) * self.mask_factor
                cond_BD_or_gss = self.shared_ada_lin(cond_BD)
                SABlock.forward
                x = next_token_map
                for b in self.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None if not self.indep else
                    self.attn_bias_for_masking[:, :, (cur_L-(pn * pn + num_sp_token) * self.mask_factor):cur_L, :cur_L])
                logits_BlV = self.get_logits(x, cond_BD)

                t = cfg * ratio
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                logits_BlV = logits_BlV[:, :, :self.V]  # ignore special tokens if used
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]

                if si > 1 and self.separator:
                    idx_Bl_ = torch.concat((idx_Bl[:, :pn*pn], idx_Bl[:, pn*pn + 1:pn*pn*2 + 1],), dim=1)  # remove special tokens
                    idx_Bl = idx_Bl_

                if not more_smooth:
                    h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
                else:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                    h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

                assert self.mask_factor <= 3, 'current visualization only support mask_factor == 2 or 1 or 3'
                if self.mask_factor == 1:
                    h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                    f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
                elif self.mask_factor == 2:
                    h_BChw = h_BChw.transpose_(1, 2)
                    h_BChw_1 = h_BChw[:, :, :pn*pn].reshape(B, self.Cvae, pn, pn)  # first part
                    h_BChw_2 = h_BChw[:, :, -pn*pn:].reshape(B, self.Cvae, pn, pn)  # second part
                    f_hat_1 = f_hat[:, :, :self.patch_nums[-1], :]
                    f_hat_2 = f_hat[:, :, self.patch_nums[-1]:, :]
                    f_hat_1, next_token_map_1 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_1, h_BChw_1)
                    f_hat_2, next_token_map_2 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_2, h_BChw_2)
                    f_hat = torch.concat((f_hat_1, f_hat_2), dim=2)  # [b, c, 2pn, pn]
                    next_token_map = torch.concat((next_token_map_1, next_token_map_2), dim=2)
                elif self.mask_factor == 3:
                    h_BChw = h_BChw.transpose_(1, 2)
                    h_BChw_1 = h_BChw[:, :, :pn*pn].reshape(B, self.Cvae, pn, pn)  # first part
                    h_BChw_2 = h_BChw[:, :, pn*pn: 2* pn*pn].reshape(B, self.Cvae, pn, pn)  # second part
                    h_BChw_3 = h_BChw[:, :, 2* pn*pn:].reshape(B, self.Cvae, pn, pn)  # third part
                    f_hat_1 = f_hat[:, :, :self.patch_nums[-1], :]
                    f_hat_2 = f_hat[:, :, self.patch_nums[-1]: 2* self.patch_nums[-1], :]
                    f_hat_3 = f_hat[:, :, 2* self.patch_nums[-1]:, :]
                    f_hat_1, next_token_map_1 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_1, h_BChw_1)
                    f_hat_2, next_token_map_2 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_2, h_BChw_2)
                    f_hat_3, next_token_map_3 = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat_3, h_BChw_3)
                    f_hat = torch.concat((f_hat_1, f_hat_2, f_hat_3), dim=2)  # [b, c, 3pn, pn]
                    next_token_map = torch.concat((next_token_map_1, next_token_map_2, next_token_map_3), dim=2)
                else:

                    raise NotImplementedError

                if si != self.num_stages_minus_1:   # prepare for next stage
                    if self.mask_factor == 1:
                        next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)

                    elif self.mask_factor == 2:
                        next_token_map_1, next_token_map_2 = next_token_map[:, :, :pn, :], next_token_map[:, :, pn:, :]
                        next_token_map_1 = next_token_map_1.view(B, self.Cvae, -1).transpose(1, 2)
                        next_token_map_2 = next_token_map_2.view(B, self.Cvae, -1).transpose(1, 2)

                        if self.separator:
                            mapping = [i for i in range(18)] if mask_first else [i + 1 if i % 2 == 0 else i - 1 for i in range(18)]
                            label1, label2 = mapping[2 * si] + self.V, mapping[2 * si + 1] + self.V
                            label1, label2 = sos.new_ones(B, ) * label1, sos.new_ones(B, ) * label2
                            label1, label2 = label1.unsqueeze(1), label2.unsqueeze(1)
                            special_token1, special_token2 = self.special_embed(label1.long()), self.special_embed(label2.long())
                            next_token_map_1 = self.word_embed(next_token_map_1)
                            next_token_map_2 = self.word_embed(next_token_map_2)
                            next_token_map = torch.concat((next_token_map_1, special_token1, next_token_map_2, special_token2), dim=1)
                            num_sp_token = 1
                        else:
                            next_token_map = torch.concat((next_token_map_1, next_token_map_2), dim=1)  # [b, c, 2pn, pn]
                            next_token_map = self.word_embed(next_token_map)
                    elif self.mask_factor == 3:
                        next_token_map_1, next_token_map_2, next_token_map_3 = next_token_map[:, :, :pn, :], next_token_map[:, :, pn:pn*2, :], next_token_map[:, :, pn*2:, :]
                        next_token_map_1 = next_token_map_1.view(B, self.Cvae, -1).transpose(1, 2)
                        next_token_map_2 = next_token_map_2.view(B, self.Cvae, -1).transpose(1, 2)
                        next_token_map_3 = next_token_map_3.view(B, self.Cvae, -1).transpose(1, 2)
            
                        next_token_map = torch.concat((next_token_map_1, next_token_map_2, next_token_map_3), dim=1)  # [b, c, 3pn, pn]
                        next_token_map = self.word_embed(next_token_map)

                    next_token_map = next_token_map + lvl_pos[:, cur_L:cur_L + (self.patch_nums[si+1] ** 2 + num_sp_token) * self.mask_factor]
                    if self.type_pos:
                        next_token_map = next_token_map + type_pos[:, cur_L:cur_L + (self.patch_nums[si + 1] ** 2 + num_sp_token) * self.mask_factor]
                    next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        for b in self.blocks: b.attn.kv_caching(False)
        img1 = self.vae_proxy[0].fhat_to_img(f_hat_1).add_(1).mul_(0.5)
        img2 = self.vae_proxy[0].fhat_to_img(f_hat_2).add_(1).mul_(0.5)
        img3 = self.vae_proxy[0].fhat_to_img(f_hat_3).add_(1).mul_(0.5)
        return torch.concat([img1, img2, img3], dim=2)   # de-normalize, from [-1, 1] to [0, 1]


    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, cond_type, context: torch.Tensor, mask_first=True) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]

        # SAFETY: KV caching is intended only for autoregressive inference.
        # If an inference call throws before it can disable caching, stale cached_k/v can leak into training
        # and cause shape-mismatch errors (e.g., when batch size changes).
        if self.training:
            for blk in self.blocks:
                if hasattr(blk, "attn") and getattr(blk.attn, "caching", False):
                    blk.attn.kv_caching(False)
                if hasattr(blk, "cross_attn") and blk.cross_attn is not None and getattr(blk.cross_attn, "caching", False):
                    blk.cross_attn.kv_caching(False)

        with torch.autocast(device_type=label_B.device.type, enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)

            if self.multi_cond and self.mask_factor == 2 or self.mask_factor == 3:
                sos = sos.unsqueeze(1).expand(B, 1, -1)
                # 0: mask, 1: canny, 2: depth, 3: normal, 4: uncond
                cond_type = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, 4, cond_type)
                cond_token = self.cond_embed(cond_type)
                cond_token = cond_token.unsqueeze(1).expand(B, 1, -1)
                sos = torch.concat([cond_token, sos, sos], dim=1) if mask_first else torch.concat([sos, sos, cond_token], dim=1)
                sos = sos + self.pos_start.expand(B, self.first_l, -1)

            # else:
            #     if self.bidirectional and self.mask_factor == 2:  # random shuffle mask and image sos
            #             sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            #             ch_sign = sos.new_ones(B, self.first_l // 2, 1)
            #             sign = -1 if mask_first else 1
            #             ch_sign = torch.cat([ch_sign * sign, -ch_sign * sign], dim=1)
            #             sos = sos * ch_sign
            #     else:
            #         sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                if self.separator:
                    mapping = [i for i in range(18)] if mask_first else [i + 1 if i % 2 == 0 else i - 1 for i in range(18)]
                    x_BLC = self.word_embed(x_BLCv_wo_first_l.float())
                    new_x = [sos,]
                    cur = 0
                    for si, pn in enumerate(self.patch_nums[1:]):  # skip first
                        label1, label2 = mapping[2 * si] + self.V, mapping[2 * si+1] + self.V
                        label1, label2 = x_BLC.new_ones(B,) * label1, x_BLC.new_ones(B,) * label2
                        label1, label2 = label1.unsqueeze(1), label2.unsqueeze(1)
                        special_token1, special_token2 = self.special_embed(label1.long()), self.special_embed(label2.long())
                        x1 = x_BLC[:, cur: cur + pn * pn]
                        x2 = x_BLC[:, cur + pn*pn: cur + pn*pn * self.mask_factor]
                        new_x.extend([x1, special_token1, x2, special_token2])
                        cur += pn*pn * self.mask_factor
                    assert cur == x_BLCv_wo_first_l.shape[1]
                    x_BLC = torch.cat(new_x, dim=1)
                else:
                    x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC
            if self.type_pos:
                x_BLC += self.type_embed(self.type_1L[:, :ed].expand(B, -1)) if mask_first else self.type_embed(self.type_1L_[:, :ed].expand(B, -1))

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        SABlock.forward, AdaLNSABlock.forward
        for i, b in enumerate(self.blocks):
            res_idx = self.block_to_resolution_idx[i]
            # Set context to None if cross-attention is disabled
            block_context = None if self.disable_cross_attention else context[res_idx]
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias, context=block_context)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size

    def special_init(self, hd0: float): # hd0: head init scale
        if hd0 >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(hd0)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(hd0)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            if True:
                self.head_nm.ada_lin[-1].weight.data.mul_(self.aln_init)
                if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                    self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: Union[AdaLNSABlock, SABlock]
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(self.aln_gamma_init)
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(self.aln_init)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, :2].mul_(self.aln_gamma_init)
                sab.ada_gss.data[:, :, 2:].mul_(self.aln_init)

    def extra_repr(self):
        gamma2_last = self.gamma2_last
        if isinstance(gamma2_last, nn.Parameter):
            gamma2_last = f'<vector {self.layer_scale}>'
        return f'drop_path_rate={self.drop_path_rate:g}, layer_scale={self.layer_scale:g}, gamma2_last={gamma2_last}'
    
    def print_trainability_status(self):
        """Print trainability status of VQVAE and fusion modules."""
        vae = self.vae_proxy[0]
        
        # Check VQVAE parameters
        vae_total = sum(p.numel() for p in vae.parameters())
        vae_trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)

        # Check optional trainable encoder copy (used only for fusion context)
        if getattr(self, "trainable_encoder", None) is not None:
            enc_total = sum(p.numel() for p in self.trainable_encoder.parameters())
            enc_trainable = sum(p.numel() for p in self.trainable_encoder.parameters() if p.requires_grad)
        else:
            enc_total = 0
            enc_trainable = 0
        
        # Check fusion modules
        fusion_total = sum(p.numel() for p in self.fusion_modules.parameters())
        fusion_trainable = sum(p.numel() for p in self.fusion_modules.parameters() if p.requires_grad)
        
        # Check RemoteVAR parameters (excluding VQVAE)
        remotevar_params = []
        for name, module in self.named_children():
            if name not in ['vae_proxy', 'vae_quant_proxy']:
                remotevar_params.extend(module.parameters())
        remotevar_total = sum(p.numel() for p in remotevar_params)
        remotevar_trainable = sum(p.numel() for p in remotevar_params if p.requires_grad)
        
        print("\n" + "="*70)
        print("MODEL TRAINABILITY STATUS")
        print("="*70)
        print(f"VQVAE Encoder/Decoder:")
        print(f"  Total params:     {vae_total:,}")
        print(f"  Trainable params: {vae_trainable:,}")
        print(f"  Status: {'✓ FROZEN' if vae_trainable == 0 else '✗ NOT FROZEN'}")

        if self.allow_trainable_encoder:
            print(f"\nTrainable Encoder Copy (for fusion context):")
            print(f"  Total params:     {enc_total:,}")
            print(f"  Trainable params: {enc_trainable:,}")
            print(f"  Status: {'✓ TRAINABLE' if enc_trainable == enc_total else '✗ PARTIALLY TRAINABLE'}")

        print(f"\nFeature Fusion Modules ({len(self.fusion_modules)} modules):")
        print(f"  Total params:     {fusion_total:,}")
        print(f"  Trainable params: {fusion_trainable:,}")
        if self.disable_cross_attention:
            print(f"  Status: ✓ DISABLED (cross-attention disabled)")
        else:
            print(f"  Status: {'✓ TRAINABLE' if fusion_trainable == fusion_total else '✗ PARTIALLY TRAINABLE'}")
        print(f"\nRemoteVAR (excluding VQVAE):")
        print(f"  Total params:     {remotevar_total:,}")
        print(f"  Trainable params: {remotevar_trainable:,}")
        if remotevar_trainable > 0:
            print(f"\nFusion modules represent {100*fusion_trainable/remotevar_trainable:.2f}% of trainable params")
        else:
            print(f"\nFusion modules represent 0.00% of trainable params (no trainable params?)")
        print("="*70 + "\n")

    def freeze_all_except_cross_and_fusion(self) -> None:
        """
        Freeze ALL parameters in RemoteVAR except:
        - per-block cross-attention modules (AdaLNSABlock.cross_attn)
        - multi-resolution context fusion modules (self.fusion_modules)
        - optional trainable encoder copy for fusion context (self.trainable_encoder) when allow_trainable_encoder=True
        """
        # Freeze everything first
        for p in self.parameters():
            p.requires_grad_(False)

        # Unfreeze fusion modules
        for p in self.fusion_modules.parameters():
            p.requires_grad_(True)

        # Unfreeze cross-attention modules (if present)
        # Additionally, initialize cross-attention output projection (proj) as zeros
        # to imitate "zero conv" behavior for efficient fine-tuning: cross-attn starts as a no-op.
        for b in self.blocks:
            if hasattr(b, "cross_attn") and b.cross_attn is not None:
                for p in b.cross_attn.parameters():
                    p.requires_grad_(True)
                # Zero-init the *output* projection so the residual branch initially contributes ~0.
                if hasattr(b.cross_attn, "proj") and isinstance(b.cross_attn.proj, nn.Linear):
                    with torch.no_grad():
                        b.cross_attn.proj.weight.zero_()
                        if b.cross_attn.proj.bias is not None:
                            b.cross_attn.proj.bias.zero_()

        # Unfreeze trainable encoder copy (if enabled)
        if getattr(self, "allow_trainable_encoder", False) and getattr(self, "trainable_encoder", None) is not None:
            for p in self.trainable_encoder.parameters():
                p.requires_grad_(True)
            # Keep unused latent head frozen (see note in __init__).
            if hasattr(self.trainable_encoder, "norm_out") and self.trainable_encoder.norm_out is not None:
                for p in self.trainable_encoder.norm_out.parameters():
                    p.requires_grad_(False)
            if hasattr(self.trainable_encoder, "conv_out") and self.trainable_encoder.conv_out is not None:
                for p in self.trainable_encoder.conv_out.parameters():
                    p.requires_grad_(False)


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))

    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x
