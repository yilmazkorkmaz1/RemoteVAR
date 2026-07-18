import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence, Tuple


# this file only defines the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder', 'ConditionedDecoder']


"""
References: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""
# swish
def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    
    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout): # conv_shortcut=False,  # conv_shortcut: always False in VAE
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()
    
    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels
        
        self.norm = Normalize(in_channels)
        self.qkv = torch.nn.Conv2d(in_channels, 3*in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape  # should be B,3C,H,W
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)
        
        # compute attention
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()     # B,HW,C
        k = k.view(B, C, H * W).contiguous()    # B,C,HW
        w = torch.bmm(q, k).mul_(self.w_ratio)  # B,HW,HW    w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
        w = F.softmax(w, dim=2)
        
        # attend to values
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()  # B,HW,HW (first HW of k, second of q)
        h = torch.bmm(v, w)  # B, C,HW (HW of q) h[B,C,j] = sum_i v[B,C,i] w[B,i,j]
        h = h.view(B, C, H, W).contiguous()
        
        return x + self.proj_out(h)


def make_attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()


class Encoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,
        z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        # Expose encoder channel multipliers so downstream models (e.g., RemoteVAR)
        # can infer per-resolution context channel dims.
        self.ch_mult = tuple(ch_mult)
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        
        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h


    def forward_context(self, x, *, return_all_levels: bool = True):
                """
                Extract multi-resolution context features from encoder.
                Returns list of features from each resolution level (coarse to fine).
                
                For a typical 4-level encoder with 256x256 input:
                - Level 0: after 1st encoder block (before 1st downsample): 128x128
                - Level 1: after 2nd encoder block (before 2nd downsample): 64x64  
                - Level 2: after 3rd encoder block (before 3rd downsample): 32x32
                - Level 3: after 4th encoder block (no more downsample): 16x16
                
                Returns list REVERSED so coarsest (16x16) comes first:
                    h_list: [16x16, 32x32, 64x64] for 3-level output
                            where h_list[0] is lowest resolution (coarsest)
                            and h_list[-1] is highest resolution (finest)
                """
                # Conditioning vector - encodes acceleration rate and mask type
                h = self.conv_in(x)
                
                h_list = []
                # -------- down path with early conditioning --------
                for lvl, d in enumerate(self.down):
                    # ResNet blocks with internal label conditioning
                    for rb in d.block:
                        h = rb(h)
                    
                    # Attention
                    for at in d.attn:
                        h = at(h)

                    # Save features from this level BEFORE downsampling.
                    # Convert from BCHW to BLC for cross-attention.
                    B, C, H, W = h.shape
                    h_flat = h.flatten(2).transpose(1, 2)  # B, H*W, C
                    h_list.append(h_flat)
                    
                    # Downsampling for next level
                    if lvl != self.num_resolutions - 1:
                        h = d.downsample(h)

                # Encoder naturally produces multiple resolutions (e.g., for 256x256 input and vq-f16 `ch_mult`,
                # the down path yields: [256, 128, 64, 32, 16] before the middle block).
                #
                # If `return_all_levels=True` (default), return all down-path levels so fusion modules can learn
                # from high-resolution features, then optionally downsample AFTER fusion.
                #
                # If `return_all_levels=False`, keep the legacy behavior and only return:
                #   [64x64, 32x32, 16x16_pre_middle] + [16x16_post_middle]
                if return_all_levels:
                    multi_res_features = list(h_list)
                else:
                    multi_res_features = h_list[-3:]  # [64x64, 32x32, 16x16]
                
                # -------- Process through middle block for 4th resolution --------
                # Middle block adds self-attention and deeper processing to the 16x16 features
                # This creates richer semantic features at same spatial resolution

                h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
                
                # Convert middle block output to BLC format (16x16 with richer features)
                h_middle_flat = h.flatten(2).transpose(1, 2)  # B, 16*16, C
                
                # Append post-middle features at the same spatial resolution as the last down level.
                multi_res_features.append(h_middle_flat)
                return multi_res_features


class Decoder(nn.Module):
    def __init__(
        self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
        dropout=0.0, in_channels=3,  # in_channels: raw img channels
        z_channels, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        
        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions-1 and using_sa:
                    attn.append(make_attn(block_in, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order
        
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, z, skips=None):
        # z to block_in
        # middle
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h


def _round_up_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError(f"multiple must be > 0, got {multiple}")
    x = int(x)
    if x <= 0:
        return int(multiple)
    return int(((x + multiple - 1) // multiple) * multiple)


class _SkipFuseResidualAdapter(nn.Module):
    """
    Extra (optional) skip-fusion adapter that:
      - takes concat([h, skip1, skip2, ...]) as input
      - predicts a delta with shape like h (B, in_ch, H, W)
      - returns h + delta

    This makes it easy to scale "new" decoder-refiner capacity via:
      - depth (num_res_blocks)
      - width (hidden_channels)

    Initialization: proj_out is zero-initialized so the adapter starts as identity (delta=0).
    """

    def __init__(
        self,
        *,
        in_ch: int,
        skip_ch: int,
        hidden_ch: int,
        num_res_blocks: int,
        dropout: float,
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.skip_ch = int(skip_ch)
        self.total_in = int(in_ch + skip_ch)
        self.hidden_ch = int(hidden_ch)
        self.num_res_blocks = int(num_res_blocks)

        if self.in_ch <= 0:
            raise ValueError(f"in_ch must be > 0, got {self.in_ch}")
        if self.skip_ch <= 0:
            raise ValueError(f"skip_ch must be > 0, got {self.skip_ch}")
        if self.hidden_ch <= 0:
            raise ValueError(f"hidden_ch must be > 0, got {self.hidden_ch}")
        if self.num_res_blocks < 0:
            raise ValueError(f"num_res_blocks must be >= 0, got {self.num_res_blocks}")

        # Project concat(h, skips) -> hidden
        self.proj_in = nn.Conv2d(self.total_in, self.hidden_ch, kernel_size=1, stride=1, padding=0)
        self.blocks = nn.ModuleList(
            [ResnetBlock(in_channels=self.hidden_ch, out_channels=self.hidden_ch, dropout=dropout) for _ in range(self.num_res_blocks)]
        )
        # Project hidden -> delta(h)
        self.proj_out = nn.Conv2d(self.hidden_ch, self.in_ch, kernel_size=1, stride=1, padding=0)

        # Start at identity: delta == 0
        with torch.no_grad():
            self.proj_out.weight.zero_()
            if self.proj_out.bias is not None:
                self.proj_out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"_SkipFuseResidualAdapter expects BCHW input; got shape={tuple(x.shape)}")
        if int(x.shape[1]) != int(self.total_in):
            raise ValueError(
                f"_SkipFuseResidualAdapter expected input channels={self.total_in} (=in_ch {self.in_ch} + skip_ch {self.skip_ch}), "
                f"got {int(x.shape[1])}. This usually means a skip tensor was missing or had unexpected channels."
            )
        h = x[:, : self.in_ch, :, :]
        y = self.proj_in(x)
        for b in self.blocks:
            y = b(y)
        delta = self.proj_out(y)
        return h + delta


class ConditionedDecoder(Decoder):
    """
    A UNet-style VQ-VAE decoder that can take *external* skip features (e.g. from RemoteVAR fusion modules).

    The external skips are concatenated at the matching decoder resolutions and fused back into the decoder
    stream using a lightweight ResnetBlock.

    Notes:
    - `skip_base_resolutions` and `skip_in_channels` must be aligned lists matching the order of the provided `skips`.
      Example (legacy 4-level context for 256px): base_res=[64,32,16,16], ch=[320,320,640,640]
      Example (high-res context for 256px):      base_res=[256,128,64,32,16,16], ch=[160,160,320,320,640,640]
    - The decoder will fuse *all* skips whose base_res matches the current decoder spatial size.
    """

    def __init__(
        self,
        *,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        in_channels=3,
        z_channels,
        using_sa=True,
        using_mid_sa=True,
        skip_base_resolutions: Sequence[int],
        skip_in_channels: Sequence[int],
        # Optional extra capacity for the *new* skip-fusion modules (decoder refiner scaling knobs).
        # Kept disabled by default so old checkpoints / behavior stay identical.
        skip_fuse_extra_depth: int = 0,
        skip_fuse_extra_width_mult: float = 1.0,
    ):
        super().__init__(
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
            in_channels=in_channels,
            z_channels=z_channels,
            using_sa=using_sa,
            using_mid_sa=using_mid_sa,
        )

        if len(skip_base_resolutions) != len(skip_in_channels):
            raise ValueError(
                f"skip_base_resolutions and skip_in_channels must have the same length, "
                f"got {len(skip_base_resolutions)} vs {len(skip_in_channels)}"
            )
        self.skip_base_resolutions: Tuple[int, ...] = tuple(int(x) for x in skip_base_resolutions)
        self.skip_in_channels: Tuple[int, ...] = tuple(int(x) for x in skip_in_channels)

        # Infer the latent spatial size from the *coarsest* skip resolution (should match z's H/W).
        # For 256px inputs with vq-f16 downsample ratio 16: latent_hw=16.
        if len(self.skip_base_resolutions) == 0:
            raise ValueError("ConditionedDecoder requires non-empty skip_base_resolutions.")
        latent_hw = int(min(self.skip_base_resolutions))
        if latent_hw <= 0:
            raise ValueError(f"Invalid latent_hw inferred from skip_base_resolutions: {latent_hw}")

        # Precompute which skip indices to fuse at each decoder level (indexed by i_level, 0..num_resolutions-1).
        # Decoder processes i_level in reversed order (coarse->fine), but module indexing matches i_level.
        self._skip_indices_per_level: List[List[int]] = []
        self.skip_fuse: nn.ModuleList = nn.ModuleList()
        self.skip_fuse_extra: nn.ModuleList = nn.ModuleList()

        extra_depth = int(skip_fuse_extra_depth)
        extra_width_mult = float(skip_fuse_extra_width_mult)
        if extra_depth < 0:
            raise ValueError(f"skip_fuse_extra_depth must be >= 0, got {extra_depth}")
        if extra_width_mult <= 0:
            raise ValueError(f"skip_fuse_extra_width_mult must be > 0, got {extra_width_mult}")

        for i_level in range(self.num_resolutions):
            # Decoder spatial size at this level:
            # i_level=num_resolutions-1 -> latent_hw
            # i_level=num_resolutions-2 -> 2*latent_hw
            # ...
            # i_level=0 -> latent_hw * 2**(num_resolutions-1)
            level_hw = int(latent_hw * (2 ** (self.num_resolutions - 1 - i_level)))
            idxs = [i for i, r in enumerate(self.skip_base_resolutions) if int(r) == level_hw]
            self._skip_indices_per_level.append(idxs)

            if len(idxs) == 0:
                self.skip_fuse.append(nn.Identity())
                self.skip_fuse_extra.append(nn.Identity())
                continue

            # Fuse into the *current* decoder channel count at the start of this level.
            # This matches the input channels expected by the first ResnetBlock at this level.
            block0: ResnetBlock = self.up[i_level].block[0]
            in_ch = int(block0.in_channels)
            skip_ch = int(sum(self.skip_in_channels[i] for i in idxs))

            # UNet-style: concat(h, skips...) then fuse back to in_ch
            fuse = ResnetBlock(in_channels=in_ch + skip_ch, out_channels=in_ch, dropout=dropout)

            # Zero-init so the conditioned decoder matches the pretrained decoder initially:
            # - Residual branch starts at ~0 (zero conv2), so block output is dominated by shortcut
            # - Shortcut is initialized to copy ONLY the original decoder channels (first in_ch),
            #   ignoring skip channels (weights=0 for skip inputs).
            with torch.no_grad():
                if hasattr(fuse, "conv2") and isinstance(fuse.conv2, nn.Conv2d):
                    fuse.conv2.weight.zero_()
                    if fuse.conv2.bias is not None:
                        fuse.conv2.bias.zero_()

                if hasattr(fuse, "nin_shortcut") and isinstance(fuse.nin_shortcut, nn.Conv2d):
                    fuse.nin_shortcut.weight.zero_()
                    if fuse.nin_shortcut.bias is not None:
                        fuse.nin_shortcut.bias.zero_()
                    # Identity for the first in_ch channels (the decoder stream `h`).
                    # Input layout is [h, skip1, skip2, ...] so h occupies the first in_ch channels.
                    for c in range(in_ch):
                        fuse.nin_shortcut.weight[c, c, 0, 0] = 1.0

            self.skip_fuse.append(fuse)

            # Optional: extra residual adapter that also sees the concatenated skips.
            if extra_depth == 0:
                self.skip_fuse_extra.append(nn.Identity())
            else:
                hidden_target = int(round(in_ch * extra_width_mult))
                # ResnetBlock uses GroupNorm(32), so make hidden channels a multiple of 32.
                hidden_ch = _round_up_to_multiple(max(32, hidden_target), 32)
                self.skip_fuse_extra.append(
                    _SkipFuseResidualAdapter(
                        in_ch=in_ch,
                        skip_ch=skip_ch,
                        hidden_ch=hidden_ch,
                        num_res_blocks=extra_depth,
                        dropout=dropout,
                    )
                )

    def forward(self, z, skips: Optional[Sequence[torch.Tensor]] = None):
        # z to block_in
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))

        # upsampling (coarse -> fine)
        for i_level in reversed(range(self.num_resolutions)):
            if skips is not None:
                idxs = self._skip_indices_per_level[i_level]
                if len(idxs) > 0:
                    parts = [h]
                    skip_parts = []
                    for j in idxs:
                        s = skips[j]
                        if s is None:
                            continue
                        if s.dim() != 4:
                            raise ValueError(
                                f"ConditionedDecoder skips must be BCHW tensors; got skip[{j}] shape={tuple(s.shape)}"
                            )
                        # Resize to match current decoder resolution if needed.
                        if tuple(s.shape[-2:]) != tuple(h.shape[-2:]):
                            s = F.interpolate(s, size=h.shape[-2:], mode="bilinear", align_corners=False)
                        # Match dtype/device (skips are often computed under no_grad).
                        s = s.to(device=h.device, dtype=h.dtype)
                        skip_parts.append(s)
                        parts.append(s)
                    if len(parts) > 1:
                        x = torch.cat(parts, dim=1)
                        h = self.skip_fuse[i_level](x)
                        # Apply optional extra adapter (also conditioned on skips).
                        # IMPORTANT: when extra modules are disabled, skip_fuse_extra[i_level] is Identity.
                        # In that case we must NOT feed concatenated [h, skips...] through Identity, or we'd
                        # change the channel count and break the base decoder blocks.
                        extra = self.skip_fuse_extra[i_level]
                        if (not isinstance(extra, nn.Identity)) and (len(skip_parts) > 0):
                            h = extra(torch.cat([h] + skip_parts, dim=1))

            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h