import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from typing import Optional


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def _make_downsample_stack(dim: int, *, num_groups: int, ratio: int) -> nn.Module:
    """
    Build a stride-2 Conv2D stack that downsamples spatial resolution by `ratio` (power of two).
    """
    ratio = int(ratio)
    if ratio == 1:
        return nn.Identity()
    if ratio < 1:
        raise ValueError(f"downsample ratio must be >= 1, got {ratio}")
    if ratio & (ratio - 1) != 0:
        raise ValueError(f"downsample ratio must be a power of two, got {ratio}")
    n = int(math.log2(ratio))
    layers = []
    for _ in range(n):
        layers.extend(
            [
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
                Normalize(dim, num_groups=num_groups),
                nn.SiLU(inplace=True),
            ]
        )
    return nn.Sequential(*layers)


# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(self.dim * 4, self.dim * 4 // reduction)
        self.fc2 = nn.Linear(self.dim * 4 // reduction, self.dim * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = F.silu(self.fc1(y), inplace=True)
        y = self.sigmoid(self.fc2(y)).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(self.dim // reduction, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        x = F.silu(self.conv1(x), inplace=True)
        spatial_weights = self.sigmoid(self.conv2(x)).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2 


# Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 

        return x1, x2


class CrossPath(nn.Module):
    def __init__(
        self,
        dim,
        reduction=1,
        num_heads=None,
        norm_layer=nn.LayerNorm,
        *,
        inner_dim: Optional[int] = None,
    ):
        super().__init__()
        if num_heads is None:
            num_heads = 8
        num_heads = int(num_heads)
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        base_dim = int(dim) // int(reduction)
        if base_dim < 1:
            raise ValueError(f"invalid base_dim={base_dim} from dim={dim}, reduction={reduction}")
        inner_dim = base_dim if inner_dim is None else int(inner_dim)
        if inner_dim < 1:
            raise ValueError(f"inner_dim must be >= 1, got {inner_dim}")
        if inner_dim % num_heads != 0:
            raise ValueError(f"inner_dim={inner_dim} must be divisible by num_heads={num_heads}")

        self.channel_proj1 = nn.Linear(dim, inner_dim * 2)
        self.channel_proj2 = nn.Linear(dim, inner_dim * 2)
        self.cross_attn = CrossAttention(inner_dim, num_heads=num_heads)
        self.end_proj1 = nn.Linear(inner_dim * 2, dim)
        self.end_proj2 = nn.Linear(inner_dim * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = F.silu(self.channel_proj1(x1), inplace=True).chunk(2, dim=-1)
        y2, u2 = F.silu(self.channel_proj2(x2), inplace=True).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, num_groups=32):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction)
        self.conv3 = nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True)
        self.norm1 = Normalize(out_channels, num_groups=num_groups)
        self.norm2 = Normalize(out_channels, num_groups=num_groups)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.conv1(x)
        x = F.silu(self.conv2(x), inplace=True)
        x = self.conv3(x)
        x = self.norm1(x)
        out = self.norm2(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(
        self,
        dim,
        reduction=1,
        num_heads=None,
        num_groups=32,
        *,
        # Optional post-fusion downsampling using stride-2 Conv2D stack.
        # Configure at init time; no dynamic module creation in forward.
        # Example: for 256->64 set downsample_ratio=4 (two stride-2 convs).
        downsample_ratio: int = 1,
        # If true and downsample_ratio>1, downsample x1/x2 BEFORE token-mixing to reduce compute.
        # (This makes enabling high-res context levels much more practical.)
        downsample_first: bool = False,
        # Stack multiple CrossPath layers before the final ChannelEmbed merge.
        # This scales capacity mostly on low-res levels (recommended: keep 64x64 at 1 for speed).
        num_cross_layers: int = 1,
        # Optional: widen CrossPath's internal token dimension (defaults to dim//reduction).
        cross_inner_dim: Optional[int] = None,
        # Optional: enable the feature-rectify module before fusion (BCHW -> BCHW).
        use_feature_rectify: bool = False,
    ):
        super().__init__()
        self.num_cross_layers = int(num_cross_layers)
        if self.num_cross_layers < 1:
            raise ValueError(f"num_cross_layers must be >= 1, got {self.num_cross_layers}")

        self.cross = nn.ModuleList(
            [
                CrossPath(
                    dim=dim,
                    reduction=reduction,
                    num_heads=num_heads,
                    inner_dim=cross_inner_dim,
                )
                for _ in range(self.num_cross_layers)
            ]
        )
        self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, num_groups=num_groups)
        self.downsample_ratio = int(downsample_ratio)
        self.downsample_first = bool(downsample_first)
        if self.downsample_ratio < 1:
            raise ValueError(f"downsample_ratio must be >= 1, got {self.downsample_ratio}")
        if self.downsample_ratio & (self.downsample_ratio - 1) != 0:
            raise ValueError(f"downsample_ratio must be a power of two, got {self.downsample_ratio}")

        self.rectify = FeatureRectifyModule(dim=dim, reduction=reduction) if bool(use_feature_rectify) else None

        # Initialize downsampler(s) in __init__ (no dynamic creation in forward).
        if self.downsample_ratio == 1:
            self.pre_downsample = nn.Identity()
            self.post_downsample = nn.Identity()
        else:
            ds = _make_downsample_stack(dim, num_groups=int(num_groups), ratio=self.downsample_ratio)
            if self.downsample_first:
                self.pre_downsample = ds
                self.post_downsample = nn.Identity()
            else:
                self.pre_downsample = nn.Identity()
                self.post_downsample = ds
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        if self.rectify is not None:
            x1, x2 = self.rectify(x1, x2)

        # Validate downsampling (if any) based on ORIGINAL resolution.
        if self.downsample_ratio != 1:
            if H != W:
                raise ValueError(f"FeatureFusionModule downsampling expects square input, got H={H}, W={W}.")
            if H % self.downsample_ratio != 0:
                raise ValueError(
                    f"FeatureFusionModule downsample_ratio={self.downsample_ratio} incompatible with H={H} (not divisible)."
                )

        # Optional: downsample before token mixing to reduce compute on high-res contexts.
        x1 = self.pre_downsample(x1)
        x2 = self.pre_downsample(x2)
        B, C, H, W = x1.shape

        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        for blk in self.cross:
            x1, x2 = blk(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        
        # Optional: downsample fused BCHW using the prebuilt conv stack.
        merge = self.post_downsample(merge)

        return merge