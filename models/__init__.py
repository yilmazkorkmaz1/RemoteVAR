from .vqvae import VQVAE
from .class_embedder import ClassEmbedder
from .var import VAR
from .remote_var import RemoteVAR

def build_var(
    vae: VQVAE, depth: int,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    aln=1, aln_gamma_init=1e-3, shared_aln=False, layer_scale=-1,
    tau=4, cos_attn=False,
    flash_if_available=True, fused_if_available=True,
    drop_path_rate: float = 0.0,
):
    return VAR(
        vae_local=vae, patch_nums=patch_nums,
        depth=depth, embed_dim=depth*64, num_heads=depth, drop_path_rate=drop_path_rate,
        aln=aln, aln_gamma_init=aln_gamma_init, shared_aln=shared_aln, layer_scale=layer_scale,
        tau=tau, cos_attn=cos_attn,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available,
    )

def build_remote_var(
    vae: VQVAE, depth: int,
    patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
    aln=1, aln_gamma_init=1e-3, shared_aln=False, layer_scale=-1,
    tau=4, cos_attn=False,
    flash_if_available=True, fused_if_available=True,
    mask_type='replace', cond_drop_rate=0.1, bidirectional=False, separate_decoding=False, separator=False,
    type_pos=False, indep=False, multi_cond=False, disable_cross_attention=False,
    enable_current_scale_tokens: bool = False,
    image_size: int = 256,
    use_high_res_context_levels: bool = True,
    fusion_downsample_ratios=None,
    fusion_num_heads=8,
    fusion_num_layers=1,
    fusion_cross_inner_dim=None,
    fusion_use_feature_rectify=False,
    fusion_downsample_first=False,
    allow_trainable_encoder: bool = False,
    drop_path_rate: float = 0.0,
    cross_attn_inner_dim=1024,
):
    if mask_type == 'replace':
        mask_factor = 1
    elif mask_type == 'interleave_append':
        mask_factor = 2
    elif mask_type == 'change_append':
        mask_factor = 3
    else:
        raise NotImplementedError

    return RemoteVAR(
        vae_local=vae, patch_nums=patch_nums,
        depth=depth, embed_dim=depth*64, num_heads=depth, drop_path_rate=drop_path_rate,
        aln=aln, aln_gamma_init=aln_gamma_init, shared_aln=shared_aln, layer_scale=layer_scale,
        tau=tau, cos_attn=cos_attn, cond_drop_rate=cond_drop_rate,
        flash_if_available=flash_if_available, fused_if_available=fused_if_available, mask_factor=mask_factor,
        bidirectional=bidirectional, separate_decoding=separate_decoding, separator=separator, type_pos=type_pos,
        indep=indep, multi_cond=multi_cond, disable_cross_attention=disable_cross_attention,
        enable_current_scale_tokens=enable_current_scale_tokens,
        image_size=image_size,
        use_high_res_context_levels=use_high_res_context_levels,
        fusion_downsample_ratios=fusion_downsample_ratios,
        fusion_num_heads=fusion_num_heads,
        fusion_num_layers=fusion_num_layers,
        fusion_cross_inner_dim=fusion_cross_inner_dim,
        fusion_use_feature_rectify=fusion_use_feature_rectify,
        fusion_downsample_first=fusion_downsample_first,
        allow_trainable_encoder=allow_trainable_encoder,
        cross_attn_inner_dim=cross_attn_inner_dim,
    )
