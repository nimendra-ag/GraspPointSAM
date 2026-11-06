"""
Factory function that builds GraspPointSAM from a config file.
Single place where all components are wired together.
Never hardcode dimensions anywhere else — always read from config.
"""

import timm
from omegaconf import DictConfig

from .pc_encoder import PointCloudEncoder, PatchEmbed
from .prompt_encoder import PointEncoder, MaskEncoder
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer
from .grasp_decoder import GraspDecoder3D
from .grasp_point_sam import GraspPointSAM


def build_model(cfg: DictConfig) -> GraspPointSAM:
    """
    Build GraspPointSAM from hydra config.

    Args:
        cfg: the full config (contents of configs/graspnet.yaml)

    Returns:
        GraspPointSAM with pretrained weights loaded and encoder frozen
        if cfg.model.freeze_encoder is True.
    """

    m = cfg.model  # shorthand

    # PatchEmbed 
    patch_embed = PatchEmbed(
        in_channels = m.patch_embed.in_channels,
        out_channels = m.patch_embed.out_channels,
        num_patches = m.patch_embed.num_patches,
        patch_size = m.patch_embed.patch_size,
    )

    # EVA-02 transformer backbone 
    transformer_backbone = timm.create_model(
        m.transformer.model_name,
        pretrained=m.transformer.pretrained,
    )

    # PointCloudEncoder
    pc_encoder = PointCloudEncoder(
        patch_embed = patch_embed,
        transformer = transformer_backbone,
        embed_dim = m.pc_encoder.embed_dim,
    )

    # PointEncoder 
    point_encoder = PointEncoder(embed_dim=m.point_encoder.embed_dim)

    # MaskEncoder
    mask_encoder = MaskEncoder(embed_dim=m.mask_encoder.embed_dim)

    # MaskDecoder (PointSAM original)
    mask_decoder = MaskDecoder(
        transformer_dim = m.mask_decoder.transformer_dim,
        transformer = TwoWayTransformer(
            depth = m.mask_decoder.depth,
            embedding_dim = m.mask_decoder.transformer_dim,
            num_heads = m.mask_decoder.num_heads,
            mlp_dim = m.mask_decoder.mlp_dim,
        ),
    )

    # GraspDecoder3D
    grasp_decoder = GraspDecoder3D(
        transformer = TwoWayTransformer(
            depth = m.grasp_decoder.depth,
            embedding_dim = m.grasp_decoder.transformer_dim,
            num_heads = m.grasp_decoder.num_heads,
            mlp_dim = m.grasp_decoder.mlp_dim,
        ),
        transformer_dim = m.grasp_decoder.transformer_dim,
        joint_mask_grasp = m.grasp_decoder.joint_mask_grasp,
        num_grasp_rot_tokens = m.grasp_decoder.num_grasp_rot_tokens,
        # 0 for configs / checkpoints from before the depth head
        num_depth_bins = m.grasp_decoder.get('num_depth_bins', 0),
    )

    # Assemble GraspPointSAM 
    model = GraspPointSAM(
        pc_encoder = pc_encoder,
        point_encoder = point_encoder,
        mask_encoder = mask_encoder,
        mask_decoder = mask_decoder,
        grasp_decoder = grasp_decoder,
        prompt_iters = m.prompt_iters,
        joint_mask_grasp = m.joint_mask_grasp,
    )

    # Load pretrained PointSAM weights 
    model.load_pretrained_pointsam(m.pretrained_ckpt)

    # Freeze encoder if specified 
    if m.freeze_encoder:
        model.freeze_encoder()

    return model