"""
3D Grasp Decoder for GraspPointSAM.

Extends PointSAM's MaskDecoder with learnable grasp tokens that predict
6-DOF grasp parameters per point. Optionally outputs segmentation masks
alongside grasp predictions (joint_mask_grasp=True).

Architecture inspired by:
- GraspSAM (ICRA 2025): grasp token concept, multi-task decoding
- PointSAM: TwoWayTransformer, point cloud embedding structure
- SAM-HQ: learnable HQ token design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass

from .common import interpolate_features, compute_interp_weights
from .mask_decoder import AuxInputs, MLP

# Data container for decoder outputs
@dataclass
class GraspPredictions:
    """
    Container for all decoder outputs.
    
    Shapes use the following notation:
        B  = batch size
        N  = number of points in the input point cloud
        M  = number of prompted instances (objects clicked on)
    
    grasp_quality : [B*M, N]
        Per-point graspability score in [0, 1].
        High score = this point is a good grasp contact point.
    
    grasp_rotation : [B*M, N, 6]
        Per-point gripper orientation as a 6D rotation representation.
        We use the first two columns of the rotation matrix (6 numbers).
        The third column is recovered via cross product (Gram-Schmidt).
        This avoids gimbal lock and discontinuities of Euler angles/quaternions.
        See: "On the Continuity of Rotation Representations in Neural Networks" (Zhou et al. 2019)
    
    grasp_width : [B*M, N, 1]
        Per-point predicted gripper opening width in meters.
    
    masks : [B*M, N] or None
        Per-point segmentation mask logits (before sigmoid).
        Only populated when joint_mask_grasp=True.
    
    iou_preds : [B*M, num_mask_tokens] or None
        IoU predictions for mask quality estimation.
        Only populated when joint_mask_grasp=True.
    """
    grasp_quality: torch.Tensor     # [B*M, N]
    grasp_rotation: torch.Tens      # [B*M, N, 6]
    grasp_width: torch.Tensor       # [B*M, N, 1]
    masks: Optional[torch.Tensor]   # [B*M, N] or None
    iou_preds: Optional[torch.Tensor] # [B*M, num_mask_tokens] or None


# Small MLP used for per-point prediction heads
class GraspMLP(nn.Module):
    """
    Lightweight MLP head for per-point grasp prediction.
    
    Takes a token embedding [D] and produces per-point predictions
    by computing a dot product with point features [N, D].
    
    This is the same "hypernetwork" pattern used in SAM's mask decoder:
    instead of running an MLP over each of N points (expensive),
    the MLP produces a weight vector that is dot-producted with all
    point features at once (cheap).
    
    Args:
        input_dim  : transformer embedding dimension (256 for base model)
        hidden_dim : hidden layer size
        output_dim : output size per point (1 for quality/width, 6 for rotation)
        num_layers : number of MLP layers
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()
        # Build MLP layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1])
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., input_dim]
        Returns:
            [..., output_dim]
        """
        for i, layer in enumerate(self.layers):
            # ReLU after all layers except the last
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# Main Decoder
class GraspDecoder3D(nn.Module):
    """
    3D Grasp Decoder that extends PointSAM's TwoWayTransformer with
    learnable grasp tokens for 6-DOF grasp prediction.

    How it works (step by step):
    
    1. TOKENS: Concatenate learnable tokens:
       [iou_token | mask_tokens | grasp_quality_token | grasp_rotation_tokens | grasp_width_token]
       These tokens will "attend" to the point cloud embeddings via cross-attention.
    
    2. TRANSFORMER: Run TwoWayTransformer — tokens and point cloud embeddings
       exchange information bidirectionally. After this, each token has
       "summarized" the point cloud from its own perspective.
    
    3. HYPERNETWORK: Pass each output token through its MLP head to get
       a weight vector [D]. Dot-product with point features [N, D] to get
       per-point predictions [N, output_dim]. This is O(N*D) not O(N*MLP_cost).
    
    4. OUTPUT: Return GraspPredictions dataclass with all predictions.

    Args:
        transformer : TwoWayTransformer instance (from PointSAM, reused)
        transformer_dim : embedding dimension, must match encoder output (256)
        joint_mask_grasp : if True, also predict segmentation masks
        num_grasp_rot_tokens : number of tokens for rotation prediction.
                               We use 6 tokens (one per rotation output dimension)
                               so each token specializes in one component.
    """

    # Number of mask tokens (kept same as original SAM for compatibility)
    NUM_MASK_TOKENS = 4

    def __init__(
        self,
        transformer: nn.Module,
        transformer_dim: int = 256,
        joint_mask_grasp: bool = True,
        num_grasp_rot_tokens: int = 6,
    ):
        super().__init__()

        self.transformer = transformer
        self.transformer_dim = transformer_dim
        self.joint_mask_grasp = joint_mask_grasp
        self.num_grasp_rot_tokens = num_grasp_rot_tokens

        # Mask tokens (from original SAM, kept for joint mode)
        # iou_token: 1 token that predicts mask IoU quality
        self.iou_token = nn.Embedding(1, transformer_dim)
        # mask_tokens: 4 tokens, one per mask output
        # (SAM uses 4: 1 single-mask + 3 multi-mask)
        self.mask_tokens = nn.Embedding(self.NUM_MASK_TOKENS, transformer_dim)

        # Grasp tokens (new, learned from scratch)
        # 1 token -> predicts per-point grasp quality score
        self.grasp_quality_token = nn.Embedding(1, transformer_dim)

        # 6 tokens -> predict 6D rotation (one token per output dimension)
        # Using multiple tokens lets each specialize in one rotation component
        self.grasp_rotation_tokens = nn.Embedding(num_grasp_rot_tokens, transformer_dim)

        # 1 token -> predicts per-point gripper width
        self.grasp_width_token = nn.Embedding(1, transformer_dim)

        # MLP heads (hypernetwork pattern) 
        # Each MLP takes a token embedding [D] -> weight vector
        # Then: weight_vector [D] dot point_features [N, D] -> per-point output [N]

        # Quality head: outputs 1 value per point (graspability score)
        self.quality_head = GraspMLP(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,
            output_dim=transformer_dim,  # produces weight vector, not final output
            num_layers=3,
        )

        # Rotation head: outputs 6 values per point (6D rotation)
        # One MLP per rotation token
        self.rotation_heads = nn.ModuleList([
            GraspMLP(
                input_dim=transformer_dim,
                hidden_dim=transformer_dim,
                output_dim=transformer_dim,
                num_layers=3,
            )
            for _ in range(num_grasp_rot_tokens)
        ])

        # Width head: outputs 1 value per point (gripper opening width)
        self.width_head = GraspMLP(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,
            output_dim=transformer_dim,
            num_layers=3,
        )

        # Mask output MLP heads (joint mode only) 
        # Same pattern as original SAM mask decoder
        if joint_mask_grasp:
            self.mask_heads = nn.ModuleList([
                MLP(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=transformer_dim,
                    num_layers=3
                )
                for _ in range(self.NUM_MASK_TOKENS)
            ])
            self.iou_prediction_head = MLP(
                input_dim=transformer_dim,
                hidden_dim=256,
                output_dim=self.NUM_MASK_TOKENS,
                num_layers=3,
            )

        # Point feature upscaling (for per-point prediction) 
        # Point cloud embeddings are patch-level [B, L, D] where L=num_patches.
        # We need point-level [B, N, D]. This MLP refines the interpolated features.
        self.point_upscale = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.LayerNorm(transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
        )

    def forward(
        self,
        pc_embeddings: torch.Tensor,  # [B, L, D] patch-level embeddings
        pc_pe: torch.Tensor,  # [B, L, D] positional encoding
        sparse_prompt_embeddings: torch.Tensor, # [B*M, K, D] point prompt embeddings
        dense_prompt_embeddings: torch.Tensor, # [B*M, L, D] mask prompt embeddings
        aux_inputs: AuxInputs,     # contains coords, features, centers
        multimask_output: bool = True,
    ) -> GraspPredictions:
        """
        Forward pass.

        The key shape to track throughout:
            B  = batch size
            M  = number of prompted instances per scene
            L  = number of patches (512 for base model)
            N  = number of points (e.g. 10000)
            D  = transformer_dim (256)
            K  = number of sparse prompt tokens
        """

        #  Step 1: Build token sequence 
        # Concatenate all learnable tokens into one sequence.
        # Order: [iou | mask_0..3 | quality | rot_0..5 | width]
        # Shape: [1, num_tokens, D]
        
        if self.joint_mask_grasp:
            output_tokens = torch.cat([
                self.iou_token.weight,            # [1, D]
                self.mask_tokens.weight,           # [4, D]
                self.grasp_quality_token.weight,   # [1, D]
                self.grasp_rotation_tokens.weight, # [6, D]
                self.grasp_width_token.weight,     # [1, D]
            ], dim=0)  # [13, D]
        else:
            # Grasp-only: no iou/mask tokens
            output_tokens = torch.cat([
                self.grasp_quality_token.weight,   # [1, D]
                self.grasp_rotation_tokens.weight, # [6, D]
                self.grasp_width_token.weight,     # [1, D]
            ], dim=0)  # [8, D]

        # Expand tokens to match batch*instances dimension
        # sparse_prompt_embeddings has shape [B*M, K, D]
        # so we expand tokens to [B*M, num_tokens, D]
        batch_size = sparse_prompt_embeddings.shape[0]
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate prompt tokens after output tokens
        # tokens: [B*M, num_tokens + K, D]
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)

        #  Step 2: Expand pc_embeddings to match B*M 
        # pc_embeddings is [B, L, D] but tokens are [B*M, ...]
        # repeat each scene embedding M times
        M = batch_size // pc_embeddings.shape[0]
        src = pc_embeddings.repeat_interleave(M, dim=0)  # [B*M, L, D]
        src = src + dense_prompt_embeddings               # add mask prompt

        pos_src = pc_pe.repeat_interleave(M, dim=0)      # [B*M, L, D]

        #  Step 3: TwoWayTransformer ─
        # Bidirectional cross-attention between tokens and point cloud.
        # After this:
        #   hs  : updated token embeddings [B*M, num_tokens+K, D]
        #   src : updated point cloud embeddings [B*M, L, D]
        hs, src = self.transformer(src, pos_src, tokens)

        #  Step 4: Extract token outputs ─
        if self.joint_mask_grasp:
            iou_token_out    = hs[:, 0, :]       # [B*M, D]
            mask_tokens_out  = hs[:, 1:5, :]     # [B*M, 4, D]
            quality_tok      = hs[:, 5, :]        # [B*M, D]
            rotation_toks    = hs[:, 6:12, :]     # [B*M, 6, D]
            width_tok        = hs[:, 12, :]        # [B*M, D]
        else:
            quality_tok      = hs[:, 0, :]        # [B*M, D]
            rotation_toks    = hs[:, 1:7, :]      # [B*M, 6, D]
            width_tok        = hs[:, 7, :]         # [B*M, D]

        #  Step 5: Interpolate patch features -> point features 
        # src is patch-level [B*M, L, D]. We need point-level [B*M, N, D].
        # Use the precomputed interpolation weights from the encoder.
        # (same mechanism PointSAM uses in its mask decoder)
        
        # Compute interpolation weights if not precomputed
        if aux_inputs.interp_index is None or aux_inputs.interp_weight is None:
            interp_index, interp_weight = compute_interp_weights(
                aux_inputs.coords,
                aux_inputs.centers,
            )
        else:
            interp_index  = aux_inputs.interp_index
            interp_weight = aux_inputs.interp_weight

        # point_features: [B*M, N, D]
        point_features = interpolate_features(src, interp_index, interp_weight)
        point_features = self.point_upscale(point_features)  # refine

        #  Step 6: Predict per-point grasp parameters ─
        # Hypernetwork pattern:
        #   token_mlp(token) -> weight_vec [D]
        #   weight_vec [D] dot point_features [N, D] -> per_point_output [N]

        # Quality: [B*M, D] -> [B*M, D] -> dot [B*M, N, D] -> [B*M, N]
        quality_weights = self.quality_head(quality_tok)          # [B*M, D]
        grasp_quality   = torch.einsum(
            'bd,bnd->bn', quality_weights, point_features
        )  # [B*M, N]
        grasp_quality   = torch.sigmoid(grasp_quality)            # -> [0, 1]

        # Rotation: 6 tokens -> 6 weight vectors -> 6 per-point values -> [B*M, N, 6]
        rot_outputs = []
        for i, rot_head in enumerate(self.rotation_heads):
            rot_weight = rot_head(rotation_toks[:, i, :])        # [B*M, D]
            rot_out    = torch.einsum(
                'bd,bnd->bn', rot_weight, point_features
            )  # [B*M, N]
            rot_outputs.append(rot_out)
        grasp_rotation = torch.stack(rot_outputs, dim=-1)         # [B*M, N, 6]

        # Width: [B*M, D] -> [B*M, D] -> dot [B*M, N, D] -> [B*M, N, 1]
        width_weights = self.width_head(width_tok)                # [B*M, D]
        grasp_width   = torch.einsum(
            'bd,bnd->bn', width_weights, point_features
        ).unsqueeze(-1)  # [B*M, N, 1]
        grasp_width   = F.softplus(grasp_width)  # ensure positive width

        #  Step 7: Mask prediction (joint mode only) 
        masks     = None
        iou_preds = None

        if self.joint_mask_grasp:
            # Same hypernetwork pattern for masks
            # mask_tokens_out: [B*M, 4, D]
            mask_weights = torch.stack([
                self.mask_heads[i](mask_tokens_out[:, i, :])
                for i in range(self.NUM_MASK_TOKENS)
            ], dim=1)  # [B*M, 4, D]

            # [B*M, 4, D] dot [B*M, N, D] -> [B*M, 4, N]
            masks = torch.einsum(
                'bmd,bnd->bmn', mask_weights, point_features
            )  # [B*M, 4, N]

            # Select single or multi mask output
            if multimask_output:
                masks = masks[:, 1:, :]   # [B*M, 3, N] — 3 multi-mask outputs
            else:
                masks = masks[:, 0:1, :]  # [B*M, 1, N] — single best mask

            # IoU prediction: how good is the predicted mask?
            iou_preds = self.iou_prediction_head(iou_token_out)  # [B*M, 4]

        return GraspPredictions(
            grasp_quality  = grasp_quality,   # [B*M, N]
            grasp_rotation = grasp_rotation,  # [B*M, N, 6]
            grasp_width    = grasp_width,     # [B*M, N, 1]
            masks          = masks,           # [B*M, num_masks, N] or None
            iou_preds      = iou_preds,       # [B*M, 4] or None
        )