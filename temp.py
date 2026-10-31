"""
GraspPointSAM — Top-level model.

Combines PointSAM's pretrained 3D point cloud encoder with a new
GraspDecoder3D head for 6-DOF grasp detection.

Pipeline:
    Input: [B, N, 6] point cloud (XYZ + RGB)
        ↓
    PointCloudEncoder  →  patch embeddings [B, L, D]
        ↓
    PromptEncoder      →  sparse + dense prompt embeddings
        ↓
    GraspDecoder3D     →  grasp quality, rotation, width (+ masks if joint)
        ↓
    Output: GraspPredictions
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from safetensors.torch import load_file

from .pc_encoder import PointCloudEncoder
from .prompt_encoder import PointEncoder, MaskEncoder
from .mask_decoder import AuxInputs
from .transformer import TwoWayTransformer
from .grasp_decoder import GraspDecoder3D, GraspPredictions
from .common import sample_prompts, repeat_interleave


class GraspPointSAM(nn.Module):
    """
    GraspPointSAM model.

    Args:
        pc_encoder       : PointCloudEncoder (from PointSAM)
        mask_encoder     : MaskEncoder for dense prompt embeddings
        grasp_decoder    : GraspDecoder3D (new)
        prompt_iters     : number of iterative prompt refinement steps
                           (same mechanism as PointSAM — each iteration
                           uses the previous mask as a new prompt)
        joint_mask_grasp : passed through to GraspDecoder3D
    """

    def __init__(
        self,
        pc_encoder:       PointCloudEncoder,
        mask_encoder:     MaskEncoder,
        grasp_decoder:    GraspDecoder3D,
        prompt_iters:     int  = 3,
        joint_mask_grasp: bool = True,
    ):
        super().__init__()

        self.pc_encoder       = pc_encoder
        self.point_encoder    = PointEncoder(pc_encoder.embed_dim)
        self.mask_encoder     = mask_encoder
        self.grasp_decoder    = grasp_decoder
        self.prompt_iters     = prompt_iters
        self.joint_mask_grasp = joint_mask_grasp

    # ─────────────────────────────────────────────────────────────────────
    # Pretrained weight loading
    # ─────────────────────────────────────────────────────────────────────

    def load_pretrained_pointsam(self, ckpt_path: str) -> None:
        """
        Load PointSAM pretrained weights into the encoder components.

        Uses strict=False so that:
        - Matching keys (pc_encoder, point_encoder, mask_encoder) are loaded
        - Missing keys (grasp_decoder.*) are left randomly initialized
        - Unexpected keys in checkpoint are ignored

        Args:
            ckpt_path: path to model.safetensors
        """
        print(f"Loading PointSAM weights from: {ckpt_path}")

        # Load safetensors format
        state_dict = load_file(ckpt_path)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        # Validate: only grasp_decoder keys should be missing
        non_grasp_missing = [
            k for k in missing
            if not k.startswith("grasp_decoder")
        ]

        if non_grasp_missing:
            print(f"WARNING: Unexpected missing keys (not grasp_decoder):")
            for k in non_grasp_missing:
                print(f"  {k}")
        else:
            print(f"PointSAM weights loaded successfully.")
            print(f"  Loaded:      {len(state_dict) - len(unexpected)} keys")
            print(f"  Initialized: {len(missing)} grasp_decoder keys (random init)")

    def freeze_encoder(self) -> None:
        """Freeze PointSAM encoder — only grasp_decoder trains."""
        for param in self.pc_encoder.parameters():
            param.requires_grad = False
        for param in self.point_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen. Only GraspDecoder3D will be trained.")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder for fine-tuning stage."""
        for param in self.pc_encoder.parameters():
            param.requires_grad = True
        for param in self.point_encoder.parameters():
            param.requires_grad = True
        for param in self.mask_encoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen.")

    # ─────────────────────────────────────────────────────────────────────
    # Core forward — shared between training and inference
    # ─────────────────────────────────────────────────────────────────────

    def predict(
        self,
        coords:          torch.Tensor,          # [B, N, 3] normalized to [-1, 1]
        features:        torch.Tensor,          # [B, N, 3] RGB normalized to [0, 1]
        prompt_coords:   torch.Tensor,          # [B*M, K, 3] 3D point prompts
        prompt_labels:   torch.Tensor,          # [B*M, K] 0=bg, 1=fg
        prompt_masks:    Optional[torch.Tensor] = None,  # [B*M, N] previous mask
        multimask_output: bool = True,
    ) -> GraspPredictions:
        """
        Single forward pass given explicit prompts.
        Used during inference and the inner loop of training.

        Args:
            coords        : [B, N, 3] XYZ coordinates, normalized to [-1, 1]
            features      : [B, N, 3] RGB values, normalized to [0, 1]
            prompt_coords : [B*M, K, 3] coordinates of prompt points
                            M = number of objects prompted per scene
                            K = number of prompt points per object
            prompt_labels : [B*M, K] label for each prompt point
                            1 = foreground (on the object)
                            0 = background
            prompt_masks  : [B*M, N] mask from previous iteration (or None)
            multimask_output : if True return 3 masks, else return 1

        Returns:
            GraspPredictions dataclass
        """

        # ── 1. Encode point cloud → patch embeddings ──────────────────────
        # pc_embeddings : [B, L, D]  patch-level features
        # patches       : dict with centers, knn_idx, fps_idx
        pc_embeddings, patches = self.pc_encoder(coords, features)

        centers  = patches["centers"]   # [B, L, 3]
        knn_idx  = patches["knn_idx"]   # [B, L, K_knn]

        aux_inputs = AuxInputs(
            coords   = coords,
            features = features,
            centers  = centers,
        )

        # ── 2. Positional encoding for patch centers ───────────────────────
        # pc_pe: [B, L, D]
        pc_pe = self.point_encoder.pe_layer(centers)

        # ── 3. Encode point prompts → sparse embeddings ────────────────────
        # sparse_embeddings: [B*M, K, D]
        sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)

        # ── 4. Encode mask prompt → dense embeddings ──────────────────────
        # dense_embeddings: [B*M, L, D] or [B, L, D] if prompt_masks=None
        dense_embeddings = self.mask_encoder(
            prompt_masks,
            coords,
            centers,
            knn_idx,
            center_idx=patches.get("fps_idx"),
        )

        # Expand dense embeddings from [B, L, D] to [B*M, L, D]
        M = sparse_embeddings.shape[0] // pc_embeddings.shape[0]
        dense_embeddings = repeat_interleave(dense_embeddings, M, dim=0)

        # ── 5. GraspDecoder3D ──────────────────────────────────────────────
        predictions = self.grasp_decoder(
            pc_embeddings            = pc_embeddings,
            pc_pe                    = pc_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings  = dense_embeddings,
            aux_inputs               = aux_inputs,
            multimask_output         = multimask_output,
        )

        return predictions

    def forward(
        self,
        coords:    torch.Tensor,   # [B, N, 3]
        features:  torch.Tensor,   # [B, N, 3]
        gt_masks:  torch.Tensor,   # [B, M, N] ground truth masks
        is_eval:   bool = False,
    ) -> List[GraspPredictions]:
        """
        Training forward pass with iterative prompt refinement.

        Same iterative strategy as PointSAM:
        - Iteration 0: sample prompts from gt_masks, use multimask_output=True
        - Iteration 1+: use previous prediction as mask prompt, multimask_output=False

        Args:
            coords   : [B, N, 3] point cloud coordinates
            features : [B, N, 3] point cloud RGB features
            gt_masks : [B, M, N] ground truth binary masks
                       M = number of objects in scene
            is_eval  : if True, use deterministic prompt sampling

        Returns:
            List of GraspPredictions, one per prompt iteration.
            Loss is computed over all iterations.
        """

        # ── Encode point cloud once — reused across all iterations ────────
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers  = patches["centers"]
        knn_idx  = patches["knn_idx"]
        pc_pe    = self.point_encoder.pe_layer(centers)

        aux_inputs = AuxInputs(
            coords   = coords,
            features = features,
            centers  = centers,
        )

        outputs      = []
        prompt_masks = None  # no mask prompt for first iteration

        for i in range(self.prompt_iters):

            # Sample prompt points from ground truth masks
            # Returns [B*M, K, 3] and [B*M, K]
            prompt_coords, prompt_labels = sample_prompts(
                coords,
                gt_masks,
                is_eval=is_eval,
            )

            # Encode prompts
            sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)
            dense_embeddings  = self.mask_encoder(
                prompt_masks,
                coords,
                centers,
                knn_idx,
                center_idx=patches.get("fps_idx"),
            )

            M = sparse_embeddings.shape[0] // pc_embeddings.shape[0]
            dense_embeddings = repeat_interleave(dense_embeddings, M, dim=0)

            # First iteration: multimask → pick best mask for next prompt
            # Later iterations: single mask → refine
            predictions = self.grasp_decoder(
                pc_embeddings            = pc_embeddings,
                pc_pe                    = pc_pe,
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings  = dense_embeddings,
                aux_inputs               = aux_inputs,
                multimask_output         = (i == 0),
            )

            outputs.append(predictions)

            # Use best mask from this iteration as prompt for next iteration
            if predictions.masks is not None:
                if i == 0:
                    # Pick the mask with highest predicted IoU
                    best_idx     = torch.argmax(predictions.iou_preds, dim=1)
                    prompt_masks = predictions.masks[
                        torch.arange(predictions.masks.shape[0]),
                        best_idx,
                    ]  # [B*M, N]
                else:
                    prompt_masks = predictions.masks[:, 0]  # [B*M, N]

        return outputs