# grasp_point_sam/model/grasp_point_sam.py
"""
GraspPointSAM - Top-level model.

Combines PointSAM's pretrained 3D point cloud encoder with a new
GraspDecoder3D head for 6-DOF grasp detection.

Pipeline:
    Input: [B, N, 6] point cloud (XYZ + RGB)
        
    PointCloudEncoder  -> patch embeddings [B, L, D]
        
    PromptEncoder      -> sparse + dense prompt embeddings
        
    GraspDecoder3D     -> grasp quality, rotation, width (+ masks if joint)
        
    Output: GraspPredictions
"""

import torch
import torch.nn as nn
from typing import List, Optional
from safetensors.torch import load_file

from .pc_encoder import PointCloudEncoder
from .prompt_encoder import PointEncoder, MaskEncoder
from .mask_decoder import AuxInputs, MaskDecoder
from .transformer import TwoWayTransformer
from .grasp_decoder import GraspDecoder3D, GraspPredictions
from .common import sample_prompts, repeat_interleave


class GraspPointSAM(nn.Module):
    """
    GraspPointSAM model.

    Args:
        pc_encoder : PointCloudEncoder (from PointSAM)
        point_encoder : PointEncoder for prompt + positional encoding
        mask_encoder : MaskEncoder for dense prompt embeddings
        mask_decoder : MaskDecoder (PointSAM original, loads pretrained weights)
        grasp_decoder : GraspDecoder3D (new, randomly initialized)
        prompt_iters : number of iterative prompt refinement steps
        joint_mask_grasp : if True, predict masks alongside grasps
    """

    def __init__(
        self,
        pc_encoder: PointCloudEncoder,
        point_encoder: PointEncoder,
        mask_encoder: MaskEncoder,
        mask_decoder: MaskDecoder,
        grasp_decoder: GraspDecoder3D,
        prompt_iters: int  = 3,
        joint_mask_grasp: bool = True,
    ):
        super().__init__()

        self.pc_encoder = pc_encoder
        self.point_encoder = point_encoder    # explicitly passed in
        self.mask_encoder = mask_encoder
        self.mask_decoder = mask_decoder
        self.grasp_decoder = grasp_decoder
        self.prompt_iters = prompt_iters
        self.joint_mask_grasp = joint_mask_grasp

    # Pretrained weight loading
    def load_pretrained_pointsam(self, ckpt_path: str) -> None:
        """
        Load PointSAM pretrained weights into the encoder components.

        Uses strict=False so that:
        - Matching keys (pc_encoder, point_encoder, mask_encoder, mask_decoder)
          are loaded from checkpoint
        - Missing keys (grasp_decoder.*) are left randomly initialized
        - Unexpected keys in checkpoint are ignored

        Args:
            ckpt_path: path to model.safetensors
        """
        print(f"Loading PointSAM weights from: {ckpt_path}")

        state_dict = load_file(ckpt_path)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        # Only grasp_decoder keys should be missing
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
            print(f"Loaded: {len(state_dict) - len(unexpected)} keys")
            print(f"Initialized: {len(missing)} grasp_decoder keys (random init)")

    # Encoder freeze / unfreeze
    def freeze_encoder(self) -> None:
        """Freeze all PointSAM components — only grasp_decoder trains."""
        for param in self.pc_encoder.parameters():
            param.requires_grad = False
        for param in self.point_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_encoder.parameters():
            param.requires_grad = False
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        print("Encoder frozen. Only GraspDecoder3D will be trained.")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all components for fine-tuning stage."""
        for param in self.pc_encoder.parameters():
            param.requires_grad = True
        for param in self.point_encoder.parameters():
            param.requires_grad = True
        for param in self.mask_encoder.parameters():
            param.requires_grad = True
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
        print("Encoder unfrozen.")

    # Core forward — used during inference
    def predict(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        prompt_coords: torch.Tensor,
        prompt_labels: torch.Tensor,
        prompt_masks: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
    ) -> GraspPredictions:
        """
        Single forward pass given explicit prompts.
        Used during inference and the inner loop of training.

        Args:
            coords : [B, N, 3] XYZ coordinates, normalized to [-1, 1]
            features : [B, N, 3] RGB values, normalized to [0, 1]
            prompt_coords : [B*M, K, 3] 3D prompt point coordinates
            prompt_labels : [B*M, K] 1=foreground, 0=background
            prompt_masks : [B*M, N] mask from previous iteration (or None)
            multimask_output : if True return 3 masks, else 1

        Returns:
            GraspPredictions dataclass
        """

        # 1. Encode point cloud → patch embeddings
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers = patches["centers"]   # [B, L, 3]
        knn_idx = patches["knn_idx"]   # [B, L, K_knn]

        aux_inputs = AuxInputs(
            coords = coords,
            features = features,
            centers = centers,
        )

        # 2. Positional encoding for patch centers
        pc_pe = self.point_encoder.pe_layer(centers)  # [B, L, D]

        # 3. Encode point prompts → sparse embeddings
        sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)

        # 4. Encode mask prompt → dense embeddings
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

        # 5. GraspDecoder3D
        predictions = self.grasp_decoder(
            pc_embeddings = pc_embeddings,
            pc_pe = pc_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings = dense_embeddings,
            aux_inputs = aux_inputs,
            multimask_output = multimask_output,
        )

        return predictions

    # Training forward — iterative prompt refinement
    def forward(
        self,
        coords:   torch.Tensor,
        features: torch.Tensor,
        gt_masks: torch.Tensor,
        is_eval:  bool = False,
    ) -> List[GraspPredictions]:
        """
        Training forward pass with iterative prompt refinement.

        Args:
            coords : [B, N, 3] point cloud coordinates
            features : [B, N, 3] point cloud RGB features
            gt_masks : [B, M, N] ground truth binary masks
            is_eval : if True, use deterministic prompt sampling

        Returns:
            List of GraspPredictions, one per prompt iteration.
        """

        # Encode point cloud once — reused across all iterations
        pc_embeddings, patches = self.pc_encoder(coords, features)
        centers = patches["centers"]
        knn_idx = patches["knn_idx"]
        pc_pe = self.point_encoder.pe_layer(centers)

        aux_inputs = AuxInputs(
            coords = coords,
            features = features,
            centers = centers,
        )

        outputs = []
        prompt_masks = None

        for i in range(self.prompt_iters):

            # Sample prompt points from ground truth masks
            prompt_coords, prompt_labels = sample_prompts(
                coords,
                gt_masks,
                is_eval=is_eval,
            )

            # Encode prompts
            sparse_embeddings = self.point_encoder(prompt_coords, prompt_labels)
            dense_embeddings = self.mask_encoder(
                prompt_masks,
                coords,
                centers,
                knn_idx,
                center_idx=patches.get("fps_idx"),
            )

            M = sparse_embeddings.shape[0] // pc_embeddings.shape[0]
            dense_embeddings = repeat_interleave(dense_embeddings, M, dim=0)

            predictions = self.grasp_decoder(
                pc_embeddings = pc_embeddings,
                pc_pe = pc_pe,
                sparse_prompt_embeddings = sparse_embeddings,
                dense_prompt_embeddings = dense_embeddings,
                aux_inputs = aux_inputs,
                multimask_output = (i == 0),
            )

            outputs.append(predictions)

            # Use best mask as prompt for next iteration
            if predictions.masks is not None:
                if i == 0:
                    best_idx = torch.argmax(predictions.iou_preds, dim=1)
                    prompt_masks = predictions.masks[
                        torch.arange(predictions.masks.shape[0]),
                        best_idx,
                    ]
                else:
                    prompt_masks = predictions.masks[:, 0]

        return outputs