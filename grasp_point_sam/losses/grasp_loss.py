# grasp_point_sam/losses/grasp_loss.py
"""
Loss functions for GraspPointSAM.

Adapted from GraspSAM's loss design but modified for 3D point clouds:
- GraspSAM uses weighted MSE on 2D grasp maps (image space)
- We use weighted MSE on per-point 3D grasp predictions (point space)

For masks, we follow PointSAM's loss:
- Focal loss + Dice loss (same as SAM)
- MSE on IoU prediction head

References:
- GraspSAM (ICRA 2025): weighted_mse, grasp loss structure
- PointSAM: focal loss, dice loss, IoU prediction loss
- SAM: multi-mask selection strategy
"""

import dataclasses
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_min
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Core loss primitives
# ─────────────────────────────────────────────────────────────────────────────

def dice_loss(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    eps:       float = 1e-3,
    reduction: str   = 'mean',
) -> torch.Tensor:
    """
    Dice loss for binary segmentation.
    Same as PointSAM/SAM implementation.

    Args:
        logits:  [..., N] predicted logits (before sigmoid)
        targets: [..., N] ground truth binary masks
        eps:     smoothing factor
        reduction: 'mean', 'none'

    Returns:
        scalar loss or [...] per-sample loss
    """
    probs       = logits.sigmoid()
    numerator   = 2 * (probs * targets).sum(-1)
    denominator = (probs + targets).sum(-1)
    loss        = 1 - (numerator + eps) / (denominator + eps)

    if reduction == 'mean':
        return loss.mean()
    return loss


def sigmoid_focal_loss(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    alpha:     float = 0.25,
    gamma:     float = 2.0,
    reduction: str   = 'mean',
) -> torch.Tensor:
    """
    Sigmoid focal loss — handles class imbalance by down-weighting
    easy negatives. Same as PointSAM's mask loss.

    For grasp quality: most points are background (quality=0),
    focal loss prevents the model from predicting all zeros.

    Args:
        logits:  [..., N] predicted logits
        targets: [..., N] ground truth in [0, 1]
        alpha:   weighting factor for positive class
                 -1 means no alpha weighting (used by PointSAM for masks)
        gamma:   focusing parameter (2.0 is standard)
        reduction: 'mean', 'sum', 'none'

    Returns:
        scalar loss
    """
    p        = torch.sigmoid(logits)
    ce_loss  = F.binary_cross_entropy_with_logits(
        logits, targets, reduction='none'
    )
    p_t      = p * targets + (1 - p) * (1 - targets)
    loss     = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss    = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


def weighted_mse(
    pred:   torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted MSE loss — same concept as GraspSAM's weighted_mse
    but adapted for point clouds instead of 2D maps.

    GraspSAM separates foreground/background and weights differently.
    We do the same: only compute regression loss where weight > 0
    (i.e. where there are valid grasps).

    Args:
        pred:   [B*M, N, D] or [B*M, N] predictions
        target: [B*M, N, D] or [B*M, N] ground truth
        weight: [B*M, N] per-point weight (grasp quality score)
                Points with weight=0 have no valid grasp — ignored.

    Returns:
        scalar weighted MSE loss
    """
    # Boolean mask of points with valid grasps
    valid = weight > 0  # [B*M, N]

    if valid.sum() == 0:
        warnings.warn(
            "weighted_mse: no point in the batch has a valid grasp "
            "(all gt_quality == 0), grasp loss term is 0 for this batch"
        )
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    # Foreground loss — only on valid grasp points
    if pred.dim() == 3:
        # [B*M, N, D] case (rotation)
        valid_exp = valid.unsqueeze(-1).expand_as(pred)
        fore_pred = pred[valid_exp].view(-1, pred.shape[-1])
        fore_gt   = target[valid_exp].view(-1, target.shape[-1])
    else:
        # [B*M, N] case (quality, width)
        fore_pred = pred[valid]
        fore_gt   = target[valid]

    fore_loss = F.mse_loss(fore_pred, fore_gt)

    # Background loss — on invalid points (no valid grasp)
    # Weight background much lower (same as GraspSAM's weight parameter)
    bg_weight = 0.1  # background contributes 10x less than foreground

    if pred.dim() == 3:
        invalid_exp = (~valid).unsqueeze(-1).expand_as(pred)
        back_pred   = pred[invalid_exp].view(-1, pred.shape[-1])
        back_gt     = target[invalid_exp].view(-1, target.shape[-1])
    else:
        back_pred = pred[~valid]
        back_gt   = target[~valid]

    if len(back_pred) > 0:
        back_loss = F.mse_loss(back_pred, back_gt)
    else:
        back_loss = torch.tensor(0.0, device=pred.device)

    return fore_loss + bg_weight * back_loss


def nearest_grasp_loss(
    pred_rotation: torch.Tensor,
    pred_width:    torch.Tensor,
    point_idx:     torch.Tensor,
    gt_rotation:   torch.Tensor,
    gt_width:      torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotation and width loss when a point has several valid grasps.

    Each labelled point is compared with ALL of its ground-truth grasps
    and only the closest one (smallest rotation error) is penalised, so
    predicting any valid grasp is correct. Regressing to one arbitrary
    grasp would instead pull the prediction towards the average of
    several valid rotations, which is often not a valid grasp.
    Same idea as Contact-GraspNet's min-over-ground-truth loss.

    Args:
        pred_rotation: [B*M, N, 6] predicted 6D rotation
        pred_width:    [B*M, N, 1] predicted width
        point_idx:     [T] flat index (into B*M*N) of each ground-truth grasp
        gt_rotation:   [T, 6] 6D rotation of each ground-truth grasp
        gt_width:      [T]    width of each ground-truth grasp

    Returns:
        rotation_loss, width_loss (scalars)
    """
    if point_idx.numel() == 0:
        warnings.warn(
            "nearest_grasp_loss: no point in the batch has a ground-truth "
            "grasp, rotation and width loss are 0 for this batch"
        )
        zero = pred_rotation.sum() * 0.0
        return zero, zero

    pred_rot = pred_rotation.reshape(-1, 6)[point_idx].float()      # [T, 6]
    rot_err  = (pred_rot - gt_rotation.float()).square().mean(-1)   # [T]

    # group the grasps by point and keep the closest one per point
    _, group      = torch.unique(point_idx, return_inverse=True)
    min_err, best = scatter_min(rot_err, group, dim=0)              # [U]

    rotation_loss = min_err.mean()

    pred_w     = pred_width.reshape(-1)[point_idx[best]].float()    # [U]
    width_loss = F.mse_loss(pred_w, gt_width[best].float())

    return rotation_loss, width_loss


def compute_iou(
    logits:  torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Compute hard IoU between predicted masks and ground truth.
    Same as PointSAM's compute_iou.

    Args:
        logits:  [..., N] predicted logits
        targets: [..., N] bool ground truth masks

    Returns:
        [...] IoU scores
    """
    preds = logits > 0  # threshold at 0
    return (
        (preds & targets).sum(-1).float() /
        ((preds | targets).sum(-1).float() + 1e-6)
    )


def compute_jaccard(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    eps:     float = 1e-3,
) -> torch.Tensor:
    """
    Soft (differentiable) IoU using probabilities.
    Used for IoU prediction head supervision.
    Same as PointSAM's compute_jaccard.

    Args:
        logits:  [..., N] predicted logits
        targets: [..., N] float ground truth masks

    Returns:
        [...] soft IoU scores
    """
    probs       = logits.sigmoid()
    numerator   = (probs * targets).sum(-1)
    denominator = (
        probs.square() + targets.square()
    ).sum(-1) - numerator
    return (numerator + eps) / (denominator + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Mask loss (from PointSAM)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mask_loss(
    logits:           torch.Tensor,
    targets:          torch.Tensor,
    loss_weight_dice: float = 1.0,
) -> torch.Tensor:
    """
    Combined focal + dice loss for segmentation masks.
    Same as PointSAM's compute_mask_loss.

    Args:
        logits:  [B*M, C, N] multi-mask predicted logits
        targets: [B*M, N] ground truth binary masks
        loss_weight_dice: weight for dice loss component

    Returns:
        [B*M, C] per-mask loss
    """
    assert logits.dim() == 3, logits.shape
    _targets = targets.unsqueeze(1).expand_as(logits).to(dtype=logits.dtype)

    loss_focal = sigmoid_focal_loss(
        logits, _targets, alpha=-1, reduction='none'
    ).mean(-1)  # [B*M, C]

    loss_dice = dice_loss(
        logits, _targets, reduction='none'
    )  # [B*M, C]

    return loss_focal + loss_weight_dice * loss_dice


# ─────────────────────────────────────────────────────────────────────────────
# Main loss criterion
# ─────────────────────────────────────────────────────────────────────────────

class GraspPointSAMLoss(nn.Module):
    """
    Combined loss for GraspPointSAM.

    Adapts GraspSAM's loss structure to 3D point clouds:

    GraspSAM (2D):
        grasp_loss = weighted_mse(pos) + weighted_mse(cos) +
                     weighted_mse(sin) + weighted_mse(width)
        mask_loss  = dice + bce

    Ours (3D):
        grasp_loss = weighted_mse(quality) +
                     nearest_grasp_loss(rotation, width)  ← all gt grasps per point
        mask_loss  = focal + dice + mse(iou_pred)  ← from PointSAM

    Total = grasp_scale * grasp_loss + mask_scale * mask_loss

    Args:
        grasp_scale:    weight for total grasp loss (default 2.0, same as GraspSAM)
        mask_scale:     weight for total mask loss  (default 1.0, same as GraspSAM)
        quality_scale:  weight for quality loss component
        rotation_scale: weight for rotation loss component
        width_scale:    weight for width loss component
        joint_mask_grasp: whether mask loss is included
        use_soft_iou:   use soft IoU for IoU head supervision (from PointSAM)
    """

    def __init__(
        self,
        grasp_scale:    float = 2.0,
        mask_scale:     float = 1.0,
        quality_scale:  float = 1.0,
        rotation_scale: float = 1.0,
        width_scale:    float = 1.0,
        joint_mask_grasp: bool = True,
        use_soft_iou:   bool  = True,
    ):
        super().__init__()
        self.grasp_scale    = grasp_scale
        self.mask_scale     = mask_scale
        self.quality_scale  = quality_scale
        self.rotation_scale = rotation_scale
        self.width_scale    = width_scale
        self.joint_mask_grasp = joint_mask_grasp
        self.use_soft_iou   = use_soft_iou

    def forward(
        self,
        outputs:   List,           # List of GraspPredictions from forward()
        gt_masks:  torch.Tensor,   # [B*M, N] ground truth masks
        gt_quality:  torch.Tensor, # [B*M, N] ground truth grasp quality
        grasp_point_idx: torch.Tensor,  # [T] flat index into B*M*N
        gt_rotation: torch.Tensor, # [T, 6] 6D rotation of each gt grasp
        gt_width:    torch.Tensor, # [T] width of each gt grasp
        valid:       Optional[torch.Tensor] = None,  # [B*M] bool, False = padding
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss over all prompt iterations.

        Same multi-iteration strategy as PointSAM:
        - Iteration 0: multimask output → select best mask
        - Later iterations: single mask → refine
        - Loss averaged over all iterations

        Args:
            outputs:     List[GraspPredictions] one per prompt iteration
            gt_masks:    [B*M, N] bool ground truth segmentation masks
            gt_quality:  [B*M, N] float grasp quality (0=no grasp, >0=quality)
            grasp_point_idx: [T] flat index (into B*M*N) of the point each
                         ground-truth grasp belongs to; a point can have
                         many grasps
            gt_rotation: [T, 6] 6D rotation of each ground-truth grasp
            gt_width:    [T] gripper width of each ground-truth grasp
            valid:       [B*M] bool, False for objects padded in by
                         collate_fn when batch_size > 1. These rows are
                         dropped so padding never affects the loss.

        Returns:
            dict with 'loss' (scalar) and sub-losses for logging
        """
        total_loss    = 0.0
        all_losses    = []

        if valid is not None:
            gt_masks    = gt_masks[valid]
            gt_quality  = gt_quality[valid]
            # gt grasps only exist on real objects, never on padded rows,
            # so grasp_point_idx / gt_rotation / gt_width need no filtering

        for i, predictions in enumerate(outputs):

            # before padding rows are dropped: grasp_point_idx indexes the
            # full B*M*N layout
            rotation_loss, width_loss = nearest_grasp_loss(
                pred_rotation = predictions.grasp_rotation,  # [B*M, N, 6]
                pred_width    = predictions.grasp_width,     # [B*M, N, 1]
                point_idx     = grasp_point_idx,             # [T]
                gt_rotation   = gt_rotation,                 # [T, 6]
                gt_width      = gt_width,                    # [T]
            )

            if valid is not None:
                predictions = dataclasses.replace(predictions, **{
                    f.name: getattr(predictions, f.name)[valid]
                    for f in dataclasses.fields(predictions)
                    if getattr(predictions, f.name) is not None
                })

            # ── Grasp losses ─────────────────────────────────────────────
            # Quality: GraspSAM's weighted_mse, gt_quality as the weight.
            # Rotation / width: nearest_grasp_loss above, only on points
            # that have grasps.

            quality_loss = weighted_mse(
                pred   = predictions.grasp_quality,   # [B*M, N]
                target = gt_quality,                   # [B*M, N]
                weight = gt_quality,                   # supervise all points
            )

            grasp_loss = (
                self.quality_scale  * quality_loss  +
                self.rotation_scale * rotation_loss +
                self.width_scale    * width_loss
            )

            # ── Mask losses (joint mode, from PointSAM) ──────────────────
            mask_loss = torch.tensor(0.0, device=gt_masks.device)

            if self.joint_mask_grasp and predictions.masks is not None:
                masks    = predictions.masks    # [B*M, C, N]
                iou_pred = predictions.iou_preds  # [B*M, C]

                loss_per_mask = compute_mask_loss(masks, gt_masks)
                # [B*M, C]

                if i == 0:
                    # loss_per_mask: [B*M, C]
                    loss_per_mask, best_idx = loss_per_mask.min(dim=1)  # [B*M]
                    batch_idx  = torch.arange(
                        best_idx.shape[0], device=masks.device
                    )
                    best_masks = masks[batch_idx, best_idx]   # [B*M, N]  ← select one mask
                    iou_pred   = iou_pred[batch_idx, best_idx] # [B*M]    ← select corresponding iou
                else:
                    loss_per_mask = loss_per_mask.squeeze(1)
                    best_masks    = masks[:, 0, :]             # [B*M, N]  ← take first mask
                    iou_pred      = iou_pred[:, 0]             # [B*M]

                loss_mask = loss_per_mask.mean()

                # IoU prediction head loss (same as PointSAM)
                if self.use_soft_iou:
                    with torch.no_grad():
                        # best_masks: [B*M, N], gt_masks: [B*M, N]
                        assert best_masks.shape == gt_masks.shape, \
                            f"Shape mismatch: {best_masks.shape} vs {gt_masks.shape}"
                        soft_iou = compute_jaccard(
                            best_masks,
                            gt_masks.to(dtype=best_masks.dtype)
                        )  # [B*M]
                    # iou_pred: [B*M]
                    assert iou_pred.shape == soft_iou.shape, \
                        f"IoU shape mismatch: {iou_pred.shape} vs {soft_iou.shape}"
                    loss_iou = F.mse_loss(iou_pred, soft_iou)
                else:
                    hard_iou = compute_iou(
                        best_masks,
                        gt_masks.bool()
                    )
                    loss_iou = F.mse_loss(iou_pred, hard_iou)

                mask_loss = loss_mask + loss_iou

            # ── Total loss for this iteration ────────────────────────────
            iter_loss = (
                self.grasp_scale * grasp_loss +
                self.mask_scale  * mask_loss
            )

            total_loss = total_loss + iter_loss
            all_losses.append({
                'iter':          i,
                'grasp_loss':    grasp_loss.item(),
                'quality_loss':  quality_loss.item(),
                'rotation_loss': rotation_loss.item(),
                'width_loss':    width_loss.item(),
                'mask_loss':     mask_loss.item()
                                 if isinstance(mask_loss, torch.Tensor)
                                 else mask_loss,
            })

        # Average over all iterations
        total_loss = total_loss / len(outputs)

        return {
            'loss':       total_loss,
            'iterations': all_losses,
        }