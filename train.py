# train.py
"""
Training script for GraspPointSAM.
Follows GraspSAM's simple training style.

Usage:
    python train.py --scene_zip /path/to/train_1.zip \
                    --grasp_zip /path/to/grasp_label.zip
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from grasp_point_sam.model.build_model import build_model
from grasp_point_sam.data.graspnet_dataset import GraspNetDataset
from grasp_point_sam.losses.grasp_loss import GraspPointSAMLoss


# ─────────────────────────────────────────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='configs/graspnet.yaml')
    parser.add_argument('--scene_zip',  type=str, required=True)
    parser.add_argument('--grasp_zip',  type=str, required=True)
    parser.add_argument('--resume',     type=str, default=None)
    parser.add_argument('--save',       type=bool, default=True)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Collate function — pads variable M across batch
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    max_M = max(s['masks'].shape[0] for s in batch)

    coords_list   = []
    features_list = []
    masks_list    = []
    quality_list  = []
    rotation_list = []
    width_list    = []
    valid_list    = []

    for s in batch:
        M, N = s['masks'].shape
        pad  = max_M - M

        coords_list.append(s['coords'])
        features_list.append(s['features'])

        masks_list.append(torch.cat([
            s['masks'], torch.zeros(pad, N)
        ], dim=0))

        quality_list.append(torch.cat([
            s['grasp_quality'], torch.zeros(pad, N)
        ], dim=0))

        rotation_list.append(torch.cat([
            s['grasp_rotation'], torch.zeros(pad, N, 6)
        ], dim=0))

        width_list.append(torch.cat([
            s['grasp_width'], torch.zeros(pad, N, 1)
        ], dim=0))

        # False for padded (fake) objects so the loss can ignore them
        valid_list.append(torch.arange(max_M) < M)

    return {
        'coords':         torch.stack(coords_list),
        'features':       torch.stack(features_list),
        'masks':          torch.stack(masks_list),
        'grasp_quality':  torch.stack(quality_list),
        'grasp_rotation': torch.stack(rotation_list),
        'grasp_width':    torch.stack(width_list),
        'mask_valid':     torch.stack(valid_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, criterion, loader, optimizer, scaler, device, epoch, cfg):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader):
        coords      = batch['coords'].to(device)
        features    = batch['features'].to(device)
        masks       = batch['masks'].to(device)
        gt_quality  = batch['grasp_quality'].to(device)
        gt_rotation = batch['grasp_rotation'].to(device)
        gt_width    = batch['grasp_width'].to(device)
        mask_valid  = batch['mask_valid'].to(device)

        # Flatten B*M
        gt_masks_flat    = masks.flatten(0, 1).bool()
        gt_quality_flat  = gt_quality.flatten(0, 1)
        gt_rotation_flat = gt_rotation.flatten(0, 1)
        gt_width_flat    = gt_width.flatten(0, 1)
        mask_valid_flat  = mask_valid.flatten(0, 1)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(cfg.training.mixed_precision == 'fp16')):
            outputs   = model(coords=coords, features=features, gt_masks=masks)
            loss_dict = criterion(
                outputs      = outputs,
                gt_masks     = gt_masks_flat,
                gt_quality   = gt_quality_flat,
                gt_rotation  = gt_rotation_flat,
                gt_width     = gt_width_flat,
                valid        = mask_valid_flat,
            )

        loss = loss_dict['loss']
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.training.max_grad_value
        )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % cfg.training.log_every_n_steps == 0:
            it = loss_dict['iterations'][-1]
            print(
                f"Epoch [{epoch}] Step [{step}/{len(loader)}] "
                f"Loss: {loss.item():.4f} | "
                f"Grasp: {it['grasp_loss']:.4f} | "
                f"Mask: {it['mask_loss']:.4f}"
            )

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, criterion, loader, device, cfg):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        coords      = batch['coords'].to(device)
        features    = batch['features'].to(device)
        masks       = batch['masks'].to(device)
        gt_quality  = batch['grasp_quality'].to(device)
        gt_rotation = batch['grasp_rotation'].to(device)
        gt_width    = batch['grasp_width'].to(device)
        mask_valid  = batch['mask_valid'].to(device)

        gt_masks_flat    = masks.flatten(0, 1).bool()
        gt_quality_flat  = gt_quality.flatten(0, 1)
        gt_rotation_flat = gt_rotation.flatten(0, 1)
        gt_width_flat    = gt_width.flatten(0, 1)
        mask_valid_flat  = mask_valid.flatten(0, 1)

        with torch.cuda.amp.autocast(enabled=(cfg.training.mixed_precision == 'fp16')):
            outputs   = model(coords=coords, features=features, gt_masks=masks)
            loss_dict = criterion(
                outputs      = outputs,
                gt_masks     = gt_masks_flat,
                gt_quality   = gt_quality_flat,
                gt_rotation  = gt_rotation_flat,
                gt_width     = gt_width_flat,
                valid        = mask_valid_flat,
            )

        total_loss += loss_dict['loss'].item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output dir
    output_dir = Path(cfg.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model
    print("Building model...")
    model = build_model(cfg).to(device)

    # Loss
    criterion = GraspPointSAMLoss(
        grasp_scale      = 2.0,
        mask_scale       = 1.0,
        joint_mask_grasp = cfg.model.joint_mask_grasp,
    ).to(device)

    # Optimizer — only trainable params (grasp_decoder)
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr           = cfg.training.learning_rate,
        weight_decay = cfg.training.weight_decay,
    )

    # Scheduler — same as GraspSAM
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg.training.max_epochs,
        eta_min = 1e-7,
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(
        enabled=(cfg.training.mixed_precision == 'fp16')
    )

    # Resume
    start_epoch   = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Datasets
    print("Loading datasets...")
    train_dataset = GraspNetDataset(
        scene_zip_path = args.scene_zip,
        grasp_zip_path = args.grasp_zip,
        num_points     = cfg.training.num_points,
        num_frames     = 1,
        split          = 'train',
        augment        = True,
    )
    val_dataset = GraspNetDataset(
        scene_zip_path = args.scene_zip,
        grasp_zip_path = args.grasp_zip,
        num_points     = cfg.training.num_points,
        num_frames     = 1,
        split          = 'val',
        augment        = False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.training.batch_size,
        shuffle     = True,
        num_workers = 0,           # ← change from cfg.data.num_workers to 0
        collate_fn  = collate_fn,
        pin_memory  = False,       # ← disable pin_memory when num_workers=0
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.training.batch_size,
        shuffle     = False,
        num_workers = 0,           # ← change to 0
        collate_fn  = collate_fn,
        pin_memory  = False,       # ← disable
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print("-" * 60)

    # Training loop
    for epoch in range(start_epoch, cfg.training.max_epochs):
        print(f"\nEpoch {epoch}/{cfg.training.max_epochs} | LR: {scheduler.get_last_lr()[0]:.2e}")
        print("-" * 60)

        train_loss = train_one_epoch(
            model, criterion, train_loader,
            optimizer, scaler, device, epoch, cfg
        )
        scheduler.step()
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if epoch % cfg.training.val_every_n_epochs == 0:
            val_loss = validate(model, criterion, val_loader, device, cfg)
            print(f"Val Loss:   {val_loss:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"New best: {best_val_loss:.4f}")
        else:
            is_best = False

        # Save
        if args.save and (epoch % cfg.training.save_freq == 0 or is_best):
            ckpt_path = output_dir / f'checkpoint_epoch_{epoch:04d}.pth'
            torch.save({
                'epoch':         epoch,
                'model':         model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'best_val_loss': best_val_loss,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

            if is_best:
                torch.save({
                    'epoch':         epoch,
                    'model':         model.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'scheduler':     scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, output_dir / 'best_model.pth')
                print(f"Saved best model.")

    print("\nTraining complete.")


if __name__ == '__main__':
    main()