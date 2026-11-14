# sim_grasp.py
"""
Tabletop grasping with GraspPointSAM and a Franka Panda in PyBullet.

  1. GraspNet-1Billion objects (--objects, model ids) are dropped at random
     poses on a table in front of the Panda
  2. a fixed RGB-D camera (GraspNet's RealSense intrinsics) takes a picture;
     the point cloud is built as in eval.py (random/FPS sampling, optional
     workspace crop), with depth noise and mm quantisation as a real sensor
  3. the model predicts grasps, one prompt click per object (from the
     simulated segmentation)
  4. the best-scoring grasp that fits the hand (<= 8 cm), has an IK solution
     and is collision-free is executed: approach, close, lift; a lifted object
     is carried aside, released and removed
  5. repeat from 2 until the table is clear, no grasp is feasible, or
     --max_attempts is reached

Run in the graspsim conda env (has pybullet):
  conda activate graspsim
  python sim_grasp.py --ckpt checkpoints_test/2026-11-08_22-56-41/best_model.safetensors \
      --config configs/graspnet_test.yaml
  python sim_grasp.py ... --objects 9 11 29 58 62 --seed 3
  python sim_grasp.py ... --headless          # no window
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p
import torch
from omegaconf import OmegaConf
from PIL import Image

from eval import checkpoint_has_depth_head, load_model, predict_grasps
from grasp_point_sam.data.graspnet_dataset import WORKSPACE_MARGIN, depth_to_pointcloud
from sim.world import CAM_K, TabletopWorld


def observe(world: TabletopWorld, args, rng):
    """Camera picture -> rgb, seg image, camera-frame points, colours, per-point labels."""
    rgb, z, seg = world.capture()
    if args.depth_noise > 0:
        z = np.where(z > 0, z + rng.normal(0.0, args.depth_noise, z.shape), 0.0)
    depth_mm = np.clip(np.round(z * 1000.0), 0, 65535).astype(np.uint16)

    coords, colors = depth_to_pointcloud(depth_mm, rgb, CAM_K, 1000.0)
    seg_flat = seg.reshape(-1)
    valid = coords[:, 2] > 0
    coords, colors, seg_flat = coords[valid], colors[valid], seg_flat[valid]

    if args.workspace_crop:   # as eval.workspace_mask: objects' bounding box + margin (world frame)
        T = world.T_world_cam
        pts = coords @ T[:3, :3].T + T[:3, 3]
        fg = pts[seg_flat > 0]
        if len(fg):
            lo, hi = fg.min(axis=0) - WORKSPACE_MARGIN, fg.max(axis=0) + WORKSPACE_MARGIN
            keep = np.all((pts > lo) & (pts < hi), axis=1)
            coords, colors, seg_flat = coords[keep], colors[keep], seg_flat[keep]
    return rgb, seg, coords, colors, seg_flat


def main():
    ap = argparse.ArgumentParser(description='GraspPointSAM + Franka Panda grasping in PyBullet')
    ap.add_argument('--ckpt', required=True, help='best_model.safetensors or a .pth checkpoint')
    ap.add_argument('--config', default='configs/graspnet_test.yaml',
                    help='model architecture + local paths; sampling/crop must match training')
    ap.add_argument('--objects', type=int, nargs='+', default=[9, 11, 29, 58, 62],
                    help='GraspNet-1Billion model ids (folders in <graspnet_root>/models)')
    ap.add_argument('--seed', type=int, default=0, help='object placement and point sampling')
    ap.add_argument('--max_attempts', type=int, default=None,
                    help='grasp attempts before giving up (default: 2 x number of objects)')
    ap.add_argument('--candidates', type=int, default=50,
                    help='best-scoring grasps checked for feasibility per attempt')
    ap.add_argument('--sampling', choices=['random', 'fps'], default=None,
                    help='default: from the config')
    ap.add_argument('--workspace_crop', action='store_true', default=None,
                    help='default: from the config')
    ap.add_argument('--depth', type=float, default=0.02,
                    help='fixed gripper depth (m), only for checkpoints without a depth head')
    ap.add_argument('--grasps_per_object', type=int, default=100,
                    help='grasps kept per object after NMS (as eval.py)')
    ap.add_argument('--friction', type=float, default=0.5, help='object friction coefficient')
    ap.add_argument('--close_force', type=float, default=20.0,
                    help='grip force per finger (N); 20 = Franka Hand URDF limit')
    ap.add_argument('--collision_tol', type=float, default=0.005,
                    help='allowed arm/scene overlap (m); object collision shapes are convex '
                         'decompositions, a few mm larger than the meshes')
    ap.add_argument('--depth_noise', type=float, default=0.001, help='depth noise std (m)')
    ap.add_argument('--headless', action='store_true', help='no PyBullet window')
    ap.add_argument('--no_fp16', dest='fp16', action='store_false')
    ap.add_argument('--asset_cache', default='sim_assets')
    ap.add_argument('--out_dir', default='sim_results',
                    help='camera pictures and a log of the attempts')
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    args.graspnet_root = cfg.data.graspnet_root
    args.num_points    = cfg.training.num_points
    args.voxel_size    = cfg.data.get('voxel_size', 0.005)
    args.sampling      = args.sampling or cfg.data.get('sampling', 'random')
    if args.workspace_crop is None:
        args.workspace_crop = bool(cfg.data.get('workspace_crop', False))
    args.has_depth     = checkpoint_has_depth_head(args.ckpt)
    max_attempts       = args.max_attempts or 2 * len(args.objects)

    out_dir = Path(args.out_dir) / time.strftime('%Y-%m-%d_%H-%M-%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'objects {args.objects}, sampling={args.sampling}, '
          f'workspace_crop={args.workspace_crop}, output: {out_dir}')

    model = load_model(cfg, args.ckpt, 'cuda', args.has_depth)
    rng   = np.random.default_rng(args.seed)
    world = TabletopWorld(gui=not args.headless, friction=args.friction)

    log = []
    try:
        world.spawn_objects(args.objects, args.graspnet_root, args.asset_cache, rng)
        n_objects = len(world.objects)
        print(f'{n_objects} objects on the table: {sorted(world.objects)}')

        for attempt in range(max_attempts):
            if not world.objects:
                break
            rgb, seg, coords, colors, seg_flat = observe(world, args, rng)
            Image.fromarray(rgb).save(out_dir / f'attempt_{attempt:02d}_rgb.png')

            np.random.seed(args.seed * 1000 + attempt)
            torch.manual_seed(args.seed * 1000 + attempt)
            grasps = predict_grasps(model, args, coords, colors, seg_flat, 'cuda')
            if len(grasps) == 0:
                print('no grasps predicted, stopping')
                break

            plan, rejected = None, Counter()
            for i in np.argsort(-grasps[:, 0])[:args.candidates]:
                res = world.plan_grasp(grasps[i], args.collision_tol)
                if isinstance(res, str):
                    rejected[res] += 1
                else:
                    plan = res
                    break
            if plan is None:
                print(f'attempt {attempt}: no feasible grasp among the top {args.candidates} '
                      f'({dict(rejected)}), stopping')
                log.append({'attempt': attempt, 'result': 'no_feasible_grasp',
                            'rejected': dict(rejected)})
                break

            world.draw_grasp(plan)
            ok = world.execute(plan, close_force=args.close_force)
            print(f'attempt {attempt}: object {plan.obj_id} score {plan.grasp[0]:.3f} '
                  f'width {plan.grasp[1] * 100:.1f} cm -> {"lifted" if ok else "failed"} '
                  f'(skipped {dict(rejected)}), {len(world.objects)} left')
            log.append({'attempt': attempt, 'object': plan.obj_id,
                        'score': float(plan.grasp[0]), 'width': float(plan.grasp[1]),
                        'result': 'lifted' if ok else 'failed', 'rejected': dict(rejected)})

        cleared = n_objects - len(world.objects)
        attempts = sum(1 for e in log if e['result'] in ('lifted', 'failed'))
        summary = {'objects': n_objects, 'cleared': cleared, 'attempts': attempts,
                   'success_rate': cleared / max(attempts, 1)}
        print(json.dumps(summary, indent=2))
        with open(out_dir / 'log.json', 'w') as f:
            json.dump({'settings': vars(args), 'summary': summary, 'attempts': log}, f, indent=2)
        print(f'Saved: {out_dir / "log.json"}')

        if not args.headless:
            print('Close the PyBullet window to exit.')
            try:
                while p.isConnected():
                    world.step(1)
            except p.error:   # window closed
                pass
    finally:
        if p.isConnected():
            world.close()


if __name__ == '__main__':
    main()
