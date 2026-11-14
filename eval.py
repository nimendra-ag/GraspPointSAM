# eval.py
"""
Evaluate GraspPointSAM on the GraspNet-1Billion test scenes with the
official graspnetAPI metric (AP, AP0.8, AP0.4).

Two stages (both by default):
  predict : model -> grasps for every test frame, saved in graspnetAPI's
            dump format: <out_dir>/dump/scene_XXXX/<camera>/XXXX.npy
  eval    : graspnetAPI evaluation of the dumped grasps (collision check
            + force closure against the object meshes)

Settings of this first version:
  - prompts: one click per object, from the ground-truth segmentation
    (5 refinement iterations, as in training / validation)
  - gripper depth: predicted by the depth head; checkpoints from before
    the depth head use a fixed depth (--depth, default 0.02 m)
  - grasp per point on the predicted mask: translation = the point
    (camera frame), rotation = predicted 6D, width = predicted,
    score = predicted quality

Point cloud preparation matches training (workspace crop, random or FPS
sampling), so pass the settings the checkpoint was trained with.

Examples:
  # quick check: 5 scenes, every 16th frame, realsense
  python eval.py --ckpt checkpoints/modal_fps_ep0/best_model.safetensors \
      --sampling fps --workspace_crop --scenes 100 105 --frame_step 16

  # full test_seen, one camera
  python eval.py --ckpt checkpoints/modal_fps_ep0/best_model.safetensors \
      --sampling fps --workspace_crop --scenes 100 130 --camera realsense

  # evaluate again without re-running the model
  python eval.py ... --stage eval
"""

import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

# graspnetAPI's bundled Dex-Net code (sdf.py, mesh.py) still uses np.int,
# removed in NumPy 1.24. Restore the alias in this process only; the
# installed packages stay unchanged.
if not hasattr(np, 'int'):
    np.int = int

from graspnetAPI import GraspGroup, GraspNet, GraspNetEval
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    create_table_points, eval_grasp, get_scene_name, load_dexnet_model,
    transform_points, voxel_sample_points,
)

from grasp_point_sam.data.graspnet_dataset import (
    WORKSPACE_MARGIN, depth_to_pointcloud, farthest_point_indices,
    voxel_downsample_indices,
)
from grasp_point_sam.model.build_model import build_model
from grasp_point_sam.model.grasp_decoder import GRASP_DEPTHS

FRICTION_COEFS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]  # graspnetAPI's list
TOP_K          = 50                              # graspnetAPI default
GRASP_HEIGHT   = 0.02                            # graspnetAPI default


# ─────────────────────────────────────────────────────────────────────────────
# Point cloud (same steps as GraspNetDataset.__getitem__, without labels)
# ─────────────────────────────────────────────────────────────────────────────

def load_frame(root: Path, scene_id: int, camera: str, ann_id: int):
    """RGB-D frame -> camera-frame points, colors and segmentation ids."""
    d     = root / 'scenes' / f'scene_{scene_id:04d}' / camera
    rgb   = np.array(Image.open(d / 'rgb'   / f'{ann_id:04d}.png'))
    depth = np.array(Image.open(d / 'depth' / f'{ann_id:04d}.png'))
    seg   = np.array(Image.open(d / 'label' / f'{ann_id:04d}.png'))
    meta  = scio.loadmat(d / 'meta' / f'{ann_id:04d}.mat')

    coords, colors = depth_to_pointcloud(
        depth, rgb, meta['intrinsic_matrix'], meta['factor_depth'].item(),
    )
    valid = coords[:, 2] > 0
    return coords[valid], colors[valid], seg.reshape(-1)[valid]


def workspace_mask(root: Path, scene_id: int, camera: str, ann_id: int,
                   coords: np.ndarray, seg_flat: np.ndarray) -> np.ndarray:
    """graspnet-baseline's get_workspace_mask (as GraspNetDataset._workspace_mask)."""
    d = root / 'scenes' / f'scene_{scene_id:04d}' / camera
    camera_pose = np.load(d / 'camera_poses.npy')[ann_id]
    align_mat   = np.load(d / 'cam0_wrt_table.npy')
    trans       = align_mat @ camera_pose

    table_coords = coords @ trans[:3, :3].T + trans[:3, 3]
    foreground   = table_coords[seg_flat > 0]
    if len(foreground) == 0:
        return np.ones(len(coords), dtype=bool)
    lo = foreground.min(axis=0) - WORKSPACE_MARGIN
    hi = foreground.max(axis=0) + WORKSPACE_MARGIN
    return np.all((table_coords > lo) & (table_coords < hi), axis=1)


def random_indices(N: int, num_points: int) -> np.ndarray:
    """As GraspNetDataset._random_indices."""
    if N > num_points:
        return np.random.choice(N, num_points, replace=False)
    return np.concatenate([np.arange(N), np.random.choice(N, num_points - N, replace=True)])


def sample_indices(coords: np.ndarray, num_points: int, sampling: str,
                   voxel_size: float) -> np.ndarray:
    """As GraspNetDataset step 4."""
    if sampling == 'random':
        return random_indices(len(coords), num_points)
    candidates = voxel_downsample_indices(coords, voxel_size)
    if len(candidates) <= num_points:
        return candidates[random_indices(len(candidates), num_points)]
    return candidates[farthest_point_indices(coords[candidates], num_points)]


def rotation_6d_to_matrix(r: np.ndarray) -> np.ndarray:
    """[K, 6] (first two columns) -> [K, 3, 3], Gram-Schmidt (Zhou et al. 2019)."""
    a1, a2 = r[:, :3], r[:, 3:]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-8)
    b2 = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # columns


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: predict
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_has_depth_head(ckpt: str) -> bool:
    """True if the checkpoint was trained with the depth head."""
    if ckpt.endswith('.safetensors'):
        from safetensors import safe_open
        with safe_open(ckpt, framework='pt') as f:
            keys = list(f.keys())
    else:
        keys = list(torch.load(ckpt, map_location='cpu')['model'].keys())
    return any(k.startswith('grasp_decoder.depth_heads') for k in keys)


def load_model(cfg, ckpt: str, device: str, has_depth: bool):
    # build the architecture the checkpoint was trained with
    cfg.model.grasp_decoder.num_depth_bins = len(GRASP_DEPTHS) if has_depth else 0
    model = build_model(cfg)
    if ckpt.endswith('.safetensors'):
        from safetensors.torch import load_model as load_safetensors
        load_safetensors(model, ckpt, strict=True)
    else:
        state = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(state['model'], strict=True)
    print(f"Loaded checkpoint: {ckpt} (depth head: {'yes' if has_depth else 'no'})")
    return model.to(device).eval()


@torch.no_grad()
def predict_frame(model, args, root: Path, scene_id: int, camera: str,
                  ann_id: int, device: str) -> np.ndarray:
    """One frame -> [G, 17] graspnetAPI grasp array (after NMS)."""
    # fixed seed per frame: sampling and prompt clicks are reproducible
    seed = scene_id * 1000 + ann_id + (0 if camera == 'realsense' else 500)
    np.random.seed(seed)
    torch.manual_seed(seed)

    coords, colors, seg_flat = load_frame(root, scene_id, camera, ann_id)
    if args.workspace_crop:
        keep = workspace_mask(root, scene_id, camera, ann_id, coords, seg_flat)
        coords, colors, seg_flat = coords[keep], colors[keep], seg_flat[keep]
    return predict_grasps(model, args, coords, colors, seg_flat, device)


@torch.no_grad()
def predict_grasps(model, args, coords: np.ndarray, colors: np.ndarray,
                   seg_flat: np.ndarray, device: str) -> np.ndarray:
    """
    Camera-frame points -> [G, 17] graspnetAPI grasp array (after NMS).

    coords [N, 3] m, colors [N, 3] in [0, 1], seg_flat [N] (obj_id + 1, 0 = background).
    One prompt per object, from seg_flat. Also used by sim_grasp.py.
    """
    idx = sample_indices(coords, args.num_points, args.sampling, args.voxel_size)
    coords, colors, seg_flat = coords[idx], colors[idx], seg_flat[idx]

    # one mask per visible object (segmentation stores obj_id + 1)
    obj_ids, masks = [], []
    for v in np.unique(seg_flat):
        if v == 0:
            continue
        mask = seg_flat == v
        if mask.sum() > 10:
            obj_ids.append(int(v) - 1)
            masks.append(mask)
    if not masks:
        return np.zeros((0, 17), dtype=np.float64)
    masks = np.stack(masks)  # [M, N]

    # normalize as in training
    coords_norm = coords - coords.mean(axis=0)
    scale = np.abs(coords_norm).max()
    if scale > 0:
        coords_norm = coords_norm / scale
    coords_norm = np.clip(coords_norm, -1.0, 1.0)

    with torch.autocast('cuda', dtype=torch.float16, enabled=args.fp16):
        outputs = model(
            coords   = torch.from_numpy(coords_norm).float()[None].to(device),
            features = torch.from_numpy(colors).float()[None].to(device),
            gt_masks = torch.from_numpy(masks).float()[None].to(device),
        )
    pred = outputs[-1]  # last prompt iteration
    quality  = pred.grasp_quality.float().cpu().numpy()       # [M, N]
    rotation = pred.grasp_rotation.float().cpu().numpy()      # [M, N, 6]
    width    = pred.grasp_width.float().cpu().numpy()[..., 0]  # [M, N]
    pred_msk = pred.masks.float().cpu().numpy()[:, 0] > 0     # [M, N]
    if pred.grasp_depth is not None:                           # [M, N] in m
        depth = np.asarray(GRASP_DEPTHS)[pred.grasp_depth.float().argmax(-1).cpu().numpy()]
    else:
        depth = np.full(quality.shape, args.depth)

    grasps = []
    for m, obj_id in enumerate(obj_ids):
        pts = np.flatnonzero(pred_msk[m])
        if len(pts) == 0:  # empty predicted mask: fall back to the prompt mask
            pts = np.flatnonzero(masks[m])
        R = rotation_6d_to_matrix(rotation[m, pts])

        g = np.zeros((len(pts), 17), dtype=np.float64)
        g[:, 0]     = quality[m, pts]
        g[:, 1]     = width[m, pts]
        g[:, 2]     = GRASP_HEIGHT
        g[:, 3]     = depth[m, pts]
        g[:, 4:13]  = R.reshape(-1, 9)
        g[:, 13:16] = coords[pts]
        g[:, 16]    = obj_id
        grasps.append(g)

    gg = GraspGroup(np.concatenate(grasps)).nms(0.03, 30.0 / 180 * np.pi)

    # keep the best grasps per object (the metric uses the top 10 per object)
    arr  = gg.grasp_group_array
    keep = []
    for obj_id in np.unique(arr[:, 16]):
        sel = np.flatnonzero(arr[:, 16] == obj_id)
        keep.append(sel[np.argsort(-arr[sel, 0])[:args.grasps_per_object]])
    return arr[np.concatenate(keep)]


def run_predict(args, cfg, scene_ids, ann_ids, dump_dir: Path):
    device = 'cuda'
    model  = load_model(cfg, args.ckpt, device, args.has_depth)
    root   = Path(args.graspnet_root)

    jobs = [(s, c, a) for s in scene_ids for c in args.cameras for a in ann_ids]
    todo = [
        j for j in jobs
        if args.overwrite or not (dump_dir / get_scene_name(j[0]) / j[1] / f'{j[2]:04d}.npy').exists()
    ]
    print(f"Predict: {len(todo)} of {len(jobs)} frames to run "
          f"({len(jobs) - len(todo)} already dumped)")

    t0 = time.time()
    for scene_id, camera, ann_id in tqdm(todo, desc='predict'):
        arr = predict_frame(model, args, root, scene_id, camera, ann_id, device)
        out = dump_dir / get_scene_name(scene_id) / camera
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f'{ann_id:04d}.npy', arr)
    if todo:
        print(f"Predict done: {(time.time() - t0) / len(todo):.2f} s/frame")

    del model
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: evaluate (graspnetAPI's GraspNetEval.eval_scene, any frame subset)
# ─────────────────────────────────────────────────────────────────────────────

class SubsetEval(GraspNetEval):
    """GraspNetEval for any scenes and frames (the official one needs all 256)."""

    def __init__(self, root: str, camera: str, scene_ids):
        GraspNet.__init__(self, root, camera, 'custom', list(scene_ids))

    def eval_frames(self, scene_id: int, ann_ids, dump_folder: str) -> np.ndarray:
        """Body of GraspNetEval.eval_scene for the given frames -> [F, 50, 6]."""
        config = get_config()
        table  = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)
        model_sampled_list = [voxel_sample_points(m, 0.008) for m in model_list]

        scene_accuracy = []
        for ann_id in ann_ids:
            gg = GraspGroup().from_npy(str(
                Path(dump_folder) / get_scene_name(scene_id) / self.camera / f'{ann_id:04d}.npy'))
            _, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(align_mat @ camera_pose))

            # clip width to [0, 0.1] as graspnetAPI
            gg.grasp_group_array[:, 1] = np.clip(gg.grasp_group_array[:, 1], 0, 0.1)

            accuracy = np.zeros((TOP_K, len(FRICTION_COEFS)))
            if len(gg) > 0:
                grasp_list, score_list, collision_list = eval_grasp(
                    gg, model_sampled_list, dexmodel_list, pose_list, config,
                    table=table_trans, voxel_size=0.008, TOP_K=TOP_K)
                grasp_list = [x for x in grasp_list if len(x) != 0]
                score_list = [x for x in score_list if len(x) != 0]

                if grasp_list:
                    grasp_list = np.concatenate(grasp_list)
                    score_list = np.concatenate(score_list)
                    score_list = score_list[np.argsort(-grasp_list[:, 0])]
                    for f, fric in enumerate(FRICTION_COEFS):
                        for k in range(TOP_K):
                            s = score_list[:k + 1]
                            accuracy[k, f] = np.sum((s <= fric) & (s > 0)) / (k + 1)

            scene_accuracy.append(accuracy)
            print(f'scene {scene_id:04d} {self.camera} frame {ann_id:04d}: '
                  f'{100 * accuracy.mean():.2f}', flush=True)
        return np.stack(scene_accuracy)


def cache_dex_models(root: Path, obj_ids):
    """Build graspnetAPI's dex_models/*.pkl cache once (it only reads it)."""
    cache = root / 'dex_models'
    cache.mkdir(exist_ok=True)
    missing = [o for o in sorted(set(obj_ids)) if not (cache / f'{o:03d}.pkl').exists()]
    for obj_id in tqdm(missing, desc='caching dex models (once)'):
        dex = load_dexnet_model(str(root / 'models' / f'{obj_id:03d}' / 'textured'))
        with open(cache / f'{obj_id:03d}.pkl', 'wb') as f:
            pickle.dump(dex, f)


def run_eval(args, scene_ids, ann_ids, dump_dir: Path, out_dir: Path):
    root    = Path(args.graspnet_root)
    summary = {}

    for camera in args.cameras:
        missing = [
            (s, a) for s in scene_ids for a in ann_ids
            if not (dump_dir / get_scene_name(s) / camera / f'{a:04d}.npy').exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} dumps missing for {camera}, e.g. scene {missing[0][0]} "
                f"frame {missing[0][1]}. Run the predict stage first.")

        ge = SubsetEval(str(root), camera, scene_ids)
        cache_dex_models(root, [o for s in scene_ids for o in ge.getObjIds(s)])

        t0 = time.time()
        if args.proc > 1:
            with Pool(args.proc) as pool:
                jobs = [pool.apply_async(ge.eval_frames, (s, ann_ids, str(dump_dir)))
                        for s in scene_ids]
                res = np.stack([j.get() for j in jobs])
        else:
            res = np.stack([ge.eval_frames(s, ann_ids, str(dump_dir)) for s in scene_ids])
        # res: [scenes, frames, TOP_K, 6 friction coefficients]

        np.save(out_dir / f'accuracy_{camera}.npy', res)
        result = {
            'AP':    float(100 * res.mean()),
            'AP0.8': float(100 * res[..., 3].mean()),
            'AP0.4': float(100 * res[..., 1].mean()),
            'per_scene_AP': {int(s): float(100 * res[i].mean()) for i, s in enumerate(scene_ids)},
            'eval_minutes': round((time.time() - t0) / 60, 1),
        }
        summary[camera] = result
        print(f"\n{camera}: AP {result['AP']:.2f} | AP0.8 {result['AP0.8']:.2f} | "
              f"AP0.4 {result['AP0.4']:.2f}  ({len(scene_ids)} scenes x {len(ann_ids)} frames)")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='GraspNet-1Billion evaluation of GraspPointSAM')
    p.add_argument('--ckpt', required=True, help='best_model.safetensors or a .pth checkpoint')
    p.add_argument('--config', default='configs/graspnet.yaml',
                   help='model architecture + local paths (graspnet_root, pretrained_ckpt)')
    p.add_argument('--stage', choices=['all', 'predict', 'eval'], default='all')
    p.add_argument('--scenes', type=int, nargs=2, default=[100, 130], metavar=('START', 'END'),
                   help='scene ids [start, end); seen 100-130, similar 130-160, novel 160-190')
    p.add_argument('--frame_step', type=int, default=1,
                   help='use every n-th of the 256 frames (1 = all, official)')
    p.add_argument('--camera', dest='cameras', nargs='+', default=['realsense'],
                   choices=['realsense', 'kinect'])
    p.add_argument('--sampling', choices=['random', 'fps'], default=None,
                   help='default: from the config; must match training')
    p.add_argument('--workspace_crop', action='store_true',
                   help='crop as in training (use if the model was trained with it)')
    p.add_argument('--depth', type=float, default=0.02,
                   help='fixed gripper depth (m), only for checkpoints without a depth head')
    p.add_argument('--grasps_per_object', type=int, default=100,
                   help='grasps kept per object after NMS')
    p.add_argument('--proc', type=int, default=2, help='processes for the eval stage')
    p.add_argument('--no_fp16', dest='fp16', action='store_false')
    p.add_argument('--overwrite', action='store_true', help='re-predict existing dumps')
    p.add_argument('--out_dir', default=None,
                   help='default: eval_results/<checkpoint folder>_<sampling>_<dpred or d<depth>>')
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    args.graspnet_root = cfg.data.graspnet_root
    args.num_points    = cfg.training.num_points
    args.voxel_size    = cfg.data.get('voxel_size', 0.005)
    args.sampling      = args.sampling or cfg.data.get('sampling', 'random')
    args.has_depth     = checkpoint_has_depth_head(args.ckpt)
    depth_tag          = 'dpred' if args.has_depth else f'd{args.depth}'

    scene_ids = list(range(*args.scenes))
    ann_ids   = list(range(0, 256, args.frame_step))

    out_dir = Path(args.out_dir or
                   f'eval_results/{Path(args.ckpt).parent.name}_{args.sampling}_{depth_tag}')
    dump_dir = out_dir / 'dump'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scenes {scene_ids[0]}-{scene_ids[-1]} ({len(scene_ids)}), "
          f"{len(ann_ids)} frames each, cameras {args.cameras}")
    print(f"sampling={args.sampling}, workspace_crop={args.workspace_crop}, "
          f"depth={'predicted' if args.has_depth else f'fixed {args.depth} m'}, "
          f"output: {out_dir}")

    if args.stage in ('all', 'predict'):
        run_predict(args, cfg, scene_ids, ann_ids, dump_dir)

    if args.stage in ('all', 'eval'):
        summary = run_eval(args, scene_ids, ann_ids, dump_dir, out_dir)
        settings = {k: v for k, v in vars(args).items()}
        name = f"results_{scene_ids[0]}-{scene_ids[-1]}_step{args.frame_step}.json"
        with open(out_dir / name, 'w') as f:
            json.dump({'settings': settings, 'results': summary}, f, indent=2)
        print(f"Saved: {out_dir / name}")


if __name__ == '__main__':
    main()
