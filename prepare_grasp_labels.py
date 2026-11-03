# prepare_grasp_labels.py
"""
Generate per-scene grasp labels for GraspPointSAM with graspnetAPI.

GraspNet.loadGrasp(format='6d') (used per frame by RGBD-Matter's
gen_label.py) does, for every object of a scene:
    1. read the object's grasp labels (points, offsets, friction coefs)
    2. keep grasps with 0 < friction <= fric_coef_thresh that do not
       collide in this scene (collision_label)
    3. decode (view, angle) into a rotation with generate_views and
       batch_viewpoint_params_to_matrix; depth/width from offsets
    4. move points and rotations into the frame's camera view

Steps 1-2 do not depend on the frame or the camera, only step 4 does.
So steps 1-2 run here once per scene, in the object frame, and the
dataset does steps 3-4 per frame with the same graspnetAPI functions.
This gives the same grasps as loadGrasp for all 256 frames of both
cameras, with one pass per scene instead of one per frame (~1 min and
~10 GB RAM each).

ALL grasps passing step 2 are kept (as RGBD-Matter does), not only the
best one per point, in loadGrasp's (point, view, angle, depth) order.

Output, one folder per scene (separate .npy files so the dataset can
memory-map them instead of loading the whole scene per sample):
    <label_dir>/scene_XXXX/
        points.npy         [K, 3]   float32  object points with >= 1 grasp,
                                             object frame (m)
        point_obj_ids.npy  [K]      int32    object id of each point
        point_offsets.npy  [K + 1]  int64    grasps of point k are
                                             [point_offsets[k], point_offsets[k+1])
        grasp_views.npy    [G]      int16    view index (0-299)
        grasp_angles.npy   [G]      float32  in-plane angle (rad)
        grasp_depths.npy   [G]      float32  grasp depth (m)
        grasp_widths.npy   [G]      float32  gripper width (m)
        grasp_fric.npy     [G]      float32  friction coefficient
        meta.npy           {fric_coef_thresh}

graspnet_root must follow graspnetAPI's layout:
    <graspnet_root>/scenes/scene_XXXX/...
    <graspnet_root>/grasp_label/NNN_labels.npz
    <graspnet_root>/collision_label/scene_XXXX/collision_labels.npz

Scenes with missing files are skipped with a warning; re-run later and
only the remaining scenes are generated.

Usage:
    python prepare_grasp_labels.py --config configs/graspnet.yaml
"""

import argparse
import gc
import shutil
import warnings
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from graspnetAPI import GraspNet
from graspnetAPI.utils.xmlhandler import xmlReader

from grasp_point_sam.data.graspnet_dataset import missing_scene_files, scene_range


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',    type=str, default='configs/graspnet.yaml')
    parser.add_argument('--overwrite', action='store_true',
                        help='regenerate scenes whose labels already exist')
    return parser.parse_args()


def object_grasps(points, offsets, fric_coefs, collision, fric_coef_thresh):
    """
    All grasps of one object that pass steps 1-2 of GraspNet.loadGrasp.

    Args:
        points:     [P, 3]
        offsets:    [P, 300, 12, 4, 3]  (angle, depth, width)
        fric_coefs: [P, 300, 12, 4]
        collision:  [P, 300, 12, 4] bool, from collision_label

    Returns:
        dict: points [K, 3], counts [K] (grasps per point) and per-grasp
        arrays [G], ordered by point then (view, angle, depth)
    """
    valid = (fric_coefs <= fric_coef_thresh) & (fric_coefs > 0) & ~collision
    p_idx, v_idx, a_idx, d_idx = np.nonzero(valid)  # C order = loadGrasp order

    off = offsets[p_idx, v_idx, a_idx, d_idx]  # [G, 3]
    kept_points, counts = np.unique(p_idx, return_counts=True)

    return {
        'points':     points[kept_points].astype(np.float32),
        'counts':     counts,
        'views':      v_idx.astype(np.int16),
        'angles':     off[:, 0].astype(np.float32),
        'depths':     off[:, 1].astype(np.float32),
        'widths':     off[:, 2].astype(np.float32),
        'fric':       fric_coefs[p_idx, v_idx, a_idx, d_idx].astype(np.float32),
    }


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    d    = cfg.data

    thresh    = float(d.fric_coef_thresh)
    label_dir = Path(d.label_dir)
    scene_ids = scene_range(d.train_scenes) + scene_range(d.val_scenes)

    print(f"graspnet_root: {d.graspnet_root}")
    print(f"label_dir:     {label_dir}")
    print(f"fric_coef_thresh: {thresh}")
    print(f"scenes: {scene_ids}")

    jobs, skipped = [], []
    for scene_id in scene_ids:
        if not args.overwrite and (label_dir / f'scene_{scene_id:04d}' / 'meta.npy').exists():
            continue
        missing = missing_scene_files(d.graspnet_root, scene_id)
        if missing:
            warnings.warn(
                f"scene_{scene_id:04d}: skipped, {len(missing)} files missing "
                f"(not extracted yet?), e.g. {missing[0]}"
            )
            skipped.append(scene_id)
            continue
        jobs.append(scene_id)
    print(f"{len(scene_ids) - len(jobs) - len(skipped)} scenes already done, "
          f"{len(jobs)} to generate")

    if jobs:
        g = GraspNet(d.graspnet_root, camera='realsense', split='custom',
                     sceneIds=jobs)
        label_dir.mkdir(parents=True, exist_ok=True)

    for scene_id in jobs:
        scene = f'scene_{scene_id:04d}'

        # collision_label is indexed by the annotation order of objects,
        # exactly as in loadGrasp (same order for every frame and camera)
        xml = Path(d.graspnet_root) / 'scenes' / scene / 'realsense' / 'annotations' / '0000.xml'
        obj_order  = [int(v[0]) for v in xmlReader(str(xml)).getposevectorlist()]
        collisions = g.loadCollisionLabels(scene_id)[scene]

        per_obj = []
        for i, obj_id in enumerate(obj_order):
            points, offsets, fric_coefs = g.loadGraspLabels([obj_id])[obj_id]
            grasps = object_grasps(points, offsets, fric_coefs, collisions[i], thresh)
            grasps['point_obj_ids'] = np.full(len(grasps['points']), obj_id, dtype=np.int32)
            per_obj.append(grasps)
            del points, offsets, fric_coefs

        cat    = lambda k: np.concatenate([o[k] for o in per_obj])
        counts = cat('counts')
        out = {
            'points':        cat('points'),
            'point_obj_ids': cat('point_obj_ids'),
            'point_offsets': np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
            'grasp_views':   cat('views'),
            'grasp_angles':  cat('angles'),
            'grasp_depths':  cat('depths'),
            'grasp_widths':  cat('widths'),
            'grasp_fric':    cat('fric'),
        }

        # write to a temp folder, then rename, so an interrupted run never
        # leaves a half-written scene behind
        tmp_dir = label_dir / f'{scene}.tmp'
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        for k, v in out.items():
            np.save(tmp_dir / f'{k}.npy', v)
        np.save(tmp_dir / 'meta.npy', {'fric_coef_thresh': thresh})
        final_dir = label_dir / scene
        if final_dir.exists():
            shutil.rmtree(final_dir)
        tmp_dir.rename(final_dir)

        print(f"{scene}: {len(out['points'])} grasp points, "
              f"{len(out['grasp_views'])} grasps, {len(obj_order)} objects")

        del per_obj, out, collisions
        gc.collect()

    if skipped:
        print(f"Skipped {len(skipped)} incomplete scenes: {skipped}. "
              f"Re-run once they are extracted.")
    print("Done.")


if __name__ == '__main__':
    main()
