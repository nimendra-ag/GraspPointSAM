# verify_labels.py
"""
Check that the grasp labels used for training are identical to the
official graspnetAPI ones.

For each (scene, camera, frame) this compares, object by object:
    reference: GraspNet.loadGrasp(format='6d', fric_coef_thresh=...)
    ours:      labels from prepare_grasp_labels.py, moved into the frame's
               camera view with GraspNetDataset._frame_poses and decoded
               the same way as GraspNetDataset._assign_grasps
on the full label set (before point subsampling / grasp_radius matching):
    number of grasps, translations, rotation matrices, widths, depths,
    scores (1.1 - friction).

Both sides keep loadGrasp's (point, view, angle, depth) order, so arrays
are compared element-wise.

Grasp and collision labels are loaded once per scene and passed to
loadGrasp, so each frame takes seconds instead of ~1 min.

Usage:
    python verify_labels.py --config configs/graspnet.yaml
    python verify_labels.py --scenes 0 45 99 --frames 0 128 255 --cameras realsense kinect
"""

import argparse
import sys

import numpy as np
from omegaconf import OmegaConf
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix

from grasp_point_sam.data.graspnet_dataset import GraspNetDataset

POS_TOL = 1e-5  # m
ROT_TOL = 1e-5
VAL_TOL = 1e-6  # widths, depths, scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  type=str, default='configs/graspnet.yaml')
    parser.add_argument('--scenes',  type=int, nargs='+', default=[0, 45, 99])
    parser.add_argument('--frames',  type=int, nargs='+', default=[0, 128, 255])
    parser.add_argument('--cameras', type=str, nargs='+', default=['realsense', 'kinect'])
    return parser.parse_args()


def our_grasps(ds, scene_id, camera, ann_id):
    """
    All grasps of one frame from our labels, per object, camera frame.

    Returns:
        obj_id -> dict(translations [G,3], rotations [G,3,3], widths [G],
                       depths [G], scores [G])
    """
    grasp_points, obj_rotations = ds._frame_poses(scene_id, camera, ann_id)
    labels        = ds._scene_labels(scene_id)
    point_obj_ids = np.asarray(labels['point_obj_ids'])
    point_offsets = np.asarray(labels['point_offsets'])
    counts        = np.diff(point_offsets)

    out = {}
    for obj_id, R_obj in obj_rotations.items():
        sel = np.flatnonzero(point_obj_ids == obj_id)
        if len(sel) == 0:
            g_idx = np.zeros(0, dtype=np.int64)
        else:
            # prepare_grasp_labels.py writes each object's points contiguously
            assert np.all(np.diff(sel) == 1), f"object {obj_id} points not contiguous"
            g_idx = np.arange(point_offsets[sel[0]], point_offsets[sel[-1] + 1])

        fric = np.asarray(labels['grasp_fric'][g_idx])
        keep = fric <= ds.fric_coef_thresh + 1e-6
        g_idx, fric = g_idx[keep], fric[keep]

        translations = np.repeat(grasp_points[sel], counts[sel], axis=0)[keep]
        R = batch_viewpoint_params_to_matrix(
            -ds.views[np.asarray(labels['grasp_views'][g_idx])],
            np.asarray(labels['grasp_angles'][g_idx]),
        )
        out[obj_id] = {
            'translations': translations,
            'rotations':    R_obj[None] @ R,
            'widths':       np.asarray(labels['grasp_widths'][g_idx]),
            'depths':       np.asarray(labels['grasp_depths'][g_idx]),
            'scores':       1.1 - fric,
        }
    return out


def ref_grasps(gg):
    """GraspGroup from loadGrasp -> same per-object dict as our_grasps."""
    out = {}
    for obj_id in np.unique(gg.object_ids):
        sel = gg.object_ids == obj_id
        out[int(obj_id)] = {
            'translations': gg.translations[sel],
            'rotations':    gg.rotation_matrices[sel],
            'widths':       gg.widths[sel],
            'depths':       gg.depths[sel],
            'scores':       gg.scores[sel],
        }
    return out


def compare(ours, ref):
    """List of error messages, empty if every object matches."""
    errors = []
    for obj_id in sorted(set(ours) | set(ref)):
        o = ours.get(obj_id)
        r = ref.get(obj_id)
        n_o = 0 if o is None else len(o['scores'])
        n_r = 0 if r is None else len(r['scores'])
        if n_o != n_r:
            errors.append(f"object {obj_id}: {n_o} grasps (ours) vs {n_r} (graspnetAPI)")
            continue
        if n_o == 0:
            continue
        for key, tol in [('translations', POS_TOL), ('rotations', ROT_TOL),
                         ('widths', VAL_TOL), ('depths', VAL_TOL), ('scores', VAL_TOL)]:
            diff = np.abs(o[key].astype(np.float64) - r[key].astype(np.float64)).max()
            if diff > tol:
                errors.append(f"object {obj_id}: {key} max diff {diff:.2e} > {tol:.0e}")
    return errors


def main():
    args = parse_args()
    cfg  = OmegaConf.load(args.config)
    d    = cfg.data
    thresh = float(d.fric_coef_thresh)

    ds = GraspNetDataset(
        graspnet_root    = d.graspnet_root,
        label_dir        = d.label_dir,
        scene_ids        = args.scenes,
        cameras          = args.cameras,
        fric_coef_thresh = thresh,
        split            = 'verify',
    )

    n_pass, n_fail = 0, 0
    for scene_id in args.scenes:
        g = ds.graspnet[args.cameras[0]]
        grasp_labels     = g.loadGraspLabels(ds.obj_ids[scene_id])
        collision_labels = g.loadCollisionLabels(scene_id)

        for camera in args.cameras:
            for ann_id in args.frames:
                name = f"scene_{scene_id:04d} {camera:9s} frame {ann_id:04d}"
                gg = ds.graspnet[camera].loadGrasp(
                    sceneId          = scene_id,
                    annId            = ann_id,
                    format           = '6d',
                    camera           = camera,
                    grasp_labels     = grasp_labels,
                    collision_labels = collision_labels,
                    fric_coef_thresh = thresh,
                )
                ours   = our_grasps(ds, scene_id, camera, ann_id)
                errors = compare(ours, ref_grasps(gg))
                total  = sum(len(o['scores']) for o in ours.values())

                if errors:
                    n_fail += 1
                    print(f"FAIL  {name}: {total} grasps (ours), {len(gg)} (graspnetAPI)")
                    for e in errors:
                        print(f"      {e}")
                else:
                    n_pass += 1
                    print(f"PASS  {name}: {total} grasps, {len(ours)} objects")

        del grasp_labels, collision_labels

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
