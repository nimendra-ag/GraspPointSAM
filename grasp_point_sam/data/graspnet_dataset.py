# grasp_point_sam/data/graspnet_dataset.py
"""
GraspNet-1Billion dataset loader built on graspnetAPI.

Scene data (RGB, depth, segmentation, camera intrinsics) is read through
graspnetAPI.GraspNet for every frame of every camera. Grasp labels come
from prepare_grasp_labels.py, which runs steps 1-2 of the official
GraspNet.loadGrasp once per scene (object frame). Here steps 3-4 are
done per frame with the same graspnetAPI functions: decode rotations
and move grasps into the frame's camera view. The result equals
loadGrasp(format='6d') for that frame.

Per sample:
    - point cloud back-projected from depth, random subsample to num_points
    - per-point RGB colors
    - per-object masks from the segmentation image
    - per-object, per-point grasp quality (1.1 - lowest friction of the
      point's grasps; 0 = no grasp)
    - every grasp of every labelled point as a flat list
      (grasp_point_idx, grasp_rotation, grasp_width), see _assign_grasps

graspnet_root must follow graspnetAPI's layout:
    <graspnet_root>/scenes/scene_XXXX/<camera>/{rgb,depth,label,meta,annotations}/
    <graspnet_root>/grasp_label/
    <graspnet_root>/collision_label/
"""

import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
from graspnetAPI import GraspNet
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix
from graspnetAPI.utils.utils import generate_views, get_obj_pose_list, transform_points
from graspnetAPI.utils.xmlhandler import xmlReader

NUM_VIEWS = 300  # as in GraspNet.loadGrasp

WORKSPACE_MARGIN = 0.02  # m, as graspnet-baseline's get_workspace_mask(outlier=0.02)

LABEL_FILES = [
    'points', 'point_obj_ids', 'point_offsets',
    'grasp_views', 'grasp_angles', 'grasp_depths', 'grasp_widths', 'grasp_fric',
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def scene_range(bounds: Sequence[int]) -> List[int]:
    """[start, end) from the config -> list of scene ids."""
    start, end = bounds
    return list(range(start, end))


def select_frames(frames_per_scene: int) -> List[int]:
    """Frame (annotation) ids used per scene, evenly spaced over the 256 frames."""
    return np.linspace(0, 255, frames_per_scene, dtype=int).tolist()


def missing_scene_files(graspnet_root: str, scene_id: int) -> List[str]:
    """
    Scene-level files that label generation and loading need but that do
    not exist, e.g. while the dataset is still being extracted.
    Empty list = complete.
    """
    root  = Path(graspnet_root)
    scene = root / 'scenes' / f'scene_{scene_id:04d}'
    paths = [
        scene / 'object_id_list.txt',
        root / 'collision_label' / f'scene_{scene_id:04d}' / 'collision_labels.npz',
    ]
    for camera in ('realsense', 'kinect'):
        paths += [
            scene / camera / 'camera_poses.npy',
            scene / camera / 'annotations' / '0000.xml',
        ]
    if (scene / 'object_id_list.txt').exists():
        obj_ids = (scene / 'object_id_list.txt').read_text().split()
        paths  += [
            root / 'grasp_label' / f'{int(i):03d}_labels.npz' for i in obj_ids
        ]
    return [str(p) for p in paths if not p.exists()]


def depth_to_pointcloud(
    depth:        np.ndarray,
    rgb:          np.ndarray,
    camK:         np.ndarray,
    factor_depth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth image to 3D point cloud using camera intrinsics.
    Same as graspnetAPI's create_point_cloud_from_depth_image.

    Args:
        depth:        [H, W] uint16 depth image
        rgb:          [H, W, 3] uint8
        camK:         [3, 3] camera intrinsic matrix
        factor_depth: depth units per meter (from meta .mat)

    Returns:
        coords: [H*W, 3] float32, XYZ in meters (camera frame)
        colors: [H*W, 3] float32, RGB normalized to [0, 1]
    """
    H, W   = depth.shape
    fx, fy = camK[0, 0], camK[1, 1]
    cx, cy = camK[0, 2], camK[1, 2]

    u  = np.arange(W, dtype=np.float32)
    v  = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z  = depth.astype(np.float32) / factor_depth
    x  = (uu - cx) * z / fx
    y  = (vv - cy) * z / fy

    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    return coords, colors


def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to 6D rotation representation.
    Takes first two columns of R.

    Reference: Zhou et al., CVPR 2019.

    Args:
        R: [..., 3, 3]

    Returns:
        [..., 6]
    """
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def nearest_neighbor(
    query: np.ndarray,
    key:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KD-tree nearest neighbor.

    Args:
        query: [N, 3]
        key:   [V, 3]

    Returns:
        nn_idx:   [N] index into key
        nn_dists: [N] squared distance to nearest key point
    """
    dists, nn_idx = cKDTree(key).query(query, k=1)
    return nn_idx.astype(np.int64), (dists ** 2).astype(np.float32)


def voxel_downsample_indices(coords: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    One random point per occupied voxel.

    Args:
        coords:     [N, 3]
        voxel_size: voxel edge length (m)

    Returns:
        [V] indices into coords
    """
    perm = np.random.permutation(len(coords))  # random point per voxel, per epoch
    q    = np.floor(coords[perm] / voxel_size).astype(np.int64)
    q   -= q.min(axis=0)
    dims = q.max(axis=0) + 1
    key  = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, first = np.unique(key, return_index=True)
    return perm[first]


def farthest_point_indices(coords: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest point sampling (open3d, C++).

    open3d returns points, not indices, so each point's index is carried
    through in its normal and read back.

    Args:
        coords:      [N, 3], N >= num_samples
        num_samples: points to keep

    Returns:
        [num_samples] indices into coords
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points  = o3d.utility.Vector3dVector(coords.astype(np.float64))
    pcd.normals = o3d.utility.Vector3dVector(
        np.repeat(np.arange(len(coords), dtype=np.float64)[:, None], 3, axis=1)
    )
    sampled = pcd.farthest_point_down_sample(num_samples)
    return np.asarray(sampled.normals)[:, 0].astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GraspNetDataset(Dataset):
    """
    GraspNet-1Billion dataset loader on top of graspnetAPI.

    One sample = one frame of one camera of one scene.

    Args:
        graspnet_root:    graspnetAPI dataset root
        label_dir:        output dir of prepare_grasp_labels.py
        scene_ids:        scene ids to use
        cameras:          any of 'realsense', 'kinect'
        num_points:       points to subsample per frame
        frames_per_scene: frames per scene and camera, evenly spaced over 256
        fric_coef_thresh: keep grasps with friction <= this (must be <= the
                          value the labels were generated with)
        grasp_radius:     max distance (m) from a scene point to a labelled
                          grasp point for the point to get that point's grasps
        sampling:         'random': num_points random points;
                          'fps': voxel downsample (voxel_size), then
                          farthest point sampling to num_points
        workspace_crop:   keep only points inside the objects' bounding box
                          (table frame) + 2 cm, as graspnet-baseline; removes
                          far background and depth outliers, which FPS
                          would otherwise pick first
        voxel_size:       voxel size (m) for sampling='fps'
        augment:          apply data augmentation
        split:            name used in log messages only
    """

    def __init__(
        self,
        graspnet_root:    str,
        label_dir:        str,
        scene_ids:        List[int],
        cameras:          Sequence[str] = ('realsense', 'kinect'),
        num_points:       int   = 10000,
        frames_per_scene: int   = 256,
        fric_coef_thresh: float = 0.1,
        grasp_radius:     float = 0.02,
        sampling:         str   = 'random',
        workspace_crop:   bool  = False,
        voxel_size:       float = 0.005,
        augment:          bool  = False,
        split:            str   = 'train',
    ):
        if sampling not in ('random', 'fps'):
            raise ValueError(f"sampling must be 'random' or 'fps', got {sampling!r}")

        self.graspnet_root    = Path(graspnet_root)
        self.label_dir        = Path(label_dir)
        self.cameras          = list(cameras)
        self.num_points       = num_points
        self.fric_coef_thresh = fric_coef_thresh
        self.grasp_radius     = grasp_radius
        self.sampling         = sampling
        self.workspace_crop   = workspace_crop
        self.voxel_size       = voxel_size
        self.augment          = augment

        self.samples = [
            (sid, camera, ann_id)
            for sid in scene_ids
            for camera in self.cameras
            for ann_id in select_frames(frames_per_scene)
        ]

        # Checked before GraspNet(), which reads every scene's files
        missing = [
            sid for sid in scene_ids
            if not (self.label_dir / f'scene_{sid:04d}' / 'meta.npy').exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"[{split}] grasp labels missing for {len(missing)} scenes: "
                f"{missing}. Run: python prepare_grasp_labels.py --config "
                f"<your config>."
            )
        for sid in scene_ids:
            label_thresh = np.load(
                self.label_dir / f'scene_{sid:04d}' / 'meta.npy', allow_pickle=True
            ).item()['fric_coef_thresh']
            if fric_coef_thresh > label_thresh + 1e-6:
                raise ValueError(
                    f"scene_{sid:04d} labels keep friction <= {label_thresh}, "
                    f"config asks for <= {fric_coef_thresh}. Re-run "
                    f"prepare_grasp_labels.py with --overwrite."
                )

        self.graspnet = {
            camera: GraspNet(graspnet_root, camera=camera, split='custom',
                             sceneIds=scene_ids)
            for camera in self.cameras
        }
        self.obj_ids = {
            sid: self.graspnet[self.cameras[0]].getObjIds(sid) for sid in scene_ids
        }
        self.views = generate_views(NUM_VIEWS)

        # filled lazily, per DataLoader worker
        self._labels       = {}
        self._camera_poses = {}
        self._align_mats   = {}

        print(f"GraspNetDataset [{split}]: "
              f"{len(scene_ids)} scenes x {len(self.cameras)} cameras x "
              f"{frames_per_scene} frames = {len(self.samples)} samples, "
              f"{num_points} points per frame, sampling={sampling}"
              f"{f' (voxel {voxel_size} m)' if sampling == 'fps' else ''}, "
              f"workspace_crop={workspace_crop}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        scene_id, camera, ann_id = self.samples[idx]
        frame = f'scene_{scene_id:04d} {camera} frame {ann_id:04d}'

        # ── 1. Load RGB-D through graspnetAPI ────────────────────────────
        rgb_path, depth_path, seg_path, meta_path, _, _, _ = \
            self.graspnet[camera].loadData(scene_id, camera, ann_id)

        rgb   = np.array(Image.open(rgb_path))
        depth = np.array(Image.open(depth_path))
        seg   = np.array(Image.open(seg_path))
        meta  = scio.loadmat(meta_path)

        # ── 2. Back-project depth → point cloud ──────────────────────────
        coords, colors = depth_to_pointcloud(
            depth, rgb,
            meta['intrinsic_matrix'],
            meta['factor_depth'].item(),
        )

        # ── 3. Remove invalid points (zero depth) ────────────────────────
        valid    = coords[:, 2] > 0
        coords   = coords[valid]
        colors   = colors[valid]
        seg_flat = seg.reshape(-1)[valid]

        # ── 4. Subsample to num_points ───────────────────────────────────
        if self.workspace_crop:
            keep     = self._workspace_mask(frame, scene_id, camera, ann_id, coords, seg_flat)
            coords   = coords[keep]
            colors   = colors[keep]
            seg_flat = seg_flat[keep]

        if self.sampling == 'fps':
            indices = self._fps_indices(coords)
        else:
            indices = self._random_indices(len(coords))
        coords   = coords[indices]
        colors   = colors[indices]
        seg_flat = seg_flat[indices]

        # ── 5. Build per-object masks ────────────────────────────────────
        masks         = []
        valid_obj_ids = []

        for obj_id in self.obj_ids[scene_id]:
            # segmentation stores obj_id + 1 (0 = background), as in graspnetAPI
            mask = seg_flat == obj_id + 1
            if mask.sum() > 10:
                masks.append(mask)
                valid_obj_ids.append(obj_id)
            else:
                warnings.warn(
                    f"{frame}: object {obj_id} dropped, "
                    f"<= 10 of the {self.num_points} sampled points are on it"
                )

        if len(masks) == 0:
            warnings.warn(
                f"{frame}: no object has a usable mask, "
                f"skipping to the next sample"
            )
            return self.__getitem__((idx + 1) % len(self))

        masks = np.stack(masks, axis=0)  # [M, N] bool

        # ── 6. Grasp points and per-object rotations in this camera view ─
        grasp_points, obj_rotations = self._frame_poses(scene_id, camera, ann_id)

        # ── 7. Augmentation (applied to points AND grasp labels) ─────────
        if self.augment:
            coords, colors, grasp_points, obj_rotations = self._augment(
                coords, colors, grasp_points, obj_rotations
            )

        # ── 8. Normalize coordinates to [-1, 1] ──────────────────────────
        centroid    = coords.mean(axis=0)
        coords_norm = coords - centroid
        scale       = np.abs(coords_norm).max()
        if scale > 0:
            coords_norm = coords_norm / scale
        coords_norm = np.clip(coords_norm, -1.0, 1.0)

        # ── 9. Assign grasps to points ───────────────────────────────────
        grasps = self._assign_grasps(
            frame, scene_id, coords, masks, valid_obj_ids,
            grasp_points, obj_rotations,
        )

        # ── 10. Convert to tensors ───────────────────────────────────────
        return {
            'coords':          torch.from_numpy(coords_norm).float(),
            'features':        torch.from_numpy(colors).float(),
            'masks':           torch.from_numpy(masks).float(),
            'grasp_quality':   grasps['quality'],
            'grasp_point_idx': grasps['point_idx'],
            'grasp_rotation':  grasps['rotation'],
            'grasp_width':     grasps['width'],
            'scene_id':        f'scene_{scene_id:04d}',
            'camera':          camera,
            'frame_id':        f'{ann_id:04d}',
            'obj_ids':         valid_obj_ids,
        }

    def _random_indices(self, N: int) -> np.ndarray:
        """num_points random indices into N points (repeats if N is smaller)."""
        if N > self.num_points:
            return np.random.choice(N, self.num_points, replace=False)
        return np.concatenate([
            np.arange(N),
            np.random.choice(N, self.num_points - N, replace=True)
        ])

    def _fps_indices(self, coords: np.ndarray) -> np.ndarray:
        """Voxel downsample, then farthest point sampling to num_points."""
        candidates = voxel_downsample_indices(coords, self.voxel_size)
        if len(candidates) <= self.num_points:
            # fewer voxels than points: keep all, pad randomly as 'random' does
            return candidates[self._random_indices(len(candidates))]
        return candidates[farthest_point_indices(coords[candidates], self.num_points)]

    def _camera_pose(self, scene_id: int, camera: str, ann_id: int) -> np.ndarray:
        """[4, 4] pose of the camera at frame ann_id w.r.t. the first frame."""
        key = (scene_id, camera)
        if key not in self._camera_poses:
            self._camera_poses[key] = np.load(
                self.graspnet_root / 'scenes' / f'scene_{scene_id:04d}'
                / camera / 'camera_poses.npy'
            )
        return self._camera_poses[key][ann_id]

    def _workspace_mask(
        self,
        frame:    str,
        scene_id: int,
        camera:   str,
        ann_id:   int,
        coords:   np.ndarray,
        seg_flat: np.ndarray,
    ) -> np.ndarray:
        """
        graspnet-baseline's get_workspace_mask: points inside the axis-aligned
        bounding box of all object points in the table frame, + 2 cm.
        """
        key = (scene_id, camera)
        if key not in self._align_mats:
            self._align_mats[key] = np.load(
                self.graspnet_root / 'scenes' / f'scene_{scene_id:04d}'
                / camera / 'cam0_wrt_table.npy'
            )
        trans = self._align_mats[key] @ self._camera_pose(scene_id, camera, ann_id)

        table_coords = coords @ trans[:3, :3].T + trans[:3, 3]
        foreground   = table_coords[seg_flat > 0]
        if len(foreground) == 0:
            warnings.warn(f"{frame}: no object points, workspace crop skipped")
            return np.ones(len(coords), dtype=bool)

        lo = foreground.min(axis=0) - WORKSPACE_MARGIN
        hi = foreground.max(axis=0) + WORKSPACE_MARGIN
        return np.all((table_coords > lo) & (table_coords < hi), axis=1)

    def _scene_labels(self, scene_id: int) -> Dict[str, np.ndarray]:
        """Per-scene labels, memory-mapped (only the rows used are read)."""
        if scene_id not in self._labels:
            scene_dir = self.label_dir / f'scene_{scene_id:04d}'
            self._labels[scene_id] = {
                k: np.load(scene_dir / f'{k}.npy', mmap_mode='r')
                for k in LABEL_FILES
            }
        return self._labels[scene_id]

    def _frame_poses(
        self,
        scene_id: int,
        camera:   str,
        ann_id:   int,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        Step 4 of GraspNet.loadGrasp for this frame: object points from the
        object frame into the camera view, and the rotation that does the
        same for gripper rotations (same operations as loadGrasp).

        Returns:
            grasp_points:  [K, 3] all labelled points of the scene, camera frame
            obj_rotations: obj_id -> [3, 3], camera_R = obj_rotations[obj] @ R
        """
        camera_pose = self._camera_pose(scene_id, camera, ann_id)

        xml =(self.graspnet_root / 'scenes' / f'scene_{scene_id:04d}' / camera
               / 'annotations' / f'{ann_id:04d}.xml')
        pose_vectors        = xmlReader(str(xml)).getposevectorlist()
        obj_list, pose_list = get_obj_pose_list(camera_pose, pose_vectors)
        inv_camera_pose     = np.linalg.inv(camera_pose)

        labels        = self._scene_labels(scene_id)
        point_obj_ids = np.asarray(labels['point_obj_ids'])
        grasp_points  = np.zeros((len(point_obj_ids), 3), dtype=np.float32)
        obj_rotations = {}

        for obj_id, trans in zip(obj_list, pose_list):
            sel = point_obj_ids == obj_id
            pts = transform_points(np.asarray(labels['points'][sel]), trans)
            grasp_points[sel]     = transform_points(pts, inv_camera_pose)
            obj_rotations[obj_id] = inv_camera_pose[:3, :3] @ trans[:3, :3]

        return grasp_points, obj_rotations

    def _assign_grasps(
        self,
        frame:         str,
        scene_id:      int,
        coords:        np.ndarray,
        masks:         np.ndarray,
        valid_obj_ids: List[int],
        grasp_points:  np.ndarray,
        obj_rotations: Dict[int, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """
        For each object m, every point n on its mask is matched to the
        nearest labelled grasp point of that object within grasp_radius,
        and gets ALL grasps of that labelled point with friction
        <= fric_coef_thresh.

        Returns:
            quality:   [M, N]  1.1 - lowest friction of the point's grasps,
                               0 where the point has no grasp
            point_idx: [T]     m * N + n of each grasp
            rotation:  [T, 6]  6D rotation, camera frame
            width:     [T]     gripper width (m)
        """
        M, N   = masks.shape
        labels = self._scene_labels(scene_id)

        point_obj_ids = np.asarray(labels['point_obj_ids'])
        point_offsets = np.asarray(labels['point_offsets'])

        quality = np.zeros((M, N), dtype=np.float32)
        out_idx, out_rot, out_width = [], [], []

        for m, obj_id in enumerate(valid_obj_ids):
            obj_points = np.flatnonzero(point_obj_ids == obj_id)
            if len(obj_points) == 0:
                warnings.warn(
                    f"{frame}: object {obj_id} has no grasp passing the "
                    f"labels' friction threshold, it gets no grasp labels"
                )
                continue

            pts = np.flatnonzero(masks[m])
            nn_idx, nn_dists = nearest_neighbor(coords[pts], grasp_points[obj_points])
            close = nn_dists < self.grasp_radius ** 2
            pts, k = pts[close], obj_points[nn_idx[close]]
            if len(pts) == 0:
                continue

            # all grasps of each matched labelled point:
            # grasps of point k are [point_offsets[k], point_offsets[k + 1])
            starts = point_offsets[k]
            counts = point_offsets[k + 1] - starts
            n_rep  = np.repeat(pts, counts)
            g_idx  = (np.repeat(starts - np.cumsum(counts) + counts, counts)
                      + np.arange(counts.sum()))

            fric = np.asarray(labels['grasp_fric'][g_idx])
            keep = fric <= self.fric_coef_thresh + 1e-6
            n_rep, g_idx, fric = n_rep[keep], g_idx[keep], fric[keep]
            if len(g_idx) == 0:
                continue

            best_fric = np.full(N, np.inf, dtype=np.float32)
            np.minimum.at(best_fric, n_rep, fric)
            has = np.isfinite(best_fric)
            quality[m, has] = 1.1 - best_fric[has]  # graspnetAPI score

            # step 3 of loadGrasp, then into this camera view
            R = batch_viewpoint_params_to_matrix(
                -self.views[np.asarray(labels['grasp_views'][g_idx])],
                np.asarray(labels['grasp_angles'][g_idx]),
            )
            R = obj_rotations[obj_id][None] @ R

            out_idx.append(m * N + n_rep)
            out_rot.append(rotation_matrix_to_6d(R).astype(np.float32))
            out_width.append(np.asarray(labels['grasp_widths'][g_idx]))

        if out_idx:
            point_idx = np.concatenate(out_idx)
            rotation  = np.concatenate(out_rot)
            width     = np.concatenate(out_width)
        else:
            point_idx = np.zeros(0, dtype=np.int64)
            rotation  = np.zeros((0, 6), dtype=np.float32)
            width     = np.zeros(0, dtype=np.float32)

        return {
            'quality':   torch.from_numpy(quality),
            'point_idx': torch.from_numpy(point_idx.astype(np.int64)),
            'rotation':  torch.from_numpy(rotation),
            'width':     torch.from_numpy(width.astype(np.float32)),
        }

    def _augment(
        self,
        coords:        np.ndarray,
        colors:        np.ndarray,
        grasp_points:  np.ndarray,
        obj_rotations: Dict[int, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        """
        Random rotation around the camera Z axis through the point cloud
        centroid, plus point and color jitter. Grasp points and grasp
        rotations are rotated with the scene so labels stay consistent.
        """
        angle        = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_aug        = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ], dtype=np.float32)
        center = coords.mean(axis=0)

        coords = (coords - center) @ R_aug.T + center
        coords = coords + np.random.normal(
            0, 0.002, coords.shape
        ).astype(np.float32)
        colors = np.clip(
            colors + np.random.normal(0, 0.02, colors.shape),
            0, 1
        ).astype(np.float32)

        grasp_points  = (grasp_points - center) @ R_aug.T + center
        obj_rotations = {o: R_aug @ R for o, R in obj_rotations.items()}

        return coords, colors, grasp_points, obj_rotations
