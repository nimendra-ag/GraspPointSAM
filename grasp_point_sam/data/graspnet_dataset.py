# grasp_point_sam/data/graspnet_dataset.py
"""
GraspNet-1B Dataset Loader — reads directly from ZIP files without extraction.

Mirrors graspnetAPI's loading logic exactly, but reads from ZIP files
instead of extracted folders. Random subsampling to num_points is applied
after loading — standard practice for all 3D point cloud GraspNet methods.

ZIP files required:
    - train_1.zip (or train_2/3/4, test_seen, test_similar, test_novel)
    - grasp_label.zip

Note: ZIP files are opened lazily per DataLoader worker to avoid
BadZipFile errors from concurrent access across workers.
"""

import io
import warnings
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def depth_to_pointcloud(
    depth: np.ndarray,
    rgb:   np.ndarray,
    camK:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth image to 3D point cloud using camera intrinsics.
    Same as graspnetAPI's loadScenePointCloud logic.

    Args:
        depth: [H, W] uint16, depth in millimeters
        rgb:   [H, W, 3] uint8
        camK:  [3, 3] camera intrinsic matrix

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

    z  = depth.astype(np.float32) / 1000.0
    x  = (uu - cx) * z / fx
    y  = (vv - cy) * z / fy

    coords = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    return coords, colors


def parse_annotation_xml(xml_bytes: bytes) -> Dict[int, Dict]:
    """
    Parse annotation XML to get object poses in world frame.

    Returns:
        dict mapping obj_id → {pos: [3], quat: [4]}
        quat is [w, x, y, z] format
    """
    root    = ET.fromstring(xml_bytes.decode())
    objects = {}

    for obj in root.findall('obj'):
        obj_id = int(obj.find('obj_id').text)
        pos    = np.array(
            obj.find('pos_in_world').text.split(),
            dtype=np.float32
        )
        quat   = np.array(
            obj.find('ori_in_world').text.split(),
            dtype=np.float32
        )
        objects[obj_id] = {'pos': pos, 'quat': quat}

    return objects


def quaternion_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = quat
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    return R.astype(np.float32)


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


def transform_points_to_camera(
    points:      np.ndarray,
    camera_pose: np.ndarray,
) -> np.ndarray:
    """
    Transform points from world frame to camera frame.

    Args:
        points:      [P, 3] in world frame
        camera_pose: [4, 4] camera-to-world transform

    Returns:
        [P, 3] in camera frame
    """
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    return ((points - t) @ R).astype(np.float32)


def _iter_npy_rows(f, chunk_rows: int):
    """
    Stream a .npy file object in chunks along axis 0.
    Avoids materializing the full array (grasp labels reach ~2.3 GB).

    Yields:
        (start_row, chunk) with chunk shape [<=chunk_rows, ...]
    """
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    assert not fortran, "Fortran-ordered .npy not supported"

    row_shape = shape[1:]
    row_bytes = int(np.prod(row_shape, dtype=np.int64)) * dtype.itemsize

    for start in range(0, shape[0], chunk_rows):
        n   = min(chunk_rows, shape[0] - start)
        buf = f.read(n * row_bytes)
        yield start, np.frombuffer(buf, dtype=dtype).reshape((n,) + row_shape)


def get_best_grasp_per_point(
    npz_file,
    chunk_rows: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each surface point, find the best valid grasp.
    Valid = score > 0 and collision=False.

    Reads the label npz member-by-member in row chunks so peak memory
    stays at a few hundred MB instead of several GB.

    Args:
        npz_file: file object of an object's *_labels.npz, containing
            points:    [P, 3]
            offsets:   [P, 300, 12, 4, 3]
            scores:    [P, 300, 12, 4]
            collision: [P, 300, 12, 4]

    Returns:
        valid_points:  [V, 3]
        valid_scores:  [V]
        valid_offsets: [V, 3]
    """
    with zipfile.ZipFile(npz_file) as npz:
        with npz.open('points.npy') as f:
            points = np.lib.format.read_array(f)
        P = points.shape[0]

        with npz.open('collision.npy') as f:
            collision = np.lib.format.read_array(f).reshape(P, -1)

        best_idx    = np.zeros(P, dtype=np.int64)
        best_scores = np.zeros(P, dtype=np.float32)
        has_valid   = np.zeros(P, dtype=bool)

        with npz.open('scores.npy') as f:
            for start, chunk in _iter_npy_rows(f, chunk_rows):
                chunk = chunk.reshape(len(chunk), -1)
                end   = start + len(chunk)
                valid = (chunk > 0) & (~collision[start:end])
                masked = np.where(valid, chunk, -np.inf)

                idx = np.argmax(masked, axis=1)
                best_idx[start:end]    = idx
                best_scores[start:end] = chunk[np.arange(len(chunk)), idx]
                has_valid[start:end]   = valid.any(axis=1)
        del collision

        has_valid &= best_scores > 0

        best_offsets = np.zeros((P, 3), dtype=np.float32)
        with npz.open('offsets.npy') as f:
            for start, chunk in _iter_npy_rows(f, chunk_rows):
                chunk = chunk.reshape(len(chunk), -1, 3)
                end   = start + len(chunk)
                best_offsets[start:end] = chunk[
                    np.arange(len(chunk)), best_idx[start:end]
                ]

    return (
        points[has_valid].astype(np.float32),
        best_scores[has_valid],
        best_offsets[has_valid],
    )


def nearest_neighbor(
    coords_norm:    np.ndarray,
    valid_pts_norm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KD-tree nearest neighbor (no [N, V, 3] intermediate).

    Args:
        coords_norm:    [N, 3] scene points normalized
        valid_pts_norm: [V, 3] grasp surface points normalized

    Returns:
        nn_idx:   [N] index into valid_pts_norm
        nn_dists: [N] squared distance to nearest grasp point
    """
    dists, nn_idx = cKDTree(valid_pts_norm).query(coords_norm, k=1)
    return nn_idx.astype(np.int64), (dists ** 2).astype(np.float32)


def approach_to_rotation(approach: np.ndarray) -> np.ndarray:
    """
    Build rotation matrices from approach vectors.
    Approach vector = z-axis of gripper frame.

    Args:
        approach: [K, 3] unit vectors

    Returns:
        R: [K, 3, 3]
    """
    K   = approach.shape[0]
    R   = np.zeros((K, 3, 3), dtype=np.float32)
    z   = approach

    ref           = np.tile([0, 1, 0], (K, 1)).astype(np.float32)
    parallel      = np.abs(np.sum(z * ref, axis=-1)) > 0.9
    ref[parallel] = [1, 0, 0]

    x  = np.cross(ref, z)
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)
    y  = np.cross(z, x)

    R[:, :, 0] = x
    R[:, :, 1] = y
    R[:, :, 2] = z

    return R


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class GraspNetDataset(Dataset):
    """
    GraspNet-1B dataset loader from ZIP files.

    Mirrors graspnetAPI loading exactly:
    - Full point cloud from depth back-projection
    - Random subsample to num_points (standard for 3D point cloud methods)
    - Per-point RGB colors
    - Per-point object masks
    - Per-point 6-DOF grasp labels

    ZIP files are opened lazily per DataLoader worker to avoid
    BadZipFile errors from concurrent access.

    Args:
        scene_zip_path: path to train_1.zip / test_seen.zip etc.
        grasp_zip_path: path to grasp_label.zip
        camera:         'realsense' or 'kinect'
        num_points:     points to subsample per frame (standard: 25600)
        num_frames:     frames per scene (None = all 256)
        split:          'train' or 'val'
        augment:        apply data augmentation
        cache_dir:      where per-object best-grasp labels are cached
    """

    def __init__(
        self,
        scene_zip_path: str,
        grasp_zip_path: str,
        camera:         str           = 'realsense',
        num_points:     int           = 25600,
        num_frames:     Optional[int] = 1,
        split:          str           = 'train',
        augment:        bool          = False,
        cache_dir:      str           = 'data/graspnet/grasp_cache',
    ):
        self.scene_zip_path = scene_zip_path
        self.grasp_zip_path = grasp_zip_path
        self.camera         = camera
        self.num_points     = num_points
        self.num_frames     = num_frames
        self.split          = split
        self.augment        = augment
        self.cache_dir      = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._grasp_cache   = {}

        # Open ZIP files temporarily just to build index
        # Workers will open their own ZIP file handles in __getitem__
        with zipfile.ZipFile(scene_zip_path, 'r') as z:
            self._scene_names = set(z.namelist())
        with zipfile.ZipFile(grasp_zip_path, 'r') as z:
            self._grasp_names = set(z.namelist())

        # Per-worker ZIP handles — None until first __getitem__ call
        self.scene_zip = None
        self.grasp_zip = None

        self.scenes = self._get_scene_list()

        n_val = max(1, len(self.scenes) // 10)
        if split == 'train':
            self.scenes = self.scenes[:-n_val]
        else:
            self.scenes = self.scenes[-n_val:]

        self.samples = self._build_sample_index()

        print(f"GraspNetDataset [{split}]: "
              f"{len(self.scenes)} scenes, "
              f"{len(self.samples)} samples, "
              f"{num_points} points per frame")

    def _get_zips(self):
        """
        Open ZIP files lazily — once per worker process.
        Called at the start of every __getitem__.
        Each DataLoader worker gets its own file handles.
        """
        if self.scene_zip is None:
            self.scene_zip = zipfile.ZipFile(self.scene_zip_path, 'r')
        if self.grasp_zip is None:
            self.grasp_zip = zipfile.ZipFile(self.grasp_zip_path, 'r')

    def _get_scene_list(self) -> List[str]:
        scenes = set()
        for name in self._scene_names:
            parts = name.split('/')
            if parts[0].startswith('scene_'):
                scenes.add(parts[0])
        return sorted(list(scenes))

    def _build_sample_index(self) -> List[Tuple[str, str]]:
        samples = []
        for scene in self.scenes:
            depth_files = sorted([
                n for n in self._scene_names
                if f'{scene}/{self.camera}/depth/' in n
                and n.endswith('.png')
            ])
            frame_ids = [Path(f).stem for f in depth_files]

            if (self.num_frames is not None
                    and len(frame_ids) > self.num_frames):
                indices   = np.linspace(
                    0, len(frame_ids) - 1,
                    self.num_frames, dtype=int
                )
                frame_ids = [frame_ids[i] for i in indices]

            for fid in frame_ids:
                samples.append((scene, fid))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        # Open ZIP files lazily per worker
        self._get_zips()

        scene_id, frame_id = self.samples[idx]

        # ── 1. Load RGB-D ────────────────────────────────────────────────
        depth, rgb, label, camK, camera_pose = self._load_rgbd(
            scene_id, frame_id
        )

        # ── 2. Back-project depth → point cloud ──────────────────────────
        coords, colors = depth_to_pointcloud(depth, rgb, camK)

        # ── 3. Remove invalid points (zero depth) ────────────────────────
        valid      = coords[:, 2] > 0
        coords     = coords[valid]
        colors     = colors[valid]
        label_flat = label.reshape(-1)[valid]

        # ── 4. Random subsample to num_points ────────────────────────────
        N = len(coords)
        if N > self.num_points:
            indices = np.random.choice(N, self.num_points, replace=False)
        else:
            indices = np.concatenate([
                np.arange(N),
                np.random.choice(N, self.num_points - N, replace=True)
            ])
        coords     = coords[indices]
        colors     = colors[indices]
        label_flat = label_flat[indices]

        # ── 5. Normalize coordinates to [-1, 1] ──────────────────────────
        centroid    = coords.mean(axis=0)
        coords_norm = coords - centroid
        scale       = np.abs(coords_norm).max()
        if scale > 0:
            coords_norm = coords_norm / scale
            
        coords_norm = np.clip(coords_norm, -1.0, 1.0)  # ← add here

        norm_params = {'centroid': centroid, 'scale': scale}

        # ── 6. Build per-object masks ────────────────────────────────────
        obj_ids       = self._load_object_ids(scene_id)
        masks         = []
        valid_obj_ids = []

        for obj_id in obj_ids:
            # label png stores obj_id + 1 (0 = background), as in graspnetAPI
            mask = (label_flat == obj_id + 1).astype(np.float32)
            if mask.sum() > 10:
                masks.append(mask)
                valid_obj_ids.append(obj_id)
            else:
                warnings.warn(
                    f"{scene_id} frame {frame_id}: object {obj_id} dropped, "
                    f"<= 10 of the {self.num_points} sampled points are on it"
                )

        if len(masks) == 0:
            warnings.warn(
                f"{scene_id} frame {frame_id}: no object has a usable mask, "
                f"skipping to the next sample"
            )
            return self.__getitem__((idx + 1) % len(self))

        masks = np.stack(masks, axis=0)  # [M, N]

        # ── 7. Load grasp labels ─────────────────────────────────────────
        grasp_labels = self._load_grasp_labels(
            scene_id, frame_id,
            valid_obj_ids,
            coords, coords_norm,
            camera_pose, norm_params,
        )

        # ── 8. Augmentation ──────────────────────────────────────────────
        if self.augment:
            coords_norm, colors, masks = self._augment(
                coords_norm, colors, masks
            )

        # ── 9. Convert to tensors ────────────────────────────────────────
        return {
            'coords':         torch.from_numpy(coords_norm).float(),
            'features':       torch.from_numpy(colors).float(),
            'masks':          torch.from_numpy(masks).float(),
            'grasp_quality':  grasp_labels['quality'],
            'grasp_rotation': grasp_labels['rotation'],
            'grasp_width':    grasp_labels['width'],
            'scene_id':       scene_id,
            'frame_id':       frame_id,
            'obj_ids':        valid_obj_ids,
        }

    def _load_rgbd(
        self,
        scene_id: str,
        frame_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base = f'{scene_id}/{self.camera}'

        depth = np.array(Image.open(io.BytesIO(
            self.scene_zip.read(f'{base}/depth/{frame_id}.png')
        )))

        rgb = np.array(Image.open(io.BytesIO(
            self.scene_zip.read(f'{base}/rgb/{frame_id}.png')
        )))

        label = np.array(Image.open(io.BytesIO(
            self.scene_zip.read(f'{base}/label/{frame_id}.png')
        )))

        camK = np.load(io.BytesIO(
            self.scene_zip.read(f'{base}/camK.npy')
        ))

        poses       = np.load(io.BytesIO(
            self.scene_zip.read(f'{base}/camera_poses.npy')
        ))
        camera_pose = poses[int(frame_id)]

        return depth, rgb, label, camK, camera_pose

    def _load_object_ids(self, scene_id: str) -> List[int]:
        txt = self.scene_zip.read(
            f'{scene_id}/object_id_list.txt'
        ).decode().strip().split()
        return [int(x) for x in txt]

    def _get_best_grasps(
        self,
        obj_id: int,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Best grasp per surface point for one object, cached in memory and
        on disk. Raw labels are up to ~2.3 GB per object, so they are
        reduced once and never loaded again.
        """
        if obj_id in self._grasp_cache:
            return self._grasp_cache[obj_id]

        label_path = f'grasp_label/{obj_id:03d}_labels.npz'
        if label_path not in self._grasp_names:
            warnings.warn(
                f"object {obj_id}: {label_path} not found in grasp zip, "
                f"it gets no grasp labels"
            )
            return None

        cache_path = self.cache_dir / f'{obj_id:03d}_best.npz'
        if cache_path.exists():
            c      = np.load(cache_path)
            result = (c['points'], c['scores'], c['offsets'])
        else:
            print(f"Caching best grasps for object {obj_id:03d}...")
            with self.grasp_zip.open(label_path) as f:
                result = get_best_grasp_per_point(f)
            tmp_path = cache_path.with_suffix('.tmp.npz')
            np.savez(tmp_path, points=result[0],
                     scores=result[1], offsets=result[2])
            tmp_path.replace(cache_path)

        self._grasp_cache[obj_id] = result
        return result

    def _load_grasp_labels(
        self,
        scene_id:      str,
        frame_id:      str,
        valid_obj_ids: List[int],
        coords:        np.ndarray,
        coords_norm:   np.ndarray,
        camera_pose:   np.ndarray,
        norm_params:   Dict,
    ) -> Dict:
        N = coords.shape[0]
        M = len(valid_obj_ids)

        quality  = np.zeros((M, N),    dtype=np.float32)
        rotation = np.zeros((M, N, 6), dtype=np.float32)
        width    = np.zeros((M, N, 1), dtype=np.float32)

        ann_bytes = self.scene_zip.read(
            f'{scene_id}/{self.camera}/annotations/{frame_id}.xml'
        )
        obj_poses = parse_annotation_xml(ann_bytes)

        for m_idx, obj_id in enumerate(valid_obj_ids):
            if obj_id not in obj_poses:
                warnings.warn(
                    f"{scene_id} frame {frame_id}: object {obj_id} has no pose "
                    f"in the annotation xml, it gets no grasp labels"
                )
                continue

            best = self._get_best_grasps(obj_id)
            if best is None:
                continue
            valid_pts, valid_scores, valid_off = best

            pose  = obj_poses[obj_id]
            R_obj = quaternion_to_matrix(pose['quat'])
            t_obj = pose['pos']

            if len(valid_pts) == 0:
                warnings.warn(
                    f"object {obj_id}: no collision-free grasp with score > 0, "
                    f"it gets no grasp labels"
                )
                continue

            valid_pts_world = valid_pts @ R_obj.T + t_obj
            valid_pts_cam   = transform_points_to_camera(
                valid_pts_world, camera_pose
            )

            valid_pts_norm = (
                valid_pts_cam - norm_params['centroid']
            ) / norm_params['scale']

            nn_idx, nn_dists = nearest_neighbor(
                coords_norm, valid_pts_norm
            )

            threshold  = 0.02 / norm_params['scale']
            close_mask = nn_dists < threshold ** 2

            quality[m_idx, close_mask] = valid_scores[nn_idx[close_mask]]

            offsets_assigned = valid_off[nn_idx[close_mask]]
            if len(offsets_assigned) > 0:
                norms    = np.linalg.norm(
                    offsets_assigned, axis=-1, keepdims=True
                )
                norms    = np.maximum(norms, 1e-6)
                approach = offsets_assigned / norms

                R_grasp  = approach_to_rotation(approach)
                r6d      = rotation_matrix_to_6d(R_grasp)
                rotation[m_idx, close_mask] = r6d
                width[m_idx, close_mask, 0] = norms.squeeze(-1)

        return {
            'quality':  torch.from_numpy(quality).float(),
            'rotation': torch.from_numpy(rotation).float(),
            'width':    torch.from_numpy(width).float(),
        }

    def _augment(
        self,
        coords:  np.ndarray,
        colors:  np.ndarray,
        masks:   np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotation around Z axis only — gravity direction preserved."""
        angle        = np.random.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_aug        = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1],
        ], dtype=np.float32)

        coords  = coords @ R_aug.T
        coords += np.random.normal(
            0, 0.002, coords.shape
        ).astype(np.float32)
        colors  = np.clip(
            colors + np.random.normal(0, 0.02, colors.shape),
            0, 1
        ).astype(np.float32)

        return coords, colors, masks