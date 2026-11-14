"""
PyBullet assets for the GraspNet-1Billion object models (built once, cached).

The original meshes are too large for PyBullet (textured.obj is ~50 MB), so
each object gets:
  visual.obj + visual.png : decimated mesh, texture baked to one colour per
                            face (one texel per face, so no UV seams)
  collision.obj           : VHACD convex decomposition of the simplified mesh
  meta.json               : mass (convex hull volume x density), centre of mass
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import pybullet as p
import trimesh
from PIL import Image

VISUAL_FACES    = 12000
COLLISION_FACES = 4000
DENSITY         = 400.0          # kg/m^3, light household objects
MASS_RANGE      = (0.02, 1.0)    # kg


def _decimate(vertices, faces, n_faces, colors=None) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
    )
    if colors is not None:
        m.vertex_colors = o3d.utility.Vector3dVector(colors)
    m.remove_duplicated_vertices()
    if len(m.triangles) > n_faces:
        m = m.simplify_quadric_decimation(n_faces)
    m.remove_degenerate_triangles()
    m.remove_unreferenced_vertices()
    m.compute_vertex_normals()
    return m


def _write_face_colour_obj(m: o3d.geometry.TriangleMesh, out: Path) -> None:
    """OBJ whose texture holds one texel per face (flat colour per face)."""
    V = np.asarray(m.vertices)
    N = np.asarray(m.vertex_normals)
    F = np.asarray(m.triangles)
    C = np.asarray(m.vertex_colors)[F].mean(axis=1)            # [F, 3]

    n    = len(F)
    side = 1 << int(np.ceil(np.log2(np.sqrt(n))))               # power of two
    tex  = np.zeros((side * side, 3), dtype=np.uint8)
    tex[:n] = np.clip(np.round(C * 255), 0, 255)
    Image.fromarray(tex.reshape(side, side, 3)).save(out / 'visual.png')

    i = np.arange(n)
    u = (i % side + 0.5) / side
    v = 1.0 - (i // side + 0.5) / side                          # OBJ: v = 0 at bottom

    with open(out / 'visual.mtl', 'w') as f:
        f.write('newmtl mat\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd visual.png\n')
    with open(out / 'visual.obj', 'w') as f:
        f.write('mtllib visual.mtl\nusemtl mat\n')
        f.write(''.join(f'v {a:.6f} {b:.6f} {c:.6f}\n' for a, b, c in V))
        f.write(''.join(f'vn {a:.6f} {b:.6f} {c:.6f}\n' for a, b, c in N))
        f.write(''.join(f'vt {a:.6f} {b:.6f}\n' for a, b in zip(u, v)))
        f.write(''.join(
            f'f {a+1}/{k+1}/{a+1} {b+1}/{k+1}/{b+1} {c+1}/{k+1}/{c+1}\n'
            for k, (a, b, c) in enumerate(F)
        ))


def prepare_object(obj_id: int, graspnet_root, cache_dir) -> dict:
    """Build (or load from cache) the assets of one object. Returns meta.json."""
    out = Path(cache_dir).resolve() / f'{obj_id:03d}'
    meta_file = out / 'meta.json'
    if meta_file.exists():
        with open(meta_file) as f:
            return json.load(f)

    src = Path(graspnet_root) / 'models' / f'{obj_id:03d}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'  building sim assets for object {obj_id:03d} ...', flush=True)

    # visual: texture -> vertex colours -> decimate -> one texel per face
    tm = trimesh.load(src / 'textured.obj', force='mesh', process=False)
    colors = tm.visual.to_color().vertex_colors[:, :3].astype(np.float64) / 255.0
    _write_face_colour_obj(_decimate(tm.vertices, tm.faces, VISUAL_FACES, colors), out)

    # collision: simplified mesh -> decimate -> VHACD
    cm = trimesh.load(src / 'nontextured_simplified.ply', force='mesh')
    cmesh = _decimate(cm.vertices, cm.faces, COLLISION_FACES)
    o3d.io.write_triangle_mesh(str(out / 'collision_src.obj'), cmesh)
    p.vhacd(str(out / 'collision_src.obj'), str(out / 'collision.obj'),
            str(out / 'vhacd.log'), resolution=200000, maxNumVerticesPerCH=64)

    hull   = cm.convex_hull
    meta   = {
        'obj_id': obj_id,
        'visual': str(out / 'visual.obj'),
        'texture': str(out / 'visual.png'),
        'collision': str(out / 'collision.obj'),
        'mass': float(np.clip(hull.volume * DENSITY, *MASS_RANGE)),
        'com': hull.centroid.tolist(),
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    return meta
