# modal_train.py
"""
Train GraspPointSAM on Modal (modal.com).

The cloud image matches the local `graspointsam` env (same package
versions); torkit3d is compiled for the cloud GPUs during the image build.

Volumes (create once with `modal volume create <name>`):
    graspnet-data -> /data
        /data/pretrained/pointsam/model.safetensors
        /data/graspnet_scene_labels/scene_XXXX/...
        /data/tars/scene_XXXX.tar          (uploaded scenes, see extract_scenes)
        /data/graspnet/scenes/scene_XXXX/  (unpacked by extract_scenes)
    graspnet-ckpt -> /ckpt
        /ckpt/checkpoints/<date_time>/...  (written by train.py)

Usage (from the `modal` conda env, in the repo root):
    modal run modal_train.py::check_env                 # image + GPU + model test, no dataset needed
    modal run modal_train.py::extract_scenes            # unpack uploaded scene tars
    modal run modal_train.py::train --config configs/graspnet_modal_test.yaml
    modal run --detach modal_train.py::train --config configs/graspnet_modal.yaml
    modal run --detach modal_train.py::train --config configs/graspnet_modal.yaml \
        --resume /ckpt/checkpoints/<run>/checkpoint_epoch_XXXX.pth
    modal run --detach modal_train.py::train --config configs/graspnet_modal.yaml --stop-epoch 10
"""

from pathlib import Path

import modal

CODE_DIR = "/root/GraspPointSAM"

# GPU for training; also e.g. "A100-80GB", "H100", "L40S"
TRAIN_GPU = "A100-40GB"

# GPUs torkit3d is compiled for: A100 (8.0), A10G (8.6), L4/L40S (8.9), H100 (9.0)
CUDA_ARCHS = "8.0;8.6;8.9;9.0"

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "libgl1", "libglib2.0-0", "libgomp1")
    .pip_install(
        "torch==2.3.1", "torchvision==0.18.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "torch_scatter==2.1.2",
        find_links="https://data.pyg.org/whl/torch-2.3.0+cu121.html",
    )
    .pip_install(
        "numpy==2.4.6", "scipy==1.17.1", "pillow==12.3.0", "tqdm==4.70.1",
        "timm==0.9.16", "einops==0.8.2", "omegaconf==2.3.0", "safetensors==0.8.0",
        "Cython==3.3.1",
        # graspnetAPI's dependencies, versions as in the local env
        "autolab-core==1.1.1", "cvxopt==1.3.3", "dill==0.4.1", "h5py==3.16.0",
        "matplotlib==3.11.2", "open3d==0.20.0", "opencv-python==5.0.0.93",
        "scikit-image==0.26.0", "scikit-learn==1.9.1", "transforms3d==0.4.2",
        "trimesh==5.2.0",
    )
    # grasp_nms's build calls clang++ (separate step so the layers above stay cached)
    .apt_install("clang")
    .pip_install("grasp_nms==1.0.2", extra_options="--no-build-isolation")
    # --no-deps as locally: its pinned extras (autolab-perception, ...) are not needed
    .pip_install("graspnetAPI==1.2.11", extra_options="--no-deps")
    # torkit3d: CUDA extension; the build machine has no GPU, so force the
    # CUDA build and name the target GPUs explicitly
    .env({"FORCE_CUDA": "1", "TORCH_CUDA_ARCH_LIST": CUDA_ARCHS})
    .add_local_dir(
        "third_party/torkit3d", "/opt/torkit3d", copy=True,
        ignore=["build", "*.egg-info", "**/__pycache__"],
    )
    .run_commands("pip install --no-build-isolation /opt/torkit3d")
    # system libraries open3d (imported by graspnetAPI) loads
    .apt_install("libegl1", "libx11-6", "libxext6", "libxrender1", "libusb-1.0-0")
    # project code last: changing it does not rebuild the image
    .add_local_dir(
        ".", CODE_DIR,
        ignore=[
            ".git", "data", "data-rgbdmatter", "pretrained", "third_party",
            "checkpoints*", "**/__pycache__", "**/*.pth",
        ],
    )
)

app = modal.App("grasp-point-sam", image=image)

data_vol = modal.Volume.from_name("graspnet-data")
ckpt_vol = modal.Volume.from_name("graspnet-ckpt")
VOLUMES  = {"/data": data_vol, "/ckpt": ckpt_vol}


@app.function(gpu="L4", volumes=VOLUMES, timeout=30 * 60)
def check_env(config: str = "configs/graspnet_modal.yaml"):
    """Packages, torkit3d on the GPU, and one model forward pass on random input."""
    import os
    import sys

    os.chdir(CODE_DIR)
    sys.path.insert(0, CODE_DIR)

    import torch
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
          f"GPU: {torch.cuda.get_device_name(0)}")

    import graspnetAPI, timm, torch_scatter  # noqa: F401
    print("graspnetAPI, timm, torch_scatter: ok")

    from torkit3d.ops.sample_farthest_points import sample_farthest_points
    idx = sample_farthest_points(torch.rand(1, 10000, 3, device='cuda'), 1024)
    print(f"torkit3d farthest point sampling on GPU: ok {tuple(idx.shape)}")

    from omegaconf import OmegaConf
    from grasp_point_sam.model.build_model import build_model

    cfg   = OmegaConf.load(config)
    model = build_model(cfg).cuda()

    N        = cfg.training.num_points
    coords   = torch.rand(1, N, 3, device='cuda') * 2 - 1
    features = torch.rand(1, N, 3, device='cuda')
    masks    = torch.stack([coords[0, :, 0] < 0, coords[0, :, 0] >= 0]).float()[None]

    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        outputs = model(coords=coords, features=features, gt_masks=masks)
    print(f"model forward: ok ({len(outputs)} prompt iterations), "
          f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")

    print("Environment OK.")


@app.function(volumes=VOLUMES, timeout=6 * 3600, cpu=4)
def extract_scenes():
    """Unpack /data/tars/scene_XXXX.tar into /data/graspnet/scenes/ (skips done scenes)."""
    import shutil
    import tarfile

    tars = sorted(Path('/data/tars').glob('scene_*.tar'))
    out  = Path('/data/graspnet/scenes')
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(tars)} scene tars found")

    for tar in tars:
        scene = tar.stem
        if (out / scene / 'object_id_list.txt').exists():
            continue
        # unpack next to the target, then rename, so a crash never leaves
        # a half-unpacked scene that looks complete
        tmp = out / f'{scene}.tmp'
        if tmp.exists():
            shutil.rmtree(tmp)
        with tarfile.open(tar) as f:
            f.extractall(tmp, filter='data')
        (tmp / scene).rename(out / scene)
        tmp.rmdir()
        data_vol.commit()
        print(f"{scene}: unpacked")

    print(f"Scenes available: {len(list(out.glob('scene_????')))}")


@app.function(gpu=TRAIN_GPU, volumes=VOLUMES, timeout=24 * 3600, cpu=12, memory=32768)
def train(config: str = "configs/graspnet_modal.yaml", resume: str = "", stop_epoch: int = 0):
    """Run train.py; checkpoints go to the graspnet-ckpt volume.

    stop_epoch > 0 ends this run after epoch stop_epoch - 1 (see train.py).
    """
    import os
    import subprocess

    cmd = ["python", "train.py", "--config", config]
    if resume:
        cmd += ["--resume", resume]
    if stop_epoch > 0:
        cmd += ["--stop_epoch", str(stop_epoch)]
    try:
        # unbuffered: step lines reach the Modal logs as they are printed
        subprocess.run(cmd, cwd=CODE_DIR, check=True,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    finally:
        # make checkpoints visible even if training crashed or timed out
        ckpt_vol.commit()
