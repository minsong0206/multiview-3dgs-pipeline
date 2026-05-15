#!/usr/bin/env python3
"""
Prepare a copied KIST HUGSIM source folder with a gravity-level meta_data.json.

This script does not generate point clouds. It only prepares the input folder
for the original HUGSIM preprocessing steps:
  python /workspace/data/utils/merge_depth_wo_ground.py ...
  python /workspace/data/utils/merge_depth_ground.py ...

The original KIST meta normalized the scene by the full first CAM_FRONT pose.
If that camera has pitch/roll, horizontal vehicle motion leaks into HUGSIM Y
(camera-down) and creates a fake multi-meter ground slope. Here we rebuild
meta_data.json from COLMAP 0_aligner using a gravity-level world:
  - HUGSIM +Y = physical down = -COLMAP aligned Z.
  - HUGSIM +Z = first CAM_FRONT forward projected onto the horizontal plane.
  - HUGSIM +X = right, completing the camera-style basis.

Optional --flatten-up removes only vehicle/rig trajectory vertical drift in
COLMAP aligned Z, applying the same per-timestamp correction to all six cameras.
Camera rotations and relative camera extrinsics are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

HUGSIM_ROOT = "/workspace" if os.path.isdir("/workspace") else "/home/ms/HUGSIM_N/HUGSIM"
import sys

sys.path.insert(0, os.path.join(HUGSIM_ROOT, "data"))
from colmap.colmap_reader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat


FRONT_CAM = "CAM_FRONT"
CAMERAS = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
]
VIDEO_FPS = 12.5


def camera_name(rgb_path: str) -> str:
    return Path(rgb_path).parent.name


def frame_key(rgb_path: str) -> str:
    return Path(rgb_path).stem


def copy_or_link(src: Path, dst: Path, copy_assets: bool) -> None:
    if os.path.lexists(dst):
        return
    if copy_assets:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def colmap_c2w(image) -> np.ndarray:
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(image.qvec)
    w2c[:3, 3] = image.tvec
    return np.linalg.inv(w2c)


def intrinsics_from_colmap(cam) -> np.ndarray:
    if cam.model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        fx, fy = f, f
    else:
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def make_level_basis(origin_c2w: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    down = -up
    forward = origin_c2w[:3, 2].astype(np.float64)
    forward = forward - np.dot(forward, up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    # Columns are HUGSIM basis vectors expressed in COLMAP coordinates.
    return np.stack([right, down, forward], axis=1)


def moving_average(values: np.ndarray, smooth: int) -> np.ndarray:
    if smooth <= 1:
        return values
    if smooth % 2 == 0:
        smooth += 1
    kernel = np.ones(smooth, dtype=np.float64) / smooth
    padded = np.pad(values, (smooth // 2, smooth // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def build_meta_from_colmap(
    colmap_path: Path,
    image_dir: Path,
    cam_height: float,
    flatten_up: bool,
    smooth: int,
) -> tuple[dict, dict]:
    cam_extrinsics = read_extrinsics_binary(str(colmap_path / "images.bin"))
    cam_intrinsics = read_intrinsics_binary(str(colmap_path / "cameras.bin"))

    name2pose = {}
    name2camid = {}
    for _, image in cam_extrinsics.items():
        name = image.name if "/" in image.name else f"{FRONT_CAM}/{image.name}"
        name2pose[name] = colmap_c2w(image)
        name2camid[name] = image.camera_id

    front_files = sorted((image_dir / FRONT_CAM).glob("*.jpg"))
    if not front_files:
        raise FileNotFoundError(f"No jpg files in {image_dir / FRONT_CAM}")

    front_names = sorted([n for n in name2pose if n.startswith(f"{FRONT_CAM}/")])
    if not front_names:
        raise RuntimeError("No CAM_FRONT frames registered in COLMAP")

    origin_name = front_names[0]
    origin_c2w = name2pose[origin_name]
    origin_t = origin_c2w[:3, 3].copy()
    basis = make_level_basis(origin_c2w)

    front_keys = [p.stem for p in front_files]
    front_z = []
    for key in front_keys:
        name = f"{FRONT_CAM}/{key}.jpg"
        if name in name2pose:
            front_z.append(name2pose[name][2, 3])
        else:
            front_z.append(np.nan)
    front_z = np.asarray(front_z, dtype=np.float64)
    valid = ~np.isnan(front_z)
    baseline_z = float(front_z[valid][0])
    z_for_drift = moving_average(front_z, smooth)
    drift_by_key = {}
    for key, z, ok in zip(front_keys, z_for_drift, valid):
        drift_by_key[key] = float(z - baseline_z) if ok and flatten_up else 0.0

    first_cam_model = next(iter(cam_intrinsics.values())).model
    meta = {
        "camera_model": first_cam_model,
        "verts": {},
        "frames": [],
        "inv_pose": np.eye(4).tolist(),
    }

    missing = []
    for local_idx, front_file in enumerate(front_files):
        key = front_file.stem
        timestamp = local_idx / VIDEO_FPS
        dz = drift_by_key.get(key, 0.0)

        for cam in CAMERAS:
            img_name = f"{cam}/{key}.jpg"
            rgb_path = f"./images/{cam}/{key}.jpg"
            if img_name not in name2pose:
                missing.append(img_name)
                continue

            c2w_col = name2pose[img_name].copy()
            c2w_col[2, 3] -= dz

            c2w = np.eye(4)
            c2w[:3, :3] = basis.T @ c2w_col[:3, :3]
            c2w[:3, 3] = basis.T @ (c2w_col[:3, 3] - origin_t)

            cid = name2camid[img_name]
            K = intrinsics_from_colmap(cam_intrinsics[cid])
            meta["frames"].append(
                {
                    "rgb_path": rgb_path,
                    "camtoworld": c2w.tolist(),
                    "intrinsics": K.tolist(),
                    "width": cam_intrinsics[cid].width,
                    "height": cam_intrinsics[cid].height,
                    "timestamp": round(timestamp, 6),
                    "dynamics": {},
                }
            )

    y_vals = []
    for fr in meta["frames"]:
        if camera_name(fr["rgb_path"]) == FRONT_CAM:
            y_vals.append(np.asarray(fr["camtoworld"])[1, 3])
    y_vals = np.asarray(y_vals)

    debug = {
        "method": "rebuild meta from COLMAP 0_aligner with gravity-level HUGSIM basis",
        "colmap_path": str(colmap_path),
        "image_dir": str(image_dir),
        "origin_name": origin_name,
        "basis_columns_colmap_right_down_forward": basis.tolist(),
        "cam_height": cam_height,
        "expected_flat_ground_y": cam_height,
        "flatten_up": flatten_up,
        "smooth": smooth,
        "colmap_front_z_min": float(np.nanmin(front_z)),
        "colmap_front_z_max": float(np.nanmax(front_z)),
        "colmap_front_z_std": float(np.nanstd(front_z)),
        "removed_colmap_z_drift_min": float(min(drift_by_key.values())),
        "removed_colmap_z_drift_max": float(max(drift_by_key.values())),
        "hugsim_front_y_min": float(y_vals.min()),
        "hugsim_front_y_max": float(y_vals.max()),
        "hugsim_front_y_std": float(y_vals.std()),
        "missing_count": len(missing),
        "missing_first_10": missing[:10],
        "changed": [
            "world basis: HUGSIM +Y is gravity down instead of first-camera down",
            "optional per-timestamp COLMAP Z drift removal for all cameras",
        ],
        "preserved": [
            "COLMAP camera rotations, expressed in the new level basis",
            "relative six-camera rig extrinsics at each timestamp",
            "intrinsics",
            "image paths and timestamps",
        ],
    }
    return meta, debug


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/media/ms/WD_BLACK_4TB/KIST/raw_data_curve/recon_HUGSIM")
    parser.add_argument("--dst", default="/media/ms/WD_BLACK_4TB/KIST/raw_data_curve_ground")
    parser.add_argument("--recon-dir", default="/media/ms/WD_BLACK_4TB/KIST/raw_data_curve/recon")
    parser.add_argument("--colmap-path", default=None)
    parser.add_argument("--image-dir", default="/media/ms/WD_BLACK_4TB/KIST/raw_data_curve/images")
    parser.add_argument("--cam-height", type=float, default=1.5)
    parser.add_argument("--smooth", type=int, default=1)
    parser.add_argument("--flatten-up", action="store_true")
    parser.add_argument("--copy-assets", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    recon_dir = Path(args.recon_dir)
    colmap_path = Path(args.colmap_path) if args.colmap_path else recon_dir / "colmap" / "sparse" / "0_aligner"
    image_dir = Path(args.image_dir)
    corrected, debug = build_meta_from_colmap(
        colmap_path=colmap_path,
        image_dir=image_dir,
        cam_height=args.cam_height,
        flatten_up=args.flatten_up,
        smooth=args.smooth,
    )

    asset_sources = {
        "images": image_dir,
        "depth": src / "depth",
        "semantics": src / "semantics",
        "masks": src / "masks",
    }
    for name, src_path in asset_sources.items():
        if src_path.exists() or src_path.is_symlink():
            copy_or_link(src_path, dst / name, args.copy_assets)

    for name in ["sparse_ba.ply"]:
        src_path = src / name
        if src_path.exists():
            copy_or_link(src_path, dst / name, args.copy_assets)

    with (dst / "meta_data.json").open("w") as f:
        json.dump(corrected, f, indent=2)
    with (dst / "flat_ground_meta_debug.json").open("w") as f:
        json.dump(debug, f, indent=2)

    print(f"Prepared: {dst}")
    print("Changed meta_data.json: rebuilt from COLMAP 0_aligner in a gravity-level HUGSIM basis.")
    print(f"COLMAP CAM_FRONT Z: {debug['colmap_front_z_min']:.3f} .. {debug['colmap_front_z_max']:.3f}, std={debug['colmap_front_z_std']:.3f}")
    print(f"Removed COLMAP Z drift: {debug['removed_colmap_z_drift_min']:.3f} .. {debug['removed_colmap_z_drift_max']:.3f}")
    print(f"New HUGSIM CAM_FRONT Y: {debug['hugsim_front_y_min']:.3f} .. {debug['hugsim_front_y_max']:.3f}, std={debug['hugsim_front_y_std']:.3f}")
    print("Next run HUGSIM merge scripts:")
    print(f"  python3 /home/ms/HUGSIM_N/HUGSIM/data/utils/merge_depth_wo_ground.py --out {dst} --total 200000 --datatype kist")
    print(f"  python3 /home/ms/HUGSIM_N/HUGSIM/data/utils/merge_depth_ground.py --out {dst} --total 200000 --datatype kist")


if __name__ == "__main__":
    main()
