#!/usr/bin/env python3
"""
COLMAP 0_aligner → HUGSIM meta_data.json  (new video pipeline)

Unlike make_meta_data_chunk.py, this script does not require a chunk name.
It works entirely from --recon_dir, which must contain:
  - images/<CAM_NAME>/<frame>.jpg
  - colmap/sparse/0_aligner/{cameras.bin, images.bin, points3D.bin}

Usage:
  python3 make_meta_data_new.py --recon_dir /data/KIST_NEW/test_chunk/recon
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HUGSIM_ROOT = '/workspace' if os.path.isdir('/workspace') else '/home/ms/HUGSIM_N/HUGSIM'
sys.path.insert(0, os.path.join(HUGSIM_ROOT, 'data'))
from colmap.colmap_reader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

CAMERAS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

VIDEO_FPS = 12.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recon_dir', required=True,
                        help='Recon root: contains images/ and colmap/sparse/0_aligner/')
    parser.add_argument('--colmap_path', default=None,
                        help='Override aligned sparse model path (default: <recon_dir>/colmap/sparse/0_aligner)')
    parser.add_argument('--out_dir', default=None,
                        help='Output directory (default: same as recon_dir)')
    parser.add_argument('--image_dir', default=None,
                        help='Override image directory (default: recon_dir/images)')
    args = parser.parse_args()

    recon_dir   = Path(args.recon_dir)
    colmap_path = Path(args.colmap_path) if args.colmap_path \
                  else recon_dir / 'colmap' / 'sparse' / '0_aligner'
    out_dir     = Path(args.out_dir) if args.out_dir else recon_dir

    # ── Validation ────────────────────────────────────────────
    for f in ['cameras.bin', 'images.bin', 'points3D.bin']:
        p = colmap_path / f
        if not p.exists():
            sys.exit(f'ERROR: {p} not found')

    images_dir = Path(args.image_dir) if args.image_dir else recon_dir / 'images'
    if not images_dir.is_dir():
        sys.exit(f'ERROR: {images_dir} not found')

    print(f'recon_dir  : {recon_dir}')
    print(f'colmap_path: {colmap_path}')
    print(f'out_dir    : {out_dir}')

    # ── Load COLMAP ───────────────────────────────────────────
    cam_extrinsics = read_extrinsics_binary(str(colmap_path / 'images.bin'))
    cam_intrinsics = read_intrinsics_binary(str(colmap_path / 'cameras.bin'))
    print(f'Loaded {len(cam_extrinsics)} images, {len(cam_intrinsics)} cameras')

    name2pose  = {}
    name2camid = {}
    for iid, image in cam_extrinsics.items():
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(image.qvec)
        w2c[:3, 3]  = image.tvec
        c2w  = np.linalg.inv(w2c)
        name = image.name
        if '/' not in name:
            name = f'CAM_FRONT/{name}'
        name2pose[name]  = c2w
        name2camid[name] = image.camera_id

    camid2intr = {}
    for cid, cam in cam_intrinsics.items():
        if cam.model in ('SIMPLE_RADIAL', 'SIMPLE_PINHOLE'):
            f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
            fx, fy = f, f
        elif cam.model in ('PINHOLE', 'OPENCV', 'OPENCV_FISHEYE',
                           'FULL_OPENCV', 'RADIAL', 'RADIAL_FISHEYE'):
            # params[0..3] = fx, fy, cx, cy (distortion params follow, ignored here)
            fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        else:
            fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        camid2intr[cid] = (K, cam.width, cam.height)

    # ── Determine frame range from actual image files ─────────
    # Frame indices are 0-based, matching images.bin names (000000.jpg ...)
    front_files = sorted((images_dir / 'CAM_FRONT').glob('*.jpg'))
    if not front_files:
        sys.exit(f'ERROR: no .jpg files in {images_dir}/CAM_FRONT')
    n_frames = len(front_files)
    print(f'Frames per camera: {n_frames}  (0 ~ {n_frames-1})')

    # ── Origin: first registered CAM_FRONT frame ─────────────
    front_names = sorted([n for n in name2pose if n.startswith('CAM_FRONT/')])
    if not front_names:
        sys.exit('ERROR: no CAM_FRONT frames registered in COLMAP')
    origin_c2w = name2pose[front_names[0]]
    inv_pose   = np.linalg.inv(origin_c2w)

    # Use the model of the first camera in the reconstruction
    first_cam_model = next(iter(cam_intrinsics.values())).model
    meta_data = {
        'camera_model': first_cam_model,
        'verts':        {},
        'frames':       [],
        'inv_pose':     inv_pose.tolist(),
    }

    missing = []
    for local_idx in range(n_frames):
        frame_str = f'{local_idx:06d}.jpg'
        timestamp = local_idx / VIDEO_FPS

        for cam in CAMERAS:
            img_name = f'{cam}/{frame_str}'
            rgb_path = f'./images/{cam}/{frame_str}'

            if img_name not in name2pose:
                missing.append(img_name)
                continue

            # model_aligner already applied metric scale — no extra scale factor.
            c2w = inv_pose @ name2pose[img_name]
            cid = name2camid[img_name]
            K, w, h = camid2intr[cid]

            meta_data['frames'].append({
                'rgb_path':   rgb_path,
                'camtoworld': c2w.tolist(),
                'intrinsics': K.tolist(),
                'width':      w,
                'height':     h,
                'timestamp':  round(timestamp, 6),
                'dynamics':   {},
            })

    print(f'Total frames written : {len(meta_data["frames"])}')
    if missing:
        pct = len(missing) / (n_frames * len(CAMERAS)) * 100
        print(f'WARNING: {len(missing)} images not in COLMAP ({pct:.1f}%) — first 5: {missing[:5]}')
        if pct > 20:
            print('ERROR: >20% missing — COLMAP reconstruction may have failed')
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'meta_data.json'
    with open(out_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    print(f'Saved: {out_path}')

    # ── Quick sanity: translation distance ───────────────────
    translations = []
    for frame in meta_data['frames']:
        if frame['rgb_path'].startswith('./images/CAM_FRONT/'):
            t = frame['camtoworld']
            translations.append([t[0][3], t[1][3], t[2][3]])
    if len(translations) >= 2:
        t0 = np.array(translations[0])
        t1 = np.array(translations[-1])
        dist = float(np.linalg.norm(t1 - t0))
        print(f'CAM_FRONT trajectory length: {dist:.2f} m  '
              f'(first→last of {len(translations)} frames)')
        if dist < 1.0:
            print('WARNING: trajectory length < 1m — vehicle may not have moved')


if __name__ == '__main__':
    main()
