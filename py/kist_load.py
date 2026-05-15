"""
COLMAP BA 결과 → HUGSIM meta_data.json 생성

nusc/load.py가 만드는 것과 동일한 형식:
{
  "camera_model": "OPENCV",
  "verts": {},
  "frames": [
    {
      "rgb_path": "./images/CAM_FRONT/000001.jpg",
      "camtoworld": [[...]], # 4x4
      "intrinsics": [[...]], # 3x3
      "width": 800,
      "height": 450,
      "timestamp": 0.0,
      "dynamics": {}
    },
    ...
  ],
  "inv_pose": [[...]]  # 첫 프레임 c2w의 역행렬
}
"""

import os
import sys
import json
import numpy as np

HUGSIM_ROOT = '/workspace' if os.path.isdir('/workspace') else '/home/ms/HUGSIM_N/HUGSIM'
DATA_ROOT   = '/data'     if os.path.isdir('/data')      else '/home/ms/260308-KIST-Videos'
sys.path.insert(0, os.path.join(HUGSIM_ROOT, 'data'))
from colmap.colmap_reader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

# ── Config ───────────────────────────────────────────────────────────────────
COLMAP_BA_DIR = os.path.join(DATA_ROOT, 'KIST_CURVE_ALL/kist_curve_all_exhaustive/sparse_with_rig/0')
OUT_DIR       = os.path.join(DATA_ROOT, 'KIST_CURVE_ALL/kist_curve_all_exhaustive_recon_v2')

N_FRAMES  = 180
VIDEO_FPS = 12.5

CAMERAS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

# ── COLMAP BA 결과 읽기 ───────────────────────────────────────────────────────
cam_extrinsics = read_extrinsics_binary(os.path.join(COLMAP_BA_DIR, 'images.bin'))
cam_intrinsics = read_intrinsics_binary(os.path.join(COLMAP_BA_DIR, 'cameras.bin'))

print(f'Loaded {len(cam_extrinsics)} images, {len(cam_intrinsics)} cameras')

# image name → c2w pose
# image.name is either 'CAM_FRONT/000001.jpg' (multi-cam) or '000001.jpg' (single-cam)
# normalize to 'CAM_FRONT/000001.jpg' format using CAMERAS list
name2pose = {}
for iid, image in cam_extrinsics.items():
    w2c = np.eye(4)
    w2c[:3, :3] = qvec2rotmat(image.qvec)
    w2c[:3, 3]  = image.tvec
    c2w = np.linalg.inv(w2c)
    # if name has no folder prefix, it was extracted per-camera → only CAM_FRONT here
    name = image.name
    if '/' not in name:
        name = f'CAM_FRONT/{name}'
    name2pose[name] = c2w

# camera_id → intrinsic 3x3
# SIMPLE_RADIAL: params = [f, cx, cy, k]  (single focal length, no separate fy)
# OPENCV / PINHOLE: params = [fx, fy, cx, cy, ...]
camid2intr = {}
for cid, cam in cam_intrinsics.items():
    if cam.model in ('SIMPLE_RADIAL', 'SIMPLE_PINHOLE'):
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        fx, fy = f, f
    else:  # OPENCV, RADIAL, PINHOLE, etc.
        fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    camid2intr[cid] = K

# image name → camera_id
name2camid = {}
for iid, image in cam_extrinsics.items():
    name = image.name
    if '/' not in name:
        name = f'CAM_FRONT/{name}'
    name2camid[name] = image.camera_id

# ── meta_data 생성 ────────────────────────────────────────────────────────────
meta_data = {
    "camera_model": "OPENCV",
    "verts": {},       # 동적 객체 없음 (PoC)
    "frames": [],
}

# 첫 번째 CAM_FRONT 프레임 c2w를 origin으로 설정
front_names = sorted([n for n in name2pose if n.startswith('CAM_FRONT/')])
origin_c2w  = name2pose[front_names[0]]
inv_pose    = np.linalg.inv(origin_c2w)

meta_data['inv_pose'] = inv_pose.tolist()

# 프레임 순서: nusc/load.py와 동일하게 frame별로 6카메라 순서로 append
missing = []
for frame_idx in range(N_FRAMES):
    frame_str = f'{frame_idx + 1:06d}.jpg'
    timestamp = frame_idx / VIDEO_FPS

    for cam in CAMERAS:
        img_name = f'{cam}/{frame_str}'
        rgb_path = f'./images/{cam}/{frame_str}'

        if img_name not in name2pose:
            missing.append(img_name)
            continue

        c2w  = inv_pose @ name2pose[img_name]   # origin-relative pose
        cid  = name2camid[img_name]
        intr = camid2intr[cid]
        w, h = cam_intrinsics[cid].width, cam_intrinsics[cid].height

        meta_data['frames'].append({
            "rgb_path":   rgb_path,
            "camtoworld": c2w.tolist(),
            "intrinsics": intr.tolist(),
            "width":      w,
            "height":     h,
            "timestamp":  timestamp,
            "dynamics":   {},
        })

print(f'Total frames written: {len(meta_data["frames"])}')
if missing:
    print(f'WARNING: {len(missing)} images not found in COLMAP: {missing[:5]}...')

# ── 저장 ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, 'meta_data.json')
with open(out_path, 'w') as f:
    json.dump(meta_data, f, indent=2)
print(f'Saved: {out_path}')
