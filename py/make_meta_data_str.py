"""
COLMAP BA 결과 → HUGSIM meta_data.json 생성 (KIST_STR_ALL, SIMPLE_RADIAL)
- bundled_sparse_txt/cameras.txt  → intrinsics (카메라별)
- bundled_sparse_txt/images.txt   → camtoworld (c2w)
- 출력: KIST_STR_ALL/kist_straight_recon/meta_data.json
"""

import os, json
import numpy as np

BA_TXT   = '/home/ms/260308-KIST-Videos/KIST_STR_ALL/KIST_STR_ALL_SEQ/kist_str_all_rig_seq/bundled_sparse_txt'
IMG_DIR  = '/home/ms/260308-KIST-Videos/KIST_STR_ALL/kist_straight_images'
OUT_JSON = '/home/ms/260308-KIST-Videos/KIST_STR_ALL/kist_straight_recon/meta_data.json'
W, H     = 800, 450
CAM_FPS  = 12.5

CAMERAS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

# ── quaternion → rotation matrix ────────────────────────────────────────────
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ])

# ── cameras.txt 읽기 → camera_id: intrinsics matrix ─────────────────────────
def read_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            cid = int(parts[0])
            # SIMPLE_RADIAL: f cx cy k
            fx = float(parts[4])
            fy = fx
            cx, cy = float(parts[5]), float(parts[6])
            K = [
                [fx,  0.0, cx,  0.0],
                [0.0, fy,  cy,  0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
            cams[cid] = K
    return cams

# ── images.txt 읽기 → name: c2w (4x4) ──────────────────────────────────────
def read_images(path):
    images = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            try:
                iid = int(parts[0])
            except ValueError:
                continue
            if len(parts) < 10 or '/' not in parts[9]:
                continue
            qw,qx,qy,qz = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
            tx,ty,tz     = float(parts[5]),float(parts[6]),float(parts[7])
            cid  = int(parts[8])
            name = parts[9]  # e.g. CAM_FRONT/000001.jpg
            R = qvec2rotmat([qw,qx,qy,qz])
            w2c = np.eye(4); w2c[:3,:3]=R; w2c[:3,3]=[tx,ty,tz]
            c2w = np.linalg.inv(w2c)
            images[name] = {'c2w': c2w, 'cam_id': cid}
    return images

print("Reading BA results...")
cam_intrinsics = read_cameras(os.path.join(BA_TXT, 'cameras.txt'))
ba_images      = read_images(os.path.join(BA_TXT, 'images.txt'))
print(f"  cameras: {len(cam_intrinsics)}, images: {len(ba_images)}")

# ── origin 정렬: CAM_FRONT/000001 기준 ──────────────────────────────────────
origin_key = 'CAM_FRONT/000001.jpg'
if origin_key not in ba_images:
    origin_key = list(ba_images.keys())[0]
origin_c2w = ba_images[origin_key]['c2w']
inv_origin  = np.linalg.inv(origin_c2w)

# ── camera_id → 카메라 이름 매핑 (images.txt에서 추론) ──────────────────────
cid2name = {}
for img_name, info in ba_images.items():
    cam = img_name.split('/')[0]
    cid2name[info['cam_id']] = cam

print("camera_id → name mapping:", cid2name)

# ── frames 생성 ──────────────────────────────────────────────────────────────
# frame_idx(1~180) x 6 cameras, timestamp = frame_idx / FPS
frames = []
missing = 0

for frame_idx in range(1, 181):
    t_sec = (frame_idx - 1) / CAM_FPS  # 0.0 ~ 14.32

    for cam in CAMERAS:
        img_name = f"{cam}/{frame_idx:06d}.jpg"

        if img_name not in ba_images:
            missing += 1
            continue

        info   = ba_images[img_name]
        c2w_world = info['c2w']
        cid    = info['cam_id']

        # origin 정렬
        c2w_local = inv_origin @ c2w_world

        K = cam_intrinsics[cid]

        # rgb_path: HUGSIM 관례 (images/ 아래 상대경로)
        rgb_path = f"./images/{cam}/{frame_idx:06d}.jpg"
        # images symlink: kist_straight_recon/images → kist_straight_images

        frames.append({
            'rgb_path':    rgb_path,
            'camtoworld':  c2w_local.tolist(),
            'intrinsics':  K,
            'width':       W,
            'height':      H,
            'timestamp':   round(t_sec, 6),
            'dynamics':    {},
        })

print(f"  frames generated: {len(frames)}  (missing: {missing})")

# ── inv_pose: world → first CAM_FRONT (scene normalization용) ───────────────
inv_pose = inv_origin.tolist()

# ── meta_data 조립 ───────────────────────────────────────────────────────────
meta = {
    'camera_model': 'OPENCV',
    'verts':        {},        # 동적 객체 없음
    'frames':       frames,
    'inv_pose':     inv_pose,
}

with open(OUT_JSON, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved: {OUT_JSON}")
print(f"Total frames: {len(frames)}")

# 샘플 확인
print("\n--- Sample frame[0] (CAM_BACK/000001) ---")
f0 = frames[0]
print(f"  rgb_path:  {f0['rgb_path']}")
print(f"  timestamp: {f0['timestamp']}")
print(f"  intrinsics fx: {f0['intrinsics'][0][0]:.3f}")
c2w = np.array(f0['camtoworld'])
print(f"  c2w pos:   {c2w[:3,3].round(3)}")

print("\n--- Sample frame CAM_FRONT/000001 ---")
for fr in frames:
    if 'CAM_FRONT/000001' in fr['rgb_path']:
        c2w = np.array(fr['camtoworld'])
        print(f"  rgb_path:  {fr['rgb_path']}")
        print(f"  c2w pos:   {c2w[:3,3].round(6)}  (should be ~[0,0,0])")
        print(f"  intrinsics fx: {fr['intrinsics'][0][0]:.3f}")
        break
