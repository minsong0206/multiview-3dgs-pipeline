"""
GPS prior 궤적 vs COLMAP 추정 궤적 비교 시각화
- GPS: prior/images.txt (qvec+tvec → c2w → ENU position)
- COLMAP: sparse/0/images.bin (qvec+tvec → c2w position)
- 출력: vis_gps_vs_colmap.png
"""

import os
import struct
import math
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────────────────────
BASE       = '/home/ms/260308-KIST-Videos/KIST_CURVE_FRONT/kist_curve_front_colmap'
PRIOR_TXT  = os.path.join(BASE, 'prior/images.txt')
SPARSE_BIN = os.path.join(BASE, 'sparse/0/images.bin')
GPS_CSV    = '/home/ms/260308-KIST-Videos/RAW_DATA/6_GPS/2_Entrance-L1.csv'
OUT_PNG    = '/home/ms/260308-KIST-Videos/py/vis_gps_vs_colmap.png'

N_FRAMES      = 180
CAM_FPS       = 12.5
CAM_START_SEC = 120.0
GPS_ROW_RATE  = 27.0


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ])

def w2c_to_pos(qvec, tvec):
    """w2c (q, t) → world position of camera"""
    R = qvec2rotmat(qvec)
    t = np.array(tvec)
    # c2w position = -R^T @ t
    return -R.T @ t


# ── 1. GPS prior 궤적 (prior/images.txt) ─────────────────────────────────────
# 형식: image_id qw qx qy qz tx ty tz camera_id name
prior_pos = {}  # frame_idx(1-based) → xyz
with open(PRIOR_TXT) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            iid  = int(parts[0])
            qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
            tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
            name = parts[9]  # e.g. 000001.jpg
            frame_idx = int(os.path.splitext(name)[0])
            pos = w2c_to_pos(qvec, tvec)
            prior_pos[frame_idx] = pos
        except Exception:
            continue

print(f"GPS prior poses loaded: {len(prior_pos)}")

# ── 2. COLMAP sparse 궤적 (sparse/0/images.bin) ───────────────────────────────
colmap_pos = {}
with open(SPARSE_BIN, 'rb') as f:
    num = struct.unpack('<Q', f.read(8))[0]
    for _ in range(num):
        iid  = struct.unpack('<i', f.read(4))[0]
        qvec = struct.unpack('<4d', f.read(32))
        tvec = struct.unpack('<3d', f.read(24))
        cid  = struct.unpack('<i', f.read(4))[0]
        name = b''
        while True:
            c = f.read(1)
            if c == b'\x00':
                break
            name += c
        name = name.decode()
        n2d  = struct.unpack('<Q', f.read(8))[0]
        f.read(n2d * 24)
        try:
            frame_idx = int(os.path.splitext(os.path.basename(name))[0])
            pos = w2c_to_pos(qvec, tvec)
            colmap_pos[frame_idx] = pos
        except Exception:
            continue

print(f"COLMAP poses loaded: {len(colmap_pos)}")

# ── 3. 원본 GPS 궤적 (CSV) ────────────────────────────────────────────────────
gps_rows = []
with open(GPS_CSV) as f:
    for row in csv.DictReader(f):
        gps_rows.append((float(row['lat_deg']), float(row['lon_deg']), float(row['alt_m'])))

all_pos_raw = np.array([
    [math.radians(r[1]-gps_rows[0][1]) * 6378137.0 * math.cos(math.radians(gps_rows[0][0])),
     math.radians(r[0]-gps_rows[0][0]) * 6378137.0,
     r[2]-gps_rows[0][2]]
    for r in gps_rows
])
diffs_raw = np.linalg.norm(np.diff(all_pos_raw, axis=0), axis=1)
change_rows = np.where(diffs_raw > 0.001)[0] + 1
change_rows = np.concatenate([[0], change_rows])
gps_times_sec = change_rows / GPS_ROW_RATE
gps_pos_enu   = all_pos_raw[change_rows]

cam_times = np.array([CAM_START_SEC + i / CAM_FPS for i in range(N_FRAMES)])
gps_interp = np.column_stack([
    np.interp(cam_times, gps_times_sec, gps_pos_enu[:, 0]),
    np.interp(cam_times, gps_times_sec, gps_pos_enu[:, 1]),
    np.interp(cam_times, gps_times_sec, gps_pos_enu[:, 2]),
])
print(f"GPS interpolated: {len(gps_interp)} frames")

# ── 4. 정렬: GPS interp(ENU, meters)와 COLMAP을 Procrustes alignment ──────────
# prior_pos는 unit-scale이므로 사용하지 않고 GPS interp를 기준으로 사용
common = sorted(colmap_pos.keys())
print(f"COLMAP frames: {len(common)}")

# GPS interp는 frame index 1-based
gps_pts    = np.array([gps_interp[i-1]   for i in common])   # (N,3) ENU meters
colmap_pts = np.array([colmap_pos[i]     for i in common])   # (N,3)

# prior_pts도 GPS와 동일하게 설정 (비교 기준)
prior_pts = gps_pts

# centroid 정렬 (Procrustes: rotation + translation)
p_mean = prior_pts.mean(0)
c_mean = colmap_pts.mean(0)
P = prior_pts  - p_mean
C = colmap_pts - c_mean

# scale: path-length 기반 (GPS 경로길이 / COLMAP 경로길이)
gps_pathlen    = np.linalg.norm(np.diff(prior_pts,  axis=0), axis=1).sum()
colmap_pathlen = np.linalg.norm(np.diff(colmap_pts, axis=0), axis=1).sum()
scale = gps_pathlen / (colmap_pathlen + 1e-9)

# rotation (SVD) - scale 제거 후 rotation만 추정
P_norm = P / (np.linalg.norm(P) + 1e-9)
C_norm = C / (np.linalg.norm(C) + 1e-9)
H = C_norm.T @ P_norm
U, S, Vt = np.linalg.svd(H)
R_align = Vt.T @ U.T
if np.linalg.det(R_align) < 0:
    Vt[-1, :] *= -1
    R_align = Vt.T @ U.T

# COLMAP → GPS 좌표계 (path-length scale + rotation + translation)
colmap_aligned = scale * (R_align @ (colmap_pts - c_mean).T).T + p_mean

print(f"Alignment scale factor (GPS_pathlen/COLMAP_pathlen): {scale:.4f}")
print(f"GPS path length: {gps_pathlen:.2f} m, COLMAP path length: {colmap_pathlen:.4f} units")

# ── 5. 시각화 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('GPS Prior vs COLMAP Trajectory (KIST_CURVE_FRONT)', fontsize=14, fontweight='bold')

plot_configs = [
    (0, 1, 'Top-down (East-North)', 'East (m)', 'North (m)', True),
    (0, 2, 'Side view (East-Up)',   'East (m)', 'Up (m)',    False),
]

for ax, (xi, yi, title, xlabel, ylabel, equal_aspect) in zip(axes, plot_configs):
    # GPS 전체 궤적 (배경)
    ax.plot(gps_pos_enu[:, xi], gps_pos_enu[:, yi],
            color='lightgray', linewidth=1.5, label='GPS full track', zorder=1)

    # GPS (해당 구간)
    ax.plot(prior_pts[:, xi], prior_pts[:, yi],
            color='royalblue', linewidth=2.5, label='GPS interpolated (180 frames)', zorder=2)
    ax.scatter(prior_pts[0, xi], prior_pts[0, yi],
               color='blue', s=100, zorder=5, marker='o')
    ax.scatter(prior_pts[-1, xi], prior_pts[-1, yi],
               color='navy', s=100, zorder=5, marker='s')

    # COLMAP aligned
    ax.plot(colmap_aligned[:, xi], colmap_aligned[:, yi],
            color='crimson', linewidth=2.5, linestyle='--', label='COLMAP (aligned)', zorder=3)
    ax.scatter(colmap_aligned[0, xi], colmap_aligned[0, yi],
               color='red', s=100, zorder=5, marker='o')
    ax.scatter(colmap_aligned[-1, xi], colmap_aligned[-1, yi],
               color='darkred', s=100, zorder=5, marker='s')

    # 프레임별 오차선 (20프레임마다)
    for k in range(0, len(common), 20):
        ax.plot([prior_pts[k, xi], colmap_aligned[k, xi]],
                [prior_pts[k, yi], colmap_aligned[k, yi]],
                color='orange', linewidth=1.0, alpha=0.7, zorder=4)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    if equal_aspect:
        ax.set_aspect('equal')
    # zoom to the 180-frame region with padding
    all_x = np.concatenate([prior_pts[:, xi], colmap_aligned[:, xi]])
    all_y = np.concatenate([prior_pts[:, yi], colmap_aligned[:, yi]])
    pad_x = max((all_x.max() - all_x.min()) * 0.15, 5)
    pad_y = max((all_y.max() - all_y.min()) * 0.15, 5)
    ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
    ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)

# 오차 통계
errs = np.linalg.norm(colmap_aligned - prior_pts, axis=1)
fig.text(0.5, 0.01,
         f'Alignment: scale={scale:.3f}x  |  RMSE={errs.mean():.3f}m  |  max_err={errs.max():.3f}m  |  common={len(common)} frames',
         ha='center', fontsize=10, color='darkgreen')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUT_PNG}")
print(f"RMSE (after alignment): {errs.mean():.4f} m")
print(f"Max error:              {errs.max():.4f} m")
