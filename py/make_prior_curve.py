"""
kist_curve용 COLMAP prior 생성
- cameras.txt: 6카메라, OPENCV, fx=444.3 (DJI Action 5 Pro Linear mode)
- images.txt:  GPS 기반 초기 pose (6카메라 전체)
- points3D.txt: 빈 파일

GPS 파라미터:
- CSV 구조: 1Hz GPS가 27행씩 중복 저장 (row/27 = 실제 시간(초))
- 실제 GPS 업데이트: ~1Hz (27행마다 값 변화)
- 카메라 시작: GPS 기준 120초 (2:00)
- 촬영 구간: 120.0 ~ 134.32초, 12.5fps, 180장
- 보간 방식: 실제 이동 row만 추출 후 카메라 시간에 선형 보간 (frozen 제거)

[수정 이력]
- GPS_ROW_RATE 기반 단순 보간 → 실제 GPS 포인트 추출 후 시간 보간으로 변경
  (이전 방식은 27행 중복 구간에서 동일 위치가 반복되어 frozen frame 발생)
"""

import os
import csv
import math
import sqlite3
import numpy as np
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
GPS_CSV       = '/home/ms/260308-KIST-Videos/6_GPS/2_Entrance-L1.csv'
OUT_DIR       = '/home/ms/260308-KIST-Videos/kist_curve'
N_FRAMES      = 180
CAM_FPS       = 12.5
CAM_START_SEC = 120.0        # 카메라 촬영 시작 시각 (GPS 시간 기준, 초)
GPS_ROW_RATE  = 27.0         # CSV 행/초 (1Hz GPS가 27배 업샘플된 파일)

# Camera intrinsics (OPENCV, 800x450, DJI Action 5 Pro Linear mode)
W, H  = 800, 450
FX = FY = 444.3
CX, CY = 400.0, 225.0

# ── GPS 읽기 ──────────────────────────────────────────────────────────────────
gps_rows = []
with open(GPS_CSV) as f:
    for row in csv.DictReader(f):
        gps_rows.append((float(row['lat_deg']), float(row['lon_deg']), float(row['alt_m'])))

print(f"GPS rows loaded: {len(gps_rows)}")

# ── WGS84 → ENU ───────────────────────────────────────────────────────────────
def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    R = 6378137.0
    east  = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * R
    up    = alt - alt0
    return np.array([east, north, up])

# ── 실제 이동 GPS 포인트만 추출 후 시간 보간 ──────────────────────────────────
# CSV 구조: row i의 시각 = i / GPS_ROW_RATE (초)
# 실제 GPS 업데이트는 ~27행마다 1번 → 중복 행을 버리고 실제 포인트만 사용
all_pos_raw = np.array([
    [math.radians(r[1]-gps_rows[0][1]) * 6378137.0 * math.cos(math.radians(gps_rows[0][0])),
     math.radians(r[0]-gps_rows[0][0]) * 6378137.0,
     r[2]-gps_rows[0][2]]
    for r in gps_rows
])
diffs_raw = np.linalg.norm(np.diff(all_pos_raw, axis=0), axis=1)

# 값이 바뀌는 row 인덱스 → 실제 GPS 포인트
change_rows = np.where(diffs_raw > 0.001)[0] + 1
change_rows = np.concatenate([[0], change_rows])
gps_times_sec = change_rows / GPS_ROW_RATE   # row → 시각(초)
gps_pos_enu   = all_pos_raw[change_rows]

print(f"실제 GPS 포인트: {len(gps_times_sec)}개 ({gps_times_sec[0]:.2f}s ~ {gps_times_sec[-1]:.2f}s)")

# ── 카메라 프레임별 ENU 위치: 실제 GPS 포인트에서 선형 보간 ───────────────────
cam_times = np.array([CAM_START_SEC + i / CAM_FPS for i in range(N_FRAMES)])

positions = np.column_stack([
    np.interp(cam_times, gps_times_sec, gps_pos_enu[:, 0]),
    np.interp(cam_times, gps_times_sec, gps_pos_enu[:, 1]),
    np.interp(cam_times, gps_times_sec, gps_pos_enu[:, 2]),
])

# origin: 첫 카메라 프레임 위치
positions = positions - positions[0]

diffs_check = np.linalg.norm(np.diff(positions, axis=0), axis=1)
print(f"보간 후 프레임간 이동: mean={diffs_check.mean():.3f}m  min={diffs_check.min():.4f}m  max={diffs_check.max():.3f}m")
print(f"Position range: {positions.min(axis=0).round(3)} ~ {positions.max(axis=0).round(3)}")

# ── Rotation 추정: 이동 방향을 카메라 forward로 ───────────────────────────────
def make_c2w(pos, forward_vec, yaw_offset_deg=0.0):
    """
    pos: ENU 위치 (origin-relative)
    forward_vec: 차량 이동 방향 (ENU)
    yaw_offset_deg: 차량 forward 기준 카메라 yaw 오프셋 (도)
                    0=FRONT, 180=BACK, 60=FRONT_LEFT, -60=FRONT_RIGHT,
                    120=BACK_LEFT, -120=BACK_RIGHT
    카메라 convention: z-forward, y-down (COLMAP)
    """
    fwd_vehicle = forward_vec / (np.linalg.norm(forward_vec) + 1e-8)
    up_enu = np.array([0.0, 0.0, 1.0])  # ENU z=Up

    cos_y = math.cos(math.radians(yaw_offset_deg))
    sin_y = math.sin(math.radians(yaw_offset_deg))
    right_vehicle = np.cross(fwd_vehicle, up_enu)
    right_vehicle = right_vehicle / (np.linalg.norm(right_vehicle) + 1e-8)
    cam_fwd = cos_y * fwd_vehicle + sin_y * right_vehicle
    cam_fwd = cam_fwd / (np.linalg.norm(cam_fwd) + 1e-8)

    right = np.cross(cam_fwd, up_enu)
    right = right / (np.linalg.norm(right) + 1e-8)
    up    = np.cross(right, cam_fwd)

    # COLMAP: x=right, y=down, z=forward
    R_c2w = np.stack([right, -up, cam_fwd], axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3]  = pos
    return c2w

# 카메라별 yaw offset (차량 forward 기준, 도)
# DJI Osmo Action 5 Pro 6카메라 배치 기준
CAM_YAW = {
    'CAM_FRONT':        0.0,
    'CAM_FRONT_RIGHT': -60.0,
    'CAM_BACK_RIGHT': -120.0,
    'CAM_BACK':        180.0,
    'CAM_BACK_LEFT':   120.0,
    'CAM_FRONT_LEFT':   60.0,
}

def rotmat2qvec(R):
    Rxx,Ryx,Rzx,Rxy,Ryy,Rzy,Rxz,Ryz,Rzz = R.flat
    K = np.array([
        [Rxx-Ryy-Rzz, 0, 0, 0],
        [Ryx+Rxy, Ryy-Rxx-Rzz, 0, 0],
        [Rzx+Rxz, Rzy+Ryz, Rzz-Rxx-Ryy, 0],
        [Ryz-Rzy, Rzx-Rxz, Rxy-Ryx, Rxx+Ryy+Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3,0,1,2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

# forward 벡터: 보간된 위치에서 중앙차분 (smooth)
def compute_forwards(positions):
    N = len(positions)
    fwds = []
    for i in range(N):
        # 중앙차분: i-1 → i+1 (경계는 단방향)
        i0 = max(0, i - 2)
        i1 = min(N - 1, i + 2)
        fwd = positions[i1] - positions[i0]
        if np.linalg.norm(fwd) < 1e-6:
            fwd = np.array([1.0, 0.0, 0.0])
        fwds.append(fwd)
    return fwds

forwards = compute_forwards(positions)

# ── prior 폴더 생성 ───────────────────────────────────────────────────────────
prior_dir = os.path.join(OUT_DIR, 'prior')
os.makedirs(prior_dir, exist_ok=True)

# cameras.txt (6카메라, OPENCV, distortion은 0으로 초기화 → BA에서 추정)
with open(os.path.join(prior_dir, 'cameras.txt'), 'w') as f:
    for cid in range(1, 7):
        f.write(f"{cid} OPENCV {W} {H} {FX} {FY} {CX} {CY} 0.0 0.0 0.0 0.0\n")
print("cameras.txt written")

# points3D.txt (빈 파일)
Path(os.path.join(prior_dir, 'points3D.txt')).touch()
print("points3D.txt written")

# database에서 image_id, name, camera_id 읽기
DB_PATH = os.path.join(OUT_DIR, 'database.db')
db = sqlite3.connect(DB_PATH)
cur = db.cursor()
db_images = cur.execute('SELECT image_id, name, camera_id FROM images').fetchall()
db.close()

name2db = {name: (iid, cid) for iid, name, cid in db_images}

CAMERAS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

# images.txt: 카메라별 yaw offset 적용한 pose
written = 0
with open(os.path.join(prior_dir, 'images.txt'), 'w') as f:
    for i in range(N_FRAMES):
        for cam in CAMERAS:
            img_name = f"{cam}/{i+1:06d}.jpg"
            if img_name not in name2db:
                continue
            iid, cid = name2db[img_name]
            yaw = CAM_YAW.get(cam, 0.0)
            c2w = make_c2w(positions[i], forwards[i], yaw_offset_deg=yaw)
            w2c = np.linalg.inv(c2w)
            q   = rotmat2qvec(w2c[:3, :3])
            t   = w2c[:3, 3]
            f.write(f"{iid} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {cid} {img_name}\n\n")
            written += 1

print(f"images.txt written ({written} entries)")
print(f"\nSample frame 1:   pos={positions[0].round(4)}")
print(f"Sample frame 90:  pos={positions[89].round(4)}")
print(f"Sample frame 180: pos={positions[179].round(4)}")
print(f"Total displacement: {np.linalg.norm(positions[-1] - positions[0]):.2f} m")
