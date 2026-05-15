"""
GPS CSV → COLMAP prior/cameras.txt + prior/images.txt

- GPS 23.93Hz, 영상 25fps → 180프레임에 맞게 균등 보간
- GPS 이동 방향으로 yaw 계산, pitch=roll=0
- 카메라 6개 모두 동일 ego 포즈 사용 (rigid_ba에서 relative pose 보정)
- 출력 좌표계: UTM (미터 단위)
"""

import csv
import math
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

# ── Config ──────────────────────────────────────────────────────────────────
GPS_CSV   = '/home/ms/260308-KIST-Videos/6_GPS/2_Entrance-L1.csv'
OUT_DIR   = '/home/ms/260308-KIST-Videos/kist_scene/prior'
IMAGE_DIR = '/home/ms/260308-KIST-Videos/kist_frames'

GPS_HZ    = 5576 / 233.0          # 23.93 Hz
VIDEO_FPS = 25.0
START_SEC = 90.0                   # 영상에서 추출 시작 시간 (00:01:30)
N_FRAMES  = 180
IMG_W, IMG_H = 800, 450

CAMERAS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

# ── UTM 변환 (간이 equirectangular, 수십 미터 오차 없음) ──────────────────
def latlon_to_utm(lat, lon):
    """단순 평면 투영 (기준점 대비 미터 단위 XY)"""
    R_EARTH = 6378137.0
    lat_rad = math.radians(lat)
    x = R_EARTH * math.radians(lon) * math.cos(math.radians(37.601))  # 기준 위도
    y = R_EARTH * math.radians(lat)
    return x, y

# ── GPS 읽기 ────────────────────────────────────────────────────────────────
gps_rows = []
with open(GPS_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        gps_rows.append({
            'doc': int(row['doc']),
            'lat': float(row['lat_deg']),
            'lon': float(row['lon_deg']),
            'alt': float(row['alt_m']),
        })

# ── GPS 구간 추출 (90~100초) ─────────────────────────────────────────────────
start_row = round(START_SEC * GPS_HZ)          # ≈ 2154 (1-indexed)
end_row   = round((START_SEC + N_FRAMES / VIDEO_FPS) * GPS_HZ)  # ≈ 2393 (1-indexed)
gps_window = gps_rows[start_row - 1 : end_row]  # 0-indexed slice
print(f'GPS window: doc {gps_window[0]["doc"]} ~ {gps_window[-1]["doc"]}  ({len(gps_window)} rows)')

# ── UTM 변환 ────────────────────────────────────────────────────────────────
gps_t = np.array([i / GPS_HZ for i in range(len(gps_window))])  # 구간 내 시간축
gps_x = np.array([latlon_to_utm(r['lat'], r['lon'])[0] for r in gps_window])
gps_y = np.array([latlon_to_utm(r['lat'], r['lon'])[1] for r in gps_window])
gps_z = np.array([r['alt'] for r in gps_window])

# ── 180프레임 시간축으로 보간 ────────────────────────────────────────────────
frame_t = np.array([i / VIDEO_FPS for i in range(N_FRAMES)])

interp_x = interp1d(gps_t, gps_x, kind='linear', fill_value='extrapolate')(frame_t)
interp_y = interp1d(gps_t, gps_y, kind='linear', fill_value='extrapolate')(frame_t)
interp_z = interp1d(gps_t, gps_z, kind='linear', fill_value='extrapolate')(frame_t)

# 첫 프레임 기준으로 원점 정규화
origin = np.array([interp_x[0], interp_y[0], interp_z[0]])
positions = np.stack([interp_x - origin[0],
                      interp_y - origin[1],
                      interp_z - origin[2]], axis=1)  # (180, 3)

print(f'Position range: x={positions[:,0].min():.2f}~{positions[:,0].max():.2f}  '
      f'y={positions[:,1].min():.2f}~{positions[:,1].max():.2f}  '
      f'z={positions[:,2].min():.2f}~{positions[:,2].max():.2f}')

# ── Yaw 계산 (이동 방향) ─────────────────────────────────────────────────────
# COLMAP 좌표계: X=right, Y=down, Z=forward (카메라 기준)
# ego 좌표계에서 yaw: 이동 방향 = world XY 평면에서의 각도
def compute_yaw(positions):
    yaws = []
    for i in range(len(positions)):
        if i < len(positions) - 1:
            dx = positions[i+1, 0] - positions[i, 0]
            dy = positions[i+1, 1] - positions[i, 1]
        else:
            dx = positions[i, 0] - positions[i-1, 0]
            dy = positions[i, 1] - positions[i-1, 1]
        yaw = math.atan2(dy, dx)
        yaws.append(yaw)
    return np.array(yaws)

yaws = compute_yaw(positions)

# ── c2w 행렬 생성 ────────────────────────────────────────────────────────────
# 차량이 +X 방향으로 전진한다고 가정
# camera2world: R_z(yaw) 적용, translation = GPS position
def make_c2w(pos, yaw):
    # World 좌표계에서 카메라 전방 = 이동 방향
    rot = R.from_euler('z', yaw).as_matrix()
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = pos
    return c2w

def rotmat2qvec(Rmat):
    r = R.from_matrix(Rmat)
    q = r.as_quat()  # [x, y, z, w]
    return np.array([q[3], q[0], q[1], q[2]])  # [w, x, y, z]

# ── prior 폴더 생성 ──────────────────────────────────────────────────────────
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# cameras.txt: DB에서 OPENCV_FISHEYE 파라미터 읽어서 그대로 쓰기
import sqlite3, numpy as np
db = '/home/ms/260308-KIST-Videos/kist_scene/database.db'
conn = sqlite3.connect(db)
cur = conn.cursor()
cam_params = {}
for row in cur.execute('SELECT camera_id, model, width, height, params FROM cameras'):
    cid, model, w, h, params_blob = row
    params = np.frombuffer(params_blob, dtype=np.float64)
    cam_params[cid] = (model, w, h, params)
conn.close()

with open(f'{OUT_DIR}/cameras.txt', 'w') as f:
    for cid in sorted(cam_params.keys()):
        model, w, h, params = cam_params[cid]
        # model 5 = OPENCV_FISHEYE: fx fy cx cy k1 k2 k3 k4
        params_str = ' '.join(f'{v:.6f}' for v in params)
        f.write(f'{cid} OPENCV_FISHEYE {w} {h} {params_str}\n')
print(f'Wrote cameras.txt (6 cameras, OPENCV_FISHEYE)')

# points3D.txt: 빈 파일
Path(f'{OUT_DIR}/points3D.txt').touch()

# images.txt: 180프레임 × 6카메라 = 1080 entries
# image_id 순서: DB의 image_id와 일치해야 함 → DB에서 읽어옴
import sqlite3
db = '/home/ms/260308-KIST-Videos/kist_scene/database.db'
conn = sqlite3.connect(db)
cur = conn.cursor()
name2id = {name: (iid, cid) for iid, name, cid in cur.execute('SELECT image_id, name, camera_id FROM images')}
conn.close()

# cam_id 순서 파악 (DB에서 폴더→cam_id 매핑)
conn = sqlite3.connect(db)
cur = conn.cursor()
folder2camid = {}
for iid, name, cid in cur.execute('SELECT image_id, name, camera_id FROM images'):
    folder = name.split('/')[0]
    if folder not in folder2camid:
        folder2camid[folder] = cid
conn.close()
print('folder → cam_id:', folder2camid)

with open(f'{OUT_DIR}/images.txt', 'w') as f:
    for frame_idx in range(N_FRAMES):
        c2w = make_c2w(positions[frame_idx], yaws[frame_idx])
        w2c = np.linalg.inv(c2w)
        qvec = rotmat2qvec(w2c[:3, :3])
        tvec = w2c[:3, 3]

        q_str = ' '.join(f'{v:.8f}' for v in qvec)
        t_str = ' '.join(f'{v:.8f}' for v in tvec)

        for cam_name in CAMERAS:
            img_name = f'{cam_name}/{frame_idx+1:06d}.jpg'
            if img_name not in name2id:
                print(f'WARNING: {img_name} not in DB, skipping')
                continue
            iid, cid = name2id[img_name]
            f.write(f'{iid} {q_str} {t_str} {cid} {img_name}\n\n')

print(f'Wrote images.txt')
print('Done. prior/ is ready.')
