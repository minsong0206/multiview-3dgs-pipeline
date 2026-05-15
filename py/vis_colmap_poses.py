"""
COLMAP BA 결과 + GPS trajectory를 Open3D로 시각화
- colmap_sparse_ba/images.bin → camera frustums (6색)
- prior/images.txt             → GPS prior poses (회색)
- GPS CSV                      → trajectory (파란선)
- point cloud                  → colmap_sparse_ba/points3D.bin (회색 점)
"""

import os, sys, struct, math
import numpy as np
import open3d as o3d

# ── Config ─────────────────────────────────────────────────────────────────
BA_DIR   = '/tmp/kist_ba_txt'
PRIOR_DIR= '/home/ms/260308-KIST-Videos/kist_curve/prior'
GPS_CSV  = '/home/ms/260308-KIST-Videos/6_GPS/2_Entrance-L1.csv'
GPS_HZ   = 27.0
GPS_START_SEC = 120.0   # 카메라 시작 = GPS 2:00

N_FRAMES = 180
CAM_FPS  = 12.5
W, H     = 800, 450
FRUSTUM_SCALE = 1.5     # frustum 크기 (m)

CAMERAS = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
           'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

CAM_COLORS = {
    'CAM_FRONT':       [1.0, 0.0, 0.0],   # 빨강
    'CAM_FRONT_RIGHT': [1.0, 0.5, 0.0],   # 주황
    'CAM_BACK_RIGHT':  [1.0, 1.0, 0.0],   # 노랑
    'CAM_BACK':        [0.0, 0.8, 0.0],   # 초록
    'CAM_BACK_LEFT':   [0.0, 0.5, 1.0],   # 하늘
    'CAM_FRONT_LEFT':  [0.5, 0.0, 1.0],   # 보라
}

# ── COLMAP binary reader ────────────────────────────────────────────────────
def read_next_bytes(f, num_bytes, fmt):
    data = f.read(num_bytes)
    return struct.unpack('<' + fmt, data)

def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-w*z),    2*(x*z+w*y)],
        [2*(x*y+w*z),    1-2*(x*x+z*z),  2*(y*z-w*x)],
        [2*(x*z-w*y),    2*(y*z+w*x),    1-2*(x*x+y*y)],
    ])

def read_images_txt(path):
    images = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            # pose 줄: 첫 토큰이 정수, 10번째 토큰이 이미지 이름(/ 포함)
            try:
                iid = int(parts[0])
            except ValueError:
                continue
            if len(parts) < 10 or '/' not in parts[9]:
                continue
            qw,qx,qy,qz = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
            tx,ty,tz     = float(parts[5]),float(parts[6]),float(parts[7])
            cid  = int(parts[8])
            name = parts[9]
            R = qvec2rotmat([qw,qx,qy,qz])
            w2c = np.eye(4); w2c[:3,:3]=R; w2c[:3,3]=[tx,ty,tz]
            c2w = np.linalg.inv(w2c)
            images[iid] = {'name': name, 'c2w': c2w, 'cam_id': cid}
    return images

def read_points3d_txt(path):
    pts = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            x,y,z = float(parts[1]),float(parts[2]),float(parts[3])
            err = float(parts[7])
            if err < 5.0:
                pts.append([x,y,z])
    return np.array(pts)

# ── GPS → ENU ───────────────────────────────────────────────────────────────
def latlon_to_enu(lat, lon, alt, lat0, lon0, alt0):
    R = 6378137.0
    east  = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * R
    up    = alt - alt0
    return np.array([east, north, up])

import csv as csv_mod

gps_rows = []
with open(GPS_CSV) as f:
    reader = csv_mod.DictReader(f)
    for row in reader:
        gps_rows.append((float(row['lat_deg']), float(row['lon_deg']), float(row['alt_m'])))

def interp_gps(t_sec):
    row_f = t_sec * GPS_HZ
    r0 = int(row_f); r1 = min(r0+1, len(gps_rows)-1)
    a = row_f - r0
    lat = gps_rows[r0][0]*(1-a) + gps_rows[r1][0]*a
    lon = gps_rows[r0][1]*(1-a) + gps_rows[r1][1]*a
    alt = gps_rows[r0][2]*(1-a) + gps_rows[r1][2]*a
    return lat, lon, alt

lat0, lon0, alt0 = interp_gps(GPS_START_SEC)

gps_enu = []
for i in range(N_FRAMES):
    t = GPS_START_SEC + i / CAM_FPS
    lat, lon, alt = interp_gps(t)
    gps_enu.append(latlon_to_enu(lat, lon, alt, lat0, lon0, alt0))
gps_enu = np.array(gps_enu)

# ── COLMAP BA 결과 읽기 ─────────────────────────────────────────────────────
print("Reading COLMAP BA results...")
ba_images = read_images_txt(os.path.join(BA_DIR, 'images.txt'))
pts3d = read_points3d_txt(os.path.join(BA_DIR, 'points3D.txt'))
print(f"  images: {len(ba_images)}, points: {len(pts3d)}")

# origin 정렬: CAM_FRONT/000001.jpg의 c2w를 origin으로
origin_c2w = None
for iid, img in ba_images.items():
    if img['name'] == 'CAM_FRONT/000001.jpg':
        origin_c2w = img['c2w']
        break
if origin_c2w is None:
    origin_c2w = list(ba_images.values())[0]['c2w']
inv_origin = np.linalg.inv(origin_c2w)

# ── Frustum 생성 ────────────────────────────────────────────────────────────
def make_frustum(c2w, color, scale=FRUSTUM_SCALE):
    fx = fy = 444.3
    cx, cy = W/2, H/2
    # 이미지 코너 → 카메라 좌표 (z=1)
    corners_cam = np.array([
        [(0-cx)/fx,   (0-cy)/fy,   1],
        [(W-cx)/fx,   (0-cy)/fy,   1],
        [(W-cx)/fx,   (H-cy)/fy,   1],
        [(0-cx)/fx,   (H-cy)/fy,   1],
    ]) * scale
    apex = np.zeros(3)
    # world 좌표로 변환
    def to_world(p):
        return (c2w[:3,:3] @ p) + c2w[:3,3]
    apex_w = to_world(apex)
    corners_w = [to_world(c) for c in corners_cam]

    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
    pts = [apex_w] + corners_w
    lset = o3d.geometry.LineSet()
    lset.points = o3d.utility.Vector3dVector(pts)
    lset.lines  = o3d.utility.Vector2iVector(lines)
    lset.paint_uniform_color(color)
    return lset

# ── Open3D 지오메트리 빌드 ──────────────────────────────────────────────────
geometries = []

# 1. COLMAP point cloud (회색) - 비활성화
# if len(pts3d) > 0:
#     pts_local = (inv_origin[:3,:3] @ pts3d.T).T + inv_origin[:3,3]
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts_local)
#     pcd.paint_uniform_color([0.6, 0.6, 0.6])
#     geometries.append(pcd)
#     print(f"Point cloud: {len(pts_local)} points")

# 2. Camera frustums (카메라별 색)
front_iid_pos = []
for iid, img in ba_images.items():
    c2w_local = inv_origin @ img['c2w']
    cam_name = img['name'].split('/')[0]
    color = CAM_COLORS.get(cam_name, [0.5,0.5,0.5])
    geom = make_frustum(c2w_local, color, scale=FRUSTUM_SCALE)
    geometries.append(geom)
    if cam_name == 'CAM_FRONT':
        front_iid_pos.append((iid, c2w_local[:3,3]))

# image_id 순서로 정렬 (시간 순서 유지)
front_iid_pos.sort(key=lambda x: x[0])
front_positions = [p for _, p in front_iid_pos]

# 3. CAM_FRONT trajectory (빨간선)
if len(front_positions) > 1:
    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(front_positions)
    traj.lines  = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(front_positions)-1)])
    traj.paint_uniform_color([1.0, 0.0, 0.0])
    geometries.append(traj)

# 4. GPS trajectory (파란선)
# COLMAP world = ENU (make_prior_curve.py가 ENU pos를 직접 c2w translation으로 사용)
# BA 후 좌표계도 동일. inv_origin으로 CAM_FRONT/000001 기준 로컬 좌표로 변환
gps_world = np.array(gps_enu, dtype=float)  # (N,3) ENU absolute (origin=frame0)
gps_local = (inv_origin[:3,:3] @ gps_world.T).T + inv_origin[:3,3]

gps_traj = o3d.geometry.LineSet()
gps_traj.points = o3d.utility.Vector3dVector(gps_local)
gps_traj.lines  = o3d.utility.Vector2iVector([[i,i+1] for i in range(len(gps_local)-1)])
gps_traj.paint_uniform_color([0.0, 0.0, 1.0])
geometries.append(gps_traj)

# 5. 좌표축
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
geometries.append(axis)

print("\nColor legend:")
print("  Red frustums:    CAM_FRONT")
print("  Orange:          CAM_FRONT_RIGHT")
print("  Yellow:          CAM_BACK_RIGHT")
print("  Green:           CAM_BACK")
print("  Cyan:            CAM_BACK_LEFT")
print("  Purple:          CAM_FRONT_LEFT")
print("  Red line:        CAM_FRONT trajectory (COLMAP BA)")
print("  Blue line:       GPS trajectory")
print("  Gray points:     COLMAP 3D point cloud")

o3d.visualization.draw_geometries(geometries,
    window_name='KIST Curve - COLMAP BA Poses',
    width=1400, height=900)
