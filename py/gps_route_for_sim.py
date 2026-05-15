"""
GPS CSV → sim 구간 metric trajectory 추출
make_prior_curve.py의 보간 로직 재활용

출력:
  - gps_route_sim.csv : sim 각 프레임의 GPS 기반 ENU 좌표 (미터)
  - gps_route_full.csv: 전체 GPS 경로 (시각화용)
"""

import csv
import math
import numpy as np
import pickle
import json
import os

# ── Config ────────────────────────────────────────────────────────────────────
GPS_CSV       = '/home/ms/260308-KIST-Videos/RAW_DATA/6_GPS/2_Entrance-L1.csv'
SIM_DATA_PKL  = '/home/ms/260308-KIST-Videos/KIST_CURVE_ALL/kist_curve_all_exhaustive_recon_model_export_v2/sim_output_drivor/scene_easy_00/data.pkl'
OUT_DIR       = '/home/ms/260308-KIST-Videos/KIST_CURVE_ALL/kist_curve_all_exhaustive_recon_model_export_v2/sim_output_drivor/scene_easy_00/bev_pred_vs_route_00_07'
CAM_START_SEC = 120.0   # kist_curve 카메라 시작 시각 (GPS 시간 기준)
GPS_ROW_RATE  = 27.0    # CSV 행/초

# ── GPS 읽기 ──────────────────────────────────────────────────────────────────
gps_rows = []
with open(GPS_CSV) as f:
    for row in csv.DictReader(f):
        gps_rows.append((float(row['lat_deg']), float(row['lon_deg']), float(row['alt_m'])))

lat0, lon0, alt0 = gps_rows[0]
R_earth = 6378137.0

def latlon_to_enu(lat, lon, alt):
    east  = math.radians(lon - lon0) * R_earth * math.cos(math.radians(lat0))
    north = math.radians(lat - lat0) * R_earth
    up    = alt - alt0
    return np.array([east, north, up])

all_pos_raw = np.array([latlon_to_enu(*r) for r in gps_rows])

# 중복 행 제거 → 실제 GPS 포인트만 추출
diffs_raw = np.linalg.norm(np.diff(all_pos_raw, axis=0), axis=1)
change_rows = np.concatenate([[0], np.where(diffs_raw > 0.001)[0] + 1])
gps_times_sec = change_rows / GPS_ROW_RATE
gps_pos_enu   = all_pos_raw[change_rows]

print(f"GPS 실제 포인트: {len(gps_times_sec)}개")
print(f"GPS 시간 범위: {gps_times_sec[0]:.2f}s ~ {gps_times_sec[-1]:.2f}s")

# ── sim 타임스탬프 로드 ────────────────────────────────────────────────────────
with open(SIM_DATA_PKL, 'rb') as f:
    data = pickle.load(f)
sim_frames = data[0]['frames']
sim_times_local = np.array([fr['time_stamp'] for fr in sim_frames])  # 0.25 ~ 4.25s

# sim 내부 시간 → GPS 절대 시간
sim_times_gps = sim_times_local + CAM_START_SEC

print(f"\nsim GPS 시간 범위: {sim_times_gps[0]:.2f}s ~ {sim_times_gps[-1]:.2f}s")
print(f"커버되는 GPS 포인트 수: {((gps_times_sec >= sim_times_gps[0]) & (gps_times_sec <= sim_times_gps[-1])).sum()}")

# ── sim 각 프레임의 GPS ENU 보간 ──────────────────────────────────────────────
sim_enu = np.column_stack([
    np.interp(sim_times_gps, gps_times_sec, gps_pos_enu[:, 0]),
    np.interp(sim_times_gps, gps_times_sec, gps_pos_enu[:, 1]),
    np.interp(sim_times_gps, gps_times_sec, gps_pos_enu[:, 2]),
])

# sim frame0 위치를 원점으로 정규화
origin_enu = sim_enu[0].copy()
sim_enu_rel = sim_enu - origin_enu

print(f"\nGPS route (sim 원점 기준):")
print(f"  east  범위: {sim_enu_rel[:,0].min():.3f} ~ {sim_enu_rel[:,0].max():.3f} m")
print(f"  north 범위: {sim_enu_rel[:,1].min():.3f} ~ {sim_enu_rel[:,1].max():.3f} m")
print(f"  총 이동거리: {np.linalg.norm(sim_enu_rel[-1]):.3f} m")

# ── sim ego_box와 비교 ────────────────────────────────────────────────────────
print(f"\n=== sim ego_box vs GPS (frame0 기준 상대거리) ===")
ego_x0 = sim_frames[0]['ego_box'][0]
ego_y0 = sim_frames[0]['ego_box'][1]
for i, fr in enumerate(sim_frames):
    eb = fr['ego_box']
    dx_sim = eb[0] - ego_x0
    dy_sim = eb[1] - ego_y0
    dist_sim = math.sqrt(dx_sim**2 + dy_sim**2)
    dist_gps = math.sqrt(sim_enu_rel[i,0]**2 + sim_enu_rel[i,1]**2)
    print(f"  frame{i:2d} t={fr['time_stamp']:.2f}s: sim_dist={dist_sim:.3f}m  gps_dist={dist_gps:.3f}m")

# ── CSV 저장 ──────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

out_sim = os.path.join(OUT_DIR, 'gps_route_sim.csv')
with open(out_sim, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'sim_time', 'gps_time', 'east_m', 'north_m', 'up_m',
                     'east_rel_m', 'north_rel_m'])
    for i, fr in enumerate(sim_frames):
        writer.writerow([
            i, fr['time_stamp'], sim_times_gps[i],
            sim_enu[i,0], sim_enu[i,1], sim_enu[i,2],
            sim_enu_rel[i,0], sim_enu_rel[i,1]
        ])
print(f"\n저장: {out_sim}")

# 전체 GPS 경로 (시각화용, CAM_START 전후 ±30s)
t_center = CAM_START_SEC + 2.0
mask = (gps_times_sec >= t_center - 30) & (gps_times_sec <= t_center + 30)
out_full = os.path.join(OUT_DIR, 'gps_route_full.csv')
with open(out_full, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['gps_time', 'east_m', 'north_m', 'east_rel_m', 'north_rel_m'])
    for t, pos in zip(gps_times_sec[mask], gps_pos_enu[mask]):
        writer.writerow([t, pos[0], pos[1], pos[0]-origin_enu[0], pos[1]-origin_enu[1]])
print(f"저장: {out_full}")
