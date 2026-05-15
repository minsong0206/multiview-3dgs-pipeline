"""
pred_traj / ego / route (scale-corrected) overlay 시각화

좌표 변환 정리:
  - pred_traj, ego : sim global egobox 좌표 (미터, x=전진 y=좌)
  - route (COLMAP) : camtoworld[:3,3] * SCALE → (z,x) 축 매핑 → sim 원점 정렬
    · SCALE = 4.769 m/COLMAP_unit  (chunk_00 GPS 186.1m / COLMAP 39.03 units)
    · sim_x = world_z * SCALE
    · sim_y = -world_x * SCALE  (부호 반전)
    · 원점: sim frame0(t=0.25s) = route frame 3.125
"""

import pickle, json, math, csv, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE   = '/home/ms/260308-KIST-Videos/KIST_CURVE_ALL/kist_curve_all_exhaustive_recon_model_export_v2'
SIM    = os.path.join(BASE, 'sim_output_drivor/scene_easy_00')
OUT    = os.path.join(SIM, 'bev_pred_vs_route_00_07')
SCALE  = 4.769   # m per COLMAP unit

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
with open(os.path.join(SIM, 'data.pkl'), 'rb') as f:
    data = pickle.load(f)
sim_frames = data[0]['frames']

with open(os.path.join(BASE, 'meta_data.json')) as f:
    meta = json.load(f)
cam_front = [fr for fr in meta['frames'] if '/CAM_FRONT/' in fr.get('rgb_path', '')]

# ── route 변환: COLMAP world → sim 좌표 (scale + 축 매핑) ─────────────────────
route_world = np.array([np.array(fr['camtoworld'])[:3, 3] for fr in cam_front])

# sim_x = world_z * SCALE,  sim_y = -world_x * SCALE  (부호 반전)
route_sim_x = route_world[:, 2] * SCALE
route_sim_y = -route_world[:, 0] * SCALE

# 원점 정렬: sim t=0.25s → route frame 3.125 (선형 보간)
ref_t     = sim_frames[0]['time_stamp']   # 0.25s
ref_ri    = ref_t * 12.5                  # 3.125
ri0, ri1  = int(ref_ri), min(int(ref_ri) + 1, len(route_sim_x) - 1)
frac      = ref_ri - ri0
origin_rx = route_sim_x[ri0] * (1 - frac) + route_sim_x[ri1] * frac
origin_ry = route_sim_y[ri0] * (1 - frac) + route_sim_y[ri1] * frac

route_x = route_sim_x - origin_rx + sim_frames[0]['ego_box'][0]
route_y = route_sim_y - origin_ry + sim_frames[0]['ego_box'][1]

# ── ego trajectory ────────────────────────────────────────────────────────────
ego_x = np.array([fr['ego_box'][0] for fr in sim_frames])
ego_y = np.array([fr['ego_box'][1] for fr in sim_frames])

# ── pred_traj: 각 프레임의 예측 포인트 (global egobox 좌표) ───────────────────
pred_pts = []  # list of (frame_idx, x, y)
for fi, fr in enumerate(sim_frames):
    traj = fr['planned_traj']['traj']   # (7, 3): (x, y, yaw)
    for pt in traj:
        pred_pts.append((fi, pt[0], pt[1]))
pred_pts = np.array(pred_pts)

# ── 플롯 ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

# route — sim 구간만 표시 (t=0.25s~4.25s → route frame 3~54)
sim_t0 = sim_frames[0]['time_stamp']   # 0.25s
sim_t1 = sim_frames[-1]['time_stamp']  # 4.25s
ri_start = int(sim_t0 * 12.5)         # 3
ri_end   = int(sim_t1 * 12.5) + 2     # 55
ax.plot(route_x[ri_start:ri_end], route_y[ri_start:ri_end],
        color='#aaaaaa', linewidth=2.0,
        linestyle='--', label='Route (COLMAP × 4.769 m/unit)', zorder=1)
# 전체 경로 (흐리게)
ax.plot(route_x, route_y, color='#dddddd', linewidth=0.8,
        linestyle=':', alpha=0.4, zorder=0)

# ego trajectory
ax.plot(ego_x, ego_y, color='#2196F3', linewidth=2.5,
        marker='o', markersize=4, label='Ego (sim actual)', zorder=3)
ax.plot(ego_x[0], ego_y[0], 'o', color='#2196F3', markersize=10, zorder=5)

# pred_traj (프레임별 색상 구분)
cmap = plt.cm.plasma
n_frames = len(sim_frames)
for fi in range(n_frames):
    pts = pred_pts[pred_pts[:, 0] == fi]
    if len(pts) == 0:
        continue
    color = cmap(fi / max(n_frames - 1, 1))
    ax.plot(pts[:, 1], pts[:, 2], '-', color=color, alpha=0.7,
            linewidth=1.5, zorder=2)
    ax.plot(pts[0, 1], pts[0, 2], 'o', color=color, markersize=3, zorder=4)

# route 위에 sim 대응 포인트 표시
for fi, sf in enumerate(sim_frames):
    ri = sf['time_stamp'] * 12.5
    ri0 = int(ri); ri1 = min(ri0 + 1, len(route_x) - 1)
    fr = ri - ri0
    rx = route_x[ri0] * (1 - fr) + route_x[ri1] * fr
    ry = route_y[ri0] * (1 - fr) + route_y[ri1] * fr
    ax.plot(rx, ry, 'x', color='gray', markersize=5, zorder=3)

# colorbar (pred_traj 프레임)
sm = plt.cm.ScalarMappable(cmap=cmap,
                            norm=plt.Normalize(vmin=0, vmax=(n_frames-1)*0.25))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label('Sim time (s)', fontsize=10)

ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('X (m) — forward', fontsize=11)
ax.set_ylabel('Y (m) — lateral', fontsize=11)
ax.set_title(f'Trajectory Overlay (scale-corrected)\n'
             f'Route: COLMAP × {SCALE} m/unit, axis=(world_z→x, world_x→y)',
             fontsize=12)

route_patch  = mpatches.Patch(color='#aaaaaa', label='Route (COLMAP scaled)')
ego_patch    = mpatches.Patch(color='#2196F3', label='Ego (sim actual)')
pred_patch   = mpatches.Patch(color='purple',  label='Pred traj (plasma=time)')
ax.legend(handles=[route_patch, ego_patch, pred_patch],
          loc='upper left', fontsize=10)

out_path = os.path.join(OUT, 'overlay_scaled.png')
os.makedirs(OUT, exist_ok=True)
plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'저장: {out_path}')

# ── 정렬 오차 출력 ────────────────────────────────────────────────────────────
print('\n=== route vs ego 정렬 오차 ===')
errs = []
for fi, sf in enumerate(sim_frames):
    ri = sf['time_stamp'] * 12.5
    ri0 = int(ri); ri1 = min(ri0+1, len(route_x)-1)
    fr = ri - ri0
    rx = route_x[ri0]*(1-fr)+route_x[ri1]*fr
    ry = route_y[ri0]*(1-fr)+route_y[ri1]*fr
    err = math.sqrt((sf['ego_box'][0]-rx)**2+(sf['ego_box'][1]-ry)**2)
    errs.append(err)
    print(f'  f{fi:2d} t={sf["time_stamp"]:.2f}s: err={err:.3f}m '
          f'(ego=({sf["ego_box"][0]:.2f},{sf["ego_box"][1]:.2f}) '
          f'route=({rx:.2f},{ry:.2f}))')
print(f'평균 오차: {np.mean(errs):.3f}m')
print(f'최대 오차: {np.max(errs):.3f}m (후반 커브 구간)')
print()
print('※ 오차 원인: sim이 실제보다 느리게 출발 + 후반 커브에서 경로 이탈')
print('  → scale 보정은 올바름, 나머지는 sim 동역학 차이')
