# score_frame_collision_debug.ply — 생성 정보

## 파일 위치

```
/home/ms/kist_curve_coord_debug/score_frame_collision_debug.ply
```

생성 스크립트:
```
/home/ms/kist_curve_coord_debug/gen_collision_debug_ply_v2.py
```

---

## 데이터 소스

| 파일 | 경로 | 용도 |
|------|------|------|
| `data.pkl` | `.../output_drivor_height=cam_height/scene_easy_00/data.pkl` | ego_box, planned_traj (per frame) |
| `ground.ply` | `.../output_drivor_height=cam_height/scene_easy_00/ground.ply` | ground point cloud (camera coords) |
| `scene.ply`  | `.../output_drivor_height=cam_height/scene_easy_00/scene.ply`  | scene background point cloud (camera coords) |
| `eval.json`  | `.../output_drivor_height=cam_height/scene_easy_00/eval.json`  | per-frame nc/ttc/dac scores |

베이스 경로:
```
/media/ms/WD_BLACK_4TB/KIST/raw_data_curve/output_drivor_height=cam_height/scene_easy_00/
```

> **이 PLY는 `height=cam_height` 수정 버전 기준입니다.**  
> `hug_sim.py`의 `ground_height()`에 `+ cam_height`를 적용한 DrivoR 실행 결과입니다.

---

## 좌표계

Score/IMU frame (score_calculator.py와 동일):

```
score_x = cam_z    (전방, forward)
score_y = -cam_x   (좌우, lateral)
score_z = -cam_y   (위쪽, up / physical up)
```

변환 공식:
```python
score_pts = np.stack([cam_pts[:,2], -cam_pts[:,0], -cam_pts[:,1]], axis=1)
```

---

## 포인트 색상

| 색상 | RGB | 의미 |
|------|-----|------|
| 회색 (gray) | (128, 128, 128) | scene background 포인트 (랜덤 샘플 30,000개) |
| 연두 (light green) | (100, 180, 100) | ground 포인트 (랜덤 샘플 5,000개) |
| 파랑 (blue) | (0, 80, 255) | 현재 ego box 엣지 (wireframe) |
| 청록 (cyan) | (0, 200, 200) | 미래 planned trajectory box 엣지 (frozen z) |
| 빨강 (red) | (220, 0, 0) | ego box 내부의 scene collision 포인트 |
| 초록 (green) | (0, 200, 0) | ego box 내부의 ground 포인트 |

---

## ego_box 구성

`data.pkl`의 각 frame에서:
```python
ego_box = (x, y, z, w, l, h, yaw)
# x, y, z : score frame 위치 (x=전방, y=좌우, z=높이)
# w=1.6m, l=3.0m, h=1.5m (차량 크기)
# yaw : score frame 기준 회전각 (z축 회전)
```

Box bottom = z, Box top = z + h.  
Canonical 8 corners × (l, w, h) 스케일링 후 R(yaw) 회전 → world 이동.

---

## planned_traj (future boxes)

각 frame의 planned_traj는 (M, 3) = (x, y, yaw) in score frame, timestep=0.5s.  
**현재 PLY에서 future box의 z는 ego_box z를 그대로 사용 (frozen z 버그 상태 그대로 시각화).**  
→ 경사로 후반부에서 future box가 실제 도로보다 낮게 위치하는 것을 확인할 수 있음.

---

## 포인트 수 (생성 결과)

- 전체 포인트: 80,961개
- 프레임 수: 38개 (key frames)
- scene bg 샘플: 30,000개
- ground 샘플: 5,000개

---

## 핵심 관찰 (per-frame 요약)

| 구간 | t (sec) | NC | TTC | DAC | scene_coll | 비고 |
|------|---------|----|----|-----|------------|------|
| 초반 정상 | 0.25~0.75 | 1.0 | 1.0 | 1.0 | 0 | 정상 주행 |
| NC/TTC 산발 실패 | 1.0~3.5 | 0/1 교차 | 0/1 교차 | 1.0 | 0 | scene_coll=0인데 NC=0 → **future box frozen-z 버그** |
| DAC 실패 시작 | 4.25~ | 1.0 | 1.0 | 0.0 | 0 | DAC 실패 (도로 이탈) |
| 후반 NC/TTC+DAC 실패 | 5.75~ | 0.0 | 0.0 | 0/1 | 0~905 | scene_coll 급증, DrivoR 궤적 이탈 |

### 중요 발견

1. **t=1.0~3.5 구간**: `scene_coll=0` (ego box 안에 충돌 포인트 없음)인데도 NC=0.  
   → 충돌이 **현재 ego box**가 아니라 **future planned trajectory box** (frozen z) 에서 발생하는 것.  
   → Fix 2 (future waypoint z 동적 계산)가 필요한 근거.

2. **t=4.25~ DAC=0**: DrivoR 차량이 ground.ply 커버리지 외부로 벗어남 (도로 이탈).  
   → DAC 계산에서 ground 포인트가 ego 내부 4분면에 미달 (30% 미만).

3. **t=8.75~ scene_coll 급증**: scene.ply의 구조물/식생 포인트가 ego box와 겹침.  
   → 후반 경사로 구간에서 ego z는 올바르지만 주변 scene 포인트 밀도가 높아 실제 충돌 가능성.

---

## 이전 PLY와의 차이

| 항목 | 이전 PLY (height=0 버그) | 현재 PLY (height=cam_height 수정) |
|------|--------------------------|-----------------------------------|
| ego_box z | +0.033 ~ +2.444 (도로 위 1.4m 떠있음) | +0.033 ~ +5.038 (도로 표면 위) |
| scene_coll 초반 | 1~594개 (false positive) | 0개 (초반 정상) |
| NC 초반 | 모두 0.0 | 0.25~0.75 모두 1.0 |
| HD score | 0.045 | 0.113 (+151%) |

---

## 생성 환경

```bash
conda run -n drivoR python3 /home/ms/kist_curve_coord_debug/gen_collision_debug_ply_v2.py
```

오픈소스 라이브러리:
- `open3d==0.19.0`
- `numpy`, `scipy`, `pickle`, `json`
