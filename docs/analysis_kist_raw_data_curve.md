# KIST raw_data_curve 분석 노트
## HUGSIM–DrivoR 디버깅 메인 기록
> 대상 씬: `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve`  
> 최종 업데이트: 2026-05-14

---

## 1. 좌표계 정리

**HUGSIM "normalized world" 좌표계** (meta_data.json camtoworld 기준)

```
world +X ≈ image right   (camera local +X)
world +Y ≈ image down    (camera local +Y, physical DOWN)
world +Z ≈ camera forward
```

근거 (CAM_FRONT/000136 c2w axes):
- +X(right)  : [ 0.9722, -0.0320,  0.2318]
- +Y(down)   : [ 0.0366,  0.9992, -0.0158]  ← world +Y 방향과 dot=0.999
- +Z(forward): [-0.2311,  0.0238,  0.9726]

따라서 **physical height = -world_Y**  
camera Y = -4.079 → height = 4.079m / road surface Y ≈ -2.58 → camera 1.5m above road ✓

**score/IMU frame** (`score_calculator.py` line 604):
```
score_x = world_z   (전방, forward)
score_y = -world_x  (좌우, lateral)
score_z = -world_y  (높이, up)
```

**ego_box 구성** (`hug_sim.py` line 204):
```python
ego_box = [vt[2], -vt[0], -vt[1], w, l, h, -yaw]
#           x       y       z
# vt[2] = world_z (forward)
# vt[1] = ground_height() (world_Y, physical down)
# -vt[1] = score_z (height, physical up)
```

**vt 정의** (`hug_sim.py` line 188-192):
```python
vt[0] = vab[0]                          # world X
vt[1] = ground_height(vab[0], vab[1])   # world Y, 지면 높이
vt[2] = vab[1]                          # world Z, forward
```

**0_aligner COLMAP 좌표계** (별도, gravity-aligned):
- Z-up (camera physical up · Z = 0.9961)
- 실측 camera-to-road height ≈ 1.11m (3D points 기준)
- config cam_height = 1.5m → ~0.39m 과대 추정 가능성 있음
- ⚠️ COLMAP Z-up과 HUGSIM world Z는 같은 축이 아님 (inv_pose 정규화로 다른 좌표계)

---

## 2. merge_depth_ground.py 분석

파일: `/home/ms/HUGSIM_N/HUGSIM/data/utils/merge_depth_ground.py`

핵심 코드 (line 151):
```python
points_local[:, 1] = front_cam_height   # 1.5 for KIST
```

검증 (frame 000136):
- camera world pos : [-3.011, -4.079, 56.863]
- local [0,1.5,0]  → world : [-2.956, -2.581, 56.839]
- world Y 차이 = +1.499m ← 1.5m BELOW camera ✓

**결론**: merge_depth_ground.py는 올바르게 동작함. world +Y = physical down이므로 `points_local[:,1]=1.5`는 카메라 아래 1.5m 도로 표면에 배치하는 것이 맞음.

---

## 3. Ground Model Gaussian 분포

체크포인트: `model/ckpts/ground_chkpnt30000.pth`  
Total ground Gaussians: 419,108

| 클래스 | 개수 | 비율 |
|--------|------|------|
| road (0) | 294,717 | 70.3% |
| sidewalk (1) | 124,391 | 29.7% |

Road Gaussian Y 분포:
- peak: Y ∈ [-2.64, -2.32] → road surface ≈ Y = -2.48m
- camera Y (frame 136) = -4.079m
- camera-to-road: 1.6m ≈ cam_height 1.5m ✓

Scale: mean=0.13m, p95=0.24m, max=41.05m (scale>5m: 34개, 0.01%)

---

## 4. 3DGS Semantic 렌더링 분석 (CAM_FRONT_000136)

비교: `export/test/ours_30000/semantic/` vs `recon_HUGSIM/semantics/CAM_FRONT/`

Mid-third (image row 150~300):
- GT road: 60.2% / 3DGS road: 85.1% → +24.9% 과잉

False road 픽셀 (render=road, GT≠road): 34,760px = 9.7%
- vegetation (8): 65.6% ← 주원인
- building (2): 13.9%
- wall (3): 8.6%

**원인**: camera(Y=-4.08)가 road surface(Y=-2.48)보다 1.5m 위에서 낮은 각도로 전방을 바라봄. 도로면에 납작한 flat Gaussian(pancake 형태)이 grazing angle에서 image-space로 옆으로 넓게 splatting → 도로 양쪽 vegetation/wall을 road 색으로 덮음.

RGB 렌더링은 이 프레임에서 치명적 blocker로 보이지 않지만, ground over-splatting은 semantic을 확실히 오염시킨다. DrivoR에는 RGB/semantic 입력 품질 악화로 간접 영향 가능성이 있고, NC/TTC 직접 원인은 별도 scoring/collision 로직 쪽 증거가 더 강함.

---

## 5. ground_height() 버그 분석

### 5.1 버그 내용

파일: `/home/ms/HUGSIM_N/HUGSIM/sim/hugsim_env/envs/hug_sim.py`

```python
def ground_height(self, u, v):
    cam_poses, cam_height, _ = self.ground_model
    ...
    uhv_local[1] = 0         # camera 중심 평면에 투영
    uhv_world = nearest_c2w[:3, :3] @ uhv_local + nearest_c2w[:3, 3]
    return uhv_world[1]      # ← 버그: camera 중심 평면의 world_Y 반환
                             #   도로 표면은 여기서 +cam_height 더 아래
```

`cam_poses`는 **카메라 중심 pose** → local Y=0 평면 = 카메라 중심 평면 (도로면이 아님).  
도로 표면은 카메라에서 cam_height=1.5m 아래(world +Y 방향).  
`plan.py`의 동일 함수는 `return uv_world[1] + cam_height`로 올바르게 구현됨.

### 5.2 ego_box z 오차 수치 검증

| 시점 | hug_sim 반환값 | 실제 도로 score_z | 오차 |
|------|--------------|-----------------|------|
| t=0.25 | ego_box_z ≈ +0.033 | ≈ −1.308 | **+1.35m 공중 부유** |
| t=4.25 | ego_box_z ≈ +2.403 | ≈ +0.9~1.0 | **+1.4m 공중 부유** |

오차 = 항상 ≈ cam_height = 1.5m → 원인 확정

### 5.3 수정 시도 및 실패 원인

**시도한 Fix 1**: `ground_height()` return 값에 `+ cam_height` 추가  
**결과**: 렌더링 이미지가 완전히 망가짐 (회색 blank 또는 지면 아래 뷰)

**실패 원인**: `ground_height()`의 리턴값이 두 곳에 동시에 사용됨:
```
ground_height() 리턴값
    ├── vt[1] → 렌더링 카메라 pose 계산  ← +cam_height 추가 시 카메라가 땅 속으로
    └── vt[1] → ego_box[2] = -vt[1]     ← 여기에만 보정이 필요
```

### 5.4 올바른 수정 방향

`ground_height()`는 건드리지 말고, **`ego_box` property에서만** score_z 계산 시 cam_height를 적용:

```python
# hug_sim.py line 202-204
@property
def ego_box(self):
    _, cam_height, _ = self.ground_model
    return [self.vt[2], -self.vt[0], -self.vt[1] - cam_height,
            self.whl[0], self.whl[2], self.whl[1], -self.vr[1]]
    # -vt[1] - cam_height : 도로 표면 기준 score_z
    # 렌더링에 쓰이는 vt[1]은 그대로 유지
```

**현재 상태**: 아직 미적용 (되돌림). 위 수정 후 test 필요.

---

## 6. score_calculator.py frozen-z 버그 (Fix 2)

### 6.1 버그 내용

파일: `/home/ms/HUGSIM_N/HUGSIM/sim/utils/score_calculator.py`

`_calculate_no_collision()` (line 373~405):
```python
ego_x, ego_y, z, ego_w, ego_l, ego_h, ego_yaw = ego_box  # z 한 번만 추출
for idx in range(planned_traj.shape[0]):
    ego_x, ego_y, ego_yaw = planned_traj[idx]  # x,y,yaw만 갱신
    ego_trans_mat[:3, 3] = np.array([ego_x, ego_y, z])  # z 고정!
```

`planned_traj`는 `traj_transform_to_global()`에서 `(x, y, yaw)`만 저장 — **z 없음**.  
→ 경사로에서 미래 waypoint의 실제 지면 높이와 무관하게 현재 프레임 z가 고정 사용됨.

### 6.2 Fix 2 적용 내용

현재 코드에 적용됨 (`score_calculator.py`):

```python
# _calculate_no_collision에 ground_xyz_score 파라미터 추가
def _calculate_no_collision(self, ego_box, planned_traj, obs_lists, scene_xyz, ground_xyz_score=None):
    ...
    for idx in range(planned_traj.shape[0]):
        ego_x, ego_y, ego_yaw = planned_traj[idx]
        if ground_xyz_score is not None and len(ground_xyz_score) > 0:
            dists = np.sum((ground_xyz_score[:, :2] - np.array([ego_x, ego_y])) ** 2, axis=1)
            z = ground_xyz_score[np.argmin(dists), 2]  # nearest-neighbor 보간
        ego_trans_mat[:3, 3] = np.array([ego_x, ego_y, z])
```

`parse_data()` / `hugsim_evaluate()`에서 `ground_xyz_score` (3D score frame) 생성 및 저장.

### 6.3 Fix 2 단독 적용 결과 (test2)

test2 = seed=42 + Fix1(cam_height, 렌더 망가진 버전) + Fix2  
→ Fix1이 렌더링을 망가뜨렸으므로 Fix2 단독 효과 측정 불가.

Fix2의 문제점:
- 후반부(t=2.5~6.75) 일부 구간에서 새로운 false collision 발생
- nearest-neighbor가 경사로에서 부정확한 z를 반환할 수 있음
- 향후 KD-tree + 반경 내 평균 z 사용으로 개선 필요

---

## 7. DrivoR 평가 실험 결과 비교

| 실험 | 설명 | NC | TTC | DAC | PDMS | RC | HD score |
|------|------|----|----|-----|------|----|----------|
| baseline | 수정 없음, seed 없음 | 0.278 | 0.222 | 1.000 | 0.238 | 0.191 | 0.045 |
| height=cam_height (이전) | ground_height+cam_height, seed 없음 | 0.526 | 0.421 | 0.579 | 0.312 | 0.363 | 0.113 |
| **test0** | seed=42만 적용 | **0.757** | **0.595** | 0.892 | **0.560** | 0.330 | **0.185** |
| test2 | seed + Fix1(렌더 망가짐) + Fix2 | 0.605 | 0.474 | 0.947 | 0.477 | 0.337 | 0.161 |

**핵심 발견**: seed=42 고정만으로 HD score 0.045 → 0.185 (약 4배 향상). Fix1이 렌더링을 망가뜨린 영향으로 test2는 test0보다 낮음.

### 실험 데이터 경로

| 실험 | 경로 |
|------|------|
| baseline | `output_drivor_height=0/scene_easy_00/` |
| height=cam_height | `output_drivor_height=cam_height/scene_easy_00/` |
| test0 (seed) | `outputdrivor_test0_seed/scene_easy_00/` |
| test1 (fix1 단독) | `outputdrivor_fix1_test1/scene_easy_00/` (crash, eval.json 없음) |
| test2 (fix1+fix2) | `outputdrivor_fix2_test2/scene_easy_00/` |

---

## 8. semantic collision 분석 (3DGS checkpoint 기준)

체크포인트: `model/ckpts/chkpnt30000.pth`

**scene.ply 전체 semantic 분포** (semantic>1 & !=10):
| 클래스 | 개수 | 비율 |
|--------|------|------|
| vegetation (8) | 328,926 | 52.1% |
| building (2) | 152,744 | 24.2% |
| terrain (9) | 60,886 | 9.6% |
| fence (4) | 34,184 | 5.4% |
| pole (5) | 28,515 | 4.5% |

**per-frame ego box collision 분석** (height=cam_height 기준):

| 구간 | coll_in_box | 주요 semantic | NC 상태 |
|------|------------|--------------|---------|
| t=0.25~0.75 | 0 | — | 1.0 (정상) |
| t=1.0~7.75 | **0** | — | **0.0 (false fail)** |
| t=8.25~9.5 | 6~905 | vegetation(52~67%), traffic_sign(33~48%) | 0.0 |

**중요**: t=1.0~7.75 구간은 현재 ego box 내 충돌 포인트가 없음에도 NC=0.  
→ 충돌이 **future planned trajectory box (frozen z)**에서 발생하는 것 확인.  
→ semantic 렌더링 오류가 NC에 직접 영향을 주지 않음.  
→ Fix2(future z 보간)가 필요한 근거.

---

## 9. 학습 결과

**Scene (배경 3DGS) — 60k iteration:**

| iter | PSNR (test) | SSIM | LPIPS |
|------|-------------|------|-------|
| 30k | 25.17 | 0.743 | 0.261 |
| 45k | 25.29 | 0.746 | 0.258 |
| 60k | 25.40 | 0.749 | 0.255 |

→ 개선의 92%는 30k 이전 달성. 45k→60k 거의 수렴.

**Ground model — 30k iteration:**

| iter | PSNR (test) | SSIM | LPIPS |
|------|-------------|------|-------|
| 7k | 25.55 | 0.866 | 0.165 |
| 15k | 27.30 | 0.878 | 0.147 |
| 30k | 29.48 | 0.889 | 0.128 |

→ Ground PSNR가 Scene보다 높은 이유: ground/save_test 이미지는 도로 픽셀만 평가 (black background).

---

## 10. seed 고정

파일: `/home/ms/HUGSIM_N/HUGSIM/closed_loop.py`

```python
SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# __main__ 진입 직후 호출
set_seed(SEED)
```

seed가 영향을 주는 코드:
- `agent_controller.py` line 287: `torch.randint(0, self.best_k, (1,))` (best-k 중 random select)
- `plan.py` line 232: `np.random.randint(len(next_lanes))` (lane random select)

---

## 11. COLMAP → HUGSIM inv_pose 정규화

```
COLMAP images.bin → c2w_colmap
inv_pose = inv(first CAM_FRONT c2w_colmap)
c2w_hugsim = inv_pose @ c2w_colmap
```

- COLMAP 0_aligner: Z-up
- HUGSIM meta_data: 첫 CAM_FRONT 기준 정규화 → +Y=down, +Z=forward
- 두 좌표계를 직접 비교 불가. 포인트 변환: `p_hugsim = inv_pose[:3,:3] @ p_colmap + inv_pose[:3,3]`

---

## 12. Ground Plane 상승 문제 (frame 161 / 461) 진단

### 12.1 증상

`compare_semantic/CAM_FRONT_000161.png`, `000461.png`: Ground plane이 화면 중간 이상을 차지.  
`model/ground/save_test/CAM_FRONT_000161.png`: 도로면이 수평선 근처까지 올라온 grazing-angle 렌더링 확인.

### 12.2 ground_points3d.ply Y std=1.19 원인 재해석

초기 해석은 “KIST 부지 실제 경사 지형”이었으나, 실제 영상에는 큰 경사 하강 구간이 없고 방지턱 정도만 존재한다는 현장 정보가 있음. 따라서 아래 camera Y 변화는 실제 지형 고도 변화보다 **COLMAP/HUGSIM pose의 vertical drift 또는 첫 CAM_FRONT 기준 정규화 artifact**로 보는 것이 더 타당하다.

```
Frame 0:   camera Y =  0.000 m  (origin)
Frame 100: camera Y = -3.231 m
Frame 200: camera Y = -4.447 m  (최저점)
Frame 300: camera Y = -3.743 m
Frame 461: camera Y = -1.063 m  (돌아옴)
```

총 Y 변화: **4.6 m**. frame-to-frame 최대 변화량 0.07 m로 부드럽게 누적되기 때문에 “실제 경사처럼 보이는” ground surface가 생성되지만, 영상 조건과 맞지 않으면 이는 정상 지형이 아니라 pose drift로 취급해야 한다.

검증: camera-local Y (Y-snap 후) → mean=1.496, std=0.015 (≈ cam_height=1.5 ✓)

즉 `merge_depth_ground.py`의 local snap 자체는 의도대로 동작하지만, 입력 camera pose의 Y가 잘못 변하면 world-space ground도 함께 잘못 기울어진다.

### 12.3 학습 후 Gaussian Y drift

| 구분 | N | Y mean | Y std |
|------|---|--------|-------|
| 초기 ground_points3d.ply | 199,584 | -1.663 | 1.189 |
| 30k 학습 후 Gaussians | 419,108 | -1.940 | 1.300 |

densification으로 2.1× 증가. Y std +0.11 증가 → **소폭 drift는 있으나 폭발적 drift는 없음**.

large-scale Gaussians (scale > 1m): 540개 (0.1%) — 극소수 이상치만 존재.

### 12.4 frame 161 / 461 상승 원인 (수치 검증)

**frame 161** (camera Y=-4.53, 최저점 근처, 경사 내려간 지점):
- expected road Y = -4.53 + 1.5 = **-3.03**
- 인근 30m Gaussian Y median = **-2.67** (expected보다 0.36m 높음 = 지면 위에 뜸)
- 인근 Gaussian 중 expected road 0.5m 이내: **44.4%**
- 카메라 앞+위쪽(camera-Y<0)에 있는 Gaussian: **18,549개 (4.4%)**, depth 10~30m
- 이 Gaussian들의 scale_x mean=0.12m — 비정상적으로 크지 않음

**원인 해석**: `GroundModel._rotation`이 항상 identity `[1,0,0,0]`로 고정된 점이 실제로 영향을 줌.  
Gaussian disk가 지형 법선이 아닌 **전역 XZ평면**에 평행하므로, pose vertical drift로 만들어진 가짜 경사 ground를 카메라가 낮은 각도로 볼 때 disk들이 grazing angle로 image-space에서 넓게 펼쳐짐. 다만 scale 폭발은 아님. 중심 위치와 scale은 대체로 정상 범위이고, pose drift + 고정 rotation + 높은 opacity가 합쳐져 road layer가 강하게 보이는 케이스.

**frame 461** (camera Y=-1.06, 경사 다시 올라온 지점):
- expected road Y = -1.06 + 1.5 = **+0.44**
- 인근 Gaussian Y median = **-1.24** (expected보다 1.68m 낮음)
- expected road 0.5m 이내: **8.9%** — 이 지점의 Gaussian들이 실제 경사와 크게 어긋남
- 카메라 앞+위쪽 Gaussian: **43개 (0.003%)** → 상승 효과 적음
- → 상승 원인이 frame 161과 다름: 이 구간은 Gaussian이 부족하거나 잘못 배치됨

### 12.5 distort_3d_loss lambda 미적용 확인

`train_ground.py` (HUGSIM_N 버전):
```python
loss += distort_3d_loss   # lambda 없이 weight=1.0으로 그냥 더함
```

config에 `lambda_dist: 1.0`이 정의되어 있으나 **실제 코드에서 `cfg.lambda_dist`를 곱하는 라인 없음** → 사실상 weight=1이 하드코딩. 별도 문제를 일으키진 않지만 조정 불가.

### 12.6 진단 평가 (사용자 가설 검증)

| 가설 | 검증 결과 |
|------|-----------|
| "sloped terrain + fixed rotation → grazing angle blocker" | ⚠️ 현상은 맞지만 실제 지형 경사가 아니라 pose drift로 생긴 가짜 경사일 가능성이 높음. frame 161에서 4.4%가 카메라 앞+위 10-30m에 존재, scale은 정상 |
| "xyz position drift during training" | ⚠️ 소폭 drift (Y std +0.11) 있으나 주 원인은 아님. 대부분 Gaussian 정상 위치 |
| "distort_3d_loss over-constraining sloped terrain" | ⚠️ 부분적으로 타당. bias sampling range (-2~+2m in cam-Z)가 pose drift로 생긴 가짜 경사에서는 교란 가능. lambda는 1.0 고정 |
| "frame 461: Gaussian 부재 또는 misalignment" | ✅ 맞음. expected road Y=+0.44 vs Gaussian median Y=-1.24, 차이 1.68m. 경사 상승 구간 coverage 부족 |

### 12.7 권장 Fix 우선순위

1. **[즉시] xyz anchor loss** — 초기 ground_points3d.ply 위치 기준 L2 penalty.  
   `loss += lambda_anchor * torch.mean((gaussians.get_xyz - anchor_xyz).pow(2))`  
   lambda_anchor = 0.01~0.1 정도. densified Gaussian에는 nearest-init anchor 배정 필요.

2. **[즉시] xyz learning rate 축소** — 현재 `lr=1.6e-4`. pose/ground drift 확대 방지를 위해 `5e-5`~`1e-4`로 감소.

3. **[중기] distort_3d_loss lambda cfg 연결** — `loss += cfg.lambda_dist * distort_3d_loss` 로 수정해서 yaml에서 조정 가능하게.

4. **[중기] 경사 구간 Gaussian 밀도 보완** — frame 461 인근 Gaussian 부족. merge_depth_ground.py에서 경사 상승 구간의 sample_per_frame 증가 또는 trajectory 끝 구간 처리 개선.

5. **[후순위] per-point rotation 초기화 또는 rotation 학습** — 근본 해결책이나 구현 복잡도 높음. 1~4 적용 후 개선량 평가 후 결정.

6. **front_info.json rectification 비활성 유지** — 적절한 pitch 보정값 없이는 오히려 악화 가능. 현재 상태 유지.

### 12.8 실제 수정 방향: pose Y drift 제거 후 재생성

실제 영상에 큰 경사 하강이 없다면 ground를 현재처럼 camera Y drift를 따라 기울어진 surface로 보면 안 된다. HUGSIM world에서 +Y는 physical down에 가깝기 때문에, 평탄 도로에서는 CAM_FRONT camera Y가 거의 일정해야 하고 road surface는 대략 `camera_Y + cam_height` 근처의 거의 수평 layer가 되어야 한다.

권장 수정 순서:

1. **corrected meta_data 생성**
   - CAM_FRONT trajectory에서 per-frame vertical drift `d_i = cam_front_y[i] - baseline_y`를 계산
   - 같은 timestamp의 모든 camera pose translation에 `t_y -= d_i` 적용
   - 먼저 rotation은 유지하고 translation Y만 보정하는 ablation을 수행
   - 방지턱은 수 cm~수십 cm 수준이므로 smoothing/median filter로 남길지 제거할지 선택

2. **corrected meta로 point cloud 재생성**
   - `merge_depth_wo_ground.py`로 `points3d.ply` 재생성
   - `merge_depth_ground.py`로 `ground_points3d.ply`, `ground_param.pkl` 재생성
   - 기대 결과: CAM_FRONT camera Y std가 크게 감소하고, ground_points3d world Y std도 현재 1.19m보다 크게 작아져야 함

3. **ground model 먼저 재학습**
   - `train_ground.py`만 먼저 돌려서 `CAM_FRONT_000126/131/136/141` semantic 상승이 사라지는지 확인
   - 이 네 프레임의 first rendered ground row가 다시 190px 전후로 내려오면 pose/ground drift 원인 검증 성공

4. **scene 3DGS 재학습 및 export**
   - 카메라 pose를 바꾸면 기존 scene/background Gaussian과도 좌표가 맞지 않으므로 최종적으로는 scene도 재학습 필요

주의: `scene.splat`에서 ground layer만 회전/평탄화하는 것은 임시 시각화 보정일 뿐이다. 렌더링 camera pose는 여전히 drift된 상태라서, ground만 수평화하면 camera-ground height가 프레임마다 달라지고 다른 artifact가 생길 수 있다.

---

## 13. `scene.splat` flat road layer 원인성 검증

대상 파일: `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve/export/vis/scene.splat`

### 13.1 Supersplat visualization-only artifact 여부

`scene.splat`은 별도의 시각화용 geometry가 아니라, `export/scene.pth`의 배경 Gaussian과 ground Gaussian을 합쳐서 만든 실제 scene geometry임.

검증:
- `scene.splat` size = 35,191,200 bytes
- splat row = 32 bytes → **1,099,725 Gaussians**
- `export/scene.pth` 구성:
  - background Gaussians: **680,617**
  - ground Gaussians: **419,108**
  - 합계: **1,099,725**

생성 경로:
```python
# eval_render/convert_scene.py
gaussians.save_splat(..., "scene.splat")

# scene/gaussian_model.py
xyz = self.get_full_xyz  # background + ground_model
scale = self.get_full_scaling
rotation = self.get_full_rotation
```

**결론**: Supersplat에서 보이는 flat road-surface layer는 가짜 표시물이 아님. 실제 HUGSIM renderer가 쓰는 ground Gaussians와 같은 출처다.

단, Supersplat의 좌표계/grid 평면 자체가 렌더링에 들어가는 것은 아님. HUGSIM viewer 코드의 `THREE.Plane(new THREE.Vector3(0, 1, 0), 0)`는 mouse ray intersection용 Y=0 reference plane이며, scene rasterizer 입력이 아니다. 따라서 “좌표계 평면이 ground를 뚫고 지나가서 그 평면이 렌더링된다”는 해석은 아님.

더 정확한 해석:
- Supersplat grid/reference plane이 ground를 관통해 보이는 것은 **원인이라기보다 증상**
- 실제 원인은 `ground_points3d.ply`/ground Gaussians가 world Y=0 기준으로 잘못 기울어지거나 내려간 것
- 이 잘못된 ground geometry가 `scene.splat`에 포함되어 실제 renderer에서도 road over-splatting을 만든다

### 13.2 ground Gaussian 상태

`export/scene.pth` 기준 ground Gaussian:

| 항목 | 값 |
|------|----|
| N | 419,108 |
| semantic | road 294,657 / sidewalk 124,384 / 기타 67 |
| alpha median / p95 / max | 0.9796 / 1.0 / 1.0 |
| max-scale median / p95 / p99 / max | 0.1115 / 0.2392 / 0.371 / 41.051 m |
| scale > 1m | 540개 |
| scale > 5m | 42개 |
| rotation | 전부 identity, diff=0 |

대부분은 작은 고투명 road/sidewalk splat이다. scale explosion이 주원인은 아니지만, opacity가 매우 높고 rotation이 고정되어 있어 경사 구간에서 시각적으로 강하게 덮을 수 있다.

### 13.3 frame별 frustum 검증

CAM_FRONT projection 기준 road/sidewalk Gaussian center:

| frame | road centers onscreen | mid-screen centers | close <30m | camera-local Y<0, Z=10~30m |
|-------|----------------------|--------------------|------------|----------------------------|
| 136 | 200,348 | 197,291 | 88,482 | 30,959 |
| 161 | 243,655 | 240,831 | 70,370 | 18,549 |
| 461 | 2,523 | 1,713 | 2,363 | 35 |

해석:
- frame 161은 실제로 road Gaussian layer가 화면 중간 영역에 대량 투영된다. Supersplat에서 보이는 flat layer가 semantic/rasterization에 영향을 줄 수 있는 구조다.
- frame 461은 onscreen ground center 수가 훨씬 적고, `camera-local Y<0` Gaussian도 35개뿐이다. 이 프레임의 문제는 “앞을 막는 dense layer”보다 coverage/misalignment와 sparse wrong projection 쪽에 가깝다.

### 13.4 RGB rendering 영향

`export/test/ours_30000/render` vs `gt`:

| frame | RGB PSNR | mean abs error | upper MAE | mid MAE | lower MAE |
|-------|----------|----------------|-----------|---------|-----------|
| 136 | 21.74 | 0.0525 | 0.0602 | 0.0611 | 0.0361 |
| 161 | 23.45 | 0.0415 | 0.0304 | 0.0426 | 0.0514 |
| 461 | 19.66 | 0.0718 | 0.0953 | 0.0713 | 0.0488 |

**결론**: flat road layer가 RGB artifact의 일부 원인일 수는 있지만, 현재 데이터만 보면 RGB rendering의 주된 blocker라고 단정하기 어렵다. frame 161은 ground layer가 강하게 투영되는데도 RGB error가 상대적으로 낮다. frame 461은 RGB error가 크지만, 그 프레임은 dense road blocker보다 coverage/misalignment와 배경 재구성 품질 문제가 섞여 있다.

### 13.5 semantic rendering 영향

Rendered semantic vs GT semantic:

| frame | region | GT ground | rendered ground | false ground |
|-------|--------|-----------|-----------------|--------------|
| 136 | mid | 0.603 | 0.856 | 0.255 |
| 161 | mid | 0.577 | 0.594 | 0.025 |
| 461 | upper | 0.000 | 0.308 | 0.308 |
| 461 | mid | 0.462 | 0.726 | 0.264 |

**결론**: semantic 문제에는 ground Gaussian layer가 실제 원인으로 관여한다. 특히 frame 136/461에서 GT가 vegetation/building/terrain/wall인 픽셀을 road/sidewalk로 덮는 false-ground가 크다. frame 161은 시각적으로 road layer가 많이 보이지만 GT도 mid/lower에 road가 많아서 false-ground 비율은 작다.

사용자가 지목한 ground point 상승 프레임:

`export/test/ours_30000/semantic/CAM_FRONT_000126.png`, `000131.png`, `000136.png`, `000141.png`

| frame | first rendered ground row | upper rendered ground | mid rendered ground | mid false ground | camera Y |
|-------|---------------------------|-----------------------|---------------------|------------------|----------|
| 126 | 191 / 450 (42.4%) | 0.000 | 0.602 | 0.004 | -3.843 |
| 131 | 192 / 450 (42.7%) | 0.000 | 0.638 | 0.029 | -3.959 |
| 136 | 126 / 450 (28.0%) | 0.035 | 0.856 | 0.255 | -4.079 |
| 141 | 0 / 450 (0.0%) | 0.437 | 0.762 | 0.160 | -4.179 |

이 네 프레임은 같은 pose-drift 구간의 연속 증상이다. 126/131은 GT와 거의 맞지만, 136부터 rendered ground가 위로 치고 올라오고 141에서는 upper-third까지 ground로 칠해진다.

추가 projection 검증:
- 126→141로 갈수록 onscreen road/sidewalk Gaussian center가 162k → 221k로 증가
- camera-local `Y<0`, `Z=10~30m` ground Gaussian도 26.7k → 47.9k로 증가
- 그러나 모든 프레임에서 ground Gaussian center의 upper-third 투영 개수는 0

따라서 이 현상은 “ground center가 하늘에 있다”가 아니라, **pose drift로 생긴 가짜 ground 기울기와 identity-rotation pancake Gaussian의 screen-space footprint가 upper/mid 영역까지 번지는 현상**으로 보는 것이 맞다. 실제 영상에 큰 경사 하강이 없다면 이 ground layer는 지형이 아니라 pose/ground 생성 artifact다.

#### 상세 검증 (frame 126/131/136/141 집중 분석)

**Gaussian-level 검증:**

Road-class Gaussian이 이상 영역(row<60, col>314)에 투영되는지 확인:
- frame 141 이상 영역: building(69%) + sky(31%) — **road-class Gaussian 0개**
- frame 131 동일 영역: vegetation(73%) + building(15%) — **road-class Gaussian 0개**

Road Gaussian의 projected 2D sigma_v (수직 방향 분산):
- p99: 10.6 pixels / max: 29 pixels (center가 row 220에서 29px 위 = row 191)
- **sigma_v 75px 이상 = top-third까지 도달 가능한 Gaussian: 0개**

따라서 scene.splat road Gaussian들의 **centers** 및 **2D sigma footprint** 모두 top-third에 도달하지 못함.

**실제 원인: near-field semantic feature bleed (alpha compositing 누출)**

| frame | 근거리(< 10m) road Gaussians | 그중 v < 225 | opacity mean |
|-------|------------------------------|-------------|-------------|
| 131 | 1,715 | 232개 | 0.443 |
| 136 | 1,556 | 192개 | 0.432 |
| 141 | **2,938** | **348개** | **0.545** |

Frame 141에서 가까운 도로 Gaussian 수가 1.7배로 증가하고 opacity도 높아짐. 이 Gaussian들의 `feats3D`(road semantic feature)가 alpha compositing에서 주변 픽셀까지 높은 weight로 기여 → 인접한 building/sky Gaussian들의 blended feature sum이 road class로 역전.

**발현 원인 (131→136 onset):**
- 카메라가 앞으로 이동하면서 5~10m 거리의 도로 Gaussians이 2~4m로 접근
- depth 감소 → image-space splatting footprint 급증
- 동시에 좌회전(-9.8°→-17.3°)으로 우측 건물 면의 occlusion 감소
- 결과: 건물/하늘 픽셀에 road feature blending 누출

**결론**: 이 현상은 “road Gaussian center가 상단에 있어서” 가 아니라, **near-field road Gaussians의 높은 alpha가 semantic feature blending을 통해 인접 non-road 픽셀까지 road class로 오염시키는 현상**. scene.splat의 geometry 자체 문제가 아니라 3DGS semantic feature blending의 near-field 특성상 한계.

### 13.6 DrivOR collision / low TTC 영향

현재 데이터 기준 결론:
- `scene.splat`의 road layer는 DrivOR 입력 RGB/semantic 품질을 악화시킬 수 있는 **간접 요인**이다.
- 하지만 기존 collision 분석에서는 t=1.0~7.75 구간에 현재 ego box 안 충돌 포인트가 없는데도 NC/TTC가 0인 케이스가 많았다.
- 이는 road semantic rendering보다 **future planned trajectory의 z 고정(frozen-z), ego_box score_z 보정, collision/scoring geometry** 쪽이 직접 원인이라는 증거가 더 강하다.
- `scene.splat`은 DrivOR scoring에서 직접 사용되는 파일이라기보다 scene geometry/export representation이고, collision은 `scene.ply`/semantic point cloud 및 score frame 변환 로직의 영향을 받는다.

**결론**: flat road-surface layer는 DrivOR low TTC의 단독 root cause로 보기는 어렵다. 다만 ground geometry가 실제 renderer에 존재하므로 RGB/semantic 입력 품질에는 영향을 주고, 이는 downstream planner 행동을 간접적으로 흔들 수 있다.

### 13.7 최종 판정

| 질문 | 판정 |
|------|------|
| 1. blurry/artifact-like RGB rendering 원인인가? | ⚠️ 부분 원인 가능. frame 161은 ground layer가 강하지만 RGB는 비교적 양호. RGB 전체 문제의 단독 원인은 아님 |
| 2. ground가 카메라 앞에 나타나는 abnormal semantic 원인인가? | ✅ 맞음. 실제 ground Gaussians가 renderer에 들어가며 false-ground를 만든다 |
| 3. frame 161/461 road/ground splatting 원인인가? | ✅ frame 161은 맞음. ⚠️ frame 461은 dense blocker보다 coverage/misalignment 성격이 강함 |
| 4. DrivOR collision/low TTC 원인인가? | ⚠️ 간접 요인. 직접 원인은 scoring/collision z/box 로직 쪽 증거가 더 강함 |

**종합 결론**: Supersplat의 flat road layer는 visualization symptom만은 아니다. 실제 ground Gaussian layer이고 semantic over-splatting의 root cause 중 하나다. 하지만 RGB 품질 저하와 DrivOR low TTC의 단일 root cause는 아니다. 현재 가장 그럴듯한 원인 체인은 `ground Gaussian layer geometry/rotation/coverage 문제 → semantic/RGB 입력 품질 일부 악화`, 그리고 별도로 `ego_box/future-z/scoring 로직 문제 → NC/TTC 직접 저하`이다.

---

## 14. 현재 미완료 항목 및 다음 단계

### 즉시 해야 할 것

1. **Fix 1 올바른 적용** (우선순위 최고)
   - `ground_height()`는 수정 금지
   - `ego_box` property에서 score_z에만 cam_height 적용:
   ```python
   @property
   def ego_box(self):
       _, cam_height, _ = self.ground_model
       return [self.vt[2], -self.vt[0], -self.vt[1] - cam_height,
               self.whl[0], self.whl[2], self.whl[1], -self.vr[1]]
   ```
   - 적용 후 test3 실행 (seed=42 + fix1_ego_box)

2. **Fix 2 개선**
   - nearest-neighbor → KD-tree + 반경 내 평균 z 사용
   - Fix1 정상 확인 후 함께 test4 실행

3. **test1 재실행** (fix1 ego_box 버전)
   - 이전 test1은 ground_height()를 수정해서 렌더링 망가짐 → crash
   - 올바른 fix1 적용 후 재실행 필요

4. **`raw_data_curve_ground_model` 재학습 (60k)** (Section 15 참고)
   - 현재 30k 학습으로 PSNR=24.31 → 60k까지 학습 시 25.0~25.3 회복 예상
   - source: `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve_ground`
   - `prepare_flat_ground_meta.py`로 생성된 meta (ground geometry 개선됨)

### 중기 과제

- DAC 후반부 하락 원인 분석 (t=4.25+ 도로 이탈)
- scene.ply의 vegetation/terrain을 collision에서 제외하는 필터링 검토
- cam_height 1.5m vs 실측 1.11m 재검토
- NEW 60k 학습 완료 후 ground artifact (frames 126/136/141) 재확인

### 관련 파일

| 파일 | 역할 |
|------|------|
| `sim/hugsim_env/envs/hug_sim.py` | ego_box, ground_height, 렌더링 환경 |
| `sim/utils/score_calculator.py` | NC/TTC/DAC/PDMS 계산 |
| `sim/utils/plan.py` | ground_height 올바른 버전 (참조용) |
| `closed_loop.py` | seed 고정, DrivoR 실행 루프 |
| `configs/sim/kist_base.yaml` | 시뮬레이션 설정 |
| `kist_curve_coord_debug/score_frame_collision_debug.ply` | 충돌 디버그 시각화 |
| `kist_curve_coord_debug/gen_collision_debug_ply_v2.py` | PLY 생성 스크립트 |
| `kist_curve_coord_debug/sem_collision_analysis.py` | semantic 충돌 분석 |
| `260308-KIST-Videos/py/prepare_flat_ground_meta.py` | gravity-level meta 생성 (raw_data_curve_ground 소스) |
| `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve_ground/flat_ground_meta_debug.json` | meta 생성 파라미터 로그 |

---

## 15. `raw_data_curve_ground_model` PSNR 하락 검증

대상:
- 입력 source: `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve_ground`
- 학습 결과: `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve_ground_model`
- 비교 기준: `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve`

### 15.1 먼저 확인해야 할 점: `make_meta_data_front_colmap.py`는 현재 source와 다름

`/home/ms/260308-KIST-Videos/py/make_meta_data_front_colmap.py`는 다음 특성을 가진다.

- `SPARSE_DIR`, `IMG_DIR`, `OUT_JSON`가 오래된 `/home/ms/260308-KIST-Videos/...` 경로로 하드코딩됨
- `CAM_FRONT`만 사용
- frame 범위가 `000001.jpg`~`000180.jpg`
- `rgb_path`가 `./images/CAM_FRONT/xxxxxx.jpg` 한 종류만 생성됨
- `camera_model: OPENCV`로 저장하지만 실제 KIST HUGSIM pipeline은 `SIMPLE_RADIAL` 형태의 6-camera meta를 사용

반면 실제 `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve_ground/meta_data.json`은:

- frame 수: 2772
- 카메라: 6개 (`CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`)
- 각 카메라 462 frame
- `camera_model: SIMPLE_RADIAL`
- image path는 기존 `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve/images` 기준으로 정상 존재
- `flat_ground_meta_debug.json` 기준 생성 방법은 `COLMAP 0_aligner + gravity-level HUGSIM basis + flatten_up`

따라서 **현재 `raw_data_curve_ground_model`의 입력 meta는 위의 단순 front-only `make_meta_data_front_colmap.py` 출력물과 동일하지 않다.**  
만약 그 스크립트를 그대로 사용해 HUGSIM 전체 scene 학습을 돌리면, 6-camera 제약이 사라지고 frame 수도 줄어들기 때문에 현재 KIST pipeline용 meta로는 부적합하다.

### 15.2 meta_data.json 검증

기존 `recon_HUGSIM` vs 새 `raw_data_curve_ground`:

| 항목 | 기존 | 새 source | 판정 |
|------|------|-----------|------|
| frames | 2772 | 2772 | 동일 |
| camera 수 | 6 | 6 | 동일 |
| camera별 frame | 462 | 462 | 동일 |
| rgb_path 순서 | 동일 | 동일 | train/test split 동일 |
| intrinsics 종류 | 6 unique K | 6 unique K | 동일 |
| CAM_FRONT world Y std | 1.120m | 0.000m | 새 meta에서 vertical drift 제거 |
| 전체 camera world Y std | 1.143m | 0.169m | 새 meta가 훨씬 수평 |
| rotation 차이 | - | 기존 대비 constant 4.956 deg | basis 재정의 영향 |
| translation 차이 median/max | - | 3.44m / 4.63m | 기존 drift 제거 영향 |

새 meta의 image path는 모두 존재하며, KIST handler의 split도 기존과 동일하다.

```
train/test = 2218 / 554
CAM_FRONT = 369 train / 93 test
```

**결론**: image path, intrinsics, frame order, train/test split 문제는 아니다. 새 meta는 의도대로 vertical drift를 제거했지만, 기존 HUGSIM 좌표계와 비교하면 모든 rotation에 약 4.956도 basis 차이가 들어가고 translation도 수 m 단위로 달라진다. 이 차이가 scene/background 학습에는 영향을 줄 수 있다.

### 15.3 ground_points3d.ply 검증

`ground_points3d.ply` 통계:

| 항목 | 기존 | 새 source |
|------|------|-----------|
| point 수 | 199,584 | 199,584 |
| Y mean | -1.663 | 1.487 |
| Y std | 1.189 | 0.227 |
| Y p1 / median / p99 | -3.277 / -1.897 / 1.500 | 1.072 / 1.507 / 1.895 |

새 source에서 CAM_FRONT pose의 world Y는 0으로 고정되어 있고, ground point median Y는 1.507이다. HUGSIM convention에서 +Y는 physical down이므로, 이는 **front camera 기준 약 1.5m 아래에 ground가 놓인 것**이다.

즉 새 ground point cloud는 “도로가 아래로 뚫고 지나가는 큰 가짜 경사면” 문제를 줄이는 방향으로 생성되었다. point 수는 기존과 같고, p1~p99 범위도 훨씬 좁아졌으므로 sparse해진 것은 아니다.

다만 완전히 깨끗한 평면은 아니다.

- 새 ground Y min/max: -5.625 / 14.198
- outlier는 여전히 존재
- frame 141 기준 near visible ground 중 camera-local `Y<0` count는 기존 15,563 → 새 11,976으로 감소했지만 0은 아님

따라서 새 ground는 기존보다 개선되었지만, segmentation/depth 기반 outlier와 multi-frame projection overlap은 남아 있다.

### 15.4 문제 frame 렌더 검증

새 ground-only render:

| frame | first non-black ground row | upper-third ground |
|-------|----------------------------|--------------------|
| 126 | 193 | 0.000 |
| 131 | 193 | 0.000 |
| 136 | 197 | 0.000 |
| 141 | 182 | 0.000 |
| 161 | 192 | 0.000 |

기존 semantic render에서 `CAM_FRONT_000141`은 ground가 row 0부터 올라오는 문제가 있었지만, 새 ground-only render에서는 상단 1/3에 ground가 나타나지 않는다.

**결론**: ground geometry correction은 사용자가 지목한 “ground가 이미지 위로 튀는 현상”을 상당히 줄였다.

### 15.5 PSNR 비교

학습 결과:

| 모델 | train/test | iter | PSNR | SSIM | LPIPS |
|------|------------|------|------|------|-------|
| 기존 ground | test | 30k | 29.481 | 0.889 | 0.128 |
| 새 ground | test | 30k | 28.937 | 0.882 | 0.135 |
| 기존 scene | test | 60k | 25.399 | 0.749 | 0.255 |
| 새 scene | test | 30k | 24.308 | 0.704 | 0.294 |

직접 이미지 PSNR 재계산:

| camera | 기존 scene | 새 scene | 변화 |
|--------|------------|----------|------|
| CAM_BACK | 24.232 | 23.078 | -1.154 |
| CAM_BACK_LEFT | 28.134 | 26.822 | -1.312 |
| CAM_BACK_RIGHT | 24.073 | 23.253 | -0.820 |
| CAM_FRONT | 25.011 | 24.025 | -0.986 |
| CAM_FRONT_LEFT | 26.634 | 25.189 | -1.445 |
| CAM_FRONT_RIGHT | 24.285 | 23.463 | -0.822 |
| 전체 | 25.396 | 24.306 | -1.090 |

하락이 특정 카메라 하나에만 집중되지 않고 6-camera 전체에서 비슷하게 발생한다.

### 15.6 PSNR 하락 원인 판정

#### 실제 사용 스크립트 확인

`raw_data_curve_ground/meta_data.json`의 실제 생성 스크립트는 `make_meta_data_front_colmap.py`가 **아니라** `/home/ms/260308-KIST-Videos/py/prepare_flat_ground_meta.py`다.

`flat_ground_meta_debug.json` 기록에 따르면:

```
method     : rebuild meta from COLMAP 0_aligner with gravity-level HUGSIM basis
colmap_path: .../raw_data_curve/recon/colmap/sparse/0_aligner  ← 기존과 동일한 COLMAP
flatten_up : true
smooth     : 1
removed COLMAP Z drift: 0.0 ~ 0.897 m
HUGSIM front Y: min=0.000, max=0.000, std=0.000
missing_count: 0
```

즉, **동일한 0_aligner COLMAP**을 사용하되, `prepare_flat_ground_meta.py`가 두 가지를 적용했다:

1. **gravity-level basis 변환**: HUGSIM +Y = gravity down (COLMAP −Z), HUGSIM +Z = first CAM_FRONT forward를 수평 평면에 투영. 결과적으로 기존 대비 4.956° 전체 rotation offset 발생.
2. **flatten_up=True**: CAM_FRONT의 COLMAP Z drift (0~0.897m)를 모든 카메라에서 동일하게 제거. 결과적으로 CAM_FRONT Y=0 완전 평탄화.

#### 원인별 판정 (최종)

| 후보 원인 | 판정 | 근거 |
|-----------|------|------|
| incorrect ground point generation | ❌ 단독 원인 아님 | Y std 1.189→0.227로 개선, ground PSNR도 28.94로 양호, outlier도 일부만 |
| wrong pose / coordinate conversion | ⚠️ 주요 기여 원인 | gravity-level basis로 전체 scene이 기존 대비 4.956° 회전. 기존 0_aligner origin 정규화와 다른 alignment. 6-camera 전체에서 균일하게 -1.09dB 하락 |
| COLMAP 자체 품질 | ❌ 해당 없음 | 동일 0_aligner COLMAP 사용 |
| front-camera-only meta 제약 | ❌ 해당 없음 | 실제 source는 6-camera/2772 frames |
| train/test mismatch | ❌ 아님 | frame order/path 동일, kist split 동일 |
| **학습 iteration 부족** | ✅ **주요 원인** | 기존 45k=25.29, 60k=25.40 / 새 30k=24.31. 새 파이프라인은 30k만 학습. 수렴 곡선(7k→15k→30k)이 정상 증가 중 (22.16→23.32→24.31, 약 +1.1dB/15k). 60k까지 학습 시 OLD 수준 도달 가능 |
| ground model 학습 영향 | ⚠️ 부분 | ground PSNR -0.54dB → scene -1.09dB. ground만으로 scene 하락 전부 설명 불가 |

#### 수렴 속도 비교

```
NEW  7k test : PSNR=22.159  (+0.000)
NEW 15k test : PSNR=23.324  (+1.165)
NEW 30k test : PSNR=24.308  (+0.984)

OLD 45k test : PSNR=25.288  (30k 기준점 없음)
OLD 60k test : PSNR=25.399  (+0.111/15k)
```

15k→30k 구간에서 +0.98dB 증가 중이므로, 30k→45k에서 추가 +0.7~0.9dB, 45k→60k에서 +0.2~0.4dB 증가 예상. **NEW 60k 예상 PSNR ≈ 25.0~25.3** — 기존 수준에 근접.

### 15.7 최종 결론

**`prepare_flat_ground_meta.py`가 생성한 meta는 ground geometry 품질을 개선했다.**

- CAM_FRONT Y=0 (완전 평탄), ground_points3d Y mean=+1.487m ≈ cam_height=1.5m ✓
- 문제 frame 126/131/136/141: ground-only render가 이미지 상단을 덮지 않음 ✓
- ground model PSNR: 28.94 (기존 29.48보다 0.54 낮지만, 기존의 ground Y drift를 제거한 결과)

**PSNR 24.3 → 25.4 하락의 주요 원인은 `학습 iteration 부족 (30k vs 60k)`이다.**

추가 원인으로 gravity-level basis 변환으로 인한 좌표 정렬 차이(4.956°)가 기존 COLMAP/depth-based initialization과 약간의 불일치를 만들어 수렴 초기 속도를 낮출 수 있다. 하지만 수렴 곡선이 정상 증가 중이므로 구조적 결함은 아니다.

**권장 다음 단계:**

| 옵션 | 설명 | 기대 효과 |
|------|------|-----------|
| **A (권장)** | `raw_data_curve_ground` source로 60k까지 재학습 | ground geometry 개선 + PSNR≈25.1~25.3 회복 |
| B | 기존 `recon_HUGSIM` pose 유지 + ground_points3d/ground_param만 새 pipeline으로 교체 | PSNR 유지 + ground artifact 일부 개선 |
| C | 새 meta + 60k + distort_3d_loss lambda 수정 + anchor loss 추가 | ground 품질 최대화 |

---

## 16. `raw_data_curve_ground_outputdrivor_test1` 충돌 분석

대상: `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve_ground_outputdrivor_test1/scene_easy_00`

### 16.1 평가 점수 요약

`eval.json` 기준:

| 지표 | 값 | 비고 |
|------|-----|------|
| NC (No Collision) | **0.833** | 42프레임 중 7개 NC=0 |
| TTC (Time-to-Collision) | **0.571** | 절반 가까이 TTC 실패 |
| DAC (Drivable Area Compliance) | 0.952 | 상대적으로 양호 |
| C (Comfort) | 1.000 | 완벽 |
| PDMS | 0.599 | NC·TTC 가중 합산 |
| RC (Route Completion) | 0.329 | 경로 완주율 낮음 |
| HD score | 0.197 | 최종 종합 점수 |

주목: **C(쾌적성)=1.0, DAC=0.952** 로 운행 자체는 원활. **NC·TTC가 발목**을 잡는 구조.

### 16.2 NC=0 및 TTC=0 발생 타임스탬프

`eval.json` details 기준:

| 타임스탬프 | NC | TTC | 비고 |
|------------|-----|-----|------|
| 1.5 | 0 | 0 | NC+TTC 동시 실패 |
| 3.5~5.75 | 1 | 0 | TTC만 실패 (NC는 통과) |
| 4.0, 4.25, 5.0 | 0 | 0 | NC+TTC 동시 실패 |
| 6.0, 7.0, 7.75 | 0 | 0 | NC+TTC 동시 실패 |
| 9.75 | 1 | 0 | TTC만 실패 |
| 10.25, 10.5 | 1 | 1 | DAC=0 (도로 이탈) |

NC=0 타임스탬프: **t=1.5, 4.0, 4.25, 5.0, 6.0, 7.0, 7.75** (7개)  
TTC=0 타임스탬프: t=3.5~9.75 대부분 (NC는 통과하면서 TTC만 실패하는 구간 존재)

### 16.3 data.pkl 프레임 분석

`data.pkl` 42 프레임 전체:

- **모든 프레임에서 `collision=False`** — 현재 타임스텝 ego box 내 충돌 없음
- ego_box [x,y,z,w,l,h,yaw]: z(높이, score_z_up) 범위 최대 -0.221 (거의 0에 가까움)
- ego 위치: x=0→65m 전진, t=7s 이후 우측 회전
- planned_traj: 7 waypoints × 0.5s 간격 (총 3.5s 선행)

```
w=1.6m, l=3.0m, h=1.5m  (고정)
```

### 16.4 NC=0 충돌 메커니즘

**핵심**: NC는 현재 ego 위치가 아닌 **미래 planned trajectory waypoint들**에서 `bg_collision_det` 호출.

`score_calculator.py _calculate_no_collision()`:
```python
ego_x, ego_y, z, ego_w, ego_l, ego_h, ego_yaw = ego_box  # z는 현재 프레임 값으로 고정
for idx in range(planned_traj.shape[0]):
    ego_x, ego_y, ego_yaw = planned_traj[idx]  # x, y, yaw만 갱신
    ego_trans_mat[:3, 3] = np.array([ego_x, ego_y, z])  # z 고정 (frozen-z 버그)
    ...
    if bg_collision_det(scene_xyz, [ego_x, ego_y, z, ...]):
        return 0.0
```

`bg_collision_det` 임계값: **>100 포인트**가 ego box 내부에 있으면 True (충돌 판정).

#### 충돌 발생 위치 (NC=0 프레임별)

| 타임스탬프 | 충돌 발생 waypoint | score_x (전방거리) | score_y (측방) | 충돌 포인트 수 |
|------------|-------------------|-------------------|----------------|----------------|
| t=1.5 | traj[6] (waypoint 6) | ~30.5m | ~-1.2m | **694 pts** |
| t=4.0 | traj[6] | ~70.4m | ~+2.1m | **1050 pts** |
| t=4.25 | traj[5~6] | ~70~75m | 유사 | >100 pts |
| t=5.0 | traj[5~6] | ~60~70m | 유사 | >100 pts |
| t=6.0 | traj[4~6] | ~50~70m | 유사 | >100 pts |
| t=7.0 | traj[4~6] | ~40~65m | 우회전 구간 | >100 pts |
| t=7.75 | traj[3~6] | ~35~65m | 우회전 구간 | >100 pts |

**모든 충돌은 전방 30~75m 범위의 미래 waypoint에서 발생**.  
현재 ego 위치(t 시점)에는 충돌 없음.

### 16.5 충돌 포인트 특성 분석

`scene.ply` 기준 (560,226 pts, camera coord → score/IMU coord 변환):

```python
# score_calculator.py parse_data() 변환
scene_xyz = np.stack([ply[:,2], -ply[:,0], -ply[:,1]], axis=1)
#                    cam_z    -cam_x      -cam_y
# = [score_x(forward), score_y(lateral), score_z(up)]
```

**t=1.5 traj[6] 충돌 포인트 694개 Z_up 분포:**

| Z_up 범위 | 비율 | 해석 |
|-----------|------|------|
| -0.9 ~ -0.25m | ~20% | 지면 아래 (scoring artifact 가능성) |
| -0.25 ~ +0.25m | ~45% | 도로 표면 레벨 |
| +0.25 ~ +1.0m | ~20% | 저층 구조물 높이 |
| +1.0 ~ +1.6m | ~15% | 차량/vegetation 높이 |

**ego box 높이 h=1.5m** → Z_up 범위 -0.9 ~ +1.6m까지 스팬 (ego box 전체 높이 포함).

충돌 포인트 분포가 도로 표면 레벨(-0.25~+0.25m) 근처에 집중되어 있으며, 이는 **scene.ply의 background Gaussian 포인트** (건물, 식생, 지형 등)가 해당 forward 위치의 ego box 높이 범위와 교차한다는 것을 의미.

**scene.ply에는 semantic 정보 없음** → 정확한 class 분류 불가. 단, Section 8에서 분석한 전체 scene Gaussian semantic 분포 참조:
- vegetation(8): 52.1%, building(2): 24.2%, terrain(9): 9.6%, fence(4): 5.4%

t=3.5~9.75 구간 TTC=0 단독 실패 원인은 0.5s/1.0s lookahead에서 동일 메커니즘으로 collision 발생.

### 16.6 생성된 디버그 파일

| 파일 | 설명 |
|------|------|
| `trajectory_score_coord.ply` (336 pts) | score/IMU 좌표계 ego 궤적 + planned_traj. 초록=ego 위치, 주황=NC=0 발생 planned traj 포인트, 파랑=정상 planned traj |
| `collision_points_cam_coord.ply` (4966 pts, 빨강) | NC=0 프레임들에서 ego box 내 충돌 포인트 (camera coord 기준) |

경로: `/media/ms/WD_BLACK_4TB/KIST/raw_data_curve_ground_outputdrivor_test1/`

### 16.7 원인 판정

| 후보 원인 | 판정 | 근거 |
|-----------|------|------|
| 현재 ego 위치 충돌 | ❌ | 모든 42 프레임 `collision=False` |
| future planned traj의 frozen-z 버그 | ✅ **주요 원인** | z는 현재 프레임 값(≈0)으로 고정; ego_box z도 cam_height 미보정(≈-0.22) → 미래 waypoint의 실제 지면 높이와 무관하게 score_z≈0 기준 box 검사 |
| scene.ply 배경 포인트가 road-level에 존재 | ✅ **직접 트리거** | 도로 전방 30~75m 범위에 scene.ply 포인트 694~1050개가 ego box 높이 범위와 겹침 |
| ego_box score_z cam_height 미보정 | ✅ **기여 원인** | ego_box z≈-0.22 (지면 0m 기준이 아니라 지면 위 1.5m로 이동해야 함); 현재 box가 지면 아래까지 내려가서 ground-level 포인트를 포함 |
| vegetation/building이 도로를 막음 | ⚠️ **부분** | 30~75m 전방 scene.ply 포인트가 도로 측방 구조물일 가능성 있음; 실제 도로를 막는 물리적 장애물인지 scoring artifact인지 구분 어려움 |

### 16.8 test0 대비 test1 점수 차이

| 실험 | NC | TTC | DAC | PDMS | HD score |
|------|-----|-----|-----|------|----------|
| test0 (seed=42) | 0.757 | 0.595 | 0.892 | 0.560 | 0.185 |
| **test1 (raw_data_curve_ground)** | **0.833** | **0.571** | **0.952** | **0.599** | **0.197** |

test1은 새 ground geometry (`prepare_flat_ground_meta.py`) 기반 scene 사용. NC·PDMS·DAC가 개선되었으나 TTC는 소폭 하락. HD score +0.012 소폭 향상.

**NC 개선 이유**: ground geometry 개선으로 scene.ply 포인트 배치가 달라지고, freeze된 ego_box z≈0 기준에서 충돌하는 포인트 수가 일부 감소했을 가능성.

**TTC 하락 이유**: 0.5s/1.0s lookahead에서 충돌 포인트가 여전히 많거나 새 scene에서 다른 위치 충돌 발생.

### 16.9 남은 과제

1. **Fix1 (ego_box cam_height 보정) 정상 적용 후 test3** — 현재 최우선과제 (Section 14 참조)
2. **Fix2 (future waypoint z 보간) 개선 후 test4**
3. scene.ply collision 포인트의 semantic class 확인: scene Gaussians의 semantic feature를 score_calculator에서 필터링 가능한지 검토
4. TTC=0이지만 NC=1인 구간(t=3.5~5.75 일부, t=9.75): 0.5s/1.0s lookahead 단계에서만 충돌 — Fix1+Fix2 적용 시 해소 여부 확인
5. DAC=0 (t=10.25, 10.5): 도로 이탈 원인 별도 분석 필요
