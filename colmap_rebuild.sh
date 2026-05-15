#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  KIST kist_curve — COLMAP 재구성 파이프라인 (디버깅용)
#
#  run.sh의 Step 1 (COLMAP) 전체를 단계별로 실행.
#  각 단계를 독립적으로 재실행할 수 있도록 --step 옵션 제공.
#
#  사용법:
#    bash colmap_rebuild.sh              # 전체 실행 (Step 0~6)
#    bash colmap_rebuild.sh --step 3     # Step 3부터 끝까지 실행
#    bash colmap_rebuild.sh --step 3 --only  # Step 3만 실행
#    bash colmap_rebuild.sh --check      # 각 단계 결과만 검사 (실행 없음)
#
#  전제 조건:
#    - /home/ms/260308-KIST-Videos/kist_curve/images/ 존재
#    - /home/ms/260308-KIST-Videos/6_GPS/2_Entrance-L1.csv 존재
#    - /usr/bin/colmap 사용 가능
#    - /home/ms/miniconda3/envs/urbansim/bin/python3 사용 가능
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── 경로 설정 ─────────────────────────────────────────────────────────────────
DATA=/home/ms/260308-KIST-Videos/kist_curve
SCRIPTS=/home/ms/260308-KIST-Videos
PYTHON=/home/ms/miniconda3/envs/urbansim/bin/python3
COLMAP=/home/ms/miniconda3/envs/urbansim/bin/colmap

# ─── 인자 파싱 ─────────────────────────────────────────────────────────────────
START_STEP=0
ONLY=0
CHECK_ONLY=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --step)  START_STEP="$2"; shift 2 ;;
        --only)  ONLY=1; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

END_STEP=6
if [[ $ONLY -eq 1 ]]; then
    END_STEP=$START_STEP
fi

# ─── 헬퍼 함수 ─────────────────────────────────────────────────────────────────
step_header() {
    echo ""
    echo "┌──────────────────────────────────────────────────────────────────"
    echo "│  STEP $1: $2"
    echo "└──────────────────────────────────────────────────────────────────"
}

should_run() {
    local s=$1
    [[ $CHECK_ONLY -eq 0 && $s -ge $START_STEP && $s -le $END_STEP ]]
}

check_db_cameras() {
    echo "  [DB] cameras:"
    $PYTHON - <<'PYEOF'
import sqlite3, struct
db = sqlite3.connect('/home/ms/260308-KIST-Videos/kist_curve/database.db')
cur = db.cursor()
for cam in cur.execute('SELECT camera_id, model, width, height, params FROM cameras').fetchall():
    params = struct.unpack('8d', cam[4][:64])
    model_name = {4:'OPENCV', 1:'SIMPLE_PINHOLE', 2:'PINHOLE'}.get(cam[1], str(cam[1]))
    print(f"    cam {cam[0]}: {model_name} {cam[2]}x{cam[3]}  fx={params[0]:.2f} fy={params[1]:.2f} cx={params[2]:.2f} cy={params[3]:.2f}  k1={params[4]:.5f}")
db.close()
PYEOF
}

check_colmap_model() {
    local label=$1
    local path=$2
    echo "  [$label] cameras.bin:"
    $PYTHON - "$path" <<'PYEOF'
import sys
sys.path.insert(0, '/home/ms/HUGSIM_N/HUGSIM/data')
from colmap.colmap_reader import read_intrinsics_binary, read_extrinsics_binary, read_points3D_binary
import numpy as np

path = sys.argv[1]
cams = read_intrinsics_binary(f'{path}/cameras.bin')
imgs = read_extrinsics_binary(f'{path}/images.bin')
xyzs, _, errors = read_points3D_binary(f'{path}/points3D.bin')

for cid, cam in sorted(cams.items()):
    p = cam.params
    print(f"    cam {cid}: {cam.model} fx={p[0]:.2f} fy={p[1]:.2f} cx={p[2]:.2f} cy={p[3]:.2f}", end='')
    if len(p) > 4:
        print(f"  k1={p[4]:.5f} k2={p[5]:.5f}", end='')
    print()
print(f"    images: {len(imgs)}  3D pts: {len(xyzs)}  repro_err median={np.median(errors):.4f} mean={np.mean(errors):.4f}")
PYEOF
}

check_prior() {
    echo "  [prior] cameras.txt:"
    cat "$DATA/prior/cameras.txt"
    echo "  [prior] images.txt (first 3 entries):"
    grep -v '^$' "$DATA/prior/images.txt" | head -3 || true
    echo "  [prior] frozen check (CAM_FRONT translation):"
    $PYTHON - <<'PYEOF'
import numpy as np

def qvec2rotmat(q):
    w,x,y,z = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

poses = {}
with open('/home/ms/260308-KIST-Videos/kist_curve/prior/images.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        p = line.split()
        if len(p) < 10 or '/' not in p[9]: continue
        name = p[9]
        if 'CAM_FRONT/' not in name or 'LEFT' in name or 'RIGHT' in name: continue
        q = [float(p[1]),float(p[2]),float(p[3]),float(p[4])]
        t = np.array([float(p[5]),float(p[6]),float(p[7])])
        R = qvec2rotmat(q)
        c2w = np.eye(4); c2w[:3,:3]=R.T; c2w[:3,3]=-R.T@t
        poses[name] = c2w[:3,3]

sorted_poses = sorted(poses.items())
positions = np.array([v for _,v in sorted_poses])
if len(positions) > 1:
    diffs = np.linalg.norm(np.diff(positions,axis=0),axis=1)
    frozen = np.sum(diffs < 0.001)
    print(f"    CAM_FRONT {len(positions)} frames: frozen={frozen}  mean_dist={diffs.mean():.4f}m  max_dist={diffs.max():.4f}m")
PYEOF
}

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "████████████████████████████████████████████████████████████████████"
echo "  KIST kist_curve COLMAP rebuild  (step ${START_STEP}~${END_STEP})"
echo "  DATA: $DATA"
echo "  CHECK_ONLY=$CHECK_ONLY"
echo "████████████████████████████████████████████████████████████████████"

# ─── Step 0: feature_extractor ────────────────────────────────────────────────
step_header 0 "feature_extractor (OPENCV, fx=444.3, single_camera_per_folder)"
echo "  변경점: OPENCV 모델 + 초기 fx=444.3 명시 (이전: default focal 사용)"
echo "  이유: DJI Action 5 Pro Linear mode의 실측 focal. 잘못된 초기값은 BA 수렴 저하."

if should_run 0; then
    # 기존 database 삭제 후 재생성
    rm -f "$DATA/database.db" "$DATA/database.db-shm" "$DATA/database.db-wal"

    $COLMAP feature_extractor \
        --database_path "$DATA/database.db" \
        --image_path    "$DATA/images" \
        --ImageReader.camera_model OPENCV \
        --ImageReader.single_camera_per_folder 1 \
        --ImageReader.camera_params "444.3,444.3,400,225,0,0,0,0" \
        --SiftExtraction.use_gpu 1 \
        --SiftExtraction.upright 1

    echo "  ✔ feature_extractor 완료"
fi

echo "  [검증]"
check_db_cameras

# ─── Step 1: exhaustive_matcher ───────────────────────────────────────────────
step_header 1 "exhaustive_matcher (모든 이미지 쌍 매칭)"
echo "  변경점: exhaustive_matcher 유지 (이전과 동일)"
echo "  이유: 6카메라가 서로 다른 방향을 향해 sequential 매칭으로는 카메라 간 매칭이 불가."
echo "  주의: 1080장 exhaustive → 약 580,000쌍. 시간 소요 큼."

if should_run 1; then
    $COLMAP exhaustive_matcher \
        --database_path "$DATA/database.db" \
        --SiftMatching.use_gpu 1

    echo "  ✔ exhaustive_matcher 완료"
fi

$PYTHON - <<'PYEOF'
import sqlite3
db = sqlite3.connect('/home/ms/260308-KIST-Videos/kist_curve/database.db')
n_matches = db.execute('SELECT COUNT(*) FROM matches').fetchone()[0]
n_two_view = db.execute('SELECT COUNT(*) FROM two_view_geometries').fetchone()[0]
db.close()
print(f"  [검증] matches: {n_matches}  two_view_geometries: {n_two_view}")
PYEOF

# ─── Step 2: make_prior_curve.py ──────────────────────────────────────────────
step_header 2 "make_prior_curve.py (GPS prior 생성)"
echo "  변경점: GPS 보간 방식 수정"
echo "  이전: GPS_ROW_RATE=27로 단순 행 인덱스 보간 → 27행 중복 구간에서 frozen 발생"
echo "  수정: 실제 이동 row만 추출(~1Hz) 후 카메라 시간(12.5fps)에 선형 보간"
echo "  의존성: Step 0(feature_extractor) 완료 후 database.db 필요"

if should_run 2; then
    $PYTHON "$SCRIPTS/make_prior_curve.py"
    echo "  ✔ make_prior_curve.py 완료"
fi

echo "  [검증]"
check_prior

# ─── Step 3: point_triangulator ───────────────────────────────────────────────
step_header 3 "point_triangulator → colmap_sparse_tri/"
echo "  변경점: 없음 (이전과 동일)"
echo "  역할: GPS prior pose를 고정한 채 3D 포인트만 삼각측량"

if should_run 3; then
    rm -rf "$DATA/colmap_sparse_tri"
    mkdir -p "$DATA/colmap_sparse_tri"

    $COLMAP point_triangulator \
        --database_path "$DATA/database.db" \
        --image_path    "$DATA/images" \
        --input_path    "$DATA/prior" \
        --output_path   "$DATA/colmap_sparse_tri"

    echo "  ✔ point_triangulator 완료"
fi

echo "  [검증]"
check_colmap_model "tri" "$DATA/colmap_sparse_tri"

# ─── Step 4: bundle_adjuster ──────────────────────────────────────────────────
step_header 4 "bundle_adjuster → colmap_sparse_ba/"
echo "  전략: GPS prior pose 고정 (refine_extrinsics=0), intrinsics만 최적화"
echo "  이유: rig constraint 없이 extrinsics 자유 최적화 시 카메라가 독립 이동 → pose 뒤틀림"
echo "        (이전 시도: frozen=27, jump=16 확인)"
echo "  최적화 대상: fx/fy (1), cx/cy (0 고정), distortion k1k2p1p2 (1), extrinsics (0 고정)"

if should_run 4; then
    rm -rf "$DATA/colmap_sparse_ba"
    mkdir -p "$DATA/colmap_sparse_ba"

    $COLMAP bundle_adjuster \
        --input_path  "$DATA/colmap_sparse_tri" \
        --output_path "$DATA/colmap_sparse_ba" \
        --BundleAdjustment.refine_focal_length    0 \
        --BundleAdjustment.refine_principal_point 0 \
        --BundleAdjustment.refine_extra_params    1 \
        --BundleAdjustment.refine_rig_from_world  0 \
        --BundleAdjustment.refine_sensor_from_rig 0 \
        --BundleAdjustment.max_num_iterations     100

    echo "  ✔ bundle_adjuster 완료"
fi

echo "  [검증]"
check_colmap_model "ba" "$DATA/colmap_sparse_ba"

# ─── Step 5: point_triangulator_ba ────────────────────────────────────────────
step_header 5 "point_triangulator_ba → colmap_sparse_ba/ (제자리 업데이트)"
echo "  변경점: 이 단계가 이전 파이프라인에서 누락되어 있었음"
echo "  이유: BA 후 카메라 pose가 바뀌면 기존 3D 포인트의 reprojection error가 폭발"
echo "        (median은 양호하나 mean=1e+140 확인됨)"
echo "  역할: BA로 정제된 pose로 3D 포인트를 재삼각측량하여 geometry 일치"

if should_run 5; then
    # colmap_sparse_ba를 input/output 모두로 사용 (in-place 업데이트)
    $COLMAP point_triangulator \
        --database_path "$DATA/database.db" \
        --image_path    "$DATA/images" \
        --input_path    "$DATA/colmap_sparse_ba" \
        --output_path   "$DATA/colmap_sparse_ba" \
        --clear_points  1

    echo "  ✔ point_triangulator_ba 완료"
fi

echo "  [검증]"
check_colmap_model "ba_after_tri" "$DATA/colmap_sparse_ba"

# ─── Step 6: make_meta_data.py ────────────────────────────────────────────────
step_header 6 "make_meta_data.py → kist_curve/meta_data.json"
echo "  변경점: colmap_sparse_ba의 cameras.bin/images.bin 읽어서 재생성"
echo "  이 단계는 kist_load.py를 사용"

if should_run 6; then
    $PYTHON "$SCRIPTS/kist_load.py"
    echo "  ✔ make_meta_data.py 완료"
fi

echo "  [검증]"
$PYTHON - <<'PYEOF'
import json, numpy as np

with open('/home/ms/260308-KIST-Videos/kist_curve/meta_data.json') as f:
    d = json.load(f)
frames = d['frames']
print(f"  총 frames: {len(frames)}")

seen = set()
for fr in frames:
    cam = fr['rgb_path'].split('/')[-2]
    if cam not in seen:
        seen.add(cam)
        K = np.array(fr['intrinsics'])
        print(f"  {cam}: fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    if len(seen) == 6: break

# frozen 확인
front = [f for f in frames if '/CAM_FRONT/' in f['rgb_path'] and 'LEFT' not in f['rgb_path'] and 'RIGHT' not in f['rgb_path']]
positions = np.array([np.array(f['camtoworld'])[:3,3] for f in front])
diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
print(f"  CAM_FRONT frozen(dist<0.001): {np.sum(diffs<0.001)}/{len(diffs)}  mean={diffs.mean():.4f}m")
PYEOF

# ─── 최종 요약 ────────────────────────────────────────────────────────────────
echo ""
echo "████████████████████████████████████████████████████████████████████"
echo "  COLMAP 재구성 완료"
echo "  다음 단계: bash run.sh  (Step 2: InverseForm semantics 부터)"
echo "  단, semantics/masks가 이미 있으면 run.sh가 자동으로 skip함"
echo "████████████████████████████████████████████████████████████████████"
