#!/bin/bash
# KIST_ALL_FULL chunk COLMAP 파이프라인 (interactive, Docker 내부에서 실행)
#
# 실행 방법:
#   1) Docker 진입:
#      docker run --gpus '"device=1"' -it --rm \
#        -v /home/ms/260308-KIST-Videos:/data \
#        colmap_cudss:latest bash
#
#   2) Docker 안에서:
#      bash /data/colmap_chunk.sh chunk_00
#
# 각 단계 완료 후 결과 출력 → "yes" 입력 시 다음 단계 진행

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

CHUNK=${1:-chunk_00}

# ─── 경로 (Docker 내부 기준) ─────────────────────────────────────────────────
DATA=/data
BASE=${DATA}/KIST_ALL_FULL/chunks/${CHUNK}
IMAGES=${BASE}/images_flat
COLMAP_DIR=${BASE}/colmap
DB=${COLMAP_DIR}/database.db
RIG_CFG=${DATA}/KIST_ALL_FULL/KIST_ALL_FULL_SEQ/rig_config.json
SPARSE=${COLMAP_DIR}/sparse/0
SPARSE_TXT=${COLMAP_DIR}/sparse/0_txt
SPARSE_BA=${COLMAP_DIR}/sparse/0_ba
GPS_TXT=${COLMAP_DIR}/gps.txt

mkdir -p "${COLMAP_DIR}" "${COLMAP_DIR}/sparse"

# ─── 헬퍼 ────────────────────────────────────────────────────────────────────
step() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  STEP $1: $2"
    echo "════════════════════════════════════════════════════════"
}

confirm() {
    echo ""
    echo "▶ 결과를 확인하고 계속하려면 'yes' 입력 (중단: Ctrl+C)"
    read -r ans
    [[ "$ans" == "yes" ]] || { echo "중단합니다."; exit 0; }
}

elapsed() {
    local secs=$(( $2 - $1 ))
    printf "%dm%ds" $(( secs/60 )) $(( secs%60 ))
}

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "████████████████████████████████████████████████████████"
echo "  KIST_ALL_FULL COLMAP pipeline  [${CHUNK}]"
echo "  DB     : ${DB}"
echo "  IMAGES : ${IMAGES}"
echo "████████████████████████████████████████████████████████"

# ─── Step 1: feature_extractor ───────────────────────────────────────────────
step 1 "feature_extractor  (GPU 1)"

rm -f "${DB}" "${DB}-shm" "${DB}-wal"

T0=$SECONDS
colmap feature_extractor \
    --database_path "${DB}" \
    --image_path    "${IMAGES}" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera_per_folder 1 \
    --FeatureExtraction.use_gpu 1
echo "  ⏱  $(elapsed $T0 $SECONDS)"

echo ""
echo "── 결과 (DB cameras) ──"
python3 - "${DB}" <<'PYEOF'
import sys, sqlite3, struct
db = sqlite3.connect(sys.argv[1])
for cam in db.execute('SELECT camera_id, model, width, height, params FROM cameras').fetchall():
    n = len(cam[4]) // 8
    params = struct.unpack(f'{n}d', cam[4])
    model = {0:'SIMPLE_PINHOLE',1:'PINHOLE',2:'SIMPLE_RADIAL',3:'RADIAL',4:'OPENCV'}.get(cam[1], str(cam[1]))
    print(f"  cam {cam[0]}: {model} {cam[2]}x{cam[3]}  params={[round(p,3) for p in params]}")
n_img = db.execute('SELECT COUNT(*) FROM images').fetchone()[0]
n_kp  = db.execute('SELECT COUNT(*) FROM keypoints').fetchone()[0]
print(f"  images: {n_img}  keypoints: {n_kp}")
db.close()
PYEOF

confirm

# ─── Step 1.5: rig_configurator ──────────────────────────────────────────────
step "1.5" "rig_configurator"

T0=$SECONDS
colmap rig_configurator \
    --database_path   "${DB}" \
    --rig_config_path "${RIG_CFG}" \
    --input_path /data/KIST_CURVE_ALL/kist_curve_all_exhaustive/sparse_with_rig/0
    
echo "  ⏱  $(elapsed $T0 $SECONDS)"

echo ""
echo "── 결과 (rig 등록 확인) ──"
python3 - "${DB}" <<'PYEOF'
import sys, sqlite3
db = sqlite3.connect(sys.argv[1])
try:
    rigs    = db.execute('SELECT COUNT(*) FROM rigs').fetchone()[0]
    sensors = db.execute('SELECT COUNT(*) FROM rig_sensors').fetchone()[0]
    print(f"  rigs: {rigs}  rig_sensors: {sensors}")
except Exception as e:
    print(f"  (rig 테이블 없음: {e})")
db.close()
PYEOF

confirm

# ─── Step 2: sequential_matcher ──────────────────────────────────────────────
step 2 "sequential_matcher  (GPU 1)"

T0=$SECONDS
colmap sequential_matcher \
    --database_path                        "${DB}" \
    --SequentialMatching.overlap           20 \
    --SequentialMatching.quadratic_overlap  0 \
    --SequentialMatching.expand_rig_images  1 \
    --FeatureMatching.use_gpu                  1 \
    --FeatureMatching.guided_matching          1
echo "  ⏱  $(elapsed $T0 $SECONDS)"

echo ""
echo "── 결과 (matches) ──"
python3 - "${DB}" <<'PYEOF'
import sys, sqlite3
db = sqlite3.connect(sys.argv[1])
n_matches  = db.execute('SELECT COUNT(*) FROM matches').fetchone()[0]
n_two_view = db.execute('SELECT COUNT(*) FROM two_view_geometries').fetchone()[0]
print(f"  matches: {n_matches}  two_view_geometries: {n_two_view}")
db.close()
PYEOF

confirm

# ─── Step 3: mapper ───────────────────────────────────────────────────────────
step 3 "mapper → sparse/0  (GPU 1)"

rm -rf "${SPARSE}"
mkdir -p "${SPARSE}"

T0=$SECONDS
colmap mapper \
    --database_path                   "${DB}" \
    --image_path                      "${IMAGES}" \
    --output_path                     "${COLMAP_DIR}/sparse" \
    --Mapper.ba_refine_focal_length    1 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params    1
echo "  ⏱  $(elapsed $T0 $SECONDS)"

echo ""
echo "── 결과 (sparse/0) ──"
colmap model_analyzer --path "${SPARSE}"

confirm

confirm

# ─── Step 4: model_converter → TXT ───────────────────────────────────────────
step 4 "model_converter → sparse/0_txt"

rm -rf "${SPARSE_TXT}"
mkdir -p "${SPARSE_TXT}"

T0=$SECONDS
colmap model_converter \
    --input_path  "${SPARSE}" \
    --output_path "${SPARSE_TXT}" \
    --output_type TXT
echo "  ⏱  $(elapsed $T0 $SECONDS)"

echo ""
echo "── 결과 (images.txt 첫 5 이미지) ──"
grep -v '^#' "${SPARSE_TXT}/images.txt" | awk 'NR%2==1' | head -5

confirm

# ─── Step 5: gps.txt 생성 ────────────────────────────────────────────────────
step 5 "gps.txt 생성 (images.txt 기준, GPS 2:1 mapping)"

T0=$SECONDS
python3 - "${SPARSE_TXT}/images.txt" "${GPS_TXT}" "${CHUNK}" <<'PYEOF'
import sys, os, csv

images_txt = sys.argv[1]
out_path   = sys.argv[2]
chunk      = sys.argv[3]

GPS_CSV = '/data/RAW_DATA/6_GPS/2_Entrance-L1.csv'

CHUNK_OFFSET = {
    'chunk_00': 0,
    'chunk_01': 525,
    'chunk_02': 987,
    'chunk_03': 1450,
    'chunk_04': 1912,
}
offset = CHUNK_OFFSET[chunk]

gps_rows = []
with open(GPS_CSV, newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        gps_rows.append(row)

def get_gps(global_frame_1based):
    doc = max(0, min((global_frame_1based - 1) * 2, len(gps_rows) - 1))
    row = gps_rows[doc]
    return float(row[2]), float(row[3]), (float(row[4]) if len(row) > 4 else 0.0)

img_names = []
with open(images_txt) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'): continue
        parts = line.split()
        if len(parts) >= 10:
            img_names.append(parts[9])

with open(out_path, 'w') as f:
    for name in sorted(img_names):
        local_idx = int(os.path.splitext(os.path.basename(name))[0])  # 0-based
        global_1based = offset + local_idx + 1
        lat, lon, alt = get_gps(global_1based)
        f.write(f'{name} {lat:.10f} {lon:.10f} {alt:.4f}\n')

print(f'  gps.txt: {out_path}  ({len(img_names)} entries)')
print('  첫 3줄:')
with open(out_path) as f:
    for i, line in enumerate(f):
        if i >= 3: break
        print(f'    {line.rstrip()}')
PYEOF
echo "  ⏱  $(elapsed $T0 $SECONDS)"

confirm

# ─── Step 6: model_aligner ────────────────────────────────────────────────────
step 6 "model_aligner (GPS 정렬) → sparse/0_ba"

rm -rf "${SPARSE_BA}"
mkdir -p "${SPARSE_BA}"

T0=$SECONDS
colmap model_aligner \
    --input_path                 "${SPARSE}" \
    --output_path                "${SPARSE_BA}" \
    --ref_images_path            "${GPS_TXT}" \
    --ref_is_gps                 1 \
    --robust_alignment           1 \
    --robust_alignment_max_error 3.0
echo "  ⏱  $(elapsed $T0 $SECONDS)"

echo ""
echo "── 결과 (sparse/0_ba) ──"
colmap model_analyzer --path "${SPARSE_BA}"

echo ""
echo "████████████████████████████████████████████████████████"
echo "  DONE: ${CHUNK} COLMAP 파이프라인 완료"
echo "  BA 결과: ${COLMAP_DIR}/sparse/0_ba"
echo "  다음: bash /data/run_chunk.sh ${CHUNK}"
echo "████████████████████████████████████████████████████████"
