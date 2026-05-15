#!/bin/bash
set -euo pipefail

CHUNK=${1:-chunk_03}
DATA=/data
BASE=${DATA}/KIST_ALL_FULL/chunks/${CHUNK}
IMAGES=${BASE}/images_flat
COLMAP_DIR=${BASE}/colmap
DB=${COLMAP_DIR}/database.db
RIG_CFG=${DATA}/KIST_ALL_FULL/KIST_ALL_FULL_SEQ/rig_config.json
SPARSE=${COLMAP_DIR}/sparse/0

mkdir -p "${COLMAP_DIR}" "${COLMAP_DIR}/sparse"
LOG_DIR=${COLMAP_DIR}/logs
mkdir -p "${LOG_DIR}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 1: Feature Extractor"
echo "════════════════════════════════════════════════════════"
rm -f "${DB}" "${DB}-shm" "${DB}-wal"
colmap feature_extractor \
    --database_path "${DB}" \
    --image_path    "${IMAGES}" \
    --ImageReader.camera_model SIMPLE_RADIAL \
    --ImageReader.single_camera_per_folder 1 \
    --FeatureExtraction.use_gpu 1

echo ""
echo "  검증..."
python3 << 'PYEOF'
import sqlite3
db = sqlite3.connect('/data/KIST_ALL_FULL/chunks/chunk_03/colmap/database.db')
n_img = db.execute('SELECT COUNT(*) FROM images').fetchone()[0]
n_kp = db.execute('SELECT COUNT(*) FROM keypoints').fetchone()[0]
print(f"  ✓ Images: {n_img}  Keypoints: {n_kp}")
if n_img > 0:
    print(f"    Avg: {n_kp/n_img:.1f} per image")
db.close()
PYEOF

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 1.5: RIG Configurator"
echo "════════════════════════════════════════════════════════"
colmap rig_configurator \
    --database_path   "${DB}" \
    --rig_config_path "${RIG_CFG}" \
    --input_path /data/KIST_CURVE_ALL/kist_curve_all_exhaustive/sparse_with_rig/0

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 2: Sequential Matcher"
echo "════════════════════════════════════════════════════════"
colmap sequential_matcher \
    --database_path                        "${DB}" \
    --SequentialMatching.overlap           20 \
    --SequentialMatching.quadratic_overlap  0 \
    --SequentialMatching.expand_rig_images  1 \
    --FeatureMatching.use_gpu                  1 \
    --FeatureMatching.guided_matching          1

echo ""
echo "  검증..."
python3 << 'PYEOF'
import sqlite3
db = sqlite3.connect('/data/KIST_ALL_FULL/chunks/chunk_03/colmap/database.db')
n_matches = db.execute('SELECT COUNT(*) FROM matches').fetchone()[0]
n_two_view = db.execute('SELECT COUNT(*) FROM two_view_geometries').fetchone()[0]
print(f"  ✓ Matches: {n_matches}  Two-View: {n_two_view}")
db.close()
PYEOF

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 3: Mapper (GPU 사용) ⭐"
echo "════════════════════════════════════════════════════════"
rm -rf "${SPARSE}"
mkdir -p "${SPARSE}"

echo ""
echo "  GPU 확인:"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -1

echo ""
echo "  Mapper 실행 중..."
colmap mapper \
    --database_path                   "${DB}" \
    --image_path                      "${IMAGES}" \
    --output_path                     "${COLMAP_DIR}/sparse" \
    --Mapper.ba_refine_focal_length    1 \
    --Mapper.ba_refine_principal_point 0 \
    --Mapper.ba_refine_extra_params    1 \
    --Mapper.ba_use_gpu                1 \
    --Mapper.ba_gpu_index              0 \
    --Mapper.fix_image_to_rig          1 \
    --Mapper.estimate_relative_affine_transform_for_rig 0

echo ""
echo "  최종 결과:"
colmap model_analyzer --path "${SPARSE}"

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✓ DONE: Mapper 완료"
echo "════════════════════════════════════════════════════════"
echo ""
echo "결과: ${SPARSE}"
echo ""
