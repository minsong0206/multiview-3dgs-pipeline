#!/bin/bash
# KIST kist_curve 데이터셋 HUGSIM 전처리 파이프라인
# Docker 환경(ganing/hugsimin:full) 기준
#
# Docker 마운트:
#   /home/ms/HUGSIM_N/HUGSIM       → /workspace
#   /home/ms/260308-KIST-Videos    → /data
#
# 실행 방법 (Docker 내부에서):
#   cd /data && bash run.sh
#
# 또는 호스트에서 직접 실행:
#   docker exec -it interesting_edison bash /data/run.sh

set -euo pipefail

# ─────────────────────────────────────────────────────────────
#  환경 변수
# ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_JIT=0   # torch.jit.script segfault workaround (torch 2.4.1+cu118)
export PYTHONPATH="/workspace/data:${PYTHONPATH:-}"

# ────────────────────────────────────────e─────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
OUT=/data/KIST_STR_ALL/kist_straight_recon                              # 출력 디렉토리 (Docker 내부 경로)
HUGSIM_DATA=/workspace/data                                      # HUGSIM data/ 디렉토리
INVERSEFORM=/workspace/data/InverseForm                          # InverseForm 디렉토리
CHECKPOINT=/data/hrnet48_OCR_HMS_IF_checkpoint.pth
N_FRAMES=180
CAM_FPS=12.5

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

step() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  STEP: $*"
    echo "════════════════════════════════════════════════════════"
}

run_py() {
    local desc="$1"; shift
    echo "▶ [Python] ${desc}"
    echo "  CMD: python3 $*"
    python3 "$@"
    local code=$?
    if [[ ${code} -ne 0 ]]; then
        echo ""
        echo "✖ FAILED: ${desc}  (exit ${code})"
        echo "  CMD: python3 $*"
        exit ${code}
    fi
    echo "✔ OK: ${desc}"
}

assert_file() {
    local path="$1"
    local desc="${2:-${path}}"
    if [[ ! -f "${path}" ]]; then
        echo "✖ VALIDATION FAILED: file not found: ${path}  [${desc}]"
        exit 1
    fi
    if [[ ! -s "${path}" ]]; then
        echo "✖ VALIDATION FAILED: file is empty: ${path}  [${desc}]"
        exit 1
    fi
    echo "  ✔ exists & non-empty: ${path}"
}

assert_dir() {
    local path="$1"
    local desc="${2:-${path}}"
    if [[ ! -d "${path}" ]]; then
        echo "✖ VALIDATION FAILED: directory not found: ${path}  [${desc}]"
        exit 1
    fi
    local count
    count=$(ls -1 "${path}" 2>/dev/null | wc -l)
    if [[ ${count} -eq 0 ]]; then
        echo "✖ VALIDATION FAILED: directory is empty: ${path}  [${desc}]"
        exit 1
    fi
    echo "  ✔ directory OK (${count} entries): ${path}"
}

# ─────────────────────────────────────────────────────────────
#  전제 조건 확인
# ─────────────────────────────────────────────────────────────
echo ""
echo "████████████████████████████████████████████████████████"
echo "  KIST kist_curve HUGSIM preprocessing pipeline"
echo "  OUT: ${OUT}"
echo "████████████████████████████████████████████████████████"

# ── Step 0: meta_data.json 생성 + images 링크 ──────────────────
step "0  kist_load.py  →  ${OUT}/meta_data.json"
mkdir -p "${OUT}"

if [[ -f "${OUT}/meta_data.json" ]]; then
    echo "  ⏭  meta_data.json already exists, skipping"
else
    run_py "Generate meta_data.json (6-cam)" \
        /data/py/kist_load.py
fi

# # images 디렉토리: kist_curve_all/images 를 심볼릭 링크
# if [[ ! -e "${OUT}/images" ]]; then
#     ln -sf /data/KIST_CURVE_ALL/kist_curve_all/images "${OUT}/images"
#     echo "  ✔ symlinked images → kist_curve_all/images"
# else
#     echo "  ⏭  images symlink already exists, skipping"
# fi

assert_file "${OUT}/meta_data.json"   "meta_data.json (kist_load.py 완료 필요)"
assert_dir  "${OUT}/images"           "images/ 디렉토리"
assert_file "${CHECKPOINT}"           "InverseForm checkpoint"

# # merge_depth_ground.py 가 front_info.json 을 OUT/ 에서 읽음
# # kist_curve_front 에서 복사 (없을 경우에만)
# if [[ ! -f "${OUT}/front_info.json" ]]; then
#     cp /data/KIST_CURVE/kist_curve_front/front_info.json "${OUT}/front_info.json"
#     echo "  ✔ copied front_info.json to ${OUT}"
# fi


# ══════════════════════════════════════════════════════════════
#  [COLMAP 전처리 — 호스트에서 수동 실행, 이미 완료됨]
#
#  Step 0: GPS prior 생성  (호스트에서)
#    python3 /home/ms/260308-KIST-Videos/make_prior_curve.py
#    → kist_curve/prior/{cameras.txt, images.txt, points3D.txt}
#
#  Step 1a: COLMAP feature extraction + matching  (호스트에서)
#    colmap feature_extractor \
#        --database_path ${HOST_OUT}/database.db \
#        --image_path    ${HOST_OUT}/images \
#        --ImageReader.camera_model OPENCV \
#        --ImageReader.single_camera_per_folder 1 \
#        --SiftExtraction.use_gpu 1
#
#    colmap exhaustive_matcher \
#        --database_path ${HOST_OUT}/database.db \
#        --SiftMatching.use_gpu 1
#
#  Step 1b: Triangulation with GPS prior  (호스트에서)
#    colmap point_triangulator \
#        --database_path   ${HOST_OUT}/database.db \
#        --image_path      ${HOST_OUT}/images \
#        --input_path      ${HOST_OUT}/prior \
#        --output_path     ${HOST_OUT}/colmap_sparse_tri \
#        --Mapper.ba_refine_focal_length 0 \
#        --Mapper.ba_refine_extra_params 0
#
#  Step 1c: Bundle Adjustment  (호스트에서)
#    colmap bundle_adjuster \
#        --input_path  ${HOST_OUT}/colmap_sparse_tri \
#        --output_path ${HOST_OUT}/colmap_sparse_ba \
#        --BundleAdjustment.refine_focal_length 1 \
#        --BundleAdjustment.refine_extra_params 1 \
#        --BundleAdjustment.refine_extrinsics 1
#
#  Step 1d: BA 결과 TXT 변환  (호스트에서)
#    mkdir -p /tmp/kist_ba_txt
#    colmap model_converter \
#        --input_path  ${HOST_OUT}/colmap_sparse_ba \
#        --output_path /tmp/kist_ba_txt \
#        --output_type TXT
#
#  Step 1e: meta_data.json 생성  (호스트에서)
#    python3 /home/ms/260308-KIST-Videos/make_meta_data.py
#    → kist_curve/meta_data.json  (1080 frames, OPENCV)
#
#  참고 시각화:
#    DISPLAY=:0 XDG_SESSION_TYPE=x11 WAYLAND_DISPLAY= \
#        python3 /home/ms/260308-KIST-Videos/vis_colmap_poses.py
# ══════════════════════════════════════════════════════════════


# ── Step 2: Semantics (InverseForm) ────────────────────────
# NOTE: --has_edge False → distance_measures_regressor.pth 로드 없이 inference만 수행
step "2  InverseForm semantics  →  ${OUT}/semantics"

set +e
cd "${INVERSEFORM}"

arr=(FRONT FRONT_LEFT FRONT_RIGHT BACK BACK_LEFT BACK_RIGHT)
port=29500
for cam in "${arr[@]}"; do
    cam_dir="CAM_${cam}"
    input_dir="${OUT}/images/${cam_dir}"
    output_dir="${OUT}/semantics/${cam_dir}"

    # 이미 완료된 카메라는 건너뜀
    existing=$(find "${output_dir}" -name "*.npy" 2>/dev/null | wc -l)
    if [[ ${existing} -ge ${N_FRAMES} ]]; then
        echo "  ⏭  ${cam_dir}: already done (${existing} files), skipping"
        port=$((port + 1))
        continue
    fi

    echo "  === ${cam_dir} ==="
    torchrun --nproc_per_node=1 --master_port=${port} validation.py \
        --input_dir  "${input_dir}" \
        --output_dir "${output_dir}" \
        --model_path "${CHECKPOINT}" \
        --arch "ocrnet.HRNet_Mscale" \
        --hrnet_base "48" \
        --has_edge False
    echo "  Done ${cam_dir}"
    port=$((port + 1))
done

set -e
assert_dir "${OUT}/semantics" "InverseForm output"
echo "✔ OK: InverseForm semantics"

cd /data


# ── Step 3: Dynamic mask ────────────────────────────────────
step "3  create_dynamic_mask.py  →  ${OUT}/masks"
run_py "Create dynamic mask" \
    "${HUGSIM_DATA}/utils/create_dynamic_mask.py" \
    --data_path "${OUT}" \
    --data_type kist

assert_dir "${OUT}/masks" "create_dynamic_mask.py output"


# ── Step 4: Depth estimation ────────────────────────────────
step "4  estimate_depth.py  →  ${OUT}/depth"
run_py "Estimate depth" \
    "${HUGSIM_DATA}/utils/estimate_depth.py" \
    --out "${OUT}"

assert_dir "${OUT}/depth" "estimate_depth.py output"

# depth .pt 파일 개수 확인
depth_count=$(find "${OUT}/depth" -name "*.pt" | wc -l)
if [[ ${depth_count} -eq 0 ]]; then
    echo "✖ VALIDATION FAILED: no .pt depth files found under ${OUT}/depth/"
    exit 1
fi
echo "  ✔ depth files OK (${depth_count} .pt files)"


# ── Step 5: Merge depth (no ground) ────────────────────────
step "5a  merge_depth_wo_ground.py  →  ${OUT}/points3d.ply"
run_py "Merge depth (no ground)" \
    "${HUGSIM_DATA}/utils/merge_depth_wo_ground.py" \
    --out "${OUT}" \
    --total 200000 \
    --datatype kist

assert_file "${OUT}/points3d.ply" "merge_depth_wo_ground.py output"


# ── Step 6: Merge depth (with ground) ──────────────────────
step "5b  merge_depth_ground.py  →  ${OUT}/ground_points3d.ply"
run_py "Merge depth (with ground)" \
    "${HUGSIM_DATA}/utils/merge_depth_ground.py" \
    --out "${OUT}" \
    --total 200000 \
    --datatype kist

assert_file "${OUT}/ground_points3d.ply" "merge_depth_ground.py output"
assert_file "${OUT}/ground_param.pkl"    "merge_depth_ground.py output"


# ─────────────────────────────────────────────────────────────
echo ""
echo "████████████████████████████████████████████████████████"
echo "  DONE: kist_curve preprocessing complete"
echo "  Output: ${OUT}"
echo "████████████████████████████████████████████████████████"
