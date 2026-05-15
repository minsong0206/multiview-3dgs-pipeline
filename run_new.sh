#!/bin/bash
# KIST_NEW test_chunk HUGSIM 전처리 파이프라인
#
# Docker 마운트:
#   /home/ms/260308-KIST-Videos  → /data
#   /home/ms/HUGSIM_N/HUGSIM     → /workspace
#
# Docker 실행:
#   docker run -it --gpus all \
#     -v /home/ms/260308-KIST-Videos:/data \
#     -v /home/ms/HUGSIM_N/HUGSIM:/workspace \
#     --name hugsim_train ganing/hugsimin:full bash
#
# Docker 내부에서 실행:
#   bash /data/run_new.sh

set -euo pipefail

# ─────────────────────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_JIT=0
export PYTHONPATH="/workspace/data:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────
SRC=/wdblack/KIST/raw_data_curve/recon
IMAGES=/wdblack/KIST/raw_data_curve/images
OUT=/wdblack/KIST/raw_data_curve/recon_HUGSIM
COLMAP_ALIGNED=${SRC}/colmap/sparse/0_aligner
HUGSIM_DATA=/workspace/data
INVERSEFORM=/workspace/data/InverseForm
CHECKPOINT=/data/hrnet48_OCR_HMS_IF_checkpoint.pth
N_FRAMES=462

# ─────────────────────────────────────────────────────────────
#  Unified log file  (shared with run_colmap.py / train_new.sh)
# ─────────────────────────────────────────────────────────────
LOG_FILE="${LOG_FILE:-/wdblack/KIST/raw_data_curve/pipeline.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
# Redirect all stdout+stderr through tee so every line goes to log
exec > >(tee -a "${LOG_FILE}") 2>&1

# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────
_STEP_START=0
step() {
    local now=$(date +%s)
    if [[ ${_STEP_START} -ne 0 ]]; then
        local elapsed=$(( now - _STEP_START ))
        echo "  ⏱  $(printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))"
    fi
    _STEP_START=${now}
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  STEP: $*"
    echo "  TIME: $(date "+%H:%M:%S")"
    echo "════════════════════════════════════════════════════════"
}

run_py() {
    local desc="$1"; shift
    echo "▶ [Python] ${desc}"
    python3 "$@"
    local code=$?
    if [[ ${code} -ne 0 ]]; then
        echo "✖ FAILED: ${desc}  (exit ${code})"
        exit ${code}
    fi
    echo "✔ OK: ${desc}"
}

assert_file() {
    local path="$1" desc="${2:-$1}"
    if [[ ! -f "${path}" ]] || [[ ! -s "${path}" ]]; then
        echo "✖ MISSING or EMPTY: ${path}  [${desc}]"
        exit 1
    fi
    echo "  ✔ exists: ${path}"
}

assert_dir() {
    local path="$1" desc="${2:-$1}"
    if [[ ! -d "${path}" ]]; then
        echo "✖ MISSING directory: ${path}  [${desc}]"
        exit 1
    fi
    local count; count=$(ls -1 "${path}" 2>/dev/null | wc -l)
    if [[ ${count} -eq 0 ]]; then
        echo "✖ EMPTY directory: ${path}  [${desc}]"
        exit 1
    fi
    echo "  ✔ directory OK (${count} entries): ${path}"
}

# ─────────────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────────────
T_START=$(date +%s)
T_START_STR=$(date "+%Y-%m-%d %H:%M:%S")

echo ""
echo "████████████████████████████████████████████████████████"
echo "  KIST_NEW test_chunk HUGSIM preprocessing  [run_new.sh]"
echo "  SRC      : ${SRC}"
echo "  OUT      : ${OUT}"
echo "  N_FRAMES : ${N_FRAMES}"
echo "  LOG      : ${LOG_FILE}"
echo "  START    : ${T_START_STR}"
echo "████████████████████████████████████████████████████████"

# ─────────────────────────────────────────────────────────────
#  Precondition checks
# ─────────────────────────────────────────────────────────────
assert_file "${SRC}/meta_data.json"  "meta_data.json"
assert_dir  "${IMAGES}"              "images/"
assert_file "${SRC}/sparse_ba.ply"   "sparse_ba.ply"
assert_file "${CHECKPOINT}"          "InverseForm checkpoint"

# ── Step 0: recon_HUGSIM 디렉토리 준비 ─────────────────────
mkdir -p "${OUT}"
for f in meta_data.json sparse_ba.ply; do
    if [[ ! -f "${OUT}/${f}" ]]; then
        cp "${SRC}/${f}" "${OUT}/${f}"
        echo "  ✔ copied: ${f}"
    else
        echo "  ⏭  already exists: ${OUT}/${f}"
    fi
done

if [[ ! -e "${OUT}/images" ]]; then
    ln -sf "${IMAGES}" "${OUT}/images"
    echo "  ✔ symlinked: images → ${IMAGES}"
else
    echo "  ⏭  already exists: ${OUT}/images"
fi
assert_file "${OUT}/meta_data.json" "meta_data.json (OUT)"
assert_file "${OUT}/sparse_ba.ply"  "sparse_ba.ply (OUT)"


# ── Step 1: Semantics (InverseForm) ────────────────────────
step "1  InverseForm semantics  →  ${OUT}/semantics"
set +o pipefail
cd "${INVERSEFORM}"

for cam in FRONT FRONT_LEFT FRONT_RIGHT BACK BACK_LEFT BACK_RIGHT; do
    cam_dir="CAM_${cam}"
    input_dir="${IMAGES}/${cam_dir}"
    output_dir="${OUT}/semantics/${cam_dir}"

    existing=$(find "${output_dir}" -name "*.png" 2>/dev/null | wc -l)
    if [[ ${existing} -ge $((N_FRAMES * 2)) ]]; then
        echo "  ⏭  ${cam_dir}: already done (${existing} files), skipping"
        continue
    fi

    echo "  === ${cam_dir} (existing: ${existing} / needed: $((N_FRAMES * 2))) ==="
    set +e
    torchrun --nproc_per_node=1 validation.py \
        --input_dir  "${input_dir}" \
        --output_dir "${output_dir}" \
        --model_path "${CHECKPOINT}" \
        --arch       "ocrnet.HRNet_Mscale" \
        --hrnet_base "48" \
        --has_edge   False
    code=$?
    set -e
    if [[ ${code} -ne 0 ]]; then
        echo "✖ FAILED: ${cam_dir} semantics  (exit ${code})"
        exit ${code}
    fi
    echo "  ✔ Done: ${cam_dir}"
done
set -o pipefail

assert_dir "${OUT}/semantics" "InverseForm output"
echo "✔ OK: InverseForm semantics"
cd /data


# ── Step 2: Dynamic mask ────────────────────────────────────
step "2  create_dynamic_mask.py  →  ${OUT}/masks"
run_py "Create dynamic mask" \
    "${HUGSIM_DATA}/utils/create_dynamic_mask.py" \
    --data_path "${OUT}" \
    --data_type kist
assert_dir "${OUT}/masks" "masks/"


# ── Step 3: Depth estimation ────────────────────────────────
step "3  estimate_depth.py  →  ${OUT}/depth"
run_py "Estimate depth" \
    "${HUGSIM_DATA}/utils/estimate_depth.py" \
    --out "${OUT}"
assert_dir "${OUT}/depth" "depth/"

depth_count=$(find "${OUT}/depth" -name "*.pt" | wc -l)
if [[ ${depth_count} -eq 0 ]]; then
    echo "✖ No .pt depth files found in ${OUT}/depth"
    exit 1
fi
echo "  ✔ depth files: ${depth_count} .pt files"


# ── Step 4a: Merge depth (no ground) ───────────────────────
step "4a  merge_depth_wo_ground.py  →  ${OUT}/points3d.ply"
run_py "Merge depth (no ground)" \
    "${HUGSIM_DATA}/utils/merge_depth_wo_ground.py" \
    --out   "${OUT}" \
    --total 200000 \
    --datatype kist
assert_file "${OUT}/points3d.ply" "points3d.ply"


# ── Step 4b: Merge depth (with ground) ─────────────────────
step "4b  merge_depth_ground.py  →  ${OUT}/ground_points3d.ply"
run_py "Merge depth (with ground)" \
    "${HUGSIM_DATA}/utils/merge_depth_ground.py" \
    --out   "${OUT}" \
    --total 200000 \
    --datatype kist
assert_file "${OUT}/ground_points3d.ply" "ground_points3d.ply"
assert_file "${OUT}/ground_param.pkl"    "ground_param.pkl"


# ─────────────────────────────────────────────────────────────
T_END=$(date +%s)
ELAPSED=$((T_END - T_START))
ELAPSED_STR=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo ""
echo "████████████████████████████████████████████████████████"
echo "  DONE: test_chunk preprocessing complete"
echo "  Output : ${OUT}"
echo "  Elapsed: ${ELAPSED_STR}"
echo "████████████████████████████████████████████████████████"
