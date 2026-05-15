#!/bin/bash
# KIST_ALL_FULL chunk HUGSIM preprocessing pipeline
# Uses model_aligner output (0_aligner) — no extra scale factor.
#
# Run inside Docker (ganing/hugsimin:full):
#   bash /data/run_chunk.sh chunk_03

set -euo pipefail

# ─────────────────────────────────────────────────────────────
#  Args
# ─────────────────────────────────────────────────────────────
CHUNK=${1:-chunk_03}

# ─────────────────────────────────────────────────────────────
#  Per-chunk frame counts (for semantics skip check)
# ─────────────────────────────────────────────────────────────
case "${CHUNK}" in
    chunk_00) N_FRAMES=525  ;;
    chunk_01) N_FRAMES=462  ;;
    chunk_02) N_FRAMES=463  ;;
    chunk_03) N_FRAMES=462  ;;
    chunk_04) N_FRAMES=876  ;;
    *) echo "✖ Unknown chunk: ${CHUNK}"; exit 1 ;;
esac

# ─────────────────────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_JIT=0
export PYTHONPATH="/workspace/data:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────
CHUNK_BASE=/data/KIST_ALL_FULL/chunks/${CHUNK}
COLMAP_ALIGNED=${CHUNK_BASE}/colmap/sparse/0_aligner   # model_aligner output
IMAGES_FLAT=${CHUNK_BASE}/images_flat                  # real JPEG files, no symlinks
OUT=${CHUNK_BASE}/recon
HUGSIM_DATA=/workspace/data
INVERSEFORM=/workspace/data/InverseForm
CHECKPOINT=/data/hrnet48_OCR_HMS_IF_checkpoint.pth
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
echo "  KIST_ALL_FULL HUGSIM preprocessing  [${CHUNK}]"
echo "  COLMAP aligned : ${COLMAP_ALIGNED}"
echo "  OUT            : ${OUT}"
echo "  N_FRAMES       : ${N_FRAMES}"
echo "  START          : ${T_START_STR}"
echo "████████████████████████████████████████████████████████"

# ─────────────────────────────────────────────────────────────
#  Precondition checks
# ─────────────────────────────────────────────────────────────
assert_dir  "${COLMAP_ALIGNED}"             "model_aligner output (run model_aligner first)"
assert_file "${COLMAP_ALIGNED}/images.bin"  "images.bin in 0_aligner"
assert_file "${COLMAP_ALIGNED}/cameras.bin" "cameras.bin in 0_aligner"
assert_file "${CHECKPOINT}"                 "InverseForm checkpoint"
assert_dir  "${IMAGES_FLAT}"                "images_flat directory (real JPEGs)"


# ── Step 0: meta_data.json ─────────────────────────────────
step "0  make_meta_data_chunk.py  →  meta_data.json"
mkdir -p "${OUT}"

# Reads from 0_aligner. Scale is already metric — no COLMAP_SCALE applied.
run_py "Generate meta_data.json" \
    /data/py/make_meta_data_chunk.py \
    --chunk      "${CHUNK}" \
    --colmap_path "${COLMAP_ALIGNED}" \
    --out_dir    "${OUT}"

assert_file "${OUT}/meta_data.json" "meta_data.json"


# ── Step 0b: images_flat → OUT/images ─────────────────────
step "0b  link images_flat  →  ${OUT}/images"
# Use images_flat (real JPEGs) to avoid Docker symlink resolution errors.
if [[ ! -e "${OUT}/images" ]]; then
    ln -sf "${IMAGES_FLAT}" "${OUT}/images"
    echo "  ✔ linked: ${OUT}/images -> ${IMAGES_FLAT}"
else
    echo "  ⏭  already exists: ${OUT}/images"
fi
assert_dir "${OUT}/images" "OUT/images -> images_flat"


# ── Step 0c: sparse_ba.ply from 0_aligner ─────────────────
step "0c  colmap model_converter  →  ${OUT}/sparse_ba.ply"
echo "▶ [colmap] model_converter (input: 0_aligner)"
colmap model_converter \
    --input_path  "${COLMAP_ALIGNED}" \
    --output_path "${OUT}/sparse_ba.ply" \
    --output_type PLY
code=$?
if [[ ${code} -ne 0 ]]; then
    echo "✖ FAILED: colmap model_converter  (exit ${code})"
    exit ${code}
fi
assert_file "${OUT}/sparse_ba.ply" "sparse_ba.ply"
echo "✔ OK: colmap model_converter"


# ── Step 1: Semantics (InverseForm) ────────────────────────
step "1  InverseForm semantics  →  ${OUT}/semantics"
set +o pipefail
cd "${INVERSEFORM}"

for cam in FRONT FRONT_LEFT FRONT_RIGHT BACK BACK_LEFT BACK_RIGHT; do
    cam_dir="CAM_${cam}"
    input_dir="${OUT}/images/${cam_dir}"
    output_dir="${OUT}/semantics/${cam_dir}"

    existing=$(find "${output_dir}" -name "*.png" 2>/dev/null | wc -l)
    # Each frame produces 2 files (seg + edge), so threshold = N_FRAMES * 2
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
T_END_STR=$(date "+%Y-%m-%d %H:%M:%S")
ELAPSED=$((T_END - T_START))
ELAPSED_STR=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo ""
echo "████████████████████████████████████████████████████████"
echo "  DONE: ${CHUNK}"
echo "  Output : ${OUT}"
echo "  Start  : ${T_START_STR}"
echo "  Finish : ${T_END_STR}"
echo "  Elapsed: ${ELAPSED_STR}  (${ELAPSED}s)"
echo "████████████████████████████████████████████████████████"
