#!/bin/bash
# KIST_NEW test_chunk HUGSIM 학습 스크립트
#
# Docker 내부에서 실행:
#   bash /data/train_new.sh
#   bash /data/train_new.sh 30000   # iter 30000 체크포인트에서 resume

set -euo pipefail

RESUME_ITER=${1:-}

# ─────────────────────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────────────────────
SOURCE_PATH=/wdblack/KIST/raw_data_curve/recon_HUGSIM
MODEL_PATH=/wdblack/KIST/raw_data_curve/model
DATA_CFG=/workspace/configs/kist.yaml
GROUND_CKPT=${MODEL_PATH}/ckpts/ground_chkpnt30000.pth

# ─────────────────────────────────────────────────────────────
#  Unified log file  (shared with run_colmap.py / run_new.sh)
# ─────────────────────────────────────────────────────────────
LOG_FILE="${LOG_FILE:-/wdblack/KIST/raw_data_curve/pipeline.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

T_START=$(date +%s)
T_START_STR=$(date "+%Y-%m-%d %H:%M:%S")

if [[ -n "${RESUME_ITER}" ]]; then
    START_CKPT="${MODEL_PATH}/ckpts/chkpnt${RESUME_ITER}.pth"
    if [[ ! -f "${START_CKPT}" ]]; then
        echo "✖ Checkpoint not found: ${START_CKPT}"
        exit 1
    fi
    echo "  ✔ Resume checkpoint: ${START_CKPT}"
else
    START_CKPT=""
fi

echo ""
echo "████████████████████████████████████████████████████████"
echo "  HUGSIM train  [test_chunk]  [train_new.sh]"
echo "  SOURCE : ${SOURCE_PATH}"
echo "  MODEL  : ${MODEL_PATH}"
echo "  RESUME : ${RESUME_ITER:-none (from scratch)}"
echo "  LOG    : ${LOG_FILE}"
echo "  START  : ${T_START_STR}"
echo "████████████████████████████████████████████████████████"

mkdir -p "${MODEL_PATH}/ckpts"

# ── Step 1: train_ground.py ─────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  STEP 1: train_ground.py"
echo "════════════════════════════════════════════════════════"

if [[ -f "${GROUND_CKPT}" ]]; then
    echo "✔ Ground checkpoint already exists: ${GROUND_CKPT}"
else
    echo "⏳ Training ground model..."
    T_S1=$(date +%s)
    cd /workspace
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    python -u train_ground.py \
        --data_cfg    "${DATA_CFG}" \
        --source_path "${SOURCE_PATH}" \
        --model_path  "${MODEL_PATH}"
    T_E1=$(date +%s)
    echo "✔ OK: train_ground.py  ($(printf "%02d:%02d:%02d" $(( (T_E1-T_S1)/3600 )) $(( (T_E1-T_S1)%3600/60 )) $(( (T_E1-T_S1)%60 ))))"
fi

# ── Step 2: train.py ────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  STEP 2: train.py"
echo "════════════════════════════════════════════════════════"

T_S2=$(date +%s)
cd /workspace

TRAIN_ARGS="--data_cfg ${DATA_CFG} --source_path ${SOURCE_PATH} --model_path ${MODEL_PATH}"
if [[ -n "${START_CKPT}" ]]; then
    TRAIN_ARGS="${TRAIN_ARGS} --start_checkpoint ${START_CKPT}"
fi

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u train.py ${TRAIN_ARGS}

T_E2=$(date +%s)
echo "✔ OK: train.py  ($(printf "%02d:%02d:%02d" $(( (T_E2-T_S2)/3600 )) $(( (T_E2-T_S2)%3600/60 )) $(( (T_E2-T_S2)%60 ))))"

T_END=$(date +%s)
T_END_STR=$(date "+%Y-%m-%d %H:%M:%S")
ELAPSED=$((T_END - T_START))
ELAPSED_STR=$(printf "%02d:%02d:%02d" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

echo ""
echo "████████████████████████████████████████████████████████"
echo "  DONE: test_chunk training complete"
echo "  Model  : ${MODEL_PATH}"
echo "  Start  : ${T_START_STR}"
echo "  Finish : ${T_END_STR}"
echo "  Elapsed: ${ELAPSED_STR}  (${ELAPSED}s)"
echo "████████████████████████████████████████████████████████"
