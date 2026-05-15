#!/bin/bash
# KIST_ALL_FULL chunk HUGSIM 학습 스크립트
# 사용법:
#   bash train_chunk.sh chunk_03               # 처음부터
#   bash train_chunk.sh chunk_03 30000         # iter 30000 체크포인트에서 resume
#
# Docker 실행 방법:
#   docker exec -it <container> bash /data/train_chunk.sh chunk_03
#   docker exec -it <container> bash /data/train_chunk.sh chunk_03 30000

set -euo pipefail

CHUNK=${1:-chunk_03}
RESUME_ITER=${2:-}   # optional: e.g. 30000

# ─────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────
SOURCE_PATH=/data/KIST_ALL_FULL/chunks/${CHUNK}/recon
MODEL_PATH=/data/KIST_ALL_FULL/chunks/${CHUNK}/model
DATA_CFG=/workspace/configs/kist.yaml
GROUND_CKPT=${MODEL_PATH}/ckpts/ground_chkpnt30000.pth

T_START=$(date +%s)
T_START_STR=$(date "+%Y-%m-%d %H:%M:%S")

# Resume checkpoint path (if RESUME_ITER set)
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
echo "  HUGSIM train  [${CHUNK}]"
echo "  SOURCE: ${SOURCE_PATH}"
echo "  MODEL : ${MODEL_PATH}"
echo "  RESUME: ${RESUME_ITER:-none (from scratch)}"
echo "  START : ${T_START_STR}"
echo "████████████████████████████████████████████████████████"

# 출력 디렉토리 생성
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
        --data_cfg "${DATA_CFG}" \
        --source_path "${SOURCE_PATH}" \
        --model_path "${MODEL_PATH}"
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
echo "  DONE: ${CHUNK} training complete"
echo "  Model  : ${MODEL_PATH}"
echo "  Start  : ${T_START_STR}"
echo "  Finish : ${T_END_STR}"
echo "  Elapsed: ${ELAPSED_STR}  (${ELAPSED}s)"
echo "████████████████████████████████████████████████████████"
