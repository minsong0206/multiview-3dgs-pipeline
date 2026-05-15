#!/bin/bash
# 이미지 파일명을 1-based → 0-based로 rename
#
# 대상:
#   1) chunks/chunk_00~04 / images / CAM_* : chunk 내 상대 0-based
#   2) kist_all_full/images / CAM_*        : 전체 0-based (1~2788 → 0~2787)
#
# 사용법:
#   dry-run:  bash rename_images.sh --dry-run
#   실제 실행: bash rename_images.sh

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[DRY-RUN MODE] 실제 rename 없이 출력만 합니다."
fi

BASE=/home/ms/260308-KIST-Videos/KIST_ALL_FULL
CAMERAS=(CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT)

rename_dir() {
    local dir="$1"
    local offset="$2"   # 기존 시작 번호 (이 값을 빼서 0-based로)

    [[ -d "$dir" ]] || { echo "  SKIP (not found): $dir"; return; }

    local files
    mapfile -t files < <(ls "$dir"/*.jpg 2>/dev/null | sort)
    local count=${#files[@]}
    echo "  $dir  ($count files, offset=$offset)"

    # 충돌 방지: 역순으로 rename (큰 번호 먼저)
    for (( i=${#files[@]}-1; i>=0; i-- )); do
        local src="${files[$i]}"
        local fname
        fname=$(basename "$src" .jpg)
        local new_idx=$(( 10#$fname - offset ))
        local dst
        dst="$(dirname "$src")/$(printf '%06d' "$new_idx").jpg"

        if [[ "$src" == "$dst" ]]; then
            continue
        fi

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "    $(basename "$src") → $(basename "$dst")"
        else
            mv "$src" "$dst"
        fi
    done
    echo "  ✔ done"
}

# ── 1) chunks/chunk_00~04 ──────────────────────────────────────
declare -A CHUNK_OFFSET=(
    [chunk_00]=1
    [chunk_01]=526
    [chunk_02]=988
    [chunk_03]=1451
    [chunk_04]=1913
)

echo ""
echo "════════════════════════════════════════════════════════"
echo "  chunks rename (chunk-relative 0-based)"
echo "════════════════════════════════════════════════════════"

for chunk in chunk_00 chunk_01 chunk_02 chunk_03 chunk_04; do
    offset=${CHUNK_OFFSET[$chunk]}
    echo ""
    echo "── $chunk (offset=$offset) ──"
    for cam in "${CAMERAS[@]}"; do
        rename_dir "${BASE}/chunks/${chunk}/images/${cam}" "$offset"
    done
done

# ── 2) kist_all_full/images ────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  kist_all_full rename (global 0-based, offset=1)"
echo "════════════════════════════════════════════════════════"

for cam in "${CAMERAS[@]}"; do
    rename_dir "${BASE}/kist_all_full/images/${cam}" 1
done

echo ""
echo "████████████████████████████████████████████████████████"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "  DRY-RUN 완료. 실제 실행: bash rename_images.sh"
else
    echo "  DONE: 모든 이미지 rename 완료"
fi
echo "████████████████████████████████████████████████████████"
