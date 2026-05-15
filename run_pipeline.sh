#!/usr/bin/env bash
# =============================================================================
#  KIST Full Automation Pipeline
#  video chunking → image extraction → COLMAP → preprocessing → training
#
#  USAGE
#  ─────
#    bash run_pipeline.sh --video_dir <dir> [options]
#
#  REQUIRED
#    --video_dir DIR        directory containing 0_front.MP4 … 5_left_front.MP4
#
#  CHUNK SELECTION  (choose one of the three forms)
#    --chunk_idx IDX        single chunk index       (e.g. --chunk_idx 0)
#    --chunk_range S E      inclusive index range    (e.g. --chunk_range 0 9)
#    --chunks all           every chunk in the video (default)
#
#  STAGE SELECTION  (all by default; subset = space-separated)
#    --stages "video colmap qgis preprocess train export"
#
#  RESUME / SKIP
#    --resume               skip any stage already marked "done" in status.json
#                           also checks fine-grained sub-stage files inside colmap/
#                           and prepro_hugsim/ so interrupted steps restart cleanly
#
#  OTHER OPTIONS
#    --chunk_base DIR       root for chunk output  [/media/ms/WD_BLACK_4TB/KIST/chunk]
#    --chunk_duration SEC   seconds per chunk      [10]
#    --gpu_colmap IDX       GPU for COLMAP         [1]
#    --gpu_train  IDX       GPU for training       [0]
#
#  FOLDER LAYOUT (per chunk)
#  ──────────────────────────
#    <chunk_base>/<chunkNN>/
#      images/                     extracted JPEG frames (6 cameras)
#      colmap/                     COLMAP output
#        database.db
#        sparse/0/, sparse/0_rig/, sparse/0_aligner/
#        sparse_ba.ply
#        gps.txt
#        meta_data.json            (gravity-level HUGSIM basis via prepare_flat_ground_meta.py)
#        flat_ground_meta_debug.json
#        camera_poses.geojson      (QGIS: COLMAP camera positions in WGS84)
#        gps_anchors.geojson       (QGIS: raw GPS anchors in WGS84)
#      prepro_hugsim/              HUGSIM preprocessing output
#        semantics/, masks/, depth/
#        points3d.ply, ground_points3d.ply, ground_param.pkl
#        meta_data.json, sparse_ba.ply
#        images -> ../images        (symlink)
#      train_hugsim/               HUGSIM training output
#        scene.pth
#        ckpts/
#      export_hugsim/              simulator-ready export
#        scene.pth, dynamic_*.pth
#        vis/semantic.ply, vis/points.ply, vis/scene.splat
#      pipeline.log                full combined log
#      stage_time_summary.txt      per-stage runtime table
#      status.json                 machine-readable per-stage status
#      error.log                   error detail + last 50 log lines
#
#  EXAMPLES
#  ─────────
#    # Test with chunk 0 only
#    bash run_pipeline.sh --video_dir /media/ms/WD_BLACK_4TB/KIST/raw_data --chunk_idx 0
#
#    # Run chunks 0–9
#    bash run_pipeline.sh --video_dir /media/ms/WD_BLACK_4TB/KIST/raw_data --chunk_range 0 9
#
#    # Continue chunks 10–19 (skip already-done stages)
#    bash run_pipeline.sh --video_dir /media/ms/WD_BLACK_4TB/KIST/raw_data \
#        --chunk_range 10 19 --resume
#
#    # Re-run only training on chunks 0–4
#    bash run_pipeline.sh --video_dir /media/ms/WD_BLACK_4TB/KIST/raw_data \
#        --chunk_range 0 4 --stages train
#
#  DOCKER CONTAINERS REQUIRED
#  ───────────────────────────
#    COLMAP stage:
#      docker run -d --gpus all --name colmap_cudss \
#        -v /home/ms/260308-KIST-Videos:/data \
#        -v /media/ms/WD_BLACK_4TB:/wdblack \
#        colmap_cudss:latest sleep infinity
#
#    Preprocess + Train stages:
#      docker run -d --gpus all --name hugsim_v3 \
#        -v /home/ms/260308-KIST-Videos:/data \
#        -v /home/ms/HUGSIM_N/HUGSIM:/workspace \
#        -v /media/ms/WD_BLACK_4TB:/wdblack \
#        ganing/hugsimin:v3 sleep infinity
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
#  Defaults
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_DIR=""
CHUNK_BASE="/media/ms/WD_BLACK_4TB/KIST/chunk"
CHUNK_DURATION=10           # seconds per chunk
CHUNK_IDX=""                # single chunk index
CHUNK_RANGE_START=""        # inclusive range start
CHUNK_RANGE_END=""          # inclusive range end
STAGES="all"                # "all" or space-separated subset
RESUME=0                    # 1 = skip stages already marked "done"
GPU_INDEX_COLMAP="1"
GPU_INDEX_TRAIN="0"
SAMPLE_EVERY=4              # 50 fps / 4 = 12.5 fps
WIDTH=800
HEIGHT=450

# Docker container names (must be running before pipeline starts)
DOCKER_COLMAP="colmap_cudss"
DOCKER_HUGSIM="hugsim_v3"

# Host paths
DATA_HOST="/home/ms/260308-KIST-Videos"
HUGSIM_HOST="/home/ms/HUGSIM_N/HUGSIM"
WDBLACK_HOST="/media/ms/WD_BLACK_4TB"
INVERSEFORM_CHECKPOINT="${DATA_HOST}/hrnet48_OCR_HMS_IF_checkpoint.pth"
DATA_CFG_CONTAINER="/workspace/configs/kist.yaml"

# ─────────────────────────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --video_dir)        VIDEO_DIR="$2";           shift 2 ;;
        --chunk_base)       CHUNK_BASE="$2";          shift 2 ;;
        --chunk_duration)   CHUNK_DURATION="$2";      shift 2 ;;
        --chunk_idx)        CHUNK_IDX="$2";           shift 2 ;;
        --chunk_range)      CHUNK_RANGE_START="$2";
                            CHUNK_RANGE_END="$3";     shift 3 ;;
        --stages)           STAGES="$2";              shift 2 ;;
        --resume)           RESUME=1;                 shift   ;;
        --gpu_colmap)       GPU_INDEX_COLMAP="$2";    shift 2 ;;
        --gpu_train)        GPU_INDEX_TRAIN="$2";     shift 2 ;;
        -h|--help)
            sed -n 's/^#  //p' "$0" | head -60
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -z "${VIDEO_DIR}" ]] && { echo "ERROR: --video_dir is required"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
#  Per-chunk context (populated by run_chunk_pipeline)
# ─────────────────────────────────────────────────────────────────────────────
_CHUNK=""
_CHUNK_DIR=""
_LOG_FILE=""
_ERROR_LOG=""
_STATUS_JSON=""
_TIME_SUMMARY=""
_LAST_ELAPSED=0

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
_ts() { date '+%Y-%m-%d %H:%M:%S'; }

_glog() {   # global log (before chunk context exists)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

_log() {    # per-chunk log → stdout + pipeline.log
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "${msg}"
    [[ -n "${_LOG_FILE}" ]] && echo "${msg}" >> "${_LOG_FILE}"
}

_log_error() {
    _log "ERROR: $*"
    [[ -n "${_ERROR_LOG}" ]] && \
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >> "${_ERROR_LOG}"
}

_banner() {
    local sep="════════════════════════════════════════════════════════════"
    _log ""
    _log "${sep}"
    _log "  $*"
    _log "  TIME: $(date '+%H:%M:%S')"
    _log "${sep}"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Stage selection
# ─────────────────────────────────────────────────────────────────────────────
_stage_enabled() {
    [[ "${STAGES}" == "all" ]] && return 0
    [[ " ${STAGES} " == *" $1 "* ]] && return 0
    return 1
}

# ─────────────────────────────────────────────────────────────────────────────
#  status.json helpers
#
#  Schema:
#    { "<stage>": { "status": "done|failed|running",
#                   "elapsed_s": 123,
#                   "timestamp": "...",
#                   "cmd": "..." } }
#
#  Sub-stage files (fine-grained resume inside stages):
#    colmap/.done_feature_extraction
#    colmap/.done_matching
#    colmap/.done_mapper
#    colmap/.done_aligner
#    colmap/.done_qgis
#    prepro_hugsim/.done_semantics_<CAM>
#    prepro_hugsim/.done_mask
#    prepro_hugsim/.done_depth
#    prepro_hugsim/.done_merge
#    export_hugsim/.done_export
# ─────────────────────────────────────────────────────────────────────────────
_status_get() {
    local stage="$1"
    [[ ! -f "${_STATUS_JSON}" ]] && echo "pending" && return
    python3 -c "
import json
try:
    d = json.load(open('${_STATUS_JSON}'))
    print(d.get('${stage}', {}).get('status', 'pending'))
except Exception:
    print('pending')
"
}

_status_set() {
    local stage="$1" status="$2" elapsed="${3:-0}" cmd="${4:-}"
    python3 - <<PYEOF
import json, os
path = "${_STATUS_JSON}"
d = json.load(open(path)) if os.path.exists(path) else {}
d["${stage}"] = {
    "status": "${status}",
    "elapsed_s": ${elapsed},
    "timestamp": "$(_ts)",
    "cmd": r"""${cmd}""",
}
with open(path, "w") as f:
    json.dump(d, f, indent=2)
PYEOF
}

# Mark a fine-grained sub-stage complete (touch a .done_* sentinel file)
_substage_done() {
    local sentinel="$1"
    touch "${sentinel}"
    _log "  ✔ sub-stage done: $(basename "${sentinel}")"
}

_substage_is_done() {
    [[ -f "$1" ]]
}

# ─────────────────────────────────────────────────────────────────────────────
#  stage_time_summary.txt
# ─────────────────────────────────────────────────────────────────────────────
_write_time_summary() {
    [[ ! -f "${_STATUS_JSON}" ]] && return
    python3 - <<PYEOF
import json
d = json.load(open("${_STATUS_JSON}"))
lines = ["Stage Time Summary  [${_CHUNK}]", "=" * 56]
total = 0
for stage, info in d.items():
    s = info.get("elapsed_s", 0)
    total += s
    h, r = divmod(int(s), 3600); m, sc = divmod(r, 60)
    st = info.get("status", "?")
    lines.append(f"  {stage:<26s}  {st:<8s}  {h:02d}:{m:02d}:{sc:02d}")
lines.append("-" * 56)
h, r = divmod(int(total), 3600); m, sc = divmod(r, 60)
lines.append(f"  {'TOTAL':<26s}  {'':8s}  {h:02d}:{m:02d}:{sc:02d}  ({int(total)}s)")
with open("${_TIME_SUMMARY}", "w") as f:
    f.write("\\n".join(lines) + "\\n")
PYEOF
    _log "  time summary: ${_TIME_SUMMARY}"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Output assertions
# ─────────────────────────────────────────────────────────────────────────────
_assert_file() {
    local path="$1" desc="${2:-$1}"
    if [[ ! -f "${path}" ]] || [[ ! -s "${path}" ]]; then
        _log_error "MISSING or EMPTY: ${path}  [${desc}]"
        exit 1
    fi
    _log "  ✔ ${desc}: ${path}"
}

_assert_dir() {
    local path="$1" desc="${2:-$1}" min="${3:-1}"
    if [[ ! -d "${path}" ]]; then
        _log_error "MISSING directory: ${path}  [${desc}]"
        exit 1
    fi
    local n; n=$(find "${path}" -maxdepth 1 -mindepth 1 | wc -l)
    if [[ ${n} -lt ${min} ]]; then
        _log_error "EMPTY (${n}/${min}): ${path}  [${desc}]"
        exit 1
    fi
    _log "  ✔ ${desc}: ${path}  (${n} entries)"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Run a command with timing and automatic failure handling
#  Usage: elapsed=$(_run_cmd STAGE DESC cmd arg arg ...)
#  On failure: writes error.log, updates status.json, exits 1
# ─────────────────────────────────────────────────────────────────────────────
_run_cmd() {
    local stage="$1" desc="$2"; shift 2
    _log "▶ [${stage}] ${desc}"
    _log "  CMD: $*"

    local t0; t0=$(date +%s)

    # Run command: tee output to log + stdout, capture real exit code
    local rc_file; rc_file=$(mktemp)
    { "$@"; echo $? > "${rc_file}"; } 2>&1 | tee -a "${_LOG_FILE}"
    local rc; rc=$(cat "${rc_file}"); rm -f "${rc_file}"

    local t1; t1=$(date +%s)
    # Store elapsed in global so callers don't need $() subshell
    _LAST_ELAPSED=$(( t1 - t0 ))

    if [[ "${rc}" -ne 0 ]]; then
        _log_error "FAILED: ${desc}  (exit=${rc}, elapsed=${_LAST_ELAPSED}s)"
        {
            echo ""
            echo "╔══ FAILED STAGE: ${stage} ══╗"
            echo "  chunk   : ${_CHUNK}"
            echo "  cmd     : $*"
            echo "  exit    : ${rc}"
            echo "  elapsed : ${_LAST_ELAPSED}s"
            echo "  time    : $(_ts)"
            echo "╠══ last 50 lines of pipeline.log ══╣"
            tail -50 "${_LOG_FILE}"
            echo "╚══════════════════════════════════╝"
        } >> "${_ERROR_LOG}"
        _status_set "${stage}" "failed" "${_LAST_ELAPSED}" "$*"
        _write_time_summary
        exit 1
    fi
}

_log_elapsed() {
    local s="$1" desc="$2"
    local h=$((s/3600)) m=$(( (s%3600)/60 )) sc=$((s%60))
    _log "✔ OK: ${desc}  ($(printf '%02d:%02d:%02d' ${h} ${m} ${sc}))"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Docker helpers
# ─────────────────────────────────────────────────────────────────────────────
_docker_running() {
    docker ps --filter "name=^/$1$" --format "{{.Names}}" 2>/dev/null \
        | grep -q "^$1$"
}

_check_docker() {
    local name="$1" hint="$2"
    if ! _docker_running "${name}"; then
        _log_error "Docker container '${name}' is not running."
        _log_error "  Start it with: ${hint}"
        exit 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
#  Video duration (float seconds via ffprobe)
# ─────────────────────────────────────────────────────────────────────────────
_video_duration_s() {
    python3 -c "
import subprocess, json, sys
r = subprocess.run(
    ['ffprobe', '-v', 'quiet', '-print_format', 'json',
     '-show_streams', '-select_streams', 'v:0', '$1'],
    capture_output=True, text=True)
st = json.loads(r.stdout)['streams'][0]
if 'duration' in st:
    print(float(st['duration']))
else:
    nb = int(st.get('nb_frames', 0))
    num, den = st.get('r_frame_rate','50/1').split('/')
    print(nb / (int(num)/int(den)))
"
}

# ─────────────────────────────────────────────────────────────────────────────
#  Chunk manifest  (chunk_base/chunk_manifest.json)
#  Stores {chunkNN: {start_s, end_s}} so runs with explicit --chunk_range
#  can look up time windows without re-parsing the video.
# ─────────────────────────────────────────────────────────────────────────────
_manifest_set() {
    local chunk="$1" start_s="$2" end_s="$3"
    python3 - <<PYEOF
import json, os
path = "${CHUNK_BASE}/chunk_manifest.json"
d = json.load(open(path)) if os.path.exists(path) else {}
d["${chunk}"] = {"start_s": ${start_s}, "end_s": ${end_s}}
with open(path, "w") as f:
    json.dump(d, f, indent=2)
PYEOF
}

_manifest_get() {
    local chunk="$1" key="$2" default="$3"
    local manifest="${CHUNK_BASE}/chunk_manifest.json"
    [[ ! -f "${manifest}" ]] && echo "${default}" && return
    local val
    val=$(python3 -c "
import json
d = json.load(open('${manifest}'))
v = d.get('${chunk}', {}).get('${key}')
print(v if v is not None else '${default}')
" 2>/dev/null)
    echo "${val:-${default}}"
}

# =============================================================================
#  STAGE: video
#  Runs on host (no Docker). Uses preprocess_new_video.py.
#  Output: <chunk>/images/<CAM>/*.jpg  +  <chunk>/colmap/gps.txt
# =============================================================================
run_stage_video() {
    local chunk="$1" start_s="$2" end_s="$3"
    local images_dir="${_CHUNK_DIR}/images"
    local gps_txt="${_CHUNK_DIR}/colmap/gps.txt"

    _banner "STAGE video  [${chunk}]  ${start_s}s – ${end_s}s"

    if [[ ${RESUME} -eq 1 ]] && [[ "$(_status_get video)" == "done" ]]; then
        _log "  SKIPPED (already done)"; return 0
    fi

    # preprocess_new_video.py writes images/ and gps.txt into --out_dir.
    # We target <chunk>/colmap so that gps.txt lands next to the COLMAP db.
    mkdir -p "${_CHUNK_DIR}/colmap"

    _run_cmd "video" "preprocess_new_video.py" \
        python3 "${DATA_HOST}/py/preprocess_new_video.py" \
            --video_dir    "${VIDEO_DIR}" \
            --out_dir      "${_CHUNK_DIR}" \
            --chunk_start  "${start_s}" \
            --chunk_end    "${end_s}" \
            --sample_every "${SAMPLE_EVERY}" \
            --width        "${WIDTH}" \
            --height       "${HEIGHT}"
    local elapsed=${_LAST_ELAPSED}

    # preprocess_new_video.py writes <out_dir>/gps.txt → move into colmap/
    if [[ -f "${_CHUNK_DIR}/gps.txt" ]] && [[ ! -f "${gps_txt}" ]]; then
        mv "${_CHUNK_DIR}/gps.txt" "${gps_txt}"
    fi

    _log_elapsed "${elapsed}" "video extraction"
    _assert_dir  "${images_dir}" "images/ (6 cameras)" 6
    _assert_file "${gps_txt}"    "colmap/gps.txt"

    _status_set "video" "done" "${elapsed}" \
        "preprocess_new_video.py --chunk_start ${start_s} --chunk_end ${end_s}"
    _write_time_summary
}

# =============================================================================
#  STAGE: colmap
#  Runs inside Docker container: colmap_cudss
#  Output: <chunk>/colmap/{database.db, sparse/, sparse_ba.ply, meta_data.json}
#
#  Fine-grained sub-stage sentinels (allow resume inside this stage):
#    colmap/.done_feature_extraction
#    colmap/.done_matching
#    colmap/.done_mapper
#    colmap/.done_rig_configurator
#    colmap/.done_mapper_rig
#    colmap/.done_aligner
#    colmap/.done_converter        (sparse_ba.ply)
#    colmap/.done_meta_data        (meta_data.json)
# =============================================================================
run_stage_colmap() {
    local chunk="$1"
    local colmap_dir="${_CHUNK_DIR}/colmap"

    _banner "STAGE colmap  [${chunk}]"

    if [[ ${RESUME} -eq 1 ]] && [[ "$(_status_get colmap)" == "done" ]]; then
        _log "  SKIPPED (already done)"; return 0
    fi

    _check_docker "${DOCKER_COLMAP}" \
        "docker run -d --gpus all --name ${DOCKER_COLMAP} \
-v ${DATA_HOST}:/data -v ${WDBLACK_HOST}:/wdblack colmap_cudss:latest sleep infinity"

    mkdir -p "${colmap_dir}/sparse"

    # Container-side paths  (/wdblack → WDBLACK_HOST)
    local colmap_c="/wdblack/KIST/chunk/${chunk}/colmap"
    local images_c="/wdblack/KIST/chunk/${chunk}/images"
    local db_c="${colmap_c}/database.db"
    local sparse_c="${colmap_c}/sparse"
    local gps_c="${colmap_c}/gps.txt"
    local rig_c="/data/rig_config.json"

    # Sub-stage sentinels (host paths)
    local s_feat="${colmap_dir}/.done_feature_extraction"
    local s_match="${colmap_dir}/.done_matching"
    local s_mapper="${colmap_dir}/.done_mapper"
    local s_best_sel="${colmap_dir}/.done_best_model_selected"
    local s_rig="${colmap_dir}/.done_rig_configurator"
    local s_mapper_rig="${colmap_dir}/.done_mapper_rig"
    local s_align="${colmap_dir}/.done_aligner"
    local s_ply="${colmap_dir}/.done_converter"
    local s_meta="${colmap_dir}/.done_meta_data"

    local t0; t0=$(date +%s)

    # ── 1. Feature extraction ──────────────────────────────────
    if _substage_is_done "${s_feat}"; then
        _log "  [colmap/feature_extraction] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] feature_extractor"
        docker exec "${DOCKER_COLMAP}" bash -c "
            colmap feature_extractor \
                --database_path ${db_c} \
                --image_path    ${images_c} \
                --ImageReader.camera_model             SIMPLE_RADIAL \
                --ImageReader.single_camera_per_folder 1 \
                --FeatureExtraction.use_gpu            1 \
                --FeatureExtraction.gpu_index          ${GPU_INDEX_COLMAP}
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${colmap_dir}/database.db" "database.db"
        _substage_done "${s_feat}"
    fi

    # ── 2. Exhaustive matching ─────────────────────────────────
    if _substage_is_done "${s_match}"; then
        _log "  [colmap/matching] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] exhaustive_matcher"
        docker exec "${DOCKER_COLMAP}" bash -c "
            colmap exhaustive_matcher \
                --database_path                    ${db_c} \
                --FeatureMatching.use_gpu          1 \
                --FeatureMatching.gpu_index        ${GPU_INDEX_COLMAP} \
                --FeatureMatching.guided_matching  1
        " 2>&1 | tee -a "${_LOG_FILE}"
        _substage_done "${s_match}"
    fi

    # ── 3. Mapper (initial, no rig) ────────────────────────────
    if _substage_is_done "${s_mapper}"; then
        _log "  [colmap/mapper] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] mapper (initial)"
        docker exec "${DOCKER_COLMAP}" bash -c "
            mkdir -p ${sparse_c}/0
            # Remove any auto-created rig tables so mapper runs free
            python3 -c \"
import sqlite3
db = sqlite3.connect('${db_c}')
for t in ('rigs','rig_sensors','frames','frame_data'):
    db.execute('DELETE FROM ' + t)
db.commit(); db.close()
print('  Cleared rig tables from DB')
\"
            colmap mapper \
                --database_path ${db_c} \
                --image_path    ${images_c} \
                --output_path   ${sparse_c} \
                --Mapper.ba_refine_focal_length    1 \
                --Mapper.ba_refine_principal_point 0 \
                --Mapper.ba_refine_extra_params    1 \
                --Mapper.ba_use_gpu                1 \
                --Mapper.ba_gpu_index              ${GPU_INDEX_COLMAP}
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_dir "${colmap_dir}/sparse/0" "sparse/0/ (mapper output)"
        _substage_done "${s_mapper}"
    fi

    # ── 3b. Select best sparse model (all 6 cameras registered) ──
    # Mapper may produce multiple sub-models (sparse/0, sparse/1, ...).
    # Find the one where all 6 camera streams are present, then rename
    # it to sparse/0_best so downstream stages use the correct model.
    local s_best_sel="${colmap_dir}/.done_best_model_selected"
    local best_sparse_host="${colmap_dir}/sparse/0_best"
    local best_sparse_c="${sparse_c}/0_best"

    if _substage_is_done "${s_best_sel}"; then
        _log "  [colmap/best_model] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] selecting best sparse model (all 6 cameras registered)"
        local best_model
        best_model=$(python3 - <<PYEOF
import sys, os
sys.path.insert(0, '/home/ms/HUGSIM_N/HUGSIM/data')
from colmap.colmap_reader import read_extrinsics_binary

REQUIRED = {'CAM_FRONT','CAM_BACK','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK_RIGHT'}
sparse_dir = '${colmap_dir}/sparse'

candidates = sorted(
    d for d in os.listdir(sparse_dir)
    if os.path.isdir(os.path.join(sparse_dir, d))
    and not d.startswith('0_')   # skip 0_rig / 0_aligner / 0_best
)

best = None
best_n = 0
for sub in candidates:
    images_bin = os.path.join(sparse_dir, sub, 'images.bin')
    if not os.path.exists(images_bin):
        continue
    imgs = read_extrinsics_binary(images_bin)
    cams_found = set()
    for iid, img in imgs.items():
        cam = img.name.split('/')[0] if '/' in img.name else 'CAM_FRONT'
        cams_found.add(cam)
    n = len(imgs)
    missing = REQUIRED - cams_found
    status = 'OK' if not missing else f'MISSING {missing}'
    print(f'  sparse/{sub}: {n} images, cameras={sorted(cams_found)}  [{status}]', file=sys.stderr)
    if not missing and n > best_n:
        best = sub
        best_n = n

if best is None:
    print('NONE')
    sys.exit(0)
print(best)
PYEOF
)
        if [[ -z "${best_model}" ]] || [[ "${best_model}" == "NONE" ]]; then
            _log_error "No sparse model with all 6 cameras found — COLMAP reconstruction failed"
            exit 1
        fi
        _log "  Best model: sparse/${best_model}  → renaming to sparse/0_best"
        mv "${colmap_dir}/sparse/${best_model}" "${best_sparse_host}"
        _substage_done "${s_best_sel}"
    fi

    # ── 4. rig_configurator ────────────────────────────────────
    if _substage_is_done "${s_rig}"; then
        _log "  [colmap/rig_configurator] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] rig_configurator"
        docker exec "${DOCKER_COLMAP}" bash -c "
            colmap rig_configurator \
                --database_path   ${db_c} \
                --rig_config_path ${rig_c} \
                --input_path      ${best_sparse_c}
        " 2>&1 | tee -a "${_LOG_FILE}"
        _substage_done "${s_rig}"
    fi

    # ── 5. Mapper with rig constraints ─────────────────────────
    if _substage_is_done "${s_mapper_rig}"; then
        _log "  [colmap/mapper_rig] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] mapper (rig)"
        docker exec "${DOCKER_COLMAP}" bash -c "
            mkdir -p ${sparse_c}/0_rig
            colmap mapper \
                --database_path ${db_c} \
                --image_path    ${images_c} \
                --output_path   ${sparse_c}/0_rig \
                --Mapper.ba_refine_focal_length    1 \
                --Mapper.ba_refine_principal_point 0 \
                --Mapper.ba_refine_extra_params    1 \
                --Mapper.ba_use_gpu                1 \
                --Mapper.ba_gpu_index              ${GPU_INDEX_COLMAP}
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_dir "${colmap_dir}/sparse/0_rig/0" "sparse/0_rig/0/ (mapper rig)"
        _substage_done "${s_mapper_rig}"
    fi

    # ── 6. model_aligner (GPS) ─────────────────────────────────
    if _substage_is_done "${s_align}"; then
        _log "  [colmap/aligner] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] model_aligner (GPS)"
        docker exec "${DOCKER_COLMAP}" bash -c "
            mkdir -p ${sparse_c}/0_aligner
            colmap model_aligner \
                --input_path      ${sparse_c}/0_rig/0 \
                --output_path     ${sparse_c}/0_aligner \
                --ref_images_path ${gps_c} \
                --ref_is_gps      1 \
                --alignment_type  enu \
                --alignment_max_error 3.0
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${colmap_dir}/sparse/0_aligner/images.bin"  "0_aligner/images.bin"
        _assert_file "${colmap_dir}/sparse/0_aligner/cameras.bin" "0_aligner/cameras.bin"
        _substage_done "${s_align}"
    fi

    # ── 7. model_converter → sparse_ba.ply ────────────────────
    if _substage_is_done "${s_ply}"; then
        _log "  [colmap/converter] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] model_converter → sparse_ba.ply"
        docker exec "${DOCKER_COLMAP}" bash -c "
            colmap model_converter \
                --input_path  ${sparse_c}/0_aligner \
                --output_path ${colmap_c}/sparse_ba.ply \
                --output_type PLY
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${colmap_dir}/sparse_ba.ply" "sparse_ba.ply"
        _substage_done "${s_ply}"
    fi

    # ── 8. prepare_flat_ground_meta.py ────────────────────────
    # Replaces make_meta_data_new.py. Rebuilds meta_data.json in a
    # gravity-level HUGSIM basis (HUGSIM +Y = physical down, +Z = forward
    # projected onto horizontal plane) so the ground sits at Y = +cam_height.
    if _substage_is_done "${s_meta}"; then
        _log "  [colmap/meta_data] SKIPPED (sentinel exists)"
    else
        _log "▶ [colmap] prepare_flat_ground_meta.py (gravity-level HUGSIM meta_data.json)"
        local chunk_root_c="/wdblack/KIST/chunk/${chunk}"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            python3 /data/py/prepare_flat_ground_meta.py \
                --recon-dir   ${chunk_root_c} \
                --colmap-path ${colmap_c}/sparse/0_aligner \
                --image-dir   ${chunk_root_c}/images \
                --src         ${chunk_root_c}/prepro_hugsim \
                --dst         ${colmap_c} \
                --cam-height  1.5 \
                --flatten-up
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${colmap_dir}/meta_data.json" "colmap/meta_data.json"
        _substage_done "${s_meta}"
    fi

    local t1; t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    _log_elapsed "${elapsed}" "COLMAP (all sub-stages)"

    _assert_file "${colmap_dir}/sparse_ba.ply"  "colmap/sparse_ba.ply"
    _assert_file "${colmap_dir}/meta_data.json" "colmap/meta_data.json"

    _status_set "colmap" "done" "${elapsed}" "run_colmap full pipeline"
    _write_time_summary
}

# =============================================================================
#  STAGE: qgis
#  Exports COLMAP camera poses and GPS anchors to GeoJSON for QGIS alignment check.
#  Runs inside Docker container: hugsim_v3  (needs numpy + pycolmap)
#  Reads from:  <chunk>/colmap/
#  Writes to:   <chunk>/colmap/camera_poses.geojson
#               <chunk>/colmap/gps_anchors.geojson
#
#  Fine-grained sub-stage sentinels:
#    colmap/.done_qgis
# =============================================================================
run_stage_qgis() {
    local chunk="$1"
    local colmap_dir="${_CHUNK_DIR}/colmap"
    local qgis_dir="${colmap_dir}/qgis"
    local colmap_c="/wdblack/KIST/chunk/${chunk}/colmap"
    local qgis_c="${colmap_c}/qgis"

    _banner "STAGE qgis  [${chunk}]"

    if [[ "${RESUME}" -eq 1 ]] && [[ "$(_status_get qgis)" == "done" ]]; then
        _log "  SKIPPED (already done)"; return 0
    fi

    _check_docker "${DOCKER_HUGSIM}" \
        "docker run -d --gpus all --name ${DOCKER_HUGSIM} \
-v ${DATA_HOST}:/data -v ${HUGSIM_HOST}:/workspace -v ${WDBLACK_HOST}:/wdblack \
ganing/hugsimin:v3 sleep infinity"

    local s_qgis="${colmap_dir}/.done_qgis"
    local t0; t0=$(date +%s)

    if _substage_is_done "${s_qgis}"; then
        _log "  [qgis] SKIPPED (sentinel exists)"
    else
        _log "▶ [qgis] export_qgis.py → qgis/camera_poses_${chunk}.geojson + qgis/gps_${chunk}.geojson"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            mkdir -p ${qgis_c}
            python3 /data/py/export_qgis.py \
                --recon_dir   ${colmap_c} \
                --colmap_path ${colmap_c}/sparse/0_aligner \
                --out_dir     ${qgis_c} \
                --out_prefix  ${chunk}
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${qgis_dir}/camera_poses_${chunk}.geojson" "colmap/qgis/camera_poses_${chunk}.geojson"
        _assert_file "${qgis_dir}/gps_${chunk}.geojson"          "colmap/qgis/gps_${chunk}.geojson"
        _substage_done "${s_qgis}"
    fi

    local t1; t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    _log_elapsed "${elapsed}" "qgis export"

    _log "  ── Alignment table (GPS anchor frames vs COLMAP CAM_FRONT) ──"
    _log "  Open in QGIS:"
    _log "    ${qgis_dir}/camera_poses_${chunk}.geojson"
    _log "    ${qgis_dir}/gps_${chunk}.geojson"

    _assert_file "${qgis_dir}/camera_poses_${chunk}.geojson" "colmap/qgis/camera_poses_${chunk}.geojson"

    _status_set "qgis" "done" "${elapsed}" "export_qgis.py"
    _write_time_summary
}

# =============================================================================
#  STAGE: preprocess
#  Runs inside Docker container: hugsim_v3  (ganing/hugsimin:v3)
#  Reads from:  <chunk>/colmap/  and  <chunk>/images/
#  Writes to:   <chunk>/prepro_hugsim/
#
#  Fine-grained sub-stage sentinels:
#    prepro_hugsim/.done_setup          (directory + symlinks ready)
#    prepro_hugsim/.done_semantics_<CAM>
#    prepro_hugsim/.done_mask
#    prepro_hugsim/.done_depth
#    prepro_hugsim/.done_merge_noground
#    prepro_hugsim/.done_merge_ground
# =============================================================================
run_stage_preprocess() {
    local chunk="$1" n_frames="$2"
    local prepro_dir="${_CHUNK_DIR}/prepro_hugsim"

    _banner "STAGE preprocess  [${chunk}]  n_frames=${n_frames}"

    if [[ ${RESUME} -eq 1 ]] && [[ "$(_status_get preprocess)" == "done" ]]; then
        _log "  SKIPPED (already done)"; return 0
    fi

    _check_docker "${DOCKER_HUGSIM}" \
        "docker run -d --gpus all --name ${DOCKER_HUGSIM} \
-v ${DATA_HOST}:/data -v ${HUGSIM_HOST}:/workspace -v ${WDBLACK_HOST}:/wdblack \
ganing/hugsimin:v3 sleep infinity"

    # Container-side paths
    local colmap_c="/wdblack/KIST/chunk/${chunk}/colmap"
    local images_c="/wdblack/KIST/chunk/${chunk}/images"
    local out_c="/wdblack/KIST/chunk/${chunk}/prepro_hugsim"

    local s_setup="${prepro_dir}/.done_setup"
    local s_mask="${prepro_dir}/.done_mask"
    local s_depth="${prepro_dir}/.done_depth"
    local s_merge_nog="${prepro_dir}/.done_merge_noground"
    local s_merge_g="${prepro_dir}/.done_merge_ground"

    local t0; t0=$(date +%s)

    # ── setup: create prepro_hugsim, copy files, symlink images ──
    if _substage_is_done "${s_setup}"; then
        _log "  [prepro/setup] SKIPPED (sentinel exists)"
    else
        _log "▶ [prepro] setup prepro_hugsim directory"
        # mkdir on host first so the directory is owned by the host user, not Docker root
        mkdir -p "${prepro_dir}"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            cp -n ${colmap_c}/meta_data.json ${out_c}/meta_data.json
            cp -n ${colmap_c}/sparse_ba.ply  ${out_c}/sparse_ba.ply
            if [[ ! -e ${out_c}/images ]]; then
                ln -sf ${images_c} ${out_c}/images
            fi
            echo 'setup OK'
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${prepro_dir}/meta_data.json" "prepro_hugsim/meta_data.json"
        _assert_file "${prepro_dir}/sparse_ba.ply"  "prepro_hugsim/sparse_ba.ply"
        _substage_done "${s_setup}"
    fi

    # ── InverseForm semantics (per-camera sentinel) ───────────
    local CAMS=(FRONT FRONT_LEFT FRONT_RIGHT BACK BACK_LEFT BACK_RIGHT)
    local all_sem_done=1
    for cam in "${CAMS[@]}"; do
        local s_sem="${prepro_dir}/.done_semantics_${cam}"
        if _substage_is_done "${s_sem}"; then
            _log "  [prepro/semantics/${cam}] SKIPPED (sentinel exists)"
            continue
        fi
        all_sem_done=0
        _log "▶ [prepro] InverseForm semantics: CAM_${cam}"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            set -eu
            export CUDA_VISIBLE_DEVICES=${GPU_INDEX_TRAIN}
            export PYTORCH_JIT=0
            cd /workspace/data/InverseForm
            cam_dir='CAM_${cam}'
            existing=\$(find '${out_c}/semantics/'\${cam_dir} -name '*.png' 2>/dev/null | wc -l || echo 0)
            if [[ \${existing} -ge \$(( ${n_frames} * 2 )) ]]; then
                echo \"  ⏭  \${cam_dir}: already done (\${existing} files)\"
                exit 0
            fi
            torchrun --nproc_per_node=1 validation.py \
                --input_dir  '${images_c}/'\${cam_dir} \
                --output_dir '${out_c}/semantics/'\${cam_dir} \
                --model_path /data/hrnet48_OCR_HMS_IF_checkpoint.pth \
                --arch       ocrnet.HRNet_Mscale \
                --hrnet_base 48 \
                --has_edge   False
            echo \"✔ \${cam_dir} done\"
        " 2>&1 | tee -a "${_LOG_FILE}"
        _substage_done "${s_sem}"
    done
    _assert_dir "${prepro_dir}/semantics" "prepro_hugsim/semantics"

    # ── Dynamic mask ──────────────────────────────────────────
    if _substage_is_done "${s_mask}"; then
        _log "  [prepro/mask] SKIPPED (sentinel exists)"
    else
        _log "▶ [prepro] create_dynamic_mask.py"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            export CUDA_VISIBLE_DEVICES=${GPU_INDEX_TRAIN}
            export PYTHONPATH='/workspace/data:\${PYTHONPATH:-}'
            python3 /workspace/data/utils/create_dynamic_mask.py \
                --data_path ${out_c} --data_type kist
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_dir "${prepro_dir}/masks" "prepro_hugsim/masks"
        _substage_done "${s_mask}"
    fi

    # ── Depth estimation ──────────────────────────────────────
    if _substage_is_done "${s_depth}"; then
        _log "  [prepro/depth] SKIPPED (sentinel exists)"
    else
        _log "▶ [prepro] estimate_depth.py"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            export CUDA_VISIBLE_DEVICES=${GPU_INDEX_TRAIN}
            export PYTHONPATH='/workspace/data:\${PYTHONPATH:-}'
            export MPLCONFIGDIR=/tmp/mpl_config && mkdir -p /tmp/mpl_config
            export HF_HOME=/home/user/.cache/huggingface
            export TRANSFORMERS_CACHE=/home/user/.cache/huggingface
            /workspace/.pixi/envs/default/bin/python3 /workspace/data/utils/estimate_depth.py \
                --out ${out_c}
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_dir "${prepro_dir}/depth" "prepro_hugsim/depth"
        _substage_done "${s_depth}"
    fi

    # ── Merge depth (no ground) ───────────────────────────────
    if _substage_is_done "${s_merge_nog}"; then
        _log "  [prepro/merge_noground] SKIPPED (sentinel exists)"
    else
        _log "▶ [prepro] merge_depth_wo_ground.py"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            export PYTHONPATH='/workspace/data:\${PYTHONPATH:-}'
            export MPLCONFIGDIR=/tmp/mpl_config
            /workspace/.pixi/envs/default/bin/python3 /workspace/data/utils/merge_depth_wo_ground.py \
                --out ${out_c} --total 200000 --datatype kist
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${prepro_dir}/points3d.ply" "prepro_hugsim/points3d.ply"
        _substage_done "${s_merge_nog}"
    fi

    # ── Merge depth (with ground) ─────────────────────────────
    if _substage_is_done "${s_merge_g}"; then
        _log "  [prepro/merge_ground] SKIPPED (sentinel exists)"
    else
        _log "▶ [prepro] merge_depth_ground.py"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            export PYTHONPATH='/workspace/data:\${PYTHONPATH:-}'
            export MPLCONFIGDIR=/tmp/mpl_config
            /workspace/.pixi/envs/default/bin/python3 /workspace/data/utils/merge_depth_ground.py \
                --out ${out_c} --total 200000 --datatype kist
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${prepro_dir}/ground_points3d.ply" "prepro_hugsim/ground_points3d.ply"
        _assert_file "${prepro_dir}/ground_param.pkl"    "prepro_hugsim/ground_param.pkl"
        _substage_done "${s_merge_g}"
    fi

    local t1; t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    _log_elapsed "${elapsed}" "preprocess (all sub-stages)"

    # Final output assertions
    _assert_file "${prepro_dir}/meta_data.json"      "prepro_hugsim/meta_data.json"
    _assert_file "${prepro_dir}/sparse_ba.ply"       "prepro_hugsim/sparse_ba.ply"
    _assert_file "${prepro_dir}/points3d.ply"        "prepro_hugsim/points3d.ply"
    _assert_file "${prepro_dir}/ground_points3d.ply" "prepro_hugsim/ground_points3d.ply"
    _assert_file "${prepro_dir}/ground_param.pkl"    "prepro_hugsim/ground_param.pkl"

    _status_set "preprocess" "done" "${elapsed}" "semantics+mask+depth+merge"
    _write_time_summary
}

# =============================================================================
#  STAGE: train
#  Runs inside Docker container: hugsim_v3  (ganing/hugsimin:v3)
#  Reads from:  <chunk>/prepro_hugsim/
#  Writes to:   <chunk>/train_hugsim/
#
#  Fine-grained sub-stage sentinels:
#    train_hugsim/.done_train_ground
#    train_hugsim/.done_train
# =============================================================================
run_stage_train() {
    local chunk="$1"
    local train_dir="${_CHUNK_DIR}/train_hugsim"

    _banner "STAGE train  [${chunk}]"

    if [[ ${RESUME} -eq 1 ]] && [[ "$(_status_get train)" == "done" ]]; then
        _log "  SKIPPED (already done)"; return 0
    fi

    _check_docker "${DOCKER_HUGSIM}" \
        "docker run -d --gpus all --name ${DOCKER_HUGSIM} \
-v ${DATA_HOST}:/data -v ${HUGSIM_HOST}:/workspace -v ${WDBLACK_HOST}:/wdblack \
ganing/hugsimin:v3 sleep infinity"

    local source_c="/wdblack/KIST/chunk/${chunk}/prepro_hugsim"
    local model_c="/wdblack/KIST/chunk/${chunk}/train_hugsim"
    local ground_ckpt_c="${model_c}/ckpts/ground_chkpnt30000.pth"

    local s_ground="${train_dir}/.done_train_ground"
    local s_train="${train_dir}/.done_train"

    local t0; t0=$(date +%s)

    mkdir -p "${train_dir}/ckpts"

    # ── train_ground.py ────────────────────────────────────────
    if _substage_is_done "${s_ground}"; then
        _log "  [train/train_ground] SKIPPED (sentinel exists)"
    else
        _log "▶ [train] train_ground.py"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            if [[ -f '${ground_ckpt_c}' ]]; then
                echo '✔ Ground checkpoint already exists, skipping'
                exit 0
            fi
            export TORCH_HOME=/home/user/.cache/torch
            export MPLCONFIGDIR=/tmp/mpl_config && mkdir -p /tmp/mpl_config
            cd /workspace
            CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_INDEX_TRAIN} \
            python -u train_ground.py \
                --data_cfg    ${DATA_CFG_CONTAINER} \
                --source_path ${source_c} \
                --model_path  ${model_c}
            echo '✔ OK: train_ground.py'
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${train_dir}/ckpts/ground_chkpnt30000.pth" "ground_chkpnt30000.pth"
        _substage_done "${s_ground}"
    fi

    # ── train.py ───────────────────────────────────────────────
    if _substage_is_done "${s_train}"; then
        _log "  [train/train] SKIPPED (sentinel exists)"
    else
        _log "▶ [train] train.py"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            export TORCH_HOME=/home/user/.cache/torch
            export MPLCONFIGDIR=/tmp/mpl_config && mkdir -p /tmp/mpl_config
            cd /workspace
            CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_INDEX_TRAIN} \
            python -u train.py \
                --data_cfg    ${DATA_CFG_CONTAINER} \
                --source_path ${source_c} \
                --model_path  ${model_c}
            echo '✔ OK: train.py'
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${train_dir}/ckpts/chkpnt30000.pth" "train_hugsim/ckpts/chkpnt30000.pth"
        _substage_done "${s_train}"
    fi

    local t1; t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    _log_elapsed "${elapsed}" "training (all sub-stages)"

    _assert_file "${train_dir}/ckpts/chkpnt30000.pth" "train_hugsim/ckpts/chkpnt30000.pth"

    _status_set "train" "done" "${elapsed}" "train_ground.py + train.py"
    _write_time_summary
}

# =============================================================================
#  Stage: export
# =============================================================================
run_stage_export() {
    local chunk="$1"
    local train_dir="${CHUNK_BASE}/${chunk}/train_hugsim"
    local export_dir="${CHUNK_BASE}/${chunk}/export_hugsim"
    local train_c="/wdblack/KIST/chunk/${chunk}/train_hugsim"
    local export_c="/wdblack/KIST/chunk/${chunk}/export_hugsim"

    _banner "STAGE export  [${chunk}]"

    if [[ "${RESUME}" -eq 1 ]] && [[ "$(_status_get export)" == "done" ]]; then
        _log "  SKIPPED (already done)"
        return
    fi

    _check_docker "${DOCKER_HUGSIM}" \
        "docker run -d --gpus all --name ${DOCKER_HUGSIM} \
-v ${DATA_HOST}:/data -v ${HUGSIM_HOST}:/workspace -v ${WDBLACK_HOST}:/wdblack \
ganing/hugsimin:v3 sleep infinity"

    local s_export="${export_dir}/.done_export"
    local t0; t0=$(date +%s)

    mkdir -p "${export_dir}"

    if _substage_is_done "${s_export}"; then
        _log "  [export] SKIPPED (sentinel exists)"
    else
        _log "▶ [export] export_scene.py + convert_scene.py"
        docker exec "${DOCKER_HUGSIM}" bash -c "
            set -eu
            mkdir -p ${export_c}
            cd /workspace
            CUDA_VISIBLE_DEVICES=${GPU_INDEX_TRAIN} \
            python eval_render/export_scene.py \
                --model_path  ${train_c} \
                --output_path ${export_c} \
                --iteration 30000
            echo '✔ OK: export_scene.py'
            python eval_render/convert_scene.py \
                --model_path ${export_c}
            echo '✔ OK: convert_scene.py'
        " 2>&1 | tee -a "${_LOG_FILE}"
        _assert_file "${export_dir}/scene.pth" "export_hugsim/scene.pth"
        _substage_done "${s_export}"
    fi

    local t1; t1=$(date +%s)
    local elapsed=$(( t1 - t0 ))
    _log_elapsed "${elapsed}" "export"

    _assert_file "${export_dir}/scene.pth" "export_hugsim/scene.pth"

    _status_set "export" "done" "${elapsed}" "export_scene.py + convert_scene.py"
    _write_time_summary
}

# =============================================================================
#  Per-chunk orchestration
# =============================================================================
run_chunk_pipeline() {
    local chunk="$1" start_s="$2" end_s="$3"

    _CHUNK="${chunk}"
    _CHUNK_DIR="${CHUNK_BASE}/${chunk}"
    _LOG_FILE="${_CHUNK_DIR}/pipeline.log"
    _ERROR_LOG="${_CHUNK_DIR}/error.log"
    _STATUS_JSON="${_CHUNK_DIR}/status.json"
    _TIME_SUMMARY="${_CHUNK_DIR}/stage_time_summary.txt"

    mkdir -p "${_CHUNK_DIR}"

    _banner "PIPELINE START  [${chunk}]"
    _log "  video_dir  : ${VIDEO_DIR}"
    _log "  chunk_dir  : ${_CHUNK_DIR}"
    _log "  window     : ${start_s}s – ${end_s}s"
    _log "  stages     : ${STAGES}"
    _log "  resume     : ${RESUME}"

    local duration_s=$(( ${end_s%.*} - ${start_s%.*} ))
    local n_frames=$(( duration_s * 50 / SAMPLE_EVERY ))
    _log "  n_frames (12.5fps): ${n_frames}"

    _stage_enabled "video"      && run_stage_video      "${chunk}" "${start_s}" "${end_s}"
    _stage_enabled "colmap"     && run_stage_colmap     "${chunk}"
    _stage_enabled "qgis"       && run_stage_qgis       "${chunk}"
    _stage_enabled "preprocess" && run_stage_preprocess "${chunk}" "${n_frames}"
    _stage_enabled "train"      && run_stage_train      "${chunk}"
    _stage_enabled "export"     && run_stage_export     "${chunk}"

    _banner "PIPELINE COMPLETE  [${chunk}]"
    _write_time_summary
    _log ""
    cat "${_TIME_SUMMARY}"
}

# =============================================================================
#  Main: resolve chunk list from arguments
# =============================================================================
_glog "════════════════════════════════════════════════════════════"
_glog "  KIST Full Automation Pipeline"
_glog "  video_dir    : ${VIDEO_DIR}"
_glog "  chunk_base   : ${CHUNK_BASE}"
_glog "  chunk_dur    : ${CHUNK_DURATION}s"
_glog "  stages       : ${STAGES}"
_glog "  resume       : ${RESUME}"
_glog "════════════════════════════════════════════════════════════"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
for f in "0_front.MP4" "1_right_front.MP4" "2_right_back.MP4" \
          "3_back.MP4" "4_left_back.MP4" "5_left_front.MP4"; do
    if [[ ! -f "${VIDEO_DIR}/${f}" ]]; then
        _glog "ERROR: Missing video file: ${VIDEO_DIR}/${f}"
        exit 1
    fi
done
_glog "  ✔ All 6 MP4 files found"

if [[ ! -f "${INVERSEFORM_CHECKPOINT}" ]]; then
    _glog "ERROR: InverseForm checkpoint not found: ${INVERSEFORM_CHECKPOINT}"
    exit 1
fi
_glog "  ✔ InverseForm checkpoint found"

mkdir -p "${CHUNK_BASE}"

# ── Determine total chunk count from video duration ───────────────────────────
_glog "  Measuring video duration (0_front.MP4)..."
TOTAL_FLOAT=$(_video_duration_s "${VIDEO_DIR}/0_front.MP4")
TOTAL_INT=${TOTAL_FLOAT%.*}
TOTAL_CHUNKS=$(( (TOTAL_INT + CHUNK_DURATION - 1) / CHUNK_DURATION ))
_glog "  Duration: ${TOTAL_FLOAT}s  →  ${TOTAL_CHUNKS} chunks of ${CHUNK_DURATION}s"

# Build manifest for all chunks (idempotent: only writes new entries)
for (( i=0; i<TOTAL_CHUNKS; i++ )); do
    cname=$(printf "chunk%02d" ${i})
    cs=$(( i * CHUNK_DURATION ))
    ce=$(( cs + CHUNK_DURATION ))
    (( ce > TOTAL_INT )) && ce=${TOTAL_INT}
    _manifest_set "${cname}" "${cs}" "${ce}"
done
_glog "  chunk_manifest.json updated: ${CHUNK_BASE}/chunk_manifest.json"

# ── Build run list from --chunk_idx / --chunk_range / default (all) ──────────
CHUNK_INDICES=()

if [[ -n "${CHUNK_IDX}" ]]; then
    # Single chunk
    CHUNK_INDICES=( "${CHUNK_IDX}" )
    _glog "  Mode: single chunk  →  chunk index ${CHUNK_IDX}"

elif [[ -n "${CHUNK_RANGE_START}" ]] && [[ -n "${CHUNK_RANGE_END}" ]]; then
    # Inclusive range
    if (( CHUNK_RANGE_START > CHUNK_RANGE_END )); then
        _glog "ERROR: --chunk_range start (${CHUNK_RANGE_START}) > end (${CHUNK_RANGE_END})"
        exit 1
    fi
    if (( CHUNK_RANGE_END >= TOTAL_CHUNKS )); then
        _glog "WARNING: --chunk_range end ${CHUNK_RANGE_END} >= total ${TOTAL_CHUNKS}; clamping."
        CHUNK_RANGE_END=$(( TOTAL_CHUNKS - 1 ))
    fi
    for (( i=CHUNK_RANGE_START; i<=CHUNK_RANGE_END; i++ )); do
        CHUNK_INDICES+=( "${i}" )
    done
    _glog "  Mode: range  →  chunks ${CHUNK_RANGE_START}–${CHUNK_RANGE_END} (${#CHUNK_INDICES[@]} chunks)"

else
    # Default: all chunks
    for (( i=0; i<TOTAL_CHUNKS; i++ )); do
        CHUNK_INDICES+=( "${i}" )
    done
    _glog "  Mode: all  →  ${TOTAL_CHUNKS} chunks"
fi

# Print run plan (read time windows directly from the index arrays built above)
_glog ""
_glog "  Run plan:"
for idx in "${CHUNK_INDICES[@]}"; do
    cname=$(printf "chunk%02d" ${idx})
    cs=$(( idx * CHUNK_DURATION ))
    ce=$(( cs + CHUNK_DURATION ))
    (( ce > TOTAL_INT )) && ce=${TOTAL_INT}
    _glog "    ${cname}  ${cs}s – ${ce}s"
done
_glog ""

# ── Execute ───────────────────────────────────────────────────────────────────
for idx in "${CHUNK_INDICES[@]}"; do
    cname=$(printf "chunk%02d" ${idx})
    start_s=$(_manifest_get "${cname}" "start_s" "0")
    end_s=$(_manifest_get   "${cname}" "end_s"   "${CHUNK_DURATION}")
    run_chunk_pipeline "${cname}" "${start_s}" "${end_s}"
done

_glog "════════════════════════════════════════════════════════════"
_glog "  ALL DONE"
_glog "════════════════════════════════════════════════════════════"