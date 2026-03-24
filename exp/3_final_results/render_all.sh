#!/bin/bash


set -e

ROOT=""
DATASET_CLEAN="${ROOT}/dataset/Nerf_Synthetic"
LOG_ROOT="${ROOT}/log/11_final_results"
RENDER="${ROOT}/victim/gaussian-splatting/render_compare.py"
PYTHON="python"

OBJECTS="chair drums ficus hotdog lego materials mic ship"

mkdir -p "${LOG_ROOT}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RENDER_LOG="${LOG_ROOT}/render_run_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${RENDER_LOG}"; }

log "======================================================"
log "  Rendering all conditions"
log "  Started : $(date)"
log "======================================================"

render_one() {
    local COND=$1
    local OBJ=$2
    local EXP_DIR="${LOG_ROOT}/${COND}/${OBJ}/exp_run_1"
    local PLY="${EXP_DIR}/victim_model.ply"
    local OUT="${EXP_DIR}/render_comparison"
    local DATA="${DATASET_CLEAN}/${OBJ}"

    if [ ! -f "${PLY}" ]; then
        log "  [SKIP] PLY not found: ${PLY}"
        return
    fi

    if [ -d "${OUT}/gt" ]; then
        n=$(ls "${OUT}/gt" | wc -l)
        log "  [SKIP] Already rendered (${n} images): ${COND}/${OBJ}"
        return
    fi

    log "  Rendering ${COND} / ${OBJ}"

    ${PYTHON} "${RENDER}" \
        -s "${DATA}" \
        -m "${EXP_DIR}" \
        --ply        "${PLY}" \
        --output_dir "${OUT}" \
        2>&1 | tee -a "${RENDER_LOG}"

    log "  Done -> ${OUT}"
}

# All conditions including all three defense methods
CONDITIONS=(
    "clean_no_defense"
    "adv_eps4_no_defense"
    "adv_eps8_no_defense"
    "adv_eps4_defense_gaussian"
    "adv_eps4_defense_median"
    "adv_eps4_defense_bilateral"
    "adv_eps8_defense_gaussian"
    "adv_eps8_defense_median"
    "adv_eps8_defense_bilateral"
)

for COND in "${CONDITIONS[@]}"; do
    log ""
    log "------------------------------------------------------"
    log "  Condition: ${COND}"
    log "------------------------------------------------------"
    for OBJ in ${OBJECTS}; do
        render_one "${COND}" "${OBJ}"
    done
done

log ""
log "======================================================"
log "  All rendering complete. Finished : $(date)"
log "  Log : ${RENDER_LOG}"
log "======================================================"