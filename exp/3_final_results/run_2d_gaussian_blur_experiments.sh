#!/bin/bash

set -e

ROOT=""

DATASET_CLEAN="${ROOT}/dataset/Nerf_Synthetic"
DATASET_EPS4="${ROOT}/dataset/Nerf_Synthetic_adversarial_eps4"
DATASET_EPS8="${ROOT}/dataset/Nerf_Synthetic_adversarial_eps8"

DATASET_EPS4_BLUR="${ROOT}/dataset/Nerf_Synthetic_adv_eps4_2d_gaussian_blur"
DATASET_EPS8_BLUR="${ROOT}/dataset/Nerf_Synthetic_adv_eps8_2d_gaussian_blur"

LOG_ROOT="${ROOT}/log/11_final_results"
BENCHMARK="${ROOT}/victim/gaussian-splatting/benchmark.py"
RENDER="${ROOT}/victim/gaussian-splatting/render_compare.py"
PYTHON="python"
GPU=0

SIGMA=1.0
KERNEL=5
OBJECTS="chair drums ficus hotdog lego materials mic ship"

mkdir -p "${LOG_ROOT}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_ROOT}/2d_blur_run_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"; }

log "======================================================"
log "  2D Gaussian Blur Baseline -- sigma=${SIGMA} kernel=${KERNEL}"
log "  Started : $(date)"
log "======================================================"


# ── Step 1: Create blurred datasets ───────────────────────────────────────
log ""
log "======================================================"
log "  STEP 1 -- Creating 2D Gaussian blurred datasets"
log "======================================================"

log "  Blurring adversarial eps4 ..."
${PYTHON} "${ROOT}/create_2d_gaussian_blur_dataset.py" \
    --input_dir  "${DATASET_EPS4}" \
    --output_dir "${DATASET_EPS4_BLUR}" \
    --sigma      ${SIGMA} \
    --kernel     ${KERNEL} \
    2>&1 | tee -a "${MAIN_LOG}"

log "  Blurring adversarial eps8 ..."
${PYTHON} "${ROOT}/create_2d_gaussian_blur_dataset.py" \
    --input_dir  "${DATASET_EPS8}" \
    --output_dir "${DATASET_EPS8_BLUR}" \
    --sigma      ${SIGMA} \
    --kernel     ${KERNEL} \
    2>&1 | tee -a "${MAIN_LOG}"

log "  Blurred datasets created."


# ── Step 2: Train 3DGS on blurred datasets ─────────────────────────────────
log ""
log "======================================================"
log "  STEP 2 -- Training 3DGS on blurred datasets"
log "======================================================"

for OBJ in ${OBJECTS}; do
    log "  [adv_eps4_2d_gaussian_blur] ${OBJ}"
    OUT="${LOG_ROOT}/adv_eps4_2d_gaussian_blur/${OBJ}"
    mkdir -p "${OUT}"
    ${PYTHON} "${BENCHMARK}" \
        -s "${DATASET_EPS4_BLUR}/${OBJ}" \
        -m "${OUT}" \
        --gpu      ${GPU} \
        --exp_runs 1 \
        2>&1 | tee -a "${LOG_ROOT}/adv_eps4_2d_gaussian_blur_${OBJ}.log" "${MAIN_LOG}"
    log "  Done: adv_eps4_2d_gaussian_blur / ${OBJ}"
done

for OBJ in ${OBJECTS}; do
    log "  [adv_eps8_2d_gaussian_blur] ${OBJ}"
    OUT="${LOG_ROOT}/adv_eps8_2d_gaussian_blur/${OBJ}"
    mkdir -p "${OUT}"
    ${PYTHON} "${BENCHMARK}" \
        -s "${DATASET_EPS8_BLUR}/${OBJ}" \
        -m "${OUT}" \
        --gpu      ${GPU} \
        --exp_runs 1 \
        2>&1 | tee -a "${LOG_ROOT}/adv_eps8_2d_gaussian_blur_${OBJ}.log" "${MAIN_LOG}"
    log "  Done: adv_eps8_2d_gaussian_blur / ${OBJ}"
done


# ── Step 3: Render test views ──────────────────────────────────────────────
log ""
log "======================================================"
log "  STEP 3 -- Rendering test views"
log "======================================================"

for COND in adv_eps4_2d_gaussian_blur adv_eps8_2d_gaussian_blur; do
    for OBJ in ${OBJECTS}; do
        EXP_DIR="${LOG_ROOT}/${COND}/${OBJ}/exp_run_1"
        PLY="${EXP_DIR}/victim_model.ply"
        OUT="${EXP_DIR}/render_comparison"

        if [ ! -f "${PLY}" ]; then
            log "  [SKIP] PLY not found: ${PLY}"
            continue
        fi

        log "  Rendering ${COND} / ${OBJ}"
        ${PYTHON} "${RENDER}" \
            -s "${DATASET_CLEAN}/${OBJ}" \
            -m "${EXP_DIR}" \
            --ply        "${PLY}" \
            --output_dir "${OUT}" \
            2>&1 | tee -a "${MAIN_LOG}"
        log "  Done -> ${OUT}"
    done
done