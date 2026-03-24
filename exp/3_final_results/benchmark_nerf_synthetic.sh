#!/bin/bash

set -e

ROOT=""

DATASET_CLEAN="${ROOT}/dataset/Nerf_Synthetic"
DATASET_EPS4="${ROOT}/dataset/Nerf_Synthetic_adversarial_eps4"
DATASET_EPS8="${ROOT}/dataset/Nerf_Synthetic_adversarial_eps8"

LOG_ROOT="${ROOT}/log/11_final_results"

BENCHMARK="${ROOT}/victim/gaussian-splatting/benchmark.py"
BENCHMARK_DEF="${ROOT}/victim/gaussian-splatting/benchmark_filter_defense.py"

PYTHON="python"
GPU=0

# ── Objects ────────────────────────────────────────────────────────────────
OBJECTS="chair drums ficus hotdog lego materials mic ship"

# ── Defense hyperparameters ────────────────────────────────────────────────
SMOOTH_K=10
SMOOTH_SIGMA=0.05
SMOOTH_SIGMA_C=0.1
SMOOTH_TARGETS="rgb opacity"

# ── Logging ────────────────────────────────────────────────────────────────
mkdir -p "${LOG_ROOT}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_ROOT}/training_run_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"
}

log "======================================================"
log "  11_final_results -- 3DGS Training"
log "  Started : $(date)"
log "  Log     : ${MAIN_LOG}"
log "======================================================"


train_object() {
    local SCRIPT=$1
    local DATA=$2
    local OUT=$3
    local LABEL=$4
    local EXTRA="${5:-}"

    log ""
    log "  [${LABEL}]"
    log "  Data   : ${DATA}"
    log "  Output : ${OUT}"

    mkdir -p "${OUT}"

    ${PYTHON} "${SCRIPT}" \
        -s "${DATA}" \
        -m "${OUT}" \
        --gpu      ${GPU} \
        --exp_runs 1 \
        ${EXTRA} \
        2>&1 | tee -a "${LOG_ROOT}/${LABEL}.log" "${MAIN_LOG}"

    log "  Done: ${LABEL}"
}


log ""
log "======================================================"
log "  [1/9] Clean -- no defense"
log "======================================================"

for OBJ in ${OBJECTS}; do
    train_object \
        "${BENCHMARK}" \
        "${DATASET_CLEAN}/${OBJ}" \
        "${LOG_ROOT}/clean_no_defense/${OBJ}" \
        "clean_no_defense_${OBJ}"
done

log ""
log "======================================================"
log "  [2/9] Adv eps=4.0 -- no defense"
log "======================================================"

for OBJ in ${OBJECTS}; do
    train_object \
        "${BENCHMARK}" \
        "${DATASET_EPS4}/${OBJ}" \
        "${LOG_ROOT}/adv_eps4_no_defense/${OBJ}" \
        "adv_eps4_no_defense_${OBJ}"
done


log ""
log "======================================================"
log "  [3/9] Adv eps=8.0 -- no defense"
log "======================================================"

for OBJ in ${OBJECTS}; do
    train_object \
        "${BENCHMARK}" \
        "${DATASET_EPS8}/${OBJ}" \
        "${LOG_ROOT}/adv_eps8_no_defense/${OBJ}" \
        "adv_eps8_no_defense_${OBJ}"
done



STEP=4
for METHOD in gaussian median bilateral; do
    log ""
    log "======================================================"
    log "  [${STEP}/9] Adv eps=4.0 -- ${METHOD} defense"
    log "======================================================"

    for OBJ in ${OBJECTS}; do
        train_object \
            "${BENCHMARK_DEF}" \
            "${DATASET_EPS4}/${OBJ}" \
            "${LOG_ROOT}/adv_eps4_defense_${METHOD}/${OBJ}" \
            "adv_eps4_defense_${METHOD}_${OBJ}" \
            "--smooth_method  ${METHOD} \
             --smooth_targets ${SMOOTH_TARGETS} \
             --smooth_k       ${SMOOTH_K} \
             --smooth_sigma   ${SMOOTH_SIGMA} \
             --smooth_sigma_c ${SMOOTH_SIGMA_C}"
    done
    STEP=$((STEP + 1))
done



for METHOD in gaussian median bilateral; do
    log ""
    log "======================================================"
    log "  [${STEP}/9] Adv eps=8.0 -- ${METHOD} defense"
    log "======================================================"

    for OBJ in ${OBJECTS}; do
        train_object \
            "${BENCHMARK_DEF}" \
            "${DATASET_EPS8}/${OBJ}" \
            "${LOG_ROOT}/adv_eps8_defense_${METHOD}/${OBJ}" \
            "adv_eps8_defense_${METHOD}_${OBJ}" \
            "--smooth_method  ${METHOD} \
             --smooth_targets ${SMOOTH_TARGETS} \
             --smooth_k       ${SMOOTH_K} \
             --smooth_sigma   ${SMOOTH_SIGMA} \
             --smooth_sigma_c ${SMOOTH_SIGMA_C}"
    done
    STEP=$((STEP + 1))
done