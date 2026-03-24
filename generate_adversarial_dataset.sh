#!/bin/bash

set -e  # stop on any error


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_ROOT="${DATASET_ROOT:-${SCRIPT_DIR}/dataset/Nerf_Synthetic}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/dataset}"
PYTHON="${PYTHON:-python}"


EPSILONS=(4.0 8.0)
NUM_ITER=100
LOSS_THRESH=20.0

# All 8 NeRF Synthetic objects
CLASSES="chair drums ficus hotdog lego materials mic ship"


LOG_DIR="${SCRIPT_DIR}/logs/adversarial_generation"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/run_${TIMESTAMP}.log"



if [ ! -d "${DATASET_ROOT}" ]; then
    log "ERROR: Dataset root not found: ${DATASET_ROOT}"
    exit 1
fi


if [ ! -f "${SCRIPT_DIR}/run_attack.py" ]; then
    log "ERROR: run_attack.py not found in ${SCRIPT_DIR}"
    exit 1
fi

TOTAL_START=$(date +%s)

for EPS in "${EPSILONS[@]}"; do

    # Format epsilon for directory name: 4.0 -> eps4, 8.0 -> eps8
    EPS_TAG="eps$(echo ${EPS} | cut -d'.' -f1)"
    OUTPUT_DIR="${OUTPUT_BASE}/Nerf_Synthetic_adversarial_${EPS_TAG}"
    EPS_LOG="${LOG_DIR}/epsilon_${EPS_TAG}_${TIMESTAMP}.log"

    log ""
    log "------------------------------------------------------"
    log "  Running epsilon = ${EPS}"
    log "  Output dir : ${OUTPUT_DIR}"
    log "  Detail log : ${EPS_LOG}"
    log "------------------------------------------------------"

    EPS_START=$(date +%s)

    ${PYTHON} "${SCRIPT_DIR}/run_attack.py" \
        --dataset_root "${DATASET_ROOT}" \
        --output_dir   "${OUTPUT_DIR}" \
        --epsilon      "${EPS}" \
        --classes      ${CLASSES} \
        --num_iter     "${NUM_ITER}" \
        --loss_thresh  "${LOSS_THRESH}" \
        2>&1 | tee -a "${EPS_LOG}" "${MAIN_LOG}"

    EPS_END=$(date +%s)
    EPS_TIME=$(( EPS_END - EPS_START ))
    log "  Finished epsilon=${EPS} in ${EPS_TIME}s"
    log "  Output: ${OUTPUT_DIR}"

done

TOTAL_END=$(date +%s)
TOTAL_TIME=$(( TOTAL_END - TOTAL_START ))

log ""
log "======================================================"
log "  All done in ${TOTAL_TIME}s"
log ""
log "  Datasets generated:"
for EPS in "${EPSILONS[@]}"; do
    EPS_TAG="eps$(echo ${EPS} | cut -d'.' -f1)"
    OUTPUT_DIR="${OUTPUT_BASE}/Nerf_Synthetic_adversarial_${EPS_TAG}"
    log "    epsilon ${EPS} -> ${OUTPUT_DIR}"
done
log ""
log "  Full log: ${MAIN_LOG}"
log "======================================================"