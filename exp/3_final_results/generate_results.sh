#!/bin/bash

set -e

ROOT=""
LOG_ROOT="${ROOT}/log/11_final_results"
PYTHON="python"

mkdir -p "${LOG_ROOT}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_ROOT}/results_run_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${MAIN_LOG}"; }

log "======================================================"
log "  Generate Results Pipeline"
log "  Started : $(date)"
log "======================================================"

# ── Step 1: Render ─────────────────────────────────────────────────────────
log ""
log "  STEP 1 -- Rendering test views"
log "------------------------------------------------------"

bash "${ROOT}/exp/11_final_results/render_all.sh" \
    2>&1 | tee -a "${MAIN_LOG}"

# ── Step 2: Compute metrics ────────────────────────────────────────────────
log ""
log "  STEP 2 -- Computing PSNR / SSIM / CLIP metrics"
log "------------------------------------------------------"

${PYTHON} "${ROOT}/compute_all_metrics.py" \
    --log_root   "${LOG_ROOT}" \
    --clean_data "${ROOT}/dataset/Nerf_Synthetic" \
    --output     "${LOG_ROOT}/all_results.csv" \
    2>&1 | tee -a "${MAIN_LOG}"

# ── Step 3: Generate tables and figures ────────────────────────────────────
log ""
log "  STEP 3 -- Generating paper tables and figures"
log "------------------------------------------------------"

${PYTHON} "${ROOT}/generate_paper_tables.py" \
    --results_csv "${LOG_ROOT}/all_results.csv" \
    --output_dir  "${LOG_ROOT}/tables" \
    2>&1 | tee -a "${MAIN_LOG}"
