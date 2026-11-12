#!/usr/bin/env bash
# Independent read-only job: no dependency on the extraction Slurm array.
set -euo pipefail
: "${JADES_RAW:?Export JADES_RAW before sbatch}"
: "${JADES_RUN:?Export JADES_RUN before sbatch}"
cd "${SLURM_SUBMIT_DIR:?Submit from the MMU repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cmd=(uv run --extra viz python -u -m scripts.jades.audit_selection
    --raw-root "$JADES_RAW"
    --mag-band "${JADES_AUDIT_MAG_BAND:-F277W}"
    --mag-cuts 21 22 23 24 25 26 27 28 29
    --output-dir "${JADES_AUDIT_OUT:-${JADES_RUN}/selection_audit_f277w}"
    --plots)
if [[ -n "${JADES_MANIFEST:-}" ]]; then
    cmd+=(--manifest "$JADES_MANIFEST")
fi
# Deliberately separate from JADES_SCRATCH: raw catalog audit is the default.
if [[ -n "${JADES_AUDIT_SCRATCH:-}" ]]; then
    cmd+=(--scratch-dir "$JADES_AUDIT_SCRATCH")
fi
exec "${cmd[@]}"
