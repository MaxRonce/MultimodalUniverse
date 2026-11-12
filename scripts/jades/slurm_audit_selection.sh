#!/usr/bin/env bash
# Independent read-only job: no dependency on the extraction Slurm array.
set -euo pipefail
: "${JADES_RAW:?Export JADES_RAW before sbatch}"
: "${JADES_RUN:?Export JADES_RUN before sbatch}"
cd "${SLURM_SUBMIT_DIR:?Submit from the MMU repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# The audit imports no MMU code. Do not sync/install the MMU project: its SCM
# version can fail on non-version Git tags, and its full dependency set is large.
# Also avoid touching the project .venv while extraction jobs are using it.
unset VIRTUAL_ENV
cmd=(uv run --no-project --isolated --python 3.11
    --with numpy --with astropy --with matplotlib --with pyarrow
    python -u -m scripts.jades.audit_selection
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
echo "Starting JADES selection audit with isolated dependencies; no MMU build."
echo "Results: ${JADES_AUDIT_OUT:-${JADES_RUN}/selection_audit_f277w}"
exec "${cmd[@]}"
