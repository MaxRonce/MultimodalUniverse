#!/usr/bin/env bash
# Submit as a Slurm array. Each task owns a disjoint contiguous catalog slice.
set -euo pipefail

: "${JADES_RAW:?Export JADES_RAW before sbatch}"
: "${JADES_RUN:?Export JADES_RUN before sbatch}"
: "${JADES_NUM_SHARDS:?Export the *total* JADES_NUM_SHARDS, including for retries}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch --array=0-N}"

cd "${SLURM_SUBMIT_DIR:?Submit from the MMU repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

scratch="${JADES_SCRATCH:-${JADES_RUN}/scratch_full_jwst_v2}"
manifest="${JADES_MANIFEST:-${JADES_RUN}/jades_dr5_manifest.json}"
cmd=(uv run python -m scripts.jades.build_parent_sample_hats
    --instrument jwst --manifest "$manifest" --raw-root "$JADES_RAW"
    --scratch-dir "$scratch" --image-size 160
    --num-shards "$JADES_NUM_SHARDS" --shard-index "$SLURM_ARRAY_TASK_ID"
    --chunk-size 4 --rows-per-file 64 --skip-hats)

# Optional small pilot: this is a GLOBAL per-field cap, not per task.
if [[ -n "${JADES_MAX_OBJECTS:-}" ]]; then
    cmd+=(--max-objects "$JADES_MAX_OBJECTS")
fi

echo "JADES extraction $(hostname): shard $SLURM_ARRAY_TASK_ID/$JADES_NUM_SHARDS; scratch=$scratch"
exec "${cmd[@]}"
