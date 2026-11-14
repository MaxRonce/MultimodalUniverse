#!/usr/bin/env bash
# Default: one task per shard, both fields (legacy, resumable).
# JADES_SPLIT_FIELDS=1: two independent tasks per shard, one per GOODS field.
# Both modes share the exact same field checkpoints and Parquet filenames.
set -euo pipefail

: "${JADES_RAW:?Export JADES_RAW before sbatch}"
: "${JADES_RUN:?Export JADES_RUN before sbatch}"
: "${JADES_NUM_SHARDS:?Export the *total* JADES_NUM_SHARDS, including for retries}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch --array=0-N}"

cd "${SLURM_SUBMIT_DIR:?Submit from the MMU repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

scratch="${JADES_SCRATCH:-${JADES_RUN}/scratch_full_jwst_v2}"
manifest="${JADES_MANIFEST:-${JADES_RUN}/jades_dr5_manifest.json}"
threads="${JADES_EXTRACT_THREADS:-1}"
spatial_batch="${JADES_SPATIAL_BATCH_SIZE:-1}"
split_fields="${JADES_SPLIT_FIELDS:-0}"

for value in "$JADES_NUM_SHARDS" "$SLURM_ARRAY_TASK_ID" "$threads" "$spatial_batch"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "JADES numeric parameters must be non-negative integers: $value" >&2
        exit 2
    fi
done
if (( JADES_NUM_SHARDS < 1 || threads < 1 || spatial_batch < 1 )); then
    echo "JADES_NUM_SHARDS, JADES_EXTRACT_THREADS, JADES_SPATIAL_BATCH_SIZE must be >= 1" >&2
    exit 2
fi

if [[ "$split_fields" == "1" ]]; then
    if (( SLURM_ARRAY_TASK_ID >= 2 * JADES_NUM_SHARDS )); then
        echo "Array index must be < 2 * JADES_NUM_SHARDS with JADES_SPLIT_FIELDS=1" >&2
        exit 2
    fi
    shard_index=$((SLURM_ARRAY_TASK_ID % JADES_NUM_SHARDS))
    field_index=$((SLURM_ARRAY_TASK_ID / JADES_NUM_SHARDS))
    if (( field_index == 0 )); then
        field=GOODS-S
    else
        field=GOODS-N
    fi
elif [[ "$split_fields" == "0" ]]; then
    if (( SLURM_ARRAY_TASK_ID >= JADES_NUM_SHARDS )); then
        echo "Array index must be < JADES_NUM_SHARDS in legacy mode" >&2
        exit 2
    fi
    shard_index=$SLURM_ARRAY_TASK_ID
    field=both
else
    echo "JADES_SPLIT_FIELDS must be 0 or 1" >&2
    exit 2
fi

cmd=(uv run python -m scripts.jades.build_parent_sample_hats
    --instrument jwst --manifest "$manifest" --raw-root "$JADES_RAW"
    --scratch-dir "$scratch" --image-size 160
    --num-shards "$JADES_NUM_SHARDS" --shard-index "$shard_index"
    --chunk-size 4 --rows-per-file 64 --skip-hats
    --extract-threads "$threads" --spatial-batch-size "$spatial_batch")

if [[ "$split_fields" == "1" ]]; then
    cmd+=(--field "$field")
fi

# Optional small pilot: this is a GLOBAL per-field cap, not per task.
if [[ -n "${JADES_MAX_OBJECTS:-}" ]]; then
    cmd+=(--max-objects "$JADES_MAX_OBJECTS")
fi

echo "JADES extraction $(hostname): array=$SLURM_ARRAY_TASK_ID shard=$shard_index/$JADES_NUM_SHARDS field=$field threads=$threads spatial_batch=$spatial_batch scratch=$scratch"

# Read-only mapping check; useful on the login node before submitting jobs.
if [[ "${JADES_PLAN_ONLY:-0}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

# Prevent *new* duplicate submissions from writing one field concurrently.
# The old Slurm job does not acquire these locks: cancel it and verify it has
# stopped before switching to the split-field array.
if [[ "$split_fields" == "1" ]]; then
    printf -v shard_dir 'shard_%04d_of_%04d' "$shard_index" "$JADES_NUM_SHARDS"
    mkdir -p "$scratch/$shard_dir"
    exec 9>"$scratch/$shard_dir/$field.lock"
    if ! flock -n 9; then
        echo "Already extracting $field in $shard_dir; refusing concurrent write" >&2
        exit 75
    fi
fi

exec "${cmd[@]}"
