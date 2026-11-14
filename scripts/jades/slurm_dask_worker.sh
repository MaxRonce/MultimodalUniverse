#!/usr/bin/env bash
# Submit as a Slurm array after the Dask scheduler has started.
set -euo pipefail

: "${JADES_DASK_FILE:?Export JADES_DASK_FILE}"
cd "${SLURM_SUBMIT_DIR:?Submit from the MMU repository root}"

for attempt in {1..120}; do
    if [[ -s "$JADES_DASK_FILE" ]]; then
        break
    fi
    sleep 5
done
if [[ ! -s "$JADES_DASK_FILE" ]]; then
    echo "Dask scheduler did not publish $JADES_DASK_FILE" >&2
    exit 1
fi

echo "Joining Dask cluster from $(hostname)"
exec uv run dask worker --scheduler-file "$JADES_DASK_FILE" \
    --nworkers 1 --nthreads "${SLURM_CPUS_PER_TASK:-1}" \
    --memory-limit "${JADES_DASK_WORKER_MEMORY:-12GiB}" --no-dashboard
