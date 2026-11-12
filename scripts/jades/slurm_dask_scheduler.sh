#!/usr/bin/env bash
# Long-lived scheduler for multi-node HATS ingestion.
set -euo pipefail

: "${JADES_DASK_FILE:?Export a fresh JADES_DASK_FILE in shared Ceph storage}"
cd "${SLURM_SUBMIT_DIR:?Submit from the MMU repository root}"

if [[ -e "$JADES_DASK_FILE" ]]; then
    echo "Refusing to reuse an existing Dask scheduler file: $JADES_DASK_FILE" >&2
    exit 1
fi
mkdir -p "$(dirname "$JADES_DASK_FILE")"
echo "Starting Dask scheduler on $(hostname): $JADES_DASK_FILE"
exec uv run dask scheduler --scheduler-file "$JADES_DASK_FILE" --no-dashboard
