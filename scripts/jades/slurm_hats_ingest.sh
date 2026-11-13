#!/usr/bin/env bash
# Client job: verify all completed Slurm shards, then launch distributed HATS import.
set -euo pipefail

: "${JADES_RAW:?Export JADES_RAW}"
: "${JADES_RUN:?Export JADES_RUN}"
: "${JADES_NUM_SHARDS:?Export JADES_NUM_SHARDS}"
: "${JADES_DASK_FILE:?Export JADES_DASK_FILE}"
: "${JADES_HATS_OUT:?Export a NEW, empty JADES_HATS_OUT path}"

cd "${SLURM_SUBMIT_DIR:?Submit from the MMU repository root}"
exec uv run python -m scripts.jades.build_parent_sample_hats \
    --instrument jwst \
    --manifest "${JADES_MANIFEST:-${JADES_RUN}/jades_dr5_manifest.json}" \
    --raw-root "$JADES_RAW" \
    --scratch-dir "${JADES_SCRATCH:-${JADES_RUN}/scratch_full_jwst_v2}" \
    --num-shards "$JADES_NUM_SHARDS" --ingest-only \
    --output-root "$JADES_HATS_OUT" \
    --tmp-dir "${JADES_HATS_TMP:-${JADES_RUN}/hats_tmp_jwst_v2}" \
    --hats-resume \
    --hats-mapping-retries "${JADES_HATS_MAPPING_RETRIES:-1}" \
    --pixel-threshold "${JADES_HATS_PIXEL_THRESHOLD:-1024}" \
    --highest-healpix-order "${JADES_HATS_MAX_ORDER:-13}" \
    --hats-batch-size "${JADES_HATS_BATCH_SIZE:-4}" \
    --scheduler-file "$JADES_DASK_FILE" \
    --min-ingest-workers "${JADES_DASK_WORKERS:-8}"
