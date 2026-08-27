#!/bin/bash
#SBATCH --job-name=mmu-lsst-dp2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=logs/lsst-dp2-%A_%a.out
#SBATCH --error=logs/lsst-dp2-%A_%a.err

set -euo pipefail

: "${WORK:?WORK must be set}"
: "${SHARD_IDX:?SHARD_IDX must be set}"

REPO="${MMU_REPO:-$WORK/MultimodalUniverse}"
PYTHON="${MMU_PYTHON:-$REPO/.venv/bin/python}"
ROOT="${LSST_DP2_ROOT:-$WORK/lsst_dp2}"
NUM_SHARDS="${NUM_SHARDS:-64}"

cd "$REPO"
mkdir -p logs "$ROOT/parquet"

"$PYTHON" -u -m scripts.lsst_dp2.build_parent_sample_hats \
  --catalog "$ROOT/catalog/objects.parquet" \
  --manifest "$ROOT/download_manifest.sqlite" \
  --scratch-dir "$ROOT/parquet" \
  --output-root "${MMU_HATS_ROOT:-$WORK/mmu_hats}/lsst_dp2" \
  --objects-per-shard 64 \
  --num-shards "$NUM_SHARDS" \
  --shard-idx "$SHARD_IDX" \
  --verify-checksums \
  --require-all-bands \
  --skip-ingest
