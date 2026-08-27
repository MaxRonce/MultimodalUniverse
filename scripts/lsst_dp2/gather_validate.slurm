#!/bin/bash
#SBATCH --job-name=mmu-lsst-gather
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --output=logs/lsst-dp2-gather-%j.out
#SBATCH --error=logs/lsst-dp2-gather-%j.err

set -euo pipefail

: "${WORK:?WORK must be set}"

REPO="${MMU_REPO:-$WORK/MultimodalUniverse}"
PYTHON="${MMU_PYTHON:-$REPO/.venv/bin/python}"
ROOT="${LSST_DP2_ROOT:-$WORK/lsst_dp2}"
HATS_ROOT="${MMU_HATS_ROOT:-$WORK/mmu_hats}/lsst_dp2"

cd "$REPO"
mkdir -p logs "$HATS_ROOT"

"$PYTHON" -u -m scripts.lsst_dp2.build_parent_sample_hats \
  --catalog "$ROOT/catalog/objects.parquet" \
  --manifest "$ROOT/download_manifest.sqlite" \
  --scratch-dir "$ROOT/parquet" \
  --output-root "$HATS_ROOT" \
  --pixel-threshold 8192 \
  --ingest-workers "${SLURM_CPUS_PER_TASK:-16}" \
  --verify-checksums \
  --only-ingest

"$PYTHON" -u -m scripts.lsst_dp2.validate_parent_sample \
  --catalog "$ROOT/catalog/objects.parquet" \
  --manifest "$ROOT/download_manifest.sqlite" \
  --scratch-dir "$ROOT/parquet" \
  --hats-root "$HATS_ROOT" \
  --verify-checksums
