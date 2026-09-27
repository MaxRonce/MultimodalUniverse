#!/bin/bash
# Offline worker for submit_offline_cutouts.sh. No installation or RSP access.
set -euo pipefail
: "${MMU_REPO:?}"
: "${LSST_DP2_OFFLINE_ROOT:?}"
: "${LSST_DP2_SOURCE_RUN:?}"
export LSST_DP2_ROOT="$LSST_DP2_OFFLINE_ROOT"
export LSST_DP2_HATS_OUTER="$LSST_DP2_ROOT/hats/lsst_dp2"
source "$MMU_REPO/scripts/lsst_dp2/jeanzay_env.sh"
unset RSP_TOKEN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 ARROW_NUM_THREADS=1
export TMPDIR="$LSST_DP2_ROOT/tmp/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-main}"
export DASK_TEMPORARY_DIRECTORY="$TMPDIR/dask"
export JOBLIB_TEMP_FOLDER="$TMPDIR/joblib"
mkdir -p "$TMPDIR" "$DASK_TEMPORARY_DIRECTORY" "$JOBLIB_TEMP_FOLDER"
cd "$MMU_REPO"

case "${1:?expected prepare, extract, or gather}" in
  prepare)
    exec "$MMU_PYTHON" -u -m scripts.lsst_dp2.offline_cutouts prepare \
      --source-run "$LSST_DP2_SOURCE_RUN" --output-run "$LSST_DP2_ROOT" \
      --shards "$LSST_DP2_OFFLINE_SHARDS" ;;
  extract)
    exec "$MMU_PYTHON" -u -m scripts.lsst_dp2.offline_cutouts extract \
      --output-run "$LSST_DP2_ROOT" --index "$SLURM_ARRAY_TASK_ID" \
      --workers 4 ;;
  gather)
    exec "$MMU_PYTHON" -u -m scripts.lsst_dp2.offline_cutouts gather \
      --output-run "$LSST_DP2_ROOT" --workers 16 --pixel-threshold 256 ;;
  *) echo "ERROR: unknown stage $1" >&2; exit 2 ;;
esac
