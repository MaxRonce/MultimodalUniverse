#!/bin/bash
# Submit sequential prepost passes for patches beyond a previous dense run.
set -euo pipefail

JOBS=10
PATCH_LIMIT=30000
SOURCE_RUN=""
while (($#)); do
  case "$1" in
    --jobs|--patch-limit|--source-run)
      if (($# < 2)); then echo "ERROR: missing value for $1" >&2; exit 2; fi
      case "$1" in
        --jobs) JOBS="$2" ;;
        --patch-limit) PATCH_LIMIT="$2" ;;
        --source-run) SOURCE_RUN="$2" ;;
      esac
      shift 2 ;;
    --help|-h)
      echo "Usage: $0 --source-run PATH [--jobs 10] [--patch-limit 30000]"
      echo "PATCH_LIMIT is the final global density rank, including prior runs."
      echo "Export LSST_DP2_RUN_NAME for the new run and RSP_TOKEN before submission."
      exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ || ! "$PATCH_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: jobs and patch limit must be positive integers" >&2
  exit 2
fi
: "${SCRATCH:?SCRATCH must be exported by Jean-Zay}"
: "${RSP_TOKEN:?Read and export RSP_TOKEN before submission}"
: "${LSST_DP2_RUN_NAME:?Export a NEW run name before submission}"
: "${SOURCE_RUN:?Provide --source-run with the original dense run path}"

export MMU_JZ_ROOT="${MMU_JZ_ROOT:-$SCRATCH/mmu_lsst_dp2}"
export MMU_REPO="${MMU_REPO:-$MMU_JZ_ROOT/MultimodalUniverse}"
export LSST_DP2_EXTEND_FROM
LSST_DP2_EXTEND_FROM=$(realpath -e -- "$SOURCE_RUN")
for FILE in catalog/objects.parquet.selection.json \
  catalog/.objects.dense_query/plan.json \
  catalog/.objects.dense_query/patch_inventory.parquet \
  catalog/.objects.dense_query/selected_patches.parquet; do
  if [[ ! -f "$LSST_DP2_EXTEND_FROM/$FILE" ]]; then
    echo "ERROR: missing source selection artifact: $LSST_DP2_EXTEND_FROM/$FILE" >&2
    exit 2
  fi
done
NEW_ROOT=$(realpath -m -- "$MMU_JZ_ROOT/runs/$LSST_DP2_RUN_NAME")
if [[ "$NEW_ROOT" == "$LSST_DP2_EXTEND_FROM" ]]; then
  echo "ERROR: the new run must differ from --source-run" >&2
  exit 2
fi

unset LSST_DP2_ROOT LSST_DP2_HATS_OUTER MMU_PYTHON
# shellcheck source=jeanzay_env.sh
source "$MMU_REPO/scripts/lsst_dp2/jeanzay_env.sh"
export LSST_DP2_DENSE_PATCH_LIMIT="$PATCH_LIMIT"
export LSST_DP2_DENSE_REQUESTS_PER_MINUTE=59
export LSST_DP2_DENSE_DOWNLOAD_WORKERS=8
export LSST_DP2_DOWNLOAD_TASK_LIMIT=18000

# Each receipt contains only job IDs; no token or Slurm environment is saved.
RECEIPT=$(mktemp "$LSST_DP2_ROOT/logs/submitted-jobs.XXXXXXXX.txt")
echo "New run: $LSST_DP2_ROOT"
echo "Source inventory: $LSST_DP2_EXTEND_FROM"
echo "Submitting $JOBS sequential passes of up to 20 hours; final density rank $PATCH_LIMIT"
echo "Job receipt: $RECEIPT"
for ((PASS=1; PASS<=JOBS; PASS++)); do
  if ! JOB_ID=$(sbatch --parsable --job-name=mmu-lsst-net --dependency=singleton \
    --account="${LSST_DP2_ACCOUNT:-jrx@cpu}" --partition=prepost \
    --nodes=1 --ntasks=1 --cpus-per-task=8 --time=20:00:00 --export=ALL \
    --output="$LSST_DP2_ROOT/logs/dense-%j.out" \
    --error="$LSST_DP2_ROOT/logs/dense-%j.err" \
    "$MMU_REPO/scripts/lsst_dp2/dense_download_prepost.slurm"); then
    echo "ERROR: submission $PASS failed; earlier job IDs are in $RECEIPT" >&2
    exit 1
  fi
  printf '%s\n' "$JOB_ID" >> "$RECEIPT"
  echo "[$PASS/$JOBS] $JOB_ID"
done
