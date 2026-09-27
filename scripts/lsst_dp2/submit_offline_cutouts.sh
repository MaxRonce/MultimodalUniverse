#!/bin/bash
# Run from a frontend. Only sbatch and lightweight filesystem checks run here.
set -euo pipefail
SOURCE_RUN=""
RUN_NAME=""
SHARDS=128
CONCURRENT=16
while (($#)); do
  case "$1" in
    --source-run|--run-name|--shards|--concurrent)
      if (($# < 2)); then echo "ERROR: missing value for $1" >&2; exit 2; fi
      case "$1" in
        --source-run) SOURCE_RUN="$2" ;;
        --run-name) RUN_NAME="$2" ;;
        --shards) SHARDS="$2" ;;
        --concurrent) CONCURRENT="$2" ;;
      esac
      shift 2 ;;
    --help|-h)
      echo "Usage: $0 --source-run PATH --run-name NEW_NAME [--shards 128] [--concurrent 16]"
      echo "Offline CPU jobs: snapshot -> extract/validate array -> HATS gather."
      echo "Repeat only after earlier jobs end; validated shards are reused."
      exit 0 ;;
    *) echo "ERROR: unknown argument $1" >&2; exit 2 ;;
  esac
done
: "${SCRATCH:?}"
: "${SOURCE_RUN:?Provide --source-run}"
: "${RUN_NAME:?Provide --run-name}"
if [[ ! "$RUN_NAME" =~ ^[a-zA-Z0-9_-]+$ || ! "$SHARDS" =~ ^[1-9][0-9]*$ || ! "$CONCURRENT" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: invalid run name, shard count, or concurrency" >&2; exit 2
fi
export MMU_JZ_ROOT="${MMU_JZ_ROOT:-$SCRATCH/mmu_lsst_dp2}"
export MMU_REPO="${MMU_REPO:-$MMU_JZ_ROOT/MultimodalUniverse}"
export LSST_DP2_OFFLINE_ROOT="$MMU_JZ_ROOT/runs/$RUN_NAME"
export LSST_DP2_SOURCE_RUN
LSST_DP2_SOURCE_RUN=$(realpath -e -- "$SOURCE_RUN")
LSST_DP2_OFFLINE_ROOT=$(realpath -m -- "$LSST_DP2_OFFLINE_ROOT")
case "$LSST_DP2_OFFLINE_ROOT/" in
  "$LSST_DP2_SOURCE_RUN/"*) echo "ERROR: output must not be inside the source run" >&2; exit 2 ;;
esac
case "$LSST_DP2_SOURCE_RUN/" in
  "$LSST_DP2_OFFLINE_ROOT/"*) echo "ERROR: source must not be inside the output run" >&2; exit 2 ;;
esac
export LSST_DP2_OFFLINE_SHARDS="$SHARDS"
for FILE in catalog/objects.parquet download_manifest.sqlite; do
  test -f "$LSST_DP2_SOURCE_RUN/$FILE" || { echo "ERROR: missing $FILE" >&2; exit 2; }
done
test -x "$MMU_REPO/.venv/bin/python" || { echo "ERROR: install the MMU environment first" >&2; exit 2; }
# Keep inherited network credentials and stale run variables out of these jobs.
unset RSP_TOKEN MMU_PYTHON
export LSST_DP2_ROOT="$LSST_DP2_OFFLINE_ROOT"
export LSST_DP2_HATS_OUTER="$LSST_DP2_ROOT/hats/lsst_dp2"
source "$MMU_REPO/scripts/lsst_dp2/jeanzay_env.sh"
PARTITION="${LSST_DP2_CPU_PARTITION:-cpu_p1}"
ACCOUNT="${LSST_DP2_ACCOUNT:-jrx@cpu}"
SCRIPT="$MMU_REPO/scripts/lsst_dp2/offline_cutouts.slurm"
RECEIPT=$(mktemp "$LSST_DP2_ROOT/logs/offline-jobs.XXXXXXXX.txt")
# A second concurrent campaign targeting this directory would waste allocations.
if [[ -n "$(squeue -h -u "$USER" -n "dp2-$RUN_NAME" -o '%A')" ]]; then
  echo "ERROR: jobs for this run already queued; use the receipt to monitor them" >&2
  exit 2
fi
COMMON=(--parsable --account="$ACCOUNT" --partition="$PARTITION" --export=ALL
  --nodes=1 --ntasks=1 --hint=nomultithread --job-name="dp2-$RUN_NAME")
PREP=$(sbatch "${COMMON[@]}" --cpus-per-task=8 --time=02:00:00 \
  --output="$LSST_DP2_ROOT/logs/prepare-%j.out" \
  --error="$LSST_DP2_ROOT/logs/prepare-%j.err" "$SCRIPT" prepare)
printf 'prepare %s\n' "$PREP" | tee -a "$RECEIPT"
ARRAY=$(sbatch "${COMMON[@]}" --dependency="afterok:${PREP%%;*}" \
  --array="0-$((SHARDS-1))%$CONCURRENT" --cpus-per-task=8 --time=20:00:00 \
  --output="$LSST_DP2_ROOT/logs/extract-%A_%a.out" \
  --error="$LSST_DP2_ROOT/logs/extract-%A_%a.err" "$SCRIPT" extract)
printf 'extract %s\n' "$ARRAY" | tee -a "$RECEIPT"
# Reserve a complete CPU node for Dask's memory, not GPU accelerators.
GATHER=$(sbatch "${COMMON[@]}" --dependency="afterok:${ARRAY%%;*}" \
  --cpus-per-task=40 --time=20:00:00 \
  --output="$LSST_DP2_ROOT/logs/gather-%j.out" \
  --error="$LSST_DP2_ROOT/logs/gather-%j.err" "$SCRIPT" gather)
printf 'gather %s\n' "$GATHER" | tee -a "$RECEIPT"
echo "Output: $LSST_DP2_ROOT"
echo "Receipt: $RECEIPT"
echo "Array ceiling: $CONCURRENT jobs x 8 allocated CPU cores; no GPUs."
