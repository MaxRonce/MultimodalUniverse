#!/bin/bash
# Run mmu.cli.build_hats for the fiducial healpix tile across all available datasets.
# Designed to run on the Flatiron cluster.
#
# Usage: bash scripts/run_fiducial_hats.sh [HEALPIX] [N_ROWS]
#
# Defaults: healpix=1177, n_rows=2000

set -uo pipefail

HEALPIX="${1:-1177}"
N_ROWS="${2:-2000}"
ROOT="${ROOT:-/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats}"

# Locate venv python (must be set up before running this script)
PYTHON="${PYTHON:-$HOME/mmu_hats_demo/.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: $PYTHON not found. Set PYTHON env var or create venv at ~/mmu_hats_demo/.venv"
    exit 1
fi

mkdir -p "$ROOT"

# Datasets known to have data at healpix=1177 (verified 2026-04-07)
# Format: "dataset:config"
DATASETS=(
    "sdss:sdss"
    "sdss:boss"
    "desi:dr1_main"
    "gaia:gaia"
    "legacysurvey:dr10_south_21"
    "allwise:allwise"
    "twomass:psc"
    "galex:ais"
    "galah:dr3"
    "apogee:apogee"
    "tess:spoc"
    "sages:dr1"
    "chandra:spectra"
)

LOG="$ROOT/run.log"
: > "$LOG"
echo "Starting batch HATS conversion for healpix=$HEALPIX (n_rows=$N_ROWS)" | tee -a "$LOG"
echo "Root: $ROOT" | tee -a "$LOG"
echo "Datasets: ${#DATASETS[@]}" | tee -a "$LOG"
echo "===" | tee -a "$LOG"

success=0
failed=0
failed_list=()

for entry in "${DATASETS[@]}"; do
    ds="${entry%%:*}"
    cfg="${entry##*:}"
    out_dir="$ROOT/$ds"
    mkdir -p "$out_dir"
    echo "" | tee -a "$LOG"
    echo "[$(date +%H:%M:%S)] $ds/$cfg -> $out_dir" | tee -a "$LOG"

    if "$PYTHON" -m mmu.cli.build_hats \
        --dataset "$ds" \
        --config "$cfg" \
        --healpix "$HEALPIX" \
        --n-rows "$N_ROWS" \
        --output "$out_dir" 2>&1 | tee -a "$LOG"; then
        success=$((success + 1))
        echo "  [$(date +%H:%M:%S)] OK $ds/$cfg" | tee -a "$LOG"
    else
        failed=$((failed + 1))
        failed_list+=("$ds/$cfg")
        echo "  [$(date +%H:%M:%S)] FAIL $ds/$cfg" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "===" | tee -a "$LOG"
echo "SUMMARY: $success ok, $failed failed" | tee -a "$LOG"
if [ $failed -gt 0 ]; then
    echo "Failed: ${failed_list[*]}" | tee -a "$LOG"
fi
