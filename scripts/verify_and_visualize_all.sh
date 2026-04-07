#!/bin/bash
# Verify and visualize all HATS catalogs in a healpix output directory.
# Run on the cluster after run_fiducial_hats.sh has finished.
#
# Usage: bash scripts/verify_and_visualize_all.sh [HEALPIX]

set -uo pipefail

HEALPIX="${1:-1177}"
ROOT="/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/healpix_${HEALPIX}"
PYTHON="${PYTHON:-$HOME/mmu_hats_demo/.venv/bin/python}"

VIS_DIR="$ROOT/visualizations"
mkdir -p "$VIS_DIR"

echo "Verifying and visualizing catalogs in $ROOT"
echo "==="

ok=0
fail=0
fail_list=()

for catalog_dir in "$ROOT"/*/; do
    name=$(basename "$catalog_dir")
    if [ "$name" = "tmp" ] || [ "$name" = "visualizations" ]; then
        continue
    fi

    inner="$catalog_dir/$name"
    if [ ! -d "$inner/dataset" ]; then
        continue
    fi

    # Figure out source HDF5 from catalog name like "sdss_sdss" -> dataset=sdss, config=sdss
    # First underscore-split tries dataset/config
    ds="${name%%_*}"
    cfg="${name#*_}"
    src="/mnt/ceph/users/polymathic/MultimodalUniverse/$ds/$cfg/healpix=$HEALPIX/001-of-001.hdf5"
    if [ ! -f "$src" ]; then
        echo "[$name] SKIP: source HDF5 not found at $src"
        continue
    fi

    echo ""
    echo "[$name] verifying..."
    if "$PYTHON" -m mmu.verify --catalog "$inner" --source-hdf5 "$src"; then
        ok=$((ok + 1))
        echo "[$name] OK"
    else
        fail=$((fail + 1))
        fail_list+=("$name")
        echo "[$name] FAIL"
    fi

    # Visualize regardless
    "$PYTHON" scripts/visualize_hats_dataset.py \
        --catalog "$inner" \
        --output "$VIS_DIR/${name}.png" || echo "  (visualization failed)"
done

echo ""
echo "==="
echo "SUMMARY: $ok ok, $fail failed"
if [ $fail -gt 0 ]; then
    echo "Failed: ${fail_list[*]}"
fi
echo "Visualizations: $VIS_DIR"
