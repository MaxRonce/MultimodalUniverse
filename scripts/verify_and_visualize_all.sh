#!/bin/bash
# Verify and visualize all HATS catalogs under the v2 root.
# Walks ROOT/{dataset}/{catalog_name}/{catalog_name}/dataset/ structure.
#
# Usage: bash scripts/verify_and_visualize_all.sh [HEALPIX]

set -uo pipefail

HEALPIX="${1:-1177}"
ROOT="${ROOT:-/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats}"
PYTHON="${PYTHON:-$HOME/mmu_hats_demo/.venv/bin/python}"

VIS_DIR="$ROOT/visualizations"
mkdir -p "$VIS_DIR"

echo "Verifying and visualizing catalogs in $ROOT"
echo "==="

ok=0
fail=0
fail_list=()

# Walk per-dataset folders
for ds_dir in "$ROOT"/*/; do
    ds=$(basename "$ds_dir")
    if [ "$ds" = "visualizations" ] || [ "$ds" = "tmp" ]; then
        continue
    fi

    for collection_dir in "$ds_dir"*/; do
        collection_name=$(basename "$collection_dir")
        inner="$collection_dir/$collection_name"
        if [ ! -d "$inner/dataset" ]; then
            continue
        fi

        # Catalog name format: {dataset}_{config}
        cfg="${collection_name#${ds}_}"
        src="/mnt/ceph/users/polymathic/MultimodalUniverse/$ds/$cfg/healpix=$HEALPIX/001-of-001.hdf5"
        if [ ! -f "$src" ]; then
            echo "[$collection_name] SKIP: source HDF5 not found at $src"
            continue
        fi

        echo ""
        echo "[$collection_name] verifying..."
        if "$PYTHON" -m mmu.verify --catalog "$inner" --source-hdf5 "$src"; then
            ok=$((ok + 1))
            echo "[$collection_name] OK"
        else
            fail=$((fail + 1))
            fail_list+=("$collection_name")
            echo "[$collection_name] FAIL"
        fi

        "$PYTHON" scripts/visualize_hats_dataset.py \
            --catalog "$inner" \
            --output "$VIS_DIR/${collection_name}.png" || echo "  (visualization failed)"
    done
done

echo ""
echo "==="
echo "SUMMARY: $ok ok, $fail failed"
if [ $fail -gt 0 ]; then
    echo "Failed: ${fail_list[*]}"
fi
echo "Visualizations: $VIS_DIR"
