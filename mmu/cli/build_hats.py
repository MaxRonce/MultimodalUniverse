"""CLI to convert one or more MMU v1 HDF5 healpix tiles to a HATS catalog.

Usage:
    python -m mmu.cli.build_hats --dataset sdss --healpix 583
    python -m mmu.cli.build_hats --dataset sdss --healpix 583 584 585
    python -m mmu.cli.build_hats --dataset desi --healpix 626 \
        --output /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats
"""

import argparse
import os
import sys

import h5py

from mmu.hats_configs import (
    DATASET_CONFIGS,
    MMU_V2_HATS_ROOT,
    get_dataset_config,
    hdf5_path,
    list_datasets,
)
from mmu.hats_import import auto_arrow_table_from_hdf5, write_hats


def convert_tiles(
    dataset: str,
    healpix_list: list[int],
    output_root: str,
    config: str | None = None,
    n_rows: int | None = None,
    pixel_threshold: int = 8192,
    debug: bool = True,
) -> str:
    """Convert N healpix HDF5 tiles to a single HATS catalog.

    Returns the catalog directory path.
    """
    cfg = get_dataset_config(dataset)
    if config is None:
        config = cfg.get("default_config")

    catalog_name = f"{dataset}_{config}" if config else dataset

    # Build PyArrow tables from each tile
    tables = []
    for hp in healpix_list:
        path = hdf5_path(dataset, hp, config=config)
        if not os.path.exists(path):
            print(f"  WARN: missing tile, skipping: {path}", file=sys.stderr)
            continue
        print(f"  Loading {path}")
        with h5py.File(path, "r") as f:
            n_obj = f[next(iter(f.keys()))].shape[0]
            print(f"    rows: {n_obj}", end="")
            if n_rows is not None and n_obj > n_rows:
                print(f" (taking first {n_rows})")
            else:
                print()
            table = auto_arrow_table_from_hdf5(f, n_rows=n_rows)
        tables.append(table)

    if not tables:
        raise RuntimeError(f"No tiles found for {dataset} healpix={healpix_list}")

    total = sum(t.num_rows for t in tables)
    print(f"\nWriting HATS for {catalog_name} ({total} rows from {len(tables)} tile(s))...")
    print(f"  Output: {output_root}/{catalog_name}")

    write_hats(
        tables,
        output_path=output_root,
        catalog_name=catalog_name,
        pixel_threshold=pixel_threshold,
        debug=debug,
    )

    return os.path.join(output_root, catalog_name, catalog_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", "-d", required=True,
        help=f"Dataset name. Available: {', '.join(list_datasets())}",
    )
    parser.add_argument(
        "--healpix", "-p", required=True, type=int, nargs="+",
        help="One or more healpix tile IDs to convert",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Sub-config name (e.g. 'sdss', 'boss'). Defaults to dataset's default.",
    )
    parser.add_argument(
        "--output", "-o", default=MMU_V2_HATS_ROOT,
        help=f"Output root directory (default: {MMU_V2_HATS_ROOT})",
    )
    parser.add_argument(
        "--n-rows", type=int, default=None,
        help="Limit to first N rows per tile (for fast testing)",
    )
    parser.add_argument(
        "--pixel-threshold", type=int, default=8192,
        help="Max rows per HATS partition",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Use multi-worker Dask client (faster, less debuggable)",
    )
    args = parser.parse_args(argv)

    if args.dataset not in DATASET_CONFIGS:
        print(
            f"Unknown dataset {args.dataset!r}. Known: {sorted(DATASET_CONFIGS)}",
            file=sys.stderr,
        )
        return 1

    catalog_dir = convert_tiles(
        dataset=args.dataset,
        healpix_list=args.healpix,
        output_root=args.output,
        config=args.config,
        n_rows=args.n_rows,
        pixel_threshold=args.pixel_threshold,
        debug=not args.no_debug,
    )
    print(f"\nDone: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
