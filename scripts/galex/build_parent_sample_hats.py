"""Convert raw GALEX GUVCat AIS catalog into a HATS catalog.

The cluster mirrors the GUVCat at:

    /mnt/ceph/users/polymathic/external_data/astro/galex/
        GUVCat_AIS_FOV055_glat*N*.fits.gz
        GUVCat_AIS_FOV055_glat*S*.fits.gz

There are 36 gzipped FITS shards split by galactic latitude. Each is a flat
photometric catalog with an ``objid`` column we use as ``object_id`` and
``ra``/``dec`` (after lowercasing).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pyarrow as pa
from astropy.table import Table

from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import to_native_endian, write_hats


CATALOG_NAME = "galex"


def find_raw_files(raw_root: str, max_files: int | None = None) -> list[str]:
    files = sorted(glob.glob(os.path.join(raw_root, "GUVCat_AIS_FOV055_glat*.fits.gz")))
    if max_files is not None:
        files = files[:max_files]
    return files


def read_shard(path: str) -> pa.Table:
    """Read one GUVCat FITS shard, lowercase all column names, normalize ids."""
    t = Table.read(path)
    # Lowercase column names to match the legacy MMU v1 schema.
    t.rename_columns(t.colnames, [c.lower() for c in t.colnames])

    if "ra" not in t.colnames or "dec" not in t.colnames:
        raise ValueError(f"{path} missing ra/dec columns")
    if "objid" not in t.colnames:
        raise ValueError(f"{path} missing objid column")

    columns: dict[str, pa.Array] = {
        "ra": pa.array(to_native_endian(np.asarray(t["ra"], dtype=np.float64))),
        "dec": pa.array(to_native_endian(np.asarray(t["dec"], dtype=np.float64))),
        "object_id": pa.array([str(int(x)) for x in t["objid"]], type=pa.string()),
    }
    for col in t.colnames:
        if col in ("ra", "dec", "objid"):
            continue
        arr = np.asarray(t[col])
        if arr.ndim != 1:
            continue
        columns[col] = pa.array(to_native_endian(arr))
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    args = parser.parse_args(argv)

    files = find_raw_files(args.raw_root, max_files=args.max_files)
    if not files:
        print(f"ERROR: no GUVCat files under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} GALEX shard(s)")

    tables = []
    total = 0
    for i, p in enumerate(files, 1):
        t = read_shard(p)
        tables.append(t)
        total += t.num_rows
        print(f"  [{i}/{len(files)}] {os.path.basename(p)}: {t.num_rows} rows")

    print(f"\nWriting HATS catalog: {total} rows from {len(tables)} shards")
    catalog_dir = write_hats(
        tables,
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
    )
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
