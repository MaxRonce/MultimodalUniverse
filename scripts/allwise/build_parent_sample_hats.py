"""Convert raw AllWISE parquet files into a HATS catalog.

The IRSA bulk-download AllWISE source catalog is already a parquet dataset
healpix-partitioned at k=0 (nside=1) inside k=5 (nside=32). On the Flatiron
cluster these files live at:

    /mnt/ceph/users/polymathic/external_data/astro/allwise/
        healpix_k0=*/healpix_k5=*/part0.snappy.parquet

Each file is ~50k rows and contains 298 columns. Conversion is essentially:
read parquet -> ensure ra/dec/object_id are typed correctly -> hand to
``write_hats``.

Usage:
    python -m scripts.allwise.build_parent_sample_hats \\
        --raw-root /mnt/ceph/users/polymathic/external_data/astro/allwise \\
        --output-root /mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats/allwise \\
        --max-files 4
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats


# IRSA's AllWISE bulk download is partitioned by healpix at two levels,
# visible in the file path: ``healpix_k0={pix}/healpix_k5={pix}/part*.parquet``.
# We use the k5 (nside=32) pixel to skip files whose footprint doesn't
# intersect the cone.
ALLWISE_K5_NSIDE = 32
ALLWISE_PATH_K5_RE = re.compile(r"healpix_k5=(\d+)")


CATALOG_NAME = "allwise"


def find_raw_files(
    raw_root: str,
    max_files: int | None = None,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> list[str]:
    """Find AllWISE raw parquet shards under ``raw_root``.

    If a cone cut is active, use ``healpy.query_disc`` to compute the set of
    ``nside=32`` pixels touching the cone, and skip any shard whose
    ``healpix_k5=N`` directory name doesn't fall in that set. A 1° cone
    typically reduces 12288 shards to ~2-4.
    """
    pattern = os.path.join(raw_root, "healpix_k0=*", "healpix_k5=*", "part*.parquet")
    files = sorted(glob.glob(pattern))

    if ra_center is not None and dec_center is not None and radius is not None:
        vec = hp.ang2vec(ra_center, dec_center, lonlat=True)
        keep = set(hp.query_disc(
            ALLWISE_K5_NSIDE, vec, np.deg2rad(radius),
            nest=True, inclusive=True,
        ).tolist())
        files = [
            f for f in files
            if (m := ALLWISE_PATH_K5_RE.search(f)) and int(m.group(1)) in keep
        ]

    if max_files is not None:
        files = files[:max_files]
    return files


def read_shard(path: str) -> pa.Table:
    """Read one AllWISE parquet shard and normalize the key columns.

    The raw schema already uses ``ra`` / ``dec`` (float64) and ``cntr`` (int64
    unique source ID). We add an ``object_id`` string column derived from
    ``cntr`` so the output matches the rest of the HATS catalogs.
    """
    table = pq.read_table(path)
    if "ra" not in table.schema.names or "dec" not in table.schema.names:
        raise ValueError(f"{path} missing ra/dec columns")
    if "object_id" not in table.schema.names:
        if "cntr" not in table.schema.names:
            raise ValueError(f"{path} missing both object_id and cntr")
        cntr = table.column("cntr").to_numpy()
        obj_id = pa.array([str(int(c)) for c in cntr], type=pa.string())
        table = table.append_column("object_id", obj_id)
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Root of the raw AllWISE parquet dataset on disk.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
        help="Where to write the HATS catalog (collection root).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit to the first N raw shards (for fast smoke tests).",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=8192,
        help="Max rows per HATS partition.",
    )
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    cone_active = (
        args.ra_center is not None
        and args.dec_center is not None
        and args.radius is not None
    )

    files = find_raw_files(
        args.raw_root,
        max_files=args.max_files,
        ra_center=args.ra_center,
        dec_center=args.dec_center,
        radius=args.radius,
    )
    if not files:
        print(f"ERROR: no raw parquet files under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} raw shard(s)")

    tables = []
    total_rows = 0
    for i, path in enumerate(files, 1):
        table = read_shard(path)
        if cone_active:
            ra = table.column("ra").to_numpy()
            dec = table.column("dec").to_numpy()
            mask = apply_cone_filter(ra, dec, args.ra_center, args.dec_center, args.radius)
            table = table.filter(pa.array(mask))
            if table.num_rows == 0:
                continue
        tables.append(table)
        total_rows += table.num_rows
        print(f"  [{i}/{len(files)}] {os.path.basename(os.path.dirname(path))}: {table.num_rows} rows")

    if not tables:
        print("ERROR: no rows survived the cone cut", file=sys.stderr)
        return 1

    print(f"\nWriting HATS catalog: {total_rows} rows from {len(tables)} shards")
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
