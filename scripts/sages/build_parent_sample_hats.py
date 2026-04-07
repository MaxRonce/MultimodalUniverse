"""Convert raw SAGES DR1 catalog into a HATS catalog.

The cluster mirrors SAGES DR1 at:

    /mnt/ceph/users/polymathic/external_data/astro/sages/
        dr1-uv.fits     # u/v photometry catalog (used here, ~4.2 GB)
        dr1s-gri.fits   # g/r/i photometry (separate paper, not used)

Conversion: read the FITS table, apply the same MAG/FLAG cuts as the legacy
``scripts/sages/process.py``, normalize RA/DEC/SAGE_ID -> ra/dec/object_id,
and feed to ``write_hats``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
from astropy.table import Table

from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import to_native_endian, write_hats


CATALOG_NAME = "sages"
DEFAULT_INPUT = "dr1-uv.fits"


def selection_fn(t: Table) -> np.ndarray:
    """Same cuts as the legacy MMU v1 SAGES process.py: keep only rows with
    valid u/v magnitudes (>-999) and clean photometric flags."""
    mask = (np.asarray(t["MAG_U"]) > -999) & (np.asarray(t["MAG_V"]) > -999)
    mask &= np.asarray(t["FLAG_U"]) == 0
    mask &= np.asarray(t["FLAG_V"]) == 0
    return mask


def read_table(raw_root: str, max_rows: int | None = None) -> pa.Table:
    """Read dr1-uv.fits, apply cuts, and return a PyArrow table normalized for HATS."""
    fits_path = os.path.join(raw_root, DEFAULT_INPUT)
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"Missing {fits_path}")

    t = Table.read(fits_path)
    if max_rows is not None:
        t = t[:max_rows]
    t = t[selection_fn(t)]

    # Build PyArrow columns explicitly so we control the schema and casting.
    columns: dict[str, pa.Array] = {
        "ra": pa.array(to_native_endian(np.asarray(t["RA"], dtype=np.float64))),
        "dec": pa.array(to_native_endian(np.asarray(t["DEC"], dtype=np.float64))),
        "object_id": pa.array([str(int(x)) for x in t["SAGE_ID"]], type=pa.string()),
    }
    # Pass everything else through with whatever dtype astropy gave us.
    for col_name in t.colnames:
        if col_name in ("RA", "DEC", "SAGE_ID"):
            continue
        col = np.asarray(t[col_name])
        if col.ndim != 1:
            continue  # SAGES is flat tabular; skip anything weird
        columns[col_name] = pa.array(to_native_endian(col))

    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Compatibility with the Snakefile interface; for SAGES this caps "
             "the number of catalog rows read (since there's only one file).",
    )
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    args = parser.parse_args(argv)

    print(f"Reading {DEFAULT_INPUT} from {args.raw_root}")
    table = read_table(args.raw_root, max_rows=args.max_files * 100_000 if args.max_files else None)
    print(f"After cuts: {table.num_rows} rows")

    catalog_dir = write_hats(
        [table],
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
    )
    print(f"Done: {catalog_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
