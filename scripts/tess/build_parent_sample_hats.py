"""Convert raw TESS-SPOC FFI lightcurves into a HATS catalog.

The cluster mirrors TESS SPOC FFI lightcurves at:

    /mnt/ceph/users/polymathic/external_data/astro/tess/
        hlsp_tess-spoc_tess_phot_{tic:016d}-s{sector:04d}_tess_v1_lc.fits

Each FITS file is one lightcurve for one TIC star observed in one sector.
The script reads:

    HDU[1] = LIGHTCURVE table  -> TIME, SAP_FLUX, SAP_FLUX_ERR, QUALITY
    HDU[1] header              -> ra_obj, dec_obj

and produces one row per (TIC, sector) with a ``lightcurve`` struct column
holding the time-series arrays.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pyarrow as pa
from astropy.io import fits

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import to_native_endian, write_hats


CATALOG_NAME = "tess"
FILENAME_RE = re.compile(
    r"hlsp_tess-spoc_tess_phot_(?P<tic>\d+)-s(?P<sector>\d+)_tess_v1_lc\.fits$"
)


def parse_filename(path: str) -> tuple[int, int] | None:
    """Extract (TIC ID, sector) from a SPOC FFI lightcurve filename."""
    m = FILENAME_RE.search(os.path.basename(path))
    if m is None:
        return None
    return int(m["tic"]), int(m["sector"])


def find_raw_files(raw_root: str, max_files: int | None = None) -> list[str]:
    files = sorted(glob.glob(os.path.join(raw_root, "hlsp_tess-spoc_tess_phot_*_tess_v1_lc.fits")))
    if max_files is not None:
        files = files[:max_files]
    return files


def read_lightcurve(
    path: str,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> dict | None:
    """Read one TESS SPOC FFI lightcurve. Returns a per-row dict, or None on
    failure (file is unreadable / missing the expected columns / no TIC in name /
    outside the cone cut if one is active).
    """
    parsed = parse_filename(path)
    if parsed is None:
        return None
    tic, sector = parsed

    with fits.open(path, memmap=False) as hdul:
        if "LIGHTCURVE" not in [h.name for h in hdul]:
            return None
        hdr = hdul[1].header
        try:
            ra = float(hdr["ra_obj"])
            dec = float(hdr["dec_obj"])
        except KeyError:
            return None

        # Cone cut BEFORE expensive array extraction.
        if ra_center is not None and dec_center is not None and radius is not None:
            if not apply_cone_filter(
                np.array([ra]), np.array([dec]),
                ra_center, dec_center, radius,
            )[0]:
                return None

        lc = hdul["LIGHTCURVE"].data

        time = np.asarray(lc["TIME"], dtype=np.float64)
        flux = np.asarray(lc["SAP_FLUX"], dtype=np.float32)
        flux_err = np.asarray(lc["SAP_FLUX_ERR"], dtype=np.float32)
        quality = np.asarray(lc["QUALITY"], dtype=np.int32)

    # Drop NaN time points (the SPOC files have ~10% NaN cadences).
    valid = np.isfinite(time) & np.isfinite(flux)
    time = time[valid]
    flux = flux[valid]
    flux_err = flux_err[valid]
    quality = quality[valid]

    return {
        "tic_id": tic,
        "sector": sector,
        "ra": ra,
        "dec": dec,
        "time": to_native_endian(time),
        "flux": to_native_endian(flux),
        "flux_err": to_native_endian(flux_err),
        "quality": to_native_endian(quality),
    }


def build_table(rows: list[dict]) -> pa.Table:
    """Stack per-lightcurve dicts into a single PyArrow table.

    Each lightcurve keeps its true length: stored as PyArrow ``list_<float>``
    columns inside a ``lightcurve`` struct. No padding, no fixed-width 2D.
    Different rows can (and typically do) have different lengths.
    """
    columns = {
        "ra": pa.array([r["ra"] for r in rows], type=pa.float64()),
        "dec": pa.array([r["dec"] for r in rows], type=pa.float64()),
        "object_id": pa.array([str(r["tic_id"]) for r in rows], type=pa.string()),
        "tic_id": pa.array([r["tic_id"] for r in rows], type=pa.int64()),
        "sector": pa.array([r["sector"] for r in rows], type=pa.int32()),
        "lightcurve": pa.StructArray.from_arrays(
            [
                pa.array([r["time"] for r in rows], type=pa.list_(pa.float64())),
                pa.array([r["flux"] for r in rows], type=pa.list_(pa.float32())),
                pa.array([r["flux_err"] for r in rows], type=pa.list_(pa.float32())),
                pa.array([r["quality"] for r in rows], type=pa.list_(pa.int32())),
            ],
            names=["time", "flux", "flux_err", "quality"],
        ),
    }
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    files = find_raw_files(args.raw_root, max_files=args.max_files)
    if not files:
        print(f"ERROR: no SPOC lightcurves under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} TESS SPOC file(s)")

    rows: list[dict] = []
    for i, p in enumerate(files, 1):
        row = read_lightcurve(
            p,
            ra_center=args.ra_center,
            dec_center=args.dec_center,
            radius=args.radius,
        )
        if row is None:
            continue
        rows.append(row)
        print(f"  [{i}/{len(files)}] TIC {row['tic_id']} s{row['sector']:04d}: {len(row['time'])} cadences")

    if not rows:
        print("ERROR: no readable lightcurves", file=sys.stderr)
        return 1

    table = build_table(rows)
    print(f"\nWriting HATS catalog: {table.num_rows} lightcurves")
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
