"""Convert JWST NIRCam deep-field cutouts into a HATS catalog.

The cluster mirrors the JWST data (already processed by v1's
``build_parent_sample.py``, which downloads FITS mosaics and extracts
96×96 cutouts) as HDF5 files under::

    /mnt/ceph/users/polymathic/external_data/astro/JWST/
        {survey}/healpix={N}/001-of-001.hdf5

where survey is one of: primer-cosmos, primer-uds, ceers, ngdeep, gds, gdn.

Each HDF5 file contains per-object arrays:
    - object_id   int64
    - ra, dec     float64
    - healpix     int64
    - image_band  (N_filters,) bytes — e.g. b'f090w'
    - image_flux  (N_filters, 96, 96) float32
    - image_ivar  (N_filters, 96, 96) float32
    - image_mask  (N_filters, 96, 96) bool
    - image_psf_fwhm (N_filters,) float32
    - image_scale (N_filters,) float32
    - mag_auto, flux_radius, flux_auto, fluxerr_auto,
      cxx_image, cyy_image, cxy_image   float32 scalars

Schema::

    image: struct<
        band:     list<string>,                   # e.g. ['f090w', ..., 'f444w']
        flux:     list<list<list<float32>>>,      # (N_filters, 96, 96)
        ivar:     list<list<list<float32>>>,      # (N_filters, 96, 96)
        mask:     list<list<list<bool>>>,         # (N_filters, 96, 96)
        psf_fwhm: list<float32>,                  # N_filters floats
        scale:    list<float32>,                  # N_filters floats
    >
    + mag_auto, flux_radius, flux_auto, fluxerr_auto,
      cxx_image, cyy_image, cxy_image  (float32)
    + object_id (string), ra (float64), dec (float64)
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _as_array(arr):
    return arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr


from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import (
    default_scratch_dir,
    write_hats_from_parquet_dir,
)


CATALOG_NAME = "jwst"

IMAGE_SIZE = 96
SURVEYS = ["primer-cosmos", "primer-uds", "ceers", "ngdeep", "gds", "gdn"]

FLOAT_FEATURES = [
    "mag_auto", "flux_radius", "flux_auto", "fluxerr_auto",
    "cxx_image", "cyy_image", "cxy_image",
]


def find_raw_files(raw_root: str, max_files: int | None = None) -> list[str]:
    """Find all HDF5 files under raw_root (any survey/healpix layout)."""
    files = sorted(glob.glob(os.path.join(raw_root, "**", "*.hdf5"), recursive=True))
    if max_files is not None:
        files = files[:max_files]
    return files


def _decode_band(val) -> str:
    """Decode a bytes or str band name to a plain Python str."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    return str(val)


def read_hdf5(
    path: str,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> pa.Table | None:
    """Read one JWST HDF5 file and return a PyArrow table, or None.

    Applies cone cut early if parameters are given.
    """
    with h5py.File(path, "r") as f:
        ra = np.asarray(f["ra"][:], dtype=np.float64)
        dec = np.asarray(f["dec"][:], dtype=np.float64)
        n_total = ra.shape[0]

        if ra_center is not None and dec_center is not None and radius is not None:
            cone_mask = apply_cone_filter(ra, dec, ra_center, dec_center, radius)
            keep = np.where(cone_mask)[0]
            if keep.size == 0:
                return None
        else:
            keep = np.arange(n_total)

        ra = ra[keep]
        dec = dec[keep]
        n_rows = len(keep)

        object_id = np.asarray(f["object_id"][keep])
        image_flux = np.asarray(f["image_flux"][keep], dtype=np.float32)  # (N, F, H, W)
        image_ivar = np.asarray(f["image_ivar"][keep], dtype=np.float32)
        image_mask = np.asarray(f["image_mask"][keep], dtype=bool)
        image_psf_fwhm = np.asarray(f["image_psf_fwhm"][keep], dtype=np.float32)
        image_scale = np.asarray(f["image_scale"][keep], dtype=np.float32)
        image_band_raw = f["image_band"][keep]  # (N, N_filters) bytes

        scalars = {}
        for feat in FLOAT_FEATURES:
            if feat in f:
                scalars[feat] = np.asarray(f[feat][keep], dtype=np.float32)

    n_filters = image_flux.shape[1]

    # Build band list per row (decode bytes → str)
    bands_per_row = [
        [_decode_band(image_band_raw[i, j]) for j in range(n_filters)]
        for i in range(n_rows)
    ]

    band_arr = pa.array(bands_per_row, type=pa.list_(pa.string()))

    # Nested list arrays: list<list<list<float32>>> shape (N_filters, H, W)
    flux_nested = [[[list(row) for row in image_flux[i, f_]] for f_ in range(n_filters)] for i in range(n_rows)]
    ivar_nested = [[[list(row) for row in image_ivar[i, f_]] for f_ in range(n_filters)] for i in range(n_rows)]
    mask_nested = [[[list(map(bool, row)) for row in image_mask[i, f_]] for f_ in range(n_filters)] for i in range(n_rows)]

    flux_arr = pa.array(flux_nested, type=pa.list_(pa.list_(pa.list_(pa.float32()))))
    ivar_arr = pa.array(ivar_nested, type=pa.list_(pa.list_(pa.list_(pa.float32()))))
    mask_arr = pa.array(mask_nested, type=pa.list_(pa.list_(pa.list_(pa.bool_()))))
    psf_fwhm_arr = pa.array(
        [list(image_psf_fwhm[i]) for i in range(n_rows)],
        type=pa.list_(pa.float32()),
    )
    scale_arr = pa.array(
        [list(image_scale[i]) for i in range(n_rows)],
        type=pa.list_(pa.float32()),
    )

    image_col = pa.StructArray.from_arrays(
            [_as_array(a) for a in [band_arr, flux_arr, ivar_arr, mask_arr, psf_fwhm_arr, scale_arr]],
        names=["band", "flux", "ivar", "mask", "psf_fwhm", "scale"],
    )

    columns: dict[str, pa.Array] = {
        "ra": pa.array(ra, type=pa.float64()),
        "dec": pa.array(dec, type=pa.float64()),
        "object_id": pa.array([str(int(v)) for v in object_id], type=pa.string()),
        "image": image_col,
    }
    for feat, arr in scalars.items():
        columns[feat] = pa.array(arr, type=pa.float32())

    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        default=DATASETS[CATALOG_NAME].raw_path,
        help="Root directory containing per-survey HDF5 healpix files.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Directory for per-file parquet shards before HATS ingest.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap on number of HDF5 files to process.",
    )
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Cone radius in degrees; requires --ra-center/--dec-center.",
    )
    args = parser.parse_args(argv)

    files = find_raw_files(args.raw_root, max_files=args.max_files)
    if not files:
        print(f"ERROR: no .hdf5 files found under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} JWST HDF5 file(s)", flush=True)

    scratch_dir = args.scratch_dir or default_scratch_dir(CATALOG_NAME)
    os.makedirs(scratch_dir, exist_ok=True)

    n_written = 0
    for idx, path in enumerate(files):
        table = read_hdf5(
            path,
            ra_center=args.ra_center,
            dec_center=args.dec_center,
            radius=args.radius,
        )
        if table is None or table.num_rows == 0:
            continue
        shard_path = os.path.join(scratch_dir, f"part-{idx:04d}.parquet")
        pq.write_table(table, shard_path)
        n_written += 1
        print(f"  {os.path.basename(path)}: {table.num_rows} rows", flush=True)

    if n_written == 0:
        print("No rows after filtering; nothing to write.", file=sys.stderr)
        return 1

    catalog_dir = write_hats_from_parquet_dir(
        scratch_dir,
        output_path=args.output_root,
        catalog_name=CATALOG_NAME,
        pixel_threshold=args.pixel_threshold,
        debug=True,
    )
    shutil.rmtree(scratch_dir, ignore_errors=True)
    print(f"Done: {catalog_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
