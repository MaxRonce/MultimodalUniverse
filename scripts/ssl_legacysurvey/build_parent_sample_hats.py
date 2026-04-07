"""Convert raw DECaLS Stein-et-al image dataset into a HATS catalog.

The cluster mirrors the Stein-et-al image chunks at:

    /mnt/ceph/users/polymathic/external_data/astro/DECALS_Stein_et_al/north/
        images_npix152_NNNNNNNNN_NNNNNNNNN.h5

Each h5 file contains up to 1,000,000 objects with:
    - inds, ra, dec
    - release, brickid, objid
    - z_spec
    - flux         shape (N, 3)     per-band flux
    - fiberflux    shape (N, 3)
    - psfdepth     shape (N, 3)
    - psfsize      shape (N, 3)
    - ebv          shape (N,)
    - images       shape (N, 3, 152, 152)   <-- the actual cutouts

The 3 bands are DES-G, DES-R, DES-Z. Image cubes are stored as a PyArrow
``fixed_shape_tensor(float32, (3, 152, 152))`` extension column. This preserves
the per-row tensor shape in the parquet schema metadata, reads back as a numpy
tensor through pyarrow natively, and is compatible with hats-import's
metadata-combination step (unlike ``Array2DExtensionType`` from HuggingFace
``datasets``, which crashes the finishing stage on nested extension types).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pyarrow as pa

from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import to_native_endian, write_hats


CATALOG_NAME = "ssl_legacysurvey"

IMAGE_SIZE = 152
BANDS = ["DES-G", "DES-R", "DES-Z"]
N_BANDS = len(BANDS)
PIXEL_SCALE = 0.262
IMAGE_TENSOR_TYPE = pa.fixed_shape_tensor(pa.float32(), (N_BANDS, IMAGE_SIZE, IMAGE_SIZE))

# Per-object scalar columns from each h5 chunk that we keep in the HATS catalog.
SCALAR_COLUMNS = [
    "ebv",
    "z_spec",
]
# Multi-band scalar columns (each is shape (N, 3)) — split into per-band columns.
PER_BAND_COLUMNS = [
    "flux",
    "fiberflux",
    "psfdepth",
]
PER_BAND_SUFFIXES = ["g", "r", "z"]


def find_raw_files(raw_root: str, max_files: int | None = None) -> list[str]:
    """Find Stein-et-al image chunk h5 files under ``raw_root``."""
    files = sorted(glob.glob(os.path.join(raw_root, "images_npix152_*.h5")))
    if max_files is not None:
        files = files[:max_files]
    return files


def _build_image_tensor(image_array: np.ndarray) -> pa.Array:
    """Build a ``fixed_shape_tensor<float32, (3, 152, 152)>`` column from a
    (N, 3, 152, 152) numpy cube.
    """
    if image_array.shape[1:] != (N_BANDS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"image_array must be (N, {N_BANDS}, {IMAGE_SIZE}, {IMAGE_SIZE}); "
            f"got {image_array.shape}"
        )
    return pa.FixedShapeTensorArray.from_numpy_ndarray(
        np.ascontiguousarray(image_array, dtype=np.float32)
    )


def read_chunk(path: str, max_rows: int | None = None) -> pa.Table:
    """Read one images_npix152_*.h5 file and return a PyArrow table."""
    with h5py.File(path, "r") as f:
        n_total = f["ra"].shape[0]
        n = min(n_total, max_rows) if max_rows else n_total

        ra = np.asarray(f["ra"][:n], dtype=np.float64)
        dec = np.asarray(f["dec"][:n], dtype=np.float64)
        # ssl_legacysurvey uses `inds` as the unique source ID.
        inds = np.asarray(f["inds"][:n])

        image_array = np.asarray(f["images"][:n], dtype=np.float32)
        psfsize = np.asarray(f["psfsize"][:n], dtype=np.float32)

        scalars: dict[str, np.ndarray] = {}
        for c in SCALAR_COLUMNS:
            if c in f:
                scalars[c] = np.asarray(f[c][:n], dtype=np.float32)

        per_band: dict[str, np.ndarray] = {}
        for c in PER_BAND_COLUMNS:
            if c in f:
                per_band[c] = np.asarray(f[c][:n], dtype=np.float32)

    columns: dict[str, pa.Array] = {
        "ra": pa.array(to_native_endian(ra)),
        "dec": pa.array(to_native_endian(dec)),
        "object_id": pa.array([str(int(i)) for i in inds], type=pa.string()),
        # Fixed-shape tensor: (3, 152, 152) float32 per row. Shape is in the
        # schema; downstream reads this as a (3,152,152) numpy array per row.
        "image": _build_image_tensor(image_array),
        # Per-band PSF size (FWHM in pixels) for the three bands.
        "psf_fwhm": pa.FixedShapeTensorArray.from_numpy_ndarray(
            np.ascontiguousarray(psfsize, dtype=np.float32)
        ),
    }
    for name, arr in scalars.items():
        columns[name] = pa.array(to_native_endian(arr))
    for name, arr in per_band.items():
        for band_idx, suffix in enumerate(PER_BAND_SUFFIXES):
            columns[f"{name}_{suffix}"] = pa.array(to_native_endian(arr[:, band_idx]))

    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=os.path.join(DATASETS[CATALOG_NAME].raw_path, "north"))
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on number of image chunks (each chunk is ~1M objects).")
    parser.add_argument("--max-rows-per-file", type=int, default=None,
                        help="Cap rows read PER chunk (useful when chunks are very large).")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    args = parser.parse_args(argv)

    files = find_raw_files(args.raw_root, max_files=args.max_files)
    if not files:
        print(f"ERROR: no images_npix152_*.h5 under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} ssl_legacysurvey chunk(s)")

    tables: list[pa.Table] = []
    total = 0
    for i, p in enumerate(files, 1):
        t = read_chunk(p, max_rows=args.max_rows_per_file)
        tables.append(t)
        total += t.num_rows
        print(f"  [{i}/{len(files)}] {os.path.basename(p)}: {t.num_rows} objects")

    print(f"\nWriting HATS catalog: {total} rows from {len(tables)} chunks")
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
