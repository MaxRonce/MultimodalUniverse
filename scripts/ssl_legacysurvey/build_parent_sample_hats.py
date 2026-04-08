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

The 3 bands are DES-G, DES-R, DES-Z.

The image column is stored as an ``image`` struct-of-parallel-lists matching
Mike's v1 SSL LegacySurvey transformer:

    image: struct<
        band:     list<string>,                      # 3 band names per row
        flux:     list<list<list<float32>>>,         # (3, 152, 152) per row
        psf_fwhm: list<float32>,                     # 3 floats per row
        scale:    list<float32>,                     # 3 floats per row
    >

Plain nested lists — no extension type — because ``nested_pandas`` (which
hats-import uses at the ``Catalog: Finishing`` stage to read the combined
``_common_metadata`` schema) has two bugs that crash when HF ``datasets``
extension types live inside a ``list<>`` or ``struct<>`` subtree:

1. ``list<Array*DExtensionType>``: nested_pandas' ``autocast_list`` packs every
   list column as a nested subframe, which round-trips through pandas → arrow
   and trips ``arrow/array/array_nested.cc:459`` (``child_data.size() == 1``).
2. ``struct<..., Array*DExtensionType, ...>``: nested_pandas' ``normalize_struct_list_type``
   calls ``pa.list_(field.type.value_type)``; ``Array3DExtensionType.value_type``
   returns the *string* ``'float32'`` (HF naming convention) rather than a
   pyarrow ``DataType``, so ``pa.list_`` raises ``TypeError``.

Plain nested lists sidestep both. ``datasets.load_dataset`` on the parquet
output reads the struct back as a list-of-items dict; callers reconstruct the
(3, 152, 152) cube via ``np.asarray(row['image']['flux'])``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import pyarrow as pa

from mmu.cone import apply_cone_filter
from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import to_native_endian, write_hats


CATALOG_NAME = "ssl_legacysurvey"

IMAGE_SIZE = 152
BANDS = ["DES-G", "DES-R", "DES-Z"]
N_BANDS = len(BANDS)
PIXEL_SCALE = 0.262

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
    """Find Stein-et-al image chunk h5 files under ``raw_root``.

    Accepts either (a) a raw root directly containing ``images_npix152_*.h5``
    files, or (b) the Stein-et-al top-level directory containing ``north/``
    and/or ``south/`` subdirectories. In case (b) we recurse into both and
    concatenate the shard lists, so cone cuts can pick up the COSMOS field
    (Dec=+2) which lives in DECaLS south, not north.
    """
    files = sorted(glob.glob(os.path.join(raw_root, "images_npix152_*.h5")))
    if not files:
        for sub in ("north", "south"):
            sub_path = os.path.join(raw_root, sub)
            if os.path.isdir(sub_path):
                files.extend(sorted(glob.glob(os.path.join(sub_path, "images_npix152_*.h5"))))
    if max_files is not None:
        files = files[:max_files]
    return files


def _build_image_struct(image_array: np.ndarray, psfsize: np.ndarray) -> pa.StructArray:
    """Build an ``image`` struct-of-parallel-lists column matching Mike's v1
    SSL LegacySurvey transformer schema::

        image: struct<
            band:     list<string>,
            flux:     list<list<list<float32>>>,
            psf_fwhm: list<float32>,
            scale:    list<float32>,
        >

    ``image_array`` must be (N, N_BANDS, IMAGE_SIZE, IMAGE_SIZE). ``psfsize``
    must be (N, N_BANDS). Pixel scale is constant (``PIXEL_SCALE``) and is
    broadcast to a list of length N_BANDS per row.
    """
    if image_array.shape[1:] != (N_BANDS, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"image_array must be (N, {N_BANDS}, {IMAGE_SIZE}, {IMAGE_SIZE}); "
            f"got {image_array.shape}"
        )
    if psfsize.shape[1:] != (N_BANDS,):
        raise ValueError(
            f"psfsize must be (N, {N_BANDS}); got {psfsize.shape}"
        )
    n = image_array.shape[0]
    arr = np.ascontiguousarray(image_array, dtype=np.float32)

    band_arr = pa.array([BANDS] * n, type=pa.list_(pa.string()))
    nested_flux = [[[list(row) for row in img[b]] for b in range(N_BANDS)] for img in arr]
    flux_arr = pa.array(nested_flux, type=pa.list_(pa.list_(pa.list_(pa.float32()))))
    psf_fwhm_arr = pa.array(
        [list(row) for row in psfsize.astype(np.float32)],
        type=pa.list_(pa.float32()),
    )
    scale_arr = pa.array(
        [[PIXEL_SCALE] * N_BANDS] * n,
        type=pa.list_(pa.float32()),
    )
    return pa.StructArray.from_arrays(
        [band_arr, flux_arr, psf_fwhm_arr, scale_arr],
        names=["band", "flux", "psf_fwhm", "scale"],
    )


def read_chunk(
    path: str,
    max_rows: int | None = None,
    ra_center: float | None = None,
    dec_center: float | None = None,
    radius: float | None = None,
) -> pa.Table | None:
    """Read one images_npix152_*.h5 file and return a PyArrow table.

    If a cone cut is active (``ra_center``/``dec_center``/``radius``), ra/dec
    are read first and the image cube / per-band arrays are read only for
    surviving indices. Returns ``None`` if no rows survive the cut.
    """
    with h5py.File(path, "r") as f:
        n_total = f["ra"].shape[0]
        n = min(n_total, max_rows) if max_rows else n_total

        # Read ra/dec first — this is cheap (a few MB per chunk).
        ra = np.asarray(f["ra"][:n], dtype=np.float64)
        dec = np.asarray(f["dec"][:n], dtype=np.float64)

        if ra_center is not None and dec_center is not None and radius is not None:
            cone_mask = apply_cone_filter(ra, dec, ra_center, dec_center, radius)
            keep = np.where(cone_mask)[0]
            if keep.size == 0:
                return None
            ra = ra[keep]
            dec = dec[keep]
            use_fancy = True
        else:
            keep = None
            use_fancy = False

        def _load(name: str, dtype=None):
            """Load a dataset with h5py fancy indexing when a cone cut is
            active, or a bulk slice otherwise.

            Without fancy indexing (`f[name][keep]`), the previous version
            read the entire chunk into memory (~277 MB for the ``images``
            cube) and then sliced in numpy — which is *orders of magnitude*
            slower than h5py's native selection against the underlying
            HDF5 chunks, especially over Ceph.
            """
            arr = f[name][keep] if use_fancy else f[name][:n]
            if dtype is not None:
                arr = np.asarray(arr, dtype=dtype)
            return arr

        # ssl_legacysurvey uses `inds` as the unique source ID.
        inds = _load("inds")

        image_array = _load("images", dtype=np.float32)
        psfsize = _load("psfsize", dtype=np.float32)

        scalars: dict[str, np.ndarray] = {}
        for c in SCALAR_COLUMNS:
            if c in f:
                scalars[c] = _load(c, dtype=np.float32)

        per_band: dict[str, np.ndarray] = {}
        for c in PER_BAND_COLUMNS:
            if c in f:
                per_band[c] = _load(c, dtype=np.float32)

    columns: dict[str, pa.Array] = {
        "ra": pa.array(to_native_endian(ra)),
        "dec": pa.array(to_native_endian(dec)),
        "object_id": pa.array([str(int(i)) for i in inds], type=pa.string()),
        # image struct-of-parallel-lists (Mike's v1 SSL transformer schema).
        "image": _build_image_struct(image_array, psfsize),
    }
    for name, arr in scalars.items():
        columns[name] = pa.array(to_native_endian(arr))
    for name, arr in per_band.items():
        for band_idx, suffix in enumerate(PER_BAND_SUFFIXES):
            columns[f"{name}_{suffix}"] = pa.array(to_native_endian(arr[:, band_idx]))

    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path,
                        help="Stein-et-al top-level dir (contains north/ and south/), "
                             "or a specific subdir.")
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap on number of image chunks (each chunk is ~1M objects).")
    parser.add_argument("--max-rows-per-file", type=int, default=None,
                        help="Cap rows read PER chunk (useful when chunks are very large).")
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ra-center", type=float, default=None)
    parser.add_argument("--dec-center", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None,
                        help="Cone radius in degrees; requires --ra-center/--dec-center.")
    args = parser.parse_args(argv)

    files = find_raw_files(args.raw_root, max_files=args.max_files)
    if not files:
        print(f"ERROR: no images_npix152_*.h5 under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} ssl_legacysurvey chunk(s)")

    tables: list[pa.Table] = []
    total = 0
    for i, p in enumerate(files, 1):
        t = read_chunk(
            p,
            max_rows=args.max_rows_per_file,
            ra_center=args.ra_center,
            dec_center=args.dec_center,
            radius=args.radius,
        )
        if t is None:
            continue
        tables.append(t)
        total += t.num_rows
        print(f"  [{i}/{len(files)}] {os.path.basename(p)}: {t.num_rows} objects")

    if not tables:
        print("ERROR: no rows survived the cone cut", file=sys.stderr)
        return 1

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
