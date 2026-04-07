"""Convert raw DESI coadd FITS files into a HATS catalog.

The cluster mirrors DESI coadd files at:

    /mnt/ceph/users/polymathic/external_data/astro/DESI_DR1/coadd-main-*.fits      # ~32k files
    /mnt/ceph/users/polymathic/external_data/astro/DESI/coadd-sv3-*.fits           # ~1k files (SV3)

Each coadd file is a multi-HDU FITS:

  HDU 1  FIBERMAP        bintable: TARGETID, TARGET_RA, TARGET_DEC, OBJTYPE,
                         COADD_FIBERSTATUS, photometry, ...
  HDU 3-7  B camera      WAVELENGTH (1D, 2751), FLUX/IVAR/MASK (2D 9x2751)
  HDU 8-12 R camera      WAVELENGTH (2326), FLUX/IVAR/MASK (9x2326)
  HDU 13-17 Z camera     WAVELENGTH (2881), FLUX/IVAR/MASK (9x2881)
  HDU 18 SCORES          per-fiber QA scores

The three cameras overlap slightly. To keep the port simple and avoid
depending on ``desispec``, we concatenate B+R+Z into one spectrum per fiber
in wavelength order. This is NOT the same as ``desispec.coadd_cameras``
(which interpolates onto a common grid in the overlap regions); it's a
``simple-stack`` that's fine for HATS-level downstream use.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pyarrow as pa
from astropy.io import fits

from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import np_to_pyarrow_list, to_native_endian, write_hats


CATALOG_NAME = "desi"

# Photometric scalar columns we keep from FIBERMAP. The full FIBERMAP has 70+
# columns; this is the subset that's most useful for downstream science.
SCALAR_FIBERMAP_COLS = [
    "EBV",
    "FLUX_G", "FLUX_R", "FLUX_Z", "FLUX_W1", "FLUX_W2",
    "FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z",
    "FIBERFLUX_G", "FIBERFLUX_R", "FIBERFLUX_Z",
    "GAIA_PHOT_G_MEAN_MAG",
    "GAIA_PHOT_BP_MEAN_MAG",
    "GAIA_PHOT_RP_MEAN_MAG",
]


def _as_str(val) -> str:
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode().strip()
    return str(val).strip()


def find_raw_files(raw_root: str, max_files: int | None = None) -> list[str]:
    files = sorted(glob.glob(os.path.join(raw_root, "coadd-*.fits")))
    if max_files is not None:
        files = files[:max_files]
    return files


def _selection_mask(fibermap) -> np.ndarray:
    """Apply DESI's standard quality cuts: science targets, good fiber status."""
    objtype = np.array([_as_str(v) for v in fibermap["OBJTYPE"]])
    fiberstatus = np.asarray(fibermap["COADD_FIBERSTATUS"])
    return (objtype == "TGT") & (fiberstatus == 0)


def read_coadd(path: str) -> pa.Table | None:
    """Read one DESI coadd FITS and return a PyArrow table with one row per
    surviving fiber. Returns None if no fibers pass the cuts."""
    with fits.open(path, memmap=False) as hdul:
        fm = hdul["FIBERMAP"].data
        mask = _selection_mask(fm)
        if not mask.any():
            return None

        # Read per-fiber arrays for the surviving rows.
        idx = np.where(mask)[0]
        target_id = np.asarray(fm["TARGETID"])[idx]
        ra = np.asarray(fm["TARGET_RA"])[idx]
        dec = np.asarray(fm["TARGET_DEC"])[idx]

        # Concatenate B/R/Z cameras (already wavelength-sorted: B<R<Z).
        cameras = []
        for cam in ("B", "R", "Z"):
            wl = np.asarray(hdul[f"{cam}_WAVELENGTH"].data, dtype=np.float32)
            flux = np.asarray(hdul[f"{cam}_FLUX"].data, dtype=np.float32)[idx]
            ivar = np.asarray(hdul[f"{cam}_IVAR"].data, dtype=np.float32)[idx]
            mbits = np.asarray(hdul[f"{cam}_MASK"].data)[idx]
            cameras.append((wl, flux, ivar, mbits))

        # Stack along the wavelength axis.
        wl_concat = np.concatenate([c[0] for c in cameras]).astype(np.float32)
        flux_concat = np.concatenate([c[1] for c in cameras], axis=1).astype(np.float32)
        ivar_concat = np.concatenate([c[2] for c in cameras], axis=1).astype(np.float32)
        mask_concat = np.concatenate([c[3] for c in cameras], axis=1).astype(np.uint32)

        n_obj = len(idx)
        lam = np.tile(wl_concat, (n_obj, 1))
        bad_mask = (mask_concat > 0) | (ivar_concat <= 1e-6)

        # Scalar metadata from FIBERMAP.
        scalar_cols: dict[str, np.ndarray] = {}
        for c in SCALAR_FIBERMAP_COLS:
            if c in fm.dtype.names:
                scalar_cols[c] = np.asarray(fm[c])[idx]

    # Build the PyArrow columns.
    columns: dict[str, pa.Array] = {
        "ra": pa.array(to_native_endian(ra.astype(np.float64))),
        "dec": pa.array(to_native_endian(dec.astype(np.float64))),
        "object_id": pa.array([str(int(t)) for t in target_id], type=pa.string()),
        "TARGETID": pa.array(to_native_endian(target_id.astype(np.int64))),
        "spectrum": pa.StructArray.from_arrays(
            [
                np_to_pyarrow_list(flux_concat),
                np_to_pyarrow_list(ivar_concat),
                np_to_pyarrow_list(lam),
                np_to_pyarrow_list(bad_mask),
            ],
            names=["flux", "ivar", "lambda", "mask"],
        ),
    }
    for c, arr in scalar_cols.items():
        columns[c] = pa.array(to_native_endian(arr))

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
        print(f"ERROR: no coadd-*.fits under {args.raw_root}", file=sys.stderr)
        return 1
    print(f"Found {len(files)} DESI coadd file(s)")

    tables: list[pa.Table] = []
    total = 0
    for i, p in enumerate(files, 1):
        t = read_coadd(p)
        if t is None:
            print(f"  [{i}/{len(files)}] {os.path.basename(p)}: no surviving fibers")
            continue
        tables.append(t)
        total += t.num_rows
        print(f"  [{i}/{len(files)}] {os.path.basename(p)}: {t.num_rows} fibers")

    if not tables:
        print("ERROR: no fibers passed the quality cuts in any file", file=sys.stderr)
        return 1

    print(f"\nWriting HATS catalog: {total} rows from {len(tables)} files")
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
