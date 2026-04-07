"""Convert raw DESI DR1 iron spectra into a HATS catalog.

Matches v1 MMU's ``scripts/desi/build_parent_sample.py`` pipeline exactly,
with HATS output instead of HDF5:

    1. Load the full redshift catalog ``zall-pix-iron.fits`` (28M rows).
    2. Apply the v1 ``selection_fn``: SURVEY=='main', MAIN_PRIMARY,
       OBJTYPE=='TGT', COADD_FIBERSTATUS==0, and exclude BAD_HANDLES
       (missing-on-server files).
    3. Group surviving rows by (SURVEY, PROGRAM, HEALPIX) so all fibers from
       the same coadd file are processed together.
    4. For each coadd file:
         - ``desispec.io.read_spectra(...).select(targets=target_ids)``
         - ``desispec.coaddition.coadd_cameras(spectra)`` to merge B/R/Z
           onto a common wavelength grid (with interpolation in the overlap
           regions — this is the thing a naive simple-stack cannot do).
         - Extract ``wave["brz"]``, ``flux["brz"]``, ``ivar["brz"]``,
           ``mask["brz"]``, and ``resolution_data["brz"]``.
         - Estimate a Gaussian LSF sigma via ``scipy.optimize.curve_fit`` on
           the mean resolution profile, matching v1.
    5. Join the per-fiber spectra with the curated scalar columns from the
       redshift catalog (Z, ZERR, ZWARN, EBV, FLUX_G/R/Z, ...).
    6. Build a PyArrow table with one row per fiber, where the ``spectrum``
       column is a struct-of-parallel-lists matching v1's HuggingFace
       schema exactly::

          spectrum: struct<
              flux:      list<float32>,
              ivar:      list<float32>,
              lsf_sigma: list<float32>,
              lambda:    list<float32>,
              mask:      list<bool>,
          >

       No extension types anywhere in the tree (see ``project_image_storage``
       memory for why: nested_pandas crashes on extension types inside
       list/struct).
    7. Write the HATS catalog via ``mmu.hats_import.write_hats``.

Cluster layout:

    /mnt/ceph/users/polymathic/external_data/astro/DESI_DR1/
        zall-pix-iron.fits                          # full redshift catalog
        coadd-main-{program}-{healpix}.fits         # ~32k coadd files, flat

The v1 raw-data layout on DESI's servers is nested by survey/program/
pix_group/healpix, but the cluster mirror is flattened to a single directory.
We find coadd files by their filename pattern, not by directory walk.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
from astropy.table import Table
from scipy.optimize import curve_fit

os.environ.setdefault("DESI_LOGLEVEL", "WARNING")

import desispec.io
from desispec import coaddition

from mmu.hats_configs import DATASETS, MMU_V2_HATS_ROOT
from mmu.hats_import import to_native_endian, write_hats


CATALOG_NAME = "desi"


# Curated scalar columns matching v1's HuggingFace schema exactly.
# See scripts/desi/desi.py _FLOAT_FEATURES and _BOOL_FEATURES.
FLOAT_FEATURES = [
    "Z",
    "ZERR",
    "EBV",
    "FLUX_G",
    "FLUX_R",
    "FLUX_Z",
    "FLUX_IVAR_G",
    "FLUX_IVAR_R",
    "FLUX_IVAR_Z",
    "FIBERFLUX_G",
    "FIBERFLUX_R",
    "FIBERFLUX_Z",
    "FIBERTOTFLUX_G",
    "FIBERTOTFLUX_R",
    "FIBERTOTFLUX_Z",
]
BOOL_FEATURES = ["ZWARN"]


# BAD_HANDLES lifted directly from v1 MMU scripts/desi/build_parent_sample.py.
# These healpix values appear in tilepix.fits but the corresponding coadd
# files don't exist on DESI's servers — see the v1 docstring for the issue
# references. Same applies to the cluster mirror.
BAD_HANDLES: dict[str, list[int]] = {
    "bright": [9836, 4802, 4561, 4730],
    "dark": [26535, 15051, 10844, 9913],
    "backup": [10786, 10810],
}


def _as_str_array(col) -> np.ndarray:
    """Decode a FITS byte/str column into a stripped-string numpy array."""
    out = np.empty(len(col), dtype=object)
    for i, v in enumerate(col):
        if isinstance(v, (bytes, np.bytes_)):
            out[i] = v.decode().strip()
        else:
            out[i] = str(v).strip()
    return out.astype(str)


def selection_fn(catalog: Table) -> np.ndarray:
    """Return a boolean mask selecting v1's standard science fibers.

    Matches scripts/desi/build_parent_sample.py:selection_fn exactly:
    SURVEY=='main' & MAIN_PRIMARY & OBJTYPE=='TGT' & COADD_FIBERSTATUS==0,
    minus the BAD_HANDLES per program.
    """
    survey = _as_str_array(catalog["SURVEY"])
    objtype = _as_str_array(catalog["OBJTYPE"])
    mask = (survey == "main")
    mask &= np.asarray(catalog["MAIN_PRIMARY"]).astype(bool)
    mask &= (objtype == "TGT")
    mask &= np.asarray(catalog["COADD_FIBERSTATUS"]) == 0
    program = _as_str_array(catalog["PROGRAM"])
    healpix = np.asarray(catalog["HEALPIX"])
    for bad_program, bad_hp in BAD_HANDLES.items():
        mask &= ~((program == bad_program) & np.isin(healpix, bad_hp))
    return mask


def find_matching_indices(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Return indices into arr2 that reorder arr2 to match arr1. Lifted
    verbatim from v1 MMU."""
    sort_idx_arr1 = np.argsort(arr1)
    sort_idx_arr2 = np.argsort(arr2)
    inverse_sort_idx_arr1 = np.argsort(sort_idx_arr1)
    return sort_idx_arr2[inverse_sort_idx_arr1]


def _gauss(x: np.ndarray, a: float, x0: float, sigma: float) -> np.ndarray:
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2))


def estimate_lsf_sigma(resolution_data: np.ndarray) -> float:
    """Estimate a single Gaussian LSF sigma (in pixel units) from the mean
    resolution profile across fibers and wavelengths. Matches v1 exactly.
    """
    # resolution_data shape: (n_fibers, ndiag, nwave)
    lsf = resolution_data.mean(axis=-1).mean(axis=0)
    popt, _ = curve_fit(_gauss, np.arange(len(lsf)), lsf, p0=[1.0, 5.0, 1.0])
    return float(popt[2])


def coadd_file_path(raw_root: str, survey: str, program: str, healpix: int) -> str:
    """Return the flat-layout coadd file path on the cluster mirror."""
    return os.path.join(raw_root, f"coadd-{survey}-{program}-{healpix}.fits")


def process_coadd(
    filename: str,
    target_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Read a coadd file, select the requested targets, run coadd_cameras,
    and return per-fiber arrays. Matches v1's processing_fn exactly, minus
    the SNR columns which aren't in the v1 HF schema.
    """
    spectra = desispec.io.read_spectra(filename).select(targets=target_ids)
    combined = coaddition.coadd_cameras(spectra)

    reordering = find_matching_indices(target_ids, np.array(combined.target_ids()))

    wavelength = np.asarray(combined.wave["brz"], dtype=np.float32)
    flux = np.asarray(combined.flux["brz"])[reordering].astype(np.float32)
    ivar = np.asarray(combined.ivar["brz"])[reordering].astype(np.float32)
    mask = np.asarray(combined.mask["brz"])[reordering].astype(np.uint32)
    resolution = np.asarray(combined.resolution_data["brz"])[reordering].astype(np.float32)
    tgt_ids = np.asarray(combined.target_ids())[reordering]

    if not np.array_equal(tgt_ids, target_ids):
        raise RuntimeError(
            f"Target ID mismatch after reordering in {filename}: "
            f"got {tgt_ids[:5]}, expected {target_ids[:5]}"
        )

    lsf_sigma_scalar = estimate_lsf_sigma(resolution)
    n = len(target_ids)
    n_wave = len(wavelength)
    lsf_sigma = np.full((n, n_wave), lsf_sigma_scalar, dtype=np.float32)
    lam = np.tile(wavelength.reshape(1, -1), (n, 1))
    bad_mask = (mask > 0) | (ivar <= 1e-6)

    return {
        "TARGETID": tgt_ids,
        "flux": flux,
        "ivar": ivar,
        "lambda": lam,
        "lsf_sigma": lsf_sigma,
        "mask": bad_mask,
    }


def _group_catalog(catalog: Table) -> list[tuple[str, str, int, np.ndarray, np.ndarray]]:
    """Group the catalog by (SURVEY, PROGRAM, HEALPIX). Returns a list of
    tuples ``(survey, program, healpix, target_ids, row_indices)`` where
    ``row_indices`` are the positions of the group's rows in the full
    catalog, so we can later stitch the scalar columns back together.
    """
    survey = _as_str_array(catalog["SURVEY"])
    program = _as_str_array(catalog["PROGRAM"])
    healpix = np.asarray(catalog["HEALPIX"])
    target_ids = np.asarray(catalog["TARGETID"])

    # Sort by (survey, program, healpix) so group boundaries are contiguous.
    order = np.lexsort((healpix, program, survey))
    groups: list[tuple[str, str, int, np.ndarray, np.ndarray]] = []
    i = 0
    while i < len(order):
        j = i
        s0, p0, h0 = survey[order[i]], program[order[i]], healpix[order[i]]
        while (
            j < len(order)
            and survey[order[j]] == s0
            and program[order[j]] == p0
            and healpix[order[j]] == h0
        ):
            j += 1
        idxs = order[i:j]
        groups.append((s0, p0, int(h0), target_ids[idxs], idxs))
        i = j
    return groups


def build_table(
    catalog: Table,
    raw_root: str,
    max_groups: int | None = None,
) -> pa.Table:
    """Process the catalog (after selection_fn) into a PyArrow table matching
    v1 MMU's HuggingFace DESI schema.
    """
    groups = _group_catalog(catalog)
    if max_groups is not None:
        groups = groups[:max_groups]

    spectrum_rows: list[dict[str, np.ndarray]] = []
    kept_row_indices: list[int] = []
    for i, (survey, program, healpix, target_ids, row_idx) in enumerate(groups, 1):
        path = coadd_file_path(raw_root, survey, program, healpix)
        if not os.path.exists(path):
            print(
                f"  [{i}/{len(groups)}] MISSING {os.path.basename(path)}"
                f" ({len(target_ids)} fibers skipped)",
                file=sys.stderr,
            )
            continue
        result = process_coadd(path, target_ids)
        spectrum_rows.append(result)
        kept_row_indices.extend(row_idx.tolist())
        print(
            f"  [{i}/{len(groups)}] {os.path.basename(path)}: {len(target_ids)} fibers"
        )

    if not spectrum_rows:
        raise RuntimeError("No spectra produced — all groups had missing coadd files.")

    # Concatenate per-file arrays (each row = one fiber).
    flux = np.concatenate([r["flux"] for r in spectrum_rows], axis=0)
    ivar = np.concatenate([r["ivar"] for r in spectrum_rows], axis=0)
    lam = np.concatenate([r["lambda"] for r in spectrum_rows], axis=0)
    lsf_sigma = np.concatenate([r["lsf_sigma"] for r in spectrum_rows], axis=0)
    mask = np.concatenate([r["mask"] for r in spectrum_rows], axis=0)
    target_ids_ordered = np.concatenate([r["TARGETID"] for r in spectrum_rows], axis=0)

    # Reorder the surviving scalar rows to match the spectrum order.
    # kept_row_indices already gives the order in which scalar rows were
    # processed (same order as spectrum_rows), so we use them as-is.
    catalog_subset = catalog[np.asarray(kept_row_indices)]

    # Sanity check: catalog target ids must match concatenated spectrum target ids.
    cat_tids = np.asarray(catalog_subset["TARGETID"])
    if not np.array_equal(cat_tids, target_ids_ordered):
        raise RuntimeError(
            "TARGETID ordering mismatch between catalog subset and processed spectra. "
            f"catalog[:5]={cat_tids[:5]}, spectra[:5]={target_ids_ordered[:5]}"
        )

    ra = np.asarray(catalog_subset["TARGET_RA"], dtype=np.float64)
    dec = np.asarray(catalog_subset["TARGET_DEC"], dtype=np.float64)

    # Build the spectrum struct (plain nested lists, no extension types).
    n = len(target_ids_ordered)
    flux_arr = pa.array([list(row) for row in flux], type=pa.list_(pa.float32()))
    ivar_arr = pa.array([list(row) for row in ivar], type=pa.list_(pa.float32()))
    lsf_arr = pa.array([list(row) for row in lsf_sigma], type=pa.list_(pa.float32()))
    lam_arr = pa.array([list(row) for row in lam], type=pa.list_(pa.float32()))
    mask_arr = pa.array([list(row) for row in mask], type=pa.list_(pa.bool_()))
    spectrum_struct = pa.StructArray.from_arrays(
        [flux_arr, ivar_arr, lsf_arr, lam_arr, mask_arr],
        names=["flux", "ivar", "lsf_sigma", "lambda", "mask"],
    )

    columns: dict[str, pa.Array] = {
        "ra": pa.array(to_native_endian(ra)),
        "dec": pa.array(to_native_endian(dec)),
        "object_id": pa.array([str(int(t)) for t in target_ids_ordered], type=pa.string()),
        "spectrum": spectrum_struct,
    }
    for f in FLOAT_FEATURES:
        arr = np.asarray(catalog_subset[f], dtype=np.float32)
        columns[f] = pa.array(to_native_endian(arr))
    for f in BOOL_FEATURES:
        arr = np.asarray(catalog_subset[f]).astype(bool)
        columns[f] = pa.array(arr)

    assert len(columns["ra"]) == n
    return pa.table(columns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=DATASETS[CATALOG_NAME].raw_path)
    parser.add_argument("--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME))
    parser.add_argument(
        "--zcatalog",
        default=None,
        help="Path to zall-pix-iron.fits; defaults to {raw_root}/zall-pix-iron.fits",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Cap on number of (survey, program, healpix) groups to process.",
    )
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    args = parser.parse_args(argv)

    zcat_path = args.zcatalog or os.path.join(args.raw_root, "zall-pix-iron.fits")
    if not os.path.exists(zcat_path):
        print(f"ERROR: missing redshift catalog {zcat_path}", file=sys.stderr)
        return 1

    print(f"Loading {zcat_path}...")
    catalog = Table.read(zcat_path)
    print(f"  {len(catalog)} total rows")
    mask = selection_fn(catalog)
    catalog = catalog[mask]
    print(f"  {len(catalog)} rows after selection_fn")

    table = build_table(catalog, args.raw_root, max_groups=args.max_groups)
    print(f"\nWriting HATS catalog: {table.num_rows} rows")
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
