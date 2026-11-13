"""Build a JADES DR5 image + photometry HATS catalog from pinned FITS inputs.

The JSON input manifest declares one catalogue per GOODS field and one or more
mosaics per (field, band), including overlapping MIRI parallel pointings.
Each mosaic explicitly declares the
SCI and real variance/ERR/IVAR HDUs.  No synthetic inverse variance is allowed.
Run ``download_data.py`` first or pass ``--download`` for the full workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from mmu.hats_configs import MMU_V2_HATS_ROOT, RAW_DATA_ROOT
from scripts.jades.download_data import fetch_manifest


DEFAULT_ROOT = Path(RAW_DATA_ROOT) / "JADES_DR5"
IMAGE_SIZE = 160
SCHEMA_VERSION = "jades-v2-multimodal-2"
MIRI_BANDS = {
    "F560W", "F770W", "F1000W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W",
}
JWST_BANDS = {
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W", "F210M",
    "F250M", "F277W", "F300M", "F335M", "F356W", "F410M", "F430M", "F444W",
    "F460M", "F480M", "F560W", "F770W", "F1000W", "F1280W", "F1500W",
    "F1800W", "F2100W", "F2550W",
}
HST_BANDS = {"F435W", "F606W", "F775W", "F814W", "F850LP", "F105W", "F125W", "F140W", "F160W"}

# EAZY photo-z products in both JADES DR5 PHOTOZ (native CIRC1) and
# PHOTOZ_KRON (common-PSF Kron) HDUs. z_spec is *not* a photo-z.
PHOTOZ_SCALAR_FIELDS = (
    "z_a", "z_ml", "chi_a", "l68", "u68", "l95", "u95", "l99", "u99",
    "nfilt", "z_peak", "chi_peak", "z025", "z160", "z500", "z840", "z975",
    "prob_gt_5", "prob_gt_6", "prob_gt_7", "prob_gt_8", "prob_gt_9",
    "chisq_z_lt_7", "z_chisq_z_lt_7", "chisq_z_lt_6", "z_chisq_z_lt_6",
    "chisq_z_lt_5", "z_chisq_z_lt_5", "chisq_z_lt_4", "z_chisq_z_lt_4",
)
PHOTOZ_ARRAY_FIELDS = ("z_bins", "prob_z_bins")
PHOTOZ_FIELDS = PHOTOZ_SCALAR_FIELDS + PHOTOZ_ARRAY_FIELDS


def _band_sort_key(band: str) -> tuple[int, str]:
    return int(re.search(r"\d+", band).group()), band


def _column_key(name: str) -> str:
    """Match FITS column spelling/case while keeping output names stable."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _column_views(table, requested: tuple[str, ...] | list[str]) -> dict[str, np.ndarray]:
    """Create zero-copy views only for required FITS columns, once per field."""
    if table is None:
        return {}
    actual = {_column_key(name): name for name in table.names}
    return {_column_key(name): table[actual[_column_key(name)]]
            for name in requested if _column_key(name) in actual}


def _value(columns: dict[str, np.ndarray], name: str, index: int,
           *, nonnegative: bool = False, positive: bool = False) -> float | None:
    values = columns.get(_column_key(name))
    if values is None:
        return None
    value = float(values[index])
    if not np.isfinite(value) or (nonnegative and value < 0) or (positive and value <= 0):
        return None
    return value


def _photoz_record(columns: dict[str, np.ndarray], index: int) -> dict:
    values = {name: _value(columns, name, index, nonnegative=True)
              for name in PHOTOZ_SCALAR_FIELDS}
    for name in PHOTOZ_ARRAY_FIELDS:
        column = columns.get(_column_key(name))
        if column is None:
            values[name] = None
        else:
            array = np.asarray(column[index], dtype=np.float32).ravel()
            values[name] = [float(v) if np.isfinite(v) else None for v in array]
    return values


def _photometry_record(index: int, bands: tuple[str, ...],
                       circ: dict, kron: dict, kron_conv: dict, miri: dict) -> dict:
    values = {}
    for band in bands:
        for method in ("circ3", "kron", "kron_conv"):
            if method == "kron_conv" and band in MIRI_BANDS:
                continue
            columns = (miri if band in MIRI_BANDS else
                       circ if method == "circ3" else
                       kron if method == "kron" else kron_conv)
            fits_name = f"{band}_{method.upper() if method != 'kron_conv' else 'KRON'}"
            for suffix, measure in (("", "flux"), ("_e", "err")):
                # KRON_CONV is an HDU; the column name still ends in _KRON.
                value = _value(columns, fits_name + suffix, index,
                               positive=(measure == "err"))
                values[f"phot_{band.lower()}_{method}_{measure}_njy"] = value
    return values


def apparent_mag_from_njy(flux):
    """AB apparent magnitude from positive total flux in nJy; others are NaN."""
    flux = np.asarray(flux, dtype=np.float64)
    result = np.full(flux.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(flux) & (flux > 0)
    result[valid] = 31.4 - 2.5 * np.log10(flux[valid])
    return result


def _input(raw_root: Path, value: str) -> Path:
    part = Path(value)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError(f"Manifest paths must be relative to raw root: {value}")
    return raw_root / part


def _hdu(hdus: fits.HDUList, name: str | int):
    try:
        return hdus[name]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Missing HDU {name}; available {[x.name for x in hdus]}") from exc


def variance_to_ivar(uncertainty: np.ndarray, kind: str) -> np.ndarray:
    """Use real uncertainty data only; invalid pixels receive zero weight."""
    x = np.asarray(uncertainty, dtype=np.float32)
    ivar = np.zeros(x.shape, dtype=np.float32)
    good = np.isfinite(x) & (x > 0)
    if kind == "sigma":
        np.divide(1.0, np.square(x), out=ivar, where=good)
    elif kind == "variance":
        np.divide(1.0, x, out=ivar, where=good)
    elif kind == "ivar":
        ivar[good] = x[good]
    else:
        raise ValueError(f"Unknown uncertainty_kind {kind!r}; expected sigma, variance, ivar")
    ivar[~np.isfinite(ivar)] = 0
    return ivar


def flux_conversion(header: fits.Header, unit: str) -> float:
    """Return the multiplicative conversion from SCI units to nJy/pixel.

    For MJy/sr the precise pixel area must be recorded in the FITS header.
    Refuse to guess a pixel area from a nominal instrument plate scale.
    """
    normalized = unit.lower().replace(" ", "")
    if normalized == "njy":
        return 1.0
    if normalized == "jy":
        return 1e9
    if normalized == "mjy/sr":
        pixar = header.get("PIXAR_SR")
        if pixar is None or not np.isfinite(pixar) or pixar <= 0:
            raise ValueError("MJy/sr mosaic requires a valid PIXAR_SR header")
        return float(pixar) * 1e15
    raise ValueError(f"Unsupported SCI unit {unit!r}; refusing implicit conversion")


def _cutout(data: np.ndarray, wcs: WCS, ra: float, dec: float, fill: float,
            image_size: int = IMAGE_SIZE) -> np.ndarray:
    x, y = wcs.all_world2pix(ra, dec, 0)
    result = _pixel_cutout(data, float(x), float(y), fill, image_size)
    if result is None:
        raise ValueError("Cutout does not overlap the mosaic")
    return result


def _pixel_cutout(data: np.ndarray, x: float, y: float, fill: float,
                  image_size: int) -> np.ndarray | None:
    """Cut out native pixels, using astropy Cutout2D's ceil rounding convention.

    Avoid per-object WCS calls and astropy Cutout2D allocations when extracting
    millions of cutouts from Ceph-backed memory-mapped mosaics.
    """
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    x0, y0 = math.ceil(x - image_size / 2), math.ceil(y - image_size / 2)
    x1, y1 = x0 + image_size, y0 + image_size
    height, width = data.shape
    sx0, sx1 = max(0, x0), min(width, x1)
    sy0, sy1 = max(0, y0), min(height, y1)
    if sx0 >= sx1 or sy0 >= sy1:
        return None
    result = np.full((image_size, image_size), fill, dtype=np.float32)
    result[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = data[sy0:sy1, sx0:sx1]
    return result


def _catalog_columns(hdus: fits.HDUList, name: str):
    data = _hdu(hdus, name).data
    return data, set(data.names)


def _ids_aligned(first, second) -> bool:
    return len(first) == len(second) and np.array_equal(first["ID"], second["ID"])


def _open_mosaics(stack: ExitStack, entries: list[dict], raw_root: Path) -> list[dict]:
    mosaics = []
    for entry in entries:
        hdus = stack.enter_context(fits.open(_input(raw_root, entry["path"]), memmap=True))
        science = _hdu(hdus, entry["sci_hdu"])
        uncertainty = _hdu(hdus, entry["uncertainty_hdu"])
        if science.data is None or uncertainty.data is None:
            raise ValueError(f"Empty SCI or uncertainty in {entry['path']}")
        if science.data.shape != uncertainty.data.shape:
            raise ValueError(f"SCI and uncertainty shape mismatch in {entry['path']}")
        unit = entry.get("unit") or science.header.get("BUNIT")
        if not unit:
            raise ValueError(f"No SCI unit for {entry['path']}")
        scale = flux_conversion(science.header, unit)
        wcs = WCS(science.header)
        psf_fwhm = entry.get("psf_fwhm_arcsec")
        if psf_fwhm is not None:
            psf_fwhm = float(psf_fwhm)
            if not np.isfinite(psf_fwhm) or psf_fwhm <= 0:
                raise ValueError(f"Invalid psf_fwhm_arcsec for {entry['path']}: {psf_fwhm}")
        # Store the raw FITS arrays; process only the requested cutouts.
        mosaics.append({"band": entry["band"].upper(), "sci": science.data,
                        "uncertainty": uncertainty.data, "kind": entry["uncertainty_kind"],
                        "wcs": wcs, "path": entry["path"], "flux_scale": scale,
                        "psf_fwhm_arcsec": psf_fwhm,
                        "scale_arcsec": float(np.mean(np.abs(proj_plane_pixel_scales(wcs.celestial))) * 3600)})
    return mosaics


def _prepare_positions(mosaics: list[dict], size, selected_indices: np.ndarray) -> None:
    """Vectorize WCS transformations once per mosaic, not once per galaxy."""
    ra = np.asarray(size["RA"][selected_indices], dtype=np.float64)
    dec = np.asarray(size["DEC"][selected_indices], dtype=np.float64)
    for mosaic in mosaics:
        mosaic["x"], mosaic["y"] = mosaic["wcs"].all_world2pix(ra, dec, 0)


def _record(index: int, size, flags, mosaics: list[dict], field: str,
            image_size: int = IMAGE_SIZE, phot_bands: tuple[str, ...] = (),
            phot_columns: dict | None = None,
            photoz_columns: dict | None = None,
            photoz_kron_columns: dict | None = None,
            coord_index: int | None = None) -> dict | None:
    ra, dec = float(size["RA"][index]), float(size["DEC"][index])
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    # Overlapping MIRI parallel pointings are resolved by local valid-pixel
    # coverage, then by local inverse variance. Do not duplicate a band.
    candidates = {}
    for m in mosaics:
        band = m["band"]
        if flags is not None and f"{band}_TEXP" in flags.names:
            if not np.isfinite(flags[f"{band}_TEXP"][index]) or flags[f"{band}_TEXP"][index] <= 0:
                continue
        # Coordinates have been transformed in one vectorized WCS call.
        coordinate = coord_index if coord_index is not None else index
        x, y = float(m["x"][coordinate]), float(m["y"][coordinate])
        f = _pixel_cutout(m["sci"], x, y, np.nan, image_size)
        if f is None:
            continue
        u = _pixel_cutout(m["uncertainty"], x, y, np.nan, image_size)
        if u is None:
            continue
        weight = variance_to_ivar(u, m["kind"])
        # Check raw validity first, then transform uncertainties with flux units.
        valid = np.isfinite(f) & (weight > 0)
        # A band is present for a *galaxy* only if its centroid is covered.
        # An image intersecting a few remote edge pixels is not an observation
        # of the source, even though the partial stamp is nonempty.
        if not valid[image_size // 2, image_size // 2]:
            continue
        weight = np.where(valid, weight / m["flux_scale"]**2, 0).astype(np.float32)
        score = (int(valid.sum()), float(np.median(weight[valid])))
        if band not in candidates or score > candidates[band][0]:
            f = np.where(valid, f * m["flux_scale"], 0).astype(np.float32)
            candidates[band] = (score, f, weight, valid, m)

    if not candidates:
        return None
    bands = sorted(candidates, key=_band_sort_key)
    fluxes, ivars, masks, scales, psfs, sources = [], [], [], [], [], []
    for band in bands:
        _, f, weight, valid, m = candidates[band]
        fluxes.append(f)
        ivars.append(weight)
        masks.append(valid)
        scales.append(m["scale_arcsec"])
        psfs.append(m["psf_fwhm_arcsec"])
        sources.append(m["path"])
    phot_columns = phot_columns or {}
    photoz_columns = photoz_columns or {}
    photoz_kron_columns = photoz_kron_columns or {}
    z_spec = _value(photoz_columns, "z_spec", index, nonnegative=True)
    if z_spec is None:
        z_spec = _value(photoz_kron_columns, "z_spec", index, nonnegative=True)
    return {"object_id": f"{field}:{int(size['ID'][index])}", "ra": ra, "dec": dec,
            "field": field, "bands": [band.lower() for band in bands], "flux": fluxes,
            "ivar": ivars, "mask": masks, "scale": scales, "psf_fwhm": psfs,
            "image_sources": sources, "photometry": _photometry_record(
                index, phot_bands, phot_columns.get("circ", {}), phot_columns.get("kron", {}),
                phot_columns.get("kron_conv", {}), phot_columns.get("miri", {})),
            "photoz": _photoz_record(photoz_columns, index),
            "photoz_kron": _photoz_record(photoz_kron_columns, index),
            "z_spec": z_spec,
            "size_a": float(size["A"][index]), "size_b": float(size["B"][index])}


def _table(records: list[dict]) -> pa.Table:
    """Build large-list image cubes without millions of nested Python lists."""
    f32 = pa.float32()

    def cube(name: str, dtype, arrow_type):
        sides = {array.shape for row in records for array in row[name]}
        if len(sides) != 1:
            raise ValueError(f"Expected identically shaped {name} images; got {sides}")
        height, width = sides.pop()
        counts = np.array([len(row[name]) for row in records], dtype=np.int64)
        flat = np.concatenate([array.astype(dtype, copy=False).ravel()
                               for row in records for array in row[name]])
        pixels = pa.array(flat, type=arrow_type)
        n_images = int(counts.sum())
        rows = pa.LargeListArray.from_arrays(
            np.arange(n_images * height + 1, dtype=np.int64) * width, pixels)
        images = pa.LargeListArray.from_arrays(
            np.arange(n_images + 1, dtype=np.int64) * height, rows)
        return pa.LargeListArray.from_arrays(np.r_[0, counts.cumsum()], images)

    image = pa.StructArray.from_arrays([
        pa.array([r["bands"] for r in records], type=pa.list_(pa.string())),
        cube("flux", np.float32, f32),
        cube("ivar", np.float32, f32),
        cube("mask", np.bool_, pa.bool_()),
        pa.array([r["psf_fwhm"] for r in records], type=pa.list_(f32)),
        pa.array([r["scale"] for r in records], type=pa.list_(f32)),
    ], names=["band", "flux", "ivar", "mask", "psf_fwhm", "scale"])

    def photoz_struct(name: str) -> pa.StructArray:
        return pa.StructArray.from_arrays([
            pa.array([r[name][field] for r in records],
                     type=pa.list_(f32) if field in PHOTOZ_ARRAY_FIELDS else
                     pa.int32() if field == "nfilt" else f32)
            for field in PHOTOZ_FIELDS
        ], names=list(PHOTOZ_FIELDS))

    columns = {"object_id": pa.array([r["object_id"] for r in records]),
               "ra": pa.array([r["ra"] for r in records], type=pa.float64()),
               "dec": pa.array([r["dec"] for r in records], type=pa.float64()),
               "field": pa.array([r["field"] for r in records]),
               "image": image,
               "image_sources": pa.array([r["image_sources"] for r in records],
                                         type=pa.list_(pa.string())),
               "photoz": photoz_struct("photoz"),
               "photoz_kron": photoz_struct("photoz_kron"),
               "z_spec": pa.array([r["z_spec"] for r in records], type=f32),
               "size_a": pa.array([r["size_a"] for r in records], type=f32),
               "size_b": pa.array([r["size_b"] for r in records], type=f32)}
    for name in records[0]["photometry"]:
        columns[name] = pa.array([r["photometry"][name] for r in records], type=f32)
    return pa.table(columns)


def shard_directory(root: Path, shard_index: int, num_shards: int) -> Path:
    """Keep each Slurm task's checkpoints and Parquet files independent."""
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError(f"Invalid shard {shard_index} of {num_shards}")
    return root if num_shards == 1 else root / f"shard_{shard_index:04d}_of_{num_shards:04d}"


def verify_completed_shards(scratch: Path, fields: list[str], num_shards: int) -> dict:
    """Refuse HATS ingestion if any extraction task is missing or unfinished."""
    totals = {field: 0 for field in fields}
    shard_counts = {}
    for index in range(num_shards):
        directory = shard_directory(scratch, index, num_shards)
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing JADES extraction shard: {directory}")
        for field in fields:
            done = directory / f"{field}.done"
            progress = directory / f"{field}.progress.json"
            if not done.is_file() or not progress.is_file():
                raise RuntimeError(f"Incomplete JADES extraction: {directory}/{field}")
            checkpoint = json.loads(progress.read_text())
            count = int(done.read_text())
            if count < 0 or checkpoint.get("count") != count:
                raise ValueError(f"Inconsistent JADES checkpoint for {field} in {directory}")
            if num_shards > 1 and (checkpoint.get("schema_version") != SCHEMA_VERSION or
                                    checkpoint.get("num_shards") != num_shards or
                                    checkpoint.get("shard_index") != index):
                raise ValueError(f"Wrong JADES shard/schema identity in {progress}")
            expected_files = {directory / f"{field}_{part:05d}.parquet"
                              for part in range(int(checkpoint["next_shard"]))}
            found_files = set(directory.glob(f"{field}_*.parquet"))
            if found_files != expected_files:
                raise RuntimeError(f"Missing, extra or misnumbered Parquet for {field} in {directory}")
            actual_rows = sum(pq.read_metadata(path).num_rows for path in sorted(found_files))
            if actual_rows != count:
                raise RuntimeError(f"Parquet row-count mismatch for {field} in {directory}: "
                                   f"checkpoint={count}, files={actual_rows}")
            totals[field] += count
            shard_counts[f"{index}:{field}"] = count
    if sum(totals.values()) == 0:
        raise ValueError("All extraction shards are empty")
    return {"num_shards": num_shards, "total_objects": sum(totals.values()),
            "field_counts": totals, "shard_counts": shard_counts}


def process_field(field: str, catalog_path: Path, mosaics: list[dict], raw_root: Path,
                  scratch: Path, chunk_size: int, max_objects: int | None,
                  mag_band: str | None = None, mag_max: float | None = None,
                  image_size: int = IMAGE_SIZE, rows_per_file: int = 64,
                  sample_mode: str = "first",
                  phot_bands: tuple[str, ...] | None = None,
                  num_shards: int = 1, shard_index: int = 0,
                  extract_threads: int = 1, spatial_batch_size: int = 1) -> int:
    """Extract a field to bounded-memory Parquet shards with crash checkpoints.

    A checkpoint is committed only after a shard is atomically published. On
    resume, at most one in-progress shard is recomputed; complete shards stay.

    Threads overlap FITS memory-map page faults. Spatial batching schedules
    nearby cutouts together but restores *catalog order* before writing. Both
    execution knobs are excluded from the fingerprint intentionally: changing
    them cannot change a source's pixels, row order or checkpoint semantics.
    """
    if extract_threads < 1 or spatial_batch_size < 1:
        raise ValueError("extract_threads and spatial_batch_size must both be >= 1")
    done = scratch / f"{field}.done"
    progress = scratch / f"{field}.progress.json"
    shard_directory(scratch, shard_index, num_shards)  # validate before doing any I/O
    phot_bands = tuple(sorted(phot_bands or JWST_BANDS, key=_band_sort_key))
    identity = {"schema_version": SCHEMA_VERSION, "field": field,
                "photometric_bands": phot_bands, "catalog": str(catalog_path),
                "catalog_size": catalog_path.stat().st_size,
                "mosaics": mosaics, "chunk_size": chunk_size,
                "rows_per_file": rows_per_file, "max_objects": max_objects,
                "sample_mode": sample_mode,
                "mag_band": mag_band, "mag_max": mag_max, "image_size": image_size}
    if num_shards > 1:
        identity.update(num_shards=num_shards, shard_index=shard_index)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if progress.exists():
        checkpoint = json.loads(progress.read_text())
        if checkpoint.get("fingerprint") != fingerprint:
            raise ValueError(f"{field}: scratch configuration differs from its checkpoint. "
                             "Choose a new --scratch-dir; do not mix runs.")
    else:
        if done.exists() or list(scratch.glob(f"{field}_*.parquet")):
            raise ValueError(f"{field}: scratch has output without a valid checkpoint; "
                             "use a fresh scratch directory to avoid mixing results")
        checkpoint = {"fingerprint": fingerprint, "next_index": 0, "next_shard": 0, "count": 0}
    if done.exists():
        return int(done.read_text())
    for index in range(checkpoint["next_shard"]):
        if not (scratch / f"{field}_{index:05d}.parquet").is_file():
            raise FileNotFoundError(f"{field}: missing completed Parquet shard {index}")
    # Ignore artifacts written after the last successfully committed checkpoint.
    uncommitted = scratch / f"{field}_{checkpoint['next_shard']:05d}.parquet"
    uncommitted.unlink(missing_ok=True)
    uncommitted.with_suffix(".parquet.part").unlink(missing_ok=True)

    def commit(next_index: int, count: int, shard: int) -> None:
        state = {"fingerprint": fingerprint, "next_index": next_index,
                 "next_shard": shard, "count": count}
        if num_shards > 1:
            state.update(schema_version=SCHEMA_VERSION, num_shards=num_shards,
                         shard_index=shard_index)
        temporary = progress.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state) + "\n")
        os.replace(temporary, progress)

    with ExitStack() as stack:
        catalog_hdus = stack.enter_context(fits.open(catalog_path, memmap=True))
        size, _ = _catalog_columns(catalog_hdus, "SIZE")
        circ, _ = _catalog_columns(catalog_hdus, "CIRC")
        if not _ids_aligned(size, circ):
            raise ValueError(f"{field}: SIZE and CIRC IDs not aligned")
        try:
            flags, _ = _catalog_columns(catalog_hdus, "FLAG")
        except ValueError:
            flags = None
        if flags is not None and not _ids_aligned(size, flags):
            raise ValueError(f"{field}: SIZE and FLAG IDs not aligned")
        try:
            miri, _ = _catalog_columns(catalog_hdus, "MIRI")
        except ValueError:
            miri = None
        if miri is not None and not _ids_aligned(size, miri):
            raise ValueError(f"{field}: SIZE and MIRI IDs not aligned")
        try:
            kron, _ = _catalog_columns(catalog_hdus, "KRON")
        except ValueError:
            kron = None
        if kron is not None and not _ids_aligned(size, kron):
            raise ValueError(f"{field}: SIZE and KRON IDs not aligned")
        try:
            kron_conv, _ = _catalog_columns(catalog_hdus, "KRON_CONV")
        except ValueError:
            kron_conv = None
        if kron_conv is not None and not _ids_aligned(size, kron_conv):
            raise ValueError(f"{field}: SIZE and KRON_CONV IDs not aligned")
        try:
            photoz, _ = _catalog_columns(catalog_hdus, "PHOTOZ")
        except ValueError:
            photoz = None
        if photoz is not None and not _ids_aligned(size, photoz):
            raise ValueError(f"{field}: SIZE and PHOTOZ IDs not aligned")
        try:
            photoz_kron, _ = _catalog_columns(catalog_hdus, "PHOTOZ_KRON")
        except ValueError:
            photoz_kron = None
        if photoz_kron is not None and not _ids_aligned(size, photoz_kron):
            raise ValueError(f"{field}: SIZE and PHOTOZ_KRON IDs not aligned")

        def phot_names(methods: tuple[str, ...], bands: tuple[str, ...]) -> list[str]:
            return [f"{band}_{method}{suffix}" for band in bands
                    for method in methods for suffix in ("", "_e")]

        non_miri_bands = tuple(b for b in phot_bands if b not in MIRI_BANDS)
        miri_bands = tuple(b for b in phot_bands if b in MIRI_BANDS)
        phot_columns = {
            "circ": _column_views(circ, phot_names(("CIRC3",), non_miri_bands)),
            "kron": _column_views(kron, phot_names(("KRON",), non_miri_bands)),
            "kron_conv": _column_views(kron_conv, phot_names(("KRON",), non_miri_bands)),
            "miri": _column_views(miri, phot_names(("CIRC3", "KRON"), miri_bands)),
        }
        photoz_columns = _column_views(photoz, (*PHOTOZ_FIELDS, "z_spec"))
        photoz_kron_columns = _column_views(photoz_kron, (*PHOTOZ_FIELDS, "z_spec"))
        magnitude = None
        if mag_max is not None:
            if kron is None:
                raise ValueError(f"{field}: magnitude selection requires KRON HDU")
            flux_column = f"{mag_band}_KRON"
            if flux_column not in kron.names:
                raise ValueError(f"{field}: requested apparent magnitude requires {flux_column}")
            magnitude = apparent_mag_from_njy(kron[flux_column])
        images = _open_mosaics(stack, mosaics, raw_root)
        if not images:
            raise ValueError(f"{field}: no mosaics configured")
        if sample_mode == "miri":
            if flags is None:
                raise ValueError(f"{field}: MIRI-focused sampling requires FLAG HDU")
            available = [f"{band}_TEXP" for band in ("F770W", "F1280W", "F1500W")
                         if f"{band}_TEXP" in flags.names]
            if not available:
                raise ValueError(f"{field}: no MIRI exposure columns in FLAG HDU")
            covered = np.zeros(len(size), dtype=bool)
            for column in available:
                exposure = flags[column]
                covered |= np.isfinite(exposure) & (exposure > 0)
            selected = np.flatnonzero(covered)
        elif sample_mode == "first":
            selected = np.arange(len(size))
        else:
            raise ValueError(f"Unknown sampling mode: {sample_mode}")
        if max_objects is not None:
            selected = selected[:max_objects]
        # A contiguous, non-overlapping partition of the *global* selection.
        # max_objects means the same limit regardless of Slurm task count.
        start = len(selected) * shard_index // num_shards
        stop = len(selected) * (shard_index + 1) // num_shards
        selected = selected[start:stop]
        n = len(selected)
        _prepare_positions(images, size, selected)
        count = checkpoint["count"]
        shard = checkpoint["next_shard"]
        chunk, shard_rows, writer = [], 0, None

        def extract_one(position: int) -> dict | None:
            i = int(selected[position])
            if magnitude is not None and (not np.isfinite(magnitude[i]) or magnitude[i] >= mag_max):
                return None
            return _record(i, size, flags, images, field, image_size,
                           phot_bands=phot_bands, phot_columns=phot_columns,
                           photoz_columns=photoz_columns,
                           photoz_kron_columns=photoz_kron_columns,
                           coord_index=position)

        def spatial_key(position: int) -> tuple[int, int, int]:
            # All GOODS mosaics share a sky footprint. One reference WCS is
            # enough to improve local page reuse without changing any pixels.
            x, y = images[0]["x"][position], images[0]["y"][position]
            if not np.isfinite(x) or not np.isfinite(y):
                return (2**62, 2**62, position)
            return (math.floor(float(y) / image_size),
                    math.floor(float(x) / image_size), position)

        started = time.monotonic()
        scanned = 0
        # Each batch bounds the number of in-flight cutouts and retains the
        # original output/checkpoint order, even when tasks run out of order.
        with ThreadPoolExecutor(max_workers=extract_threads) as executor:
            for start_index in range(checkpoint["next_index"], n, spatial_batch_size):
                positions = list(range(start_index, min(n, start_index + spatial_batch_size)))
                ordered = (sorted(positions, key=spatial_key)
                           if spatial_batch_size > 1 else positions)
                if extract_threads > 1:
                    futures = {pos: executor.submit(extract_one, pos) for pos in ordered}
                    rows = {pos: futures[pos].result() for pos in positions}
                else:
                    rows = {pos: extract_one(pos) for pos in ordered}

                for position in positions:
                    scanned += 1
                    row = rows.pop(position)
                    if row is None:
                        continue
                    chunk.append(row)
                    if len(chunk) >= chunk_size:
                        batch = _table(chunk)
                        if writer is None:
                            target = scratch / f"{field}_{shard:05d}.parquet"
                            temporary = target.with_suffix(".parquet.part")
                            writer = pq.ParquetWriter(temporary, batch.schema, compression="zstd")
                        writer.write_table(batch)
                        shard_rows += len(chunk)
                        chunk.clear()
                        if shard_rows >= rows_per_file:
                            writer.close()
                            os.replace(temporary, target)
                            count += shard_rows
                            shard += 1
                            writer, shard_rows = None, 0
                            commit(position + 1, count, shard)
                            rate = scanned / max(time.monotonic() - started, 1e-6)
                            print(f"{field}: checkpoint {position+1}/{n} scanned, "
                                  f"{count} cutouts, {shard} parquet files "
                                  f"({rate:.2f} scanned/s)", flush=True)
        if chunk:
            batch = _table(chunk)
            if writer is None:
                target = scratch / f"{field}_{shard:05d}.parquet"
                temporary = target.with_suffix(".parquet.part")
                writer = pq.ParquetWriter(temporary, batch.schema, compression="zstd")
            writer.write_table(batch)
            shard_rows += len(chunk)
        if writer is not None:
            writer.close()
            os.replace(temporary, target)
            count += shard_rows
            shard += 1
        commit(n, count, shard)
    # Atomic completion marker: a Slurm cancellation must never leave an
    # empty/partially written .done file that prevents future resumption.
    done_temporary = done.with_name(done.name + ".part")
    done_temporary.write_text(str(count))
    os.replace(done_temporary, done)
    return count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instrument", choices=("jwst", "hst"), required=True,
                   help="Build a separate MMU catalogue for JWST (NIRCam+MIRI) or HST (ACS+WFC3)")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-root", type=Path,
                   help="Defaults to MMU_V2_HATS_ROOT/jades_{instrument}")
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--max-objects", type=int)
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split each field's selected galaxies into this many independent Slurm tasks")
    p.add_argument("--shard-index", type=int, default=0,
                   help="Zero-based task number, 0 <= index < num-shards")
    p.add_argument("--field", choices=("GOODS-S", "GOODS-N"),
                   help="Extract only one GOODS field; allows 2*num-shards independent Slurm tasks")
    p.add_argument("--extract-threads", type=int, default=1,
                   help="Concurrent FITS cutout readers within a shard (I/O-bound)")
    p.add_argument("--spatial-batch-size", type=int, default=1,
                   help="Schedule this many nearby galaxies together; output order unchanged")
    p.add_argument("--ingest-only", action="store_true",
                   help="Verify all extraction shards, then build the unified HATS collection")
    p.add_argument("--sample-mode", choices=("first", "miri"), default="first",
                   help="Use first catalog rows or sources with MIRI exposure (for focused smoke tests)")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                   help="Native-pixel cutout side length; configure separately for JWST/HST")
    p.add_argument("--mag-band", help="Reference filter for the apparent-magnitude cut")
    p.add_argument("--mag-max", type=float, help="Optional AB Kron apparent magnitude upper limit")
    p.add_argument("--chunk-size", type=int, default=4,
                   help="Maximum image records held in memory for Arrow serialization")
    p.add_argument("--rows-per-file", type=int, default=64,
                   help="Target records per completed Parquet checkpoint shard")
    p.add_argument("--pixel-threshold", type=int, default=1024)
    p.add_argument("--highest-healpix-order", type=int, default=13,
                   help="Maximum HEALPix order for dense JADES fields (default: 13; preflight first)")
    p.add_argument("--ingest-workers", type=int, default=2)
    p.add_argument("--hats-batch-size", type=int, default=4,
                   help="Maximum records in a HATS import batch per worker")
    p.add_argument("--skip-hats", action="store_true",
                   help="Only extract and checkpoint cutouts as Parquet; defer the HATS ingest")
    p.add_argument("--scheduler-file", type=Path,
                   help="Connect HATS ingest to external multi-node Dask workers via a scheduler JSON")
    p.add_argument("--min-ingest-workers", type=int, default=1,
                   help="Wait for this many external Dask workers before starting HATS import")
    p.add_argument("--tmp-dir", type=Path)
    p.add_argument("--hats-resume", action="store_true",
                   help="Keep HATS intermediate files on failure and resume an identical import")
    p.add_argument("--hats-mapping-retries", type=int, default=0,
                   help="Retry incomplete HATS mapping with retained checkpoints (requires --hats-resume)")
    args = p.parse_args()
    if args.chunk_size <= 0:
        p.error("--chunk-size must be > 0")
    if args.extract_threads <= 0 or args.spatial_batch_size <= 0:
        p.error("--extract-threads and --spatial-batch-size must be > 0")
    if args.max_objects is not None and args.max_objects <= 0:
        p.error("--max-objects must be > 0")
    if args.rows_per_file <= 0 or args.hats_batch_size <= 0:
        p.error("--rows-per-file and --hats-batch-size must be > 0")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        p.error("Require --num-shards >= 1 and 0 <= --shard-index < --num-shards")
    if args.num_shards > 1 and not (args.skip_hats or args.ingest_only):
        p.error("Distributed extraction requires --skip-hats; run --ingest-only after all shards finish")
    if args.ingest_only and args.skip_hats:
        p.error("--ingest-only and --skip-hats are mutually exclusive")
    if args.ingest_only and args.shard_index:
        p.error("--shard-index applies only to extraction, not --ingest-only")
    if args.ingest_only and args.field:
        p.error("--field applies only to extraction, not --ingest-only")
    if args.min_ingest_workers <= 0:
        p.error("--min-ingest-workers must be > 0")
    if args.min_ingest_workers > 1 and args.scheduler_file is None:
        p.error("--min-ingest-workers > 1 requires --scheduler-file")
    if args.hats_mapping_retries < 0:
        p.error("--hats-mapping-retries must be >= 0")
    if args.hats_resume and args.tmp_dir is None:
        p.error("--hats-resume requires --tmp-dir")
    if args.hats_mapping_retries and not args.hats_resume:
        p.error("--hats-mapping-retries requires --hats-resume")
    if args.image_size <= 0:
        p.error("--image-size must be > 0")
    if args.highest_healpix_order < 4:
        p.error("--highest-healpix-order must be >= 4")
    if (args.mag_band is None) != (args.mag_max is None):
        p.error("--mag-band and --mag-max must be set together")
    allowed = JWST_BANDS if args.instrument == "jwst" else HST_BANDS
    if args.mag_band and args.mag_band.upper() not in allowed:
        p.error(f"--mag-band {args.mag_band} is not a {args.instrument.upper()} filter")
    mag_band = args.mag_band.upper() if args.mag_band else None
    catalog_name = f"jades_{args.instrument}"
    output_root = args.output_root or Path(MMU_V2_HATS_ROOT) / catalog_name
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("release") != "JADES_DR5":
        p.error("Manifest must pin release JADES_DR5")
    if args.download:
        fetch_manifest(args.manifest, args.raw_root)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    if args.ingest_only:
        report = verify_completed_shards(args.scratch_dir,
                                         [cat["field"] for cat in manifest["catalogs"]],
                                         args.num_shards)
        total = report["total_objects"]
        print(f"Verified all {args.num_shards} JADES extraction shards: "
              f"{report['field_counts']} ({total} total objects)", flush=True)
    else:
        scratch = shard_directory(args.scratch_dir, args.shard_index, args.num_shards)
        scratch.mkdir(parents=True, exist_ok=True)
        total = 0
        catalogs = [cat for cat in manifest["catalogs"]
                    if args.field is None or cat["field"] == args.field]
        if not catalogs:
            p.error(f"Field {args.field} is missing from the manifest")
        for cat in catalogs:
            field = cat["field"]
            entries = [m for m in manifest["mosaics"]
                       if m["field"] == field and m["band"].upper() in allowed]
            n = process_field(field, _input(args.raw_root, cat["path"]), entries,
                              args.raw_root, scratch, args.chunk_size, args.max_objects,
                              mag_band, args.mag_max, args.image_size, args.rows_per_file,
                              sample_mode=args.sample_mode,
                              phot_bands=tuple(sorted(allowed, key=_band_sort_key)),
                              num_shards=args.num_shards, shard_index=args.shard_index,
                              extract_threads=args.extract_threads,
                              spatial_batch_size=args.spatial_batch_size)
            print(f"{field} task {args.shard_index}/{args.num_shards}: "
                  f"{n} objects with real image inverse variance", flush=True)
            total += n
        if total == 0 and args.num_shards == 1:
            raise RuntimeError("No valid JADES images; refusing to write empty HATS catalog")
        if args.skip_hats:
            print(f"JADES extraction task {args.shard_index}/{args.num_shards} complete: "
                  f"{scratch} ({total} objects); run --ingest-only after all tasks finish",
                  flush=True)
            return

    if output_root.exists() and any(output_root.iterdir()):
        resume_identity = (
            args.tmp_dir / f"hats_import_{catalog_name}_resume" / "_mmu_resume_identity.json"
            if args.hats_resume else None
        )
        if resume_identity is None or not resume_identity.is_file():
            raise FileExistsError(f"HATS output root already contains data: {output_root}. "
                                  "Choose a new --output-root to preserve existing results")
    from mmu.hats_import import write_hats_from_parquet_dir

    kwargs = {"output_path": str(output_root), "catalog_name": catalog_name,
              "pixel_threshold": args.pixel_threshold, "n_workers": args.ingest_workers,
              "chunksize": args.hats_batch_size,
              "highest_healpix_order": args.highest_healpix_order,
              "expected_total_rows": total,
              "tmp_dir": str(args.tmp_dir) if args.tmp_dir else None,
              "resumable": args.hats_resume,
              "mapping_retries": args.hats_mapping_retries}
    if args.scheduler_file:
        from dask.distributed import Client

        with Client(scheduler_file=str(args.scheduler_file)) as client:
            print(f"Waiting for {args.min_ingest_workers} Dask workers", flush=True)
            client.wait_for_workers(n_workers=args.min_ingest_workers, timeout="20m")
            write_hats_from_parquet_dir(str(args.scratch_dir), client=client, **kwargs)
    else:
        write_hats_from_parquet_dir(str(args.scratch_dir), **kwargs)
    print(f"JADES HATS output: {output_root} ({total} objects)")


if __name__ == "__main__":
    main()
