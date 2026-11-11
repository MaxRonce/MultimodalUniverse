"""Build a JADES DR5 image + photometry HATS catalog from pinned FITS inputs.

The JSON input manifest declares one catalogue per GOODS field and one or more
non-overlapping mosaics per (field, band).  Each mosaic explicitly declares the
SCI and real variance/ERR/IVAR HDUs.  No synthetic inverse variance is allowed.
Run ``download_data.py`` first or pass ``--download`` for the full workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
JWST_BANDS = {
    "F070W", "F090W", "F115W", "F150W", "F162M", "F182M", "F200W", "F210M",
    "F250M", "F277W", "F300M", "F335M", "F356W", "F410M", "F430M", "F444W",
    "F460M", "F480M", "F560W", "F770W", "F1000W", "F1280W", "F1500W",
    "F1800W", "F2100W", "F2550W",
}
HST_BANDS = {"F435W", "F606W", "F775W", "F814W", "F850LP", "F105W", "F125W", "F140W", "F160W"}


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
        # Store the raw FITS arrays; process only the requested cutouts.
        mosaics.append({"band": entry["band"].upper(), "sci": science.data,
                        "uncertainty": uncertainty.data, "kind": entry["uncertainty_kind"],
                        "wcs": wcs, "path": entry["path"], "flux_scale": scale,
                        "scale_arcsec": float(np.mean(np.abs(proj_plane_pixel_scales(wcs.celestial))) * 3600)})
    return mosaics


def _prepare_positions(mosaics: list[dict], size, selected_indices: np.ndarray) -> None:
    """Vectorize WCS transformations once per mosaic, not once per galaxy."""
    ra = np.asarray(size["RA"][selected_indices], dtype=np.float64)
    dec = np.asarray(size["DEC"][selected_indices], dtype=np.float64)
    for mosaic in mosaics:
        mosaic["x"], mosaic["y"] = mosaic["wcs"].all_world2pix(ra, dec, 0)


def _catalog_photoz(photoz, index: int) -> dict:
    result = {}
    for name in ("z_spec", "z_peak", "z_ml", "l68", "u68"):
        value = float(photoz[name][index]) if photoz is not None and name in photoz.names else np.nan
        result[name] = value if np.isfinite(value) and value >= 0 else np.nan
    return result


def _record(index: int, size, circ, flags, mosaics: list[dict], field: str,
            image_size: int = IMAGE_SIZE, miri=None, photoz=None,
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
        if not np.any(valid):
            continue
        weight = np.where(valid, weight / m["flux_scale"]**2, 0).astype(np.float32)
        score = (int(valid.sum()), float(np.median(weight[valid])))
        if band not in candidates or score > candidates[band][0]:
            f = np.where(valid, f * m["flux_scale"], 0).astype(np.float32)
            candidates[band] = (score, f, weight, valid, m)

    if not candidates:
        return None
    bands = list(candidates)
    fluxes, ivars, masks, scales, sources, scalar_flux, scalar_err = [], [], [], [], [], [], []
    for band, (_, f, weight, valid, m) in candidates.items():
        fluxes.append(f)
        ivars.append(weight)
        masks.append(valid)
        scales.append(m["scale_arcsec"])
        sources.append(m["path"])
        phot_table = miri if band in {"F560W", "F770W", "F1000W", "F1280W", "F1500W", "F1800W", "F2100W", "F2550W"} else circ
        phot = f"{band}_CIRC3"
        err = f"{band}_CIRC3_e"
        scalar_flux.append(float(phot_table[phot][index]) if phot_table is not None and phot in phot_table.names else np.nan)
        scalar_err.append(float(phot_table[err][index]) if phot_table is not None and err in phot_table.names else np.nan)
    return {"object_id": f"{field}:{int(size['ID'][index])}", "ra": ra, "dec": dec,
            "field": field, "bands": bands, "flux": fluxes, "ivar": ivars,
            "mask": masks, "scale": scales, "image_sources": sources, "phot_flux": scalar_flux,
            "phot_err": scalar_err, "size_a": float(size["A"][index]),
            "size_b": float(size["B"][index]), "redshift": _catalog_photoz(photoz, index)}


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
        pa.array([r["scale"] for r in records], type=pa.list_(f32)),
    ], names=["band", "flux", "ivar", "mask", "scale"])
    phot = pa.StructArray.from_arrays([
        pa.array([r["bands"] for r in records], type=pa.list_(pa.string())),
        pa.array([r["phot_flux"] for r in records], type=pa.list_(f32)),
        pa.array([r["phot_err"] for r in records], type=pa.list_(f32)),
    ], names=["band", "flux_circ3_njy", "flux_err_circ3_njy"])
    return pa.table({"object_id": pa.array([r["object_id"] for r in records]),
                     "ra": pa.array([r["ra"] for r in records], type=pa.float64()),
                     "dec": pa.array([r["dec"] for r in records], type=pa.float64()),
                     "field": pa.array([r["field"] for r in records]), "image": image,
                     "image_sources": pa.array([r["image_sources"] for r in records],
                                               type=pa.list_(pa.string())),
                     "photometry": phot,
                     "redshift": pa.StructArray.from_arrays([
                         pa.array([r["redshift"][name] for r in records], type=f32)
                         for name in ("z_spec", "z_peak", "z_ml", "l68", "u68")
                     ], names=["z_spec", "z_peak", "z_ml", "l68", "u68"]),
                     "size_a": pa.array([r["size_a"] for r in records], type=f32),
                     "size_b": pa.array([r["size_b"] for r in records], type=f32)})


def process_field(field: str, catalog_path: Path, mosaics: list[dict], raw_root: Path,
                  scratch: Path, chunk_size: int, max_objects: int | None,
                  mag_band: str | None = None, mag_max: float | None = None,
                  image_size: int = IMAGE_SIZE, rows_per_file: int = 64,
                  sample_mode: str = "first") -> int:
    """Extract a field to bounded-memory Parquet shards with crash checkpoints.

    A checkpoint is committed only after a shard is atomically published. On
    resume, at most one in-progress shard is recomputed; complete shards stay.
    """
    done = scratch / f"{field}.done"
    progress = scratch / f"{field}.progress.json"
    identity = {"field": field, "catalog": str(catalog_path),
                "catalog_size": catalog_path.stat().st_size,
                "mosaics": mosaics, "chunk_size": chunk_size,
                "rows_per_file": rows_per_file, "max_objects": max_objects,
                "sample_mode": sample_mode,
                "mag_band": mag_band, "mag_max": mag_max, "image_size": image_size}
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
            photoz, _ = _catalog_columns(catalog_hdus, "PHOTOZ")
        except ValueError:
            photoz = None
        if photoz is not None and not _ids_aligned(size, photoz):
            raise ValueError(f"{field}: SIZE and PHOTOZ IDs not aligned")
        magnitude = None
        if mag_max is not None:
            kron, kron_columns = _catalog_columns(catalog_hdus, "KRON")
            if not _ids_aligned(size, kron):
                raise ValueError(f"{field}: SIZE and KRON IDs not aligned")
            flux_column = f"{mag_band}_KRON"
            if flux_column not in kron_columns:
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
        n = len(selected)
        _prepare_positions(images, size, selected)
        count = checkpoint["count"]
        shard = checkpoint["next_shard"]
        chunk, shard_rows, writer = [], 0, None
        for position in range(checkpoint["next_index"], n):
            i = int(selected[position])
            if magnitude is not None and (not np.isfinite(magnitude[i]) or magnitude[i] >= mag_max):
                continue
            row = _record(i, size, circ, flags, images, field, image_size,
                          miri=miri, photoz=photoz, coord_index=position)
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
                    print(f"{field}: checkpoint {position+1}/{n} scanned, "
                          f"{count} cutouts, {shard} parquet files", flush=True)
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
    done.write_text(str(count))
    return count


def main() -> None:
    from mmu.hats_import import write_hats_from_parquet_dir

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
    p.add_argument("--pixel-threshold", type=int, default=256)
    p.add_argument("--ingest-workers", type=int, default=2)
    p.add_argument("--hats-batch-size", type=int, default=4,
                   help="Maximum records in a HATS import batch per worker")
    p.add_argument("--skip-hats", action="store_true",
                   help="Only extract and checkpoint cutouts as Parquet; defer the HATS ingest")
    p.add_argument("--tmp-dir", type=Path)
    args = p.parse_args()
    if args.chunk_size <= 0:
        p.error("--chunk-size must be > 0")
    if args.max_objects is not None and args.max_objects <= 0:
        p.error("--max-objects must be > 0")
    if args.rows_per_file <= 0 or args.hats_batch_size <= 0:
        p.error("--rows-per-file and --hats-batch-size must be > 0")
    if args.image_size <= 0:
        p.error("--image-size must be > 0")
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
    total = 0
    for cat in manifest["catalogs"]:
        field = cat["field"]
        entries = [m for m in manifest["mosaics"]
                   if m["field"] == field and m["band"].upper() in allowed]
        n = process_field(field, _input(args.raw_root, cat["path"]), entries,
                          args.raw_root, args.scratch_dir, args.chunk_size, args.max_objects,
                          mag_band, args.mag_max, args.image_size, args.rows_per_file,
                          sample_mode=args.sample_mode)
        print(f"{field}: {n} objects with real image inverse variance", flush=True)
        total += n
    if total == 0:
        raise RuntimeError("No valid JADES images; refusing to write empty HATS catalog")
    if args.skip_hats:
        print(f"JADES Parquet extraction complete: {args.scratch_dir} ({total} objects); "
              "rerun without --skip-hats to build HATS", flush=True)
        return
    write_hats_from_parquet_dir(str(args.scratch_dir), output_path=str(output_root),
                               catalog_name=catalog_name, pixel_threshold=args.pixel_threshold,
                               n_workers=args.ingest_workers,
                               chunksize=args.hats_batch_size,
                               tmp_dir=str(args.tmp_dir) if args.tmp_dir else None)
    print(f"JADES HATS output: {output_root} ({total} objects)")


if __name__ == "__main__":
    main()
