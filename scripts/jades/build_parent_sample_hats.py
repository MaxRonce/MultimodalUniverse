"""Build a JADES DR5 image + photometry HATS catalog from pinned FITS inputs.

The JSON input manifest declares one catalogue per GOODS field and one or more
non-overlapping mosaics per (field, band).  Each mosaic explicitly declares the
SCI and real variance/ERR/IVAR HDUs.  No synthetic inverse variance is allowed.
Run ``download_data.py`` first or pass ``--download`` for the full workflow.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from mmu.hats_configs import MMU_V2_HATS_ROOT, RAW_DATA_ROOT
from scripts.jades.download_data import fetch_manifest


CATALOG_NAME = "jades"
DEFAULT_ROOT = Path(RAW_DATA_ROOT) / "JADES_DR5"
IMAGE_SIZE = 160


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


def _cutout(data: np.ndarray, wcs: WCS, ra: float, dec: float, fill: float) -> np.ndarray:
    x, y = wcs.all_world2pix(ra, dec, 0)
    return np.asarray(Cutout2D(data, (x, y), (IMAGE_SIZE, IMAGE_SIZE), mode="partial", fill_value=fill).data)


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
                        "wcs": wcs, "flux_scale": scale,
                        "scale_arcsec": float(np.mean(np.abs(proj_plane_pixel_scales(wcs.celestial))) * 3600)})
    return mosaics


def _record(index: int, size, circ, flags, mosaics: list[dict], field: str) -> dict | None:
    ra, dec = float(size["RA"][index]), float(size["DEC"][index])
    if not np.isfinite(ra) or not np.isfinite(dec):
        return None
    bands, fluxes, ivars, masks, scales, scalar_flux, scalar_err = [], [], [], [], [], [], []
    for m in mosaics:
        band = m["band"]
        if band in bands:
            continue  # Another subregion already supplied this band for this source.
        if flags is not None and f"{band}_TEXP" in flags.names:
            if not np.isfinite(flags[f"{band}_TEXP"][index]) or flags[f"{band}_TEXP"][index] <= 0:
                continue
        try:
            f = _cutout(m["sci"], m["wcs"], ra, dec, np.nan).astype(np.float32)
            u = _cutout(m["uncertainty"], m["wcs"], ra, dec, np.nan).astype(np.float32)
        except (ValueError, TypeError, IndexError):
            continue  # Outside mosaic footprint; try the next non-overlapping mosaic.
        weight = variance_to_ivar(u, m["kind"])
        # Check raw validity first, then transform uncertainties with flux units.
        valid = np.isfinite(f) & (weight > 0)
        if not np.any(valid):
            continue
        f = np.where(valid, f * m["flux_scale"], 0).astype(np.float32)
        weight = np.where(valid, weight / m["flux_scale"]**2, 0).astype(np.float32)
        bands.append(band)
        fluxes.append(f)
        ivars.append(weight)
        masks.append(valid)
        scales.append(m["scale_arcsec"])
        phot = f"{band}_CIRC3"
        err = f"{band}_CIRC3_e"
        scalar_flux.append(float(circ[phot][index]) if phot in circ.names else np.nan)
        scalar_err.append(float(circ[err][index]) if err in circ.names else np.nan)
    if not bands:
        return None
    return {"object_id": f"{field}:{int(size['ID'][index])}", "ra": ra, "dec": dec,
            "field": field, "bands": bands, "flux": fluxes, "ivar": ivars,
            "mask": masks, "scale": scales, "phot_flux": scalar_flux,
            "phot_err": scalar_err, "size_a": float(size["A"][index]),
            "size_b": float(size["B"][index])}


def _table(records: list[dict]) -> pa.Table:
    f32 = pa.float32()
    cube = pa.large_list(pa.large_list(pa.large_list(f32)))
    bool_cube = pa.large_list(pa.large_list(pa.large_list(pa.bool_())))
    image = pa.StructArray.from_arrays([
        pa.array([r["bands"] for r in records], type=pa.list_(pa.string())),
        pa.array([[x.tolist() for x in r["flux"]] for r in records], type=cube),
        pa.array([[x.tolist() for x in r["ivar"]] for r in records], type=cube),
        pa.array([[x.tolist() for x in r["mask"]] for r in records], type=bool_cube),
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
                     "photometry": phot,
                     "size_a": pa.array([r["size_a"] for r in records], type=f32),
                     "size_b": pa.array([r["size_b"] for r in records], type=f32)})


def process_field(field: str, catalog_path: Path, mosaics: list[dict], raw_root: Path,
                  scratch: Path, chunk_size: int, max_objects: int | None) -> int:
    done = scratch / f"{field}.done"
    if done.exists():
        return int(done.read_text())
    for old in scratch.glob(f"{field}_*.parquet"):
        old.unlink()  # Incomplete previous build; rewrite this field from scratch.
    # Existing partial shards are overwritten atomically, then the done marker is written.
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
        images = _open_mosaics(stack, mosaics, raw_root)
        if not images:
            raise ValueError(f"{field}: no mosaics configured")
        count, chunk, shard = 0, [], 0
        n = min(len(size), max_objects) if max_objects else len(size)
        for i in range(n):
            row = _record(i, size, circ, flags, images, field)
            if row is None:
                continue
            chunk.append(row)
            if len(chunk) >= chunk_size or i == n - 1:
                target = scratch / f"{field}_{shard:05d}.parquet"
                tmp = target.with_suffix(".parquet.tmp")
                pq.write_table(_table(chunk), tmp)
                os.replace(tmp, target)
                count += len(chunk)
                chunk.clear()
                shard += 1
        if chunk:
            target = scratch / f"{field}_{shard:05d}.parquet"
            tmp = target.with_suffix(".parquet.tmp")
            pq.write_table(_table(chunk), tmp)
            os.replace(tmp, target)
            count += len(chunk)
    done.write_text(str(count))
    return count


def main() -> None:
    from mmu.hats_import import write_hats_from_parquet_dir

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--output-root", type=Path, default=Path(MMU_V2_HATS_ROOT) / CATALOG_NAME)
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--max-objects", type=int)
    p.add_argument("--chunk-size", type=int, default=50)
    p.add_argument("--pixel-threshold", type=int, default=8192)
    p.add_argument("--ingest-workers", type=int, default=4)
    p.add_argument("--tmp-dir", type=Path)
    args = p.parse_args()
    if args.chunk_size <= 0:
        p.error("--chunk-size must be > 0")
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("release") != "JADES_DR5":
        p.error("Manifest must pin release JADES_DR5")
    if args.download:
        fetch_manifest(args.manifest, args.raw_root)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for cat in manifest["catalogs"]:
        field = cat["field"]
        entries = [m for m in manifest["mosaics"] if m["field"] == field]
        n = process_field(field, _input(args.raw_root, cat["path"]), entries,
                          args.raw_root, args.scratch_dir, args.chunk_size, args.max_objects)
        print(f"{field}: {n} objects with real image inverse variance", flush=True)
        total += n
    if total == 0:
        raise RuntimeError("No valid JADES images; refusing to write empty HATS catalog")
    write_hats_from_parquet_dir(str(args.scratch_dir), output_path=str(args.output_root),
                               catalog_name=CATALOG_NAME, pixel_threshold=args.pixel_threshold,
                               n_workers=args.ingest_workers,
                               tmp_dir=str(args.tmp_dir) if args.tmp_dir else None)
    print(f"JADES HATS output: {args.output_root} ({total} objects)")


if __name__ == "__main__":
    main()
