"""Validate LSST DP2 cutouts from parent catalog through final HATS output."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS

from scripts.lsst_dp2.build_parent_sample_hats import (
    PSF_SIZE,
    clean_mask_from_bits,
    load_manifest,
)
from scripts.lsst_dp2.common import (
    BANDS,
    IMAGE_SIZE,
    PIXEL_SCALE_ARCSEC,
    read_catalog,
    validate_catalog,
)


def _hats_row_count(root: str) -> tuple[int, str]:
    matches = []
    for path in Path(root).rglob("hats.properties"):
        properties = {}
        for line in path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                properties[key] = value
        if properties.get("obs_collection") == "lsst_dp2":
            matches.append((int(properties["hats_nrows"]), str(path)))
    if len(matches) != 1:
        raise ValueError(f"expected one lsst_dp2 HATS catalog, found {len(matches)}")
    return matches[0]


def _validate_coadds(
    catalog, columns, manifest: dict, progress_every: int = 500
) -> dict:
    wcs_cache = {}
    max_axis_offset = 0.0
    max_radial_offset = 0.0
    checked = 0
    units = set()
    for row_index, row in enumerate(catalog, 1):
        tract = int(row[columns["tract"]])
        patch = int(row[columns["patch"]])
        ra = float(row[columns["ra"]])
        dec = float(row[columns["dec"]])
        for band in BANDS:
            info = manifest.get((tract, patch, band))
            if info is None:
                continue
            path = info["output_path"]
            if path not in wcs_cache:
                with fits.open(path, memmap=True) as hdul:
                    image_hdu = hdul["IMAGE"]
                    variance_hdu = hdul["VARIANCE"]
                    image_unit = str(image_hdu.header.get("BUNIT", "")).strip()
                    variance_unit = str(variance_hdu.header.get("BUNIT", "")).strip()
                    units.add((image_unit, variance_unit))
                    if (image_unit, variance_unit) != ("nJy", "nJy2"):
                        raise ValueError(
                            f"unexpected units in {path}: {image_unit!r}, {variance_unit!r}"
                        )
                    wcs_cache[path] = WCS(image_hdu.header)
            x, y = wcs_cache[path].world_to_pixel_values(ra, dec)
            local_x = x - math.ceil(x - IMAGE_SIZE / 2)
            local_y = y - math.ceil(y - IMAGE_SIZE / 2)
            dx = local_x - (IMAGE_SIZE - 1) / 2
            dy = local_y - (IMAGE_SIZE - 1) / 2
            max_axis_offset = max(max_axis_offset, abs(dx), abs(dy))
            max_radial_offset = max(max_radial_offset, math.hypot(dx, dy))
            if abs(dx) > 0.500001 or abs(dy) > 0.500001:
                raise ValueError(
                    f"off-center cutout for {(tract, patch, band)}: dx={dx}, dy={dy}"
                )
            checked += 1
        if progress_every and (
            row_index % progress_every == 0 or row_index == len(catalog)
        ):
            print(f"[validate coadds] {row_index}/{len(catalog)} objects", flush=True)
    return {
        "coadd_products": len(wcs_cache),
        "center_checks": checked,
        "max_center_axis_offset_pix": max_axis_offset,
        "max_center_radial_offset_pix": max_radial_offset,
        "max_center_radial_offset_arcsec": max_radial_offset * PIXEL_SCALE_ARCSEC,
        "fits_units": [list(item) for item in sorted(units)],
    }


def _validate_image(
    object_id: str, image: dict
) -> tuple[float, np.ndarray, np.ndarray]:
    """Validate one nested MMU image struct and return aggregate statistics."""
    if image["band"] != list(BANDS):
        raise ValueError(f"unexpected bands for {object_id}: {image['band']}")

    flux = np.asarray(image["flux"], dtype=np.float32)
    ivar = np.asarray(image["ivar"], dtype=np.float32)
    mask = np.asarray(image["mask"], dtype=bool)
    mask_bits = np.asarray(image["mask_bits"], dtype=np.int32)
    psf_image = np.asarray(image["psf_image"], dtype=np.float32)
    psf_image_valid = np.asarray(image["psf_image_valid"], dtype=bool)
    band_present = np.asarray(image["band_present"], dtype=bool)
    expected_shape = (len(BANDS), IMAGE_SIZE, IMAGE_SIZE)

    if any(array.shape != expected_shape for array in (flux, ivar, mask, mask_bits)):
        raise ValueError(f"invalid image shape for {object_id}")
    if psf_image.shape != (len(BANDS), PSF_SIZE, PSF_SIZE):
        raise ValueError(f"invalid PSF image shape for {object_id}")
    if psf_image_valid.shape != (len(BANDS),) or band_present.shape != (len(BANDS),):
        raise ValueError(f"invalid band/PSF validity shape for {object_id}")
    if not band_present.any():
        raise ValueError(f"all bands are missing for {object_id}")
    if not np.isfinite(psf_image).all():
        raise ValueError(f"non-finite PSF image for {object_id}")

    psf_sums = psf_image.sum(axis=(-2, -1), dtype=np.float64)
    if not np.allclose(psf_sums[psf_image_valid], 1.0, rtol=1e-5, atol=1e-5):
        raise ValueError(f"PSF image is not normalized for {object_id}: {psf_sums}")
    if np.any(psf_sums[~psf_image_valid] != 0):
        raise ValueError(f"invalid PSF image is not zero-filled for {object_id}")
    if not np.isfinite(flux).all() or not np.isfinite(ivar).all():
        raise ValueError(f"non-finite flux/ivar for {object_id}")
    if (ivar < 0).any() or (ivar[~mask] != 0).any():
        raise ValueError(f"invalid ivar/mask relation for {object_id}")
    plane_maps = image["mask_plane_map"]
    if len(plane_maps) != len(BANDS):
        raise ValueError(f"invalid mask-plane provenance for {object_id}")
    for band_index, encoded_mapping in enumerate(plane_maps):
        if not band_present[band_index]:
            continue
        mapping = json.loads(encoded_mapping)
        clean_from_bits = clean_mask_from_bits(mask_bits[band_index], mapping)
        if np.any(mask[band_index] & ~clean_from_bits):
            raise ValueError(
                f"valid mask includes rejected bits for {object_id}/{BANDS[band_index]}"
            )
    psf = np.asarray(image["psf_fwhm"], dtype=np.float32)
    scale = np.asarray(image["scale"], dtype=np.float32)
    if not np.isfinite(psf).all() or (psf[band_present] <= 0).any():
        raise ValueError(f"missing or invalid PSF FWHM for {object_id}")
    if not np.allclose(scale, PIXEL_SCALE_ARCSEC):
        raise ValueError(f"invalid pixel scale for {object_id}: {scale}")
    dataset_ids = image["dataset_id"]
    checksums = image["sha256"]
    if any(not dataset_ids[index] or not checksums[index] for index in np.flatnonzero(band_present)):
        raise ValueError(f"missing provenance for {object_id}")
    absent = np.flatnonzero(~band_present)
    for band_index in absent:
        if (
            flux[band_index].any()
            or ivar[band_index].any()
            or mask[band_index].any()
            or mask_bits[band_index].any()
            or psf_image[band_index].any()
            or psf_image_valid[band_index]
            or psf[band_index] != 0
            or dataset_ids[band_index]
            or checksums[band_index]
            or plane_maps[band_index] != "{}"
            or image["psf_source"][band_index] != "missing"
        ):
            raise ValueError(
                f"absent band is not zero-padded for {object_id}/{BANDS[band_index]}"
            )
    if image["flux_unit"] != ["nJy"] * len(BANDS):
        raise ValueError(f"invalid flux units for {object_id}")
    if image["ivar_unit"] != ["nJy^-2"] * len(BANDS):
        raise ValueError(f"invalid ivar units for {object_id}")
    return float(mask[band_present].mean()), psf_image_valid, band_present


def _validate_parquet(
    scratch_dir: str,
    expected_ids: set[str],
    expected_band_presence: dict[str, np.ndarray],
    progress_every: int = 500,
) -> dict:
    paths = sorted(Path(scratch_dir).glob("part-*.parquet"))
    if not paths:
        raise ValueError(f"no Parquet shards in {scratch_dir}")
    found_ids = set()
    checked_rows = 0
    min_clean_fraction = 1.0
    max_clean_fraction = 0.0
    psf_valid_counts = np.zeros(len(BANDS), dtype=np.int64)
    band_present_counts = np.zeros(len(BANDS), dtype=np.int64)
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=["object_id", "image"], batch_size=64
        ):
            for object_scalar, image_scalar in zip(batch["object_id"], batch["image"]):
                object_id = object_scalar.as_py()
                if object_id in found_ids:
                    raise ValueError(f"duplicate object_id in shards: {object_id}")
                found_ids.add(object_id)
                clean_fraction, psf_valid, band_present = _validate_image(
                    object_id, image_scalar.as_py()
                )
                if not np.array_equal(
                    band_present, expected_band_presence[object_id]
                ):
                    raise ValueError(
                        f"band availability differs from manifest for {object_id}"
                    )
                psf_valid_counts += psf_valid
                band_present_counts += band_present
                min_clean_fraction = min(min_clean_fraction, clean_fraction)
                max_clean_fraction = max(max_clean_fraction, clean_fraction)
                checked_rows += 1
                if progress_every and (
                    checked_rows % progress_every == 0
                    or checked_rows == len(expected_ids)
                ):
                    print(
                        f"[validate parquet] {checked_rows}/{len(expected_ids)} objects",
                        flush=True,
                    )
    if found_ids != expected_ids:
        missing = sorted(expected_ids - found_ids)[:10]
        extra = sorted(found_ids - expected_ids)[:10]
        raise ValueError(
            f"Parquet/catalog object IDs differ; missing={missing}, extra={extra}"
        )
    return {
        "parquet_shards": len(paths),
        "parquet_rows": checked_rows,
        "unique_object_ids": len(found_ids),
        "min_clean_fraction": min_clean_fraction,
        "max_clean_fraction": max_clean_fraction,
        "psf_valid_counts": dict(zip(BANDS, psf_valid_counts.tolist())),
        "psf_valid_fractions": dict(
            zip(BANDS, (psf_valid_counts / checked_rows).tolist())
        ),
        "band_present_counts": dict(zip(BANDS, band_present_counts.tolist())),
        "band_present_fractions": dict(
            zip(BANDS, (band_present_counts / checked_rows).tolist())
        ),
    }


def _manifest_status_counts(path: str) -> dict[str, int]:
    con = sqlite3.connect(path)
    try:
        counts = dict(
            con.execute(
                "SELECT status, COUNT(*) FROM coadds GROUP BY status ORDER BY status"
            ).fetchall()
        )
    finally:
        con.close()
    incomplete = {
        status: count
        for status, count in counts.items()
        if status not in {"complete", "unavailable"}
    }
    if incomplete:
        raise ValueError(f"download manifest has incomplete tasks: {incomplete}")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="input DP2 Object Parquet")
    parser.add_argument("--manifest", required=True, help="completed download manifest")
    parser.add_argument(
        "--scratch-dir", required=True, help="directory of MMU Parquet shards"
    )
    parser.add_argument(
        "--hats-root", required=True, help="outer LSST HATS output root"
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="rehash every mirrored FITS product",
    )
    parser.add_argument("--report", default=None, help="validation JSON output path")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="print progress after this many objects; zero disables it",
    )
    args = parser.parse_args(argv)
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")

    try:
        started = time.monotonic()
        print("[1/4] reading catalog and manifest", flush=True)
        catalog = read_catalog(args.catalog)
        columns = validate_catalog(catalog)
        object_ids = [str(value) for value in catalog[columns["object_id"]]]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("parent catalog contains duplicate object IDs")
        manifest_status_counts = _manifest_status_counts(args.manifest)
        manifest = load_manifest(args.manifest, args.verify_checksums)
        expected_band_presence = {}
        for object_id, row in zip(object_ids, catalog):
            tract = int(row[columns["tract"]])
            patch = int(row[columns["patch"]])
            expected_band_presence[object_id] = np.asarray(
                [(tract, patch, band) in manifest for band in BANDS], dtype=bool
            )
        catalog_elapsed = time.monotonic() - started

        print("[2/4] validating coadds, units, WCS, and centering", flush=True)
        stage_started = time.monotonic()
        coadd_report = _validate_coadds(catalog, columns, manifest, args.progress_every)
        coadd_elapsed = time.monotonic() - stage_started

        print("[3/4] validating Parquet image records", flush=True)
        stage_started = time.monotonic()
        parquet_report = _validate_parquet(
            args.scratch_dir,
            set(object_ids),
            expected_band_presence,
            args.progress_every,
        )
        parquet_elapsed = time.monotonic() - stage_started

        print("[4/4] validating HATS metadata", flush=True)
        stage_started = time.monotonic()
        hats_rows, properties_path = _hats_row_count(args.hats_root)
        if hats_rows != len(catalog):
            raise ValueError(
                f"HATS/catalog row mismatch: {hats_rows} != {len(catalog)}"
            )
        hats_elapsed = time.monotonic() - stage_started
        report = {
            "status": "PASS",
            "catalog_rows": len(catalog),
            "manifest_status_counts": manifest_status_counts,
            **coadd_report,
            **parquet_report,
            "hats_rows": hats_rows,
            "hats_properties": properties_path,
            "elapsed_seconds": {
                "catalog_and_manifest": round(catalog_elapsed, 3),
                "coadds": round(coadd_elapsed, 3),
                "parquet": round(parquet_elapsed, 3),
                "hats": round(hats_elapsed, 3),
                "total": round(time.monotonic() - started, 3),
            },
        }
        report_path = Path(
            args.report or Path(args.scratch_dir) / "validation_report.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_suffix(report_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, report_path)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"PASS: {report_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
