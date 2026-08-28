"""Validate LSST DP2 cutouts from parent catalog through final HATS output."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS

from scripts.lsst_dp2.build_parent_sample_hats import PSF_SIZE, load_manifest
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


def _validate_coadds(catalog, columns, manifest: dict) -> dict:
    wcs_cache = {}
    max_axis_offset = 0.0
    max_radial_offset = 0.0
    checked = 0
    units = set()
    for row in catalog:
        tract = int(row[columns["tract"]])
        patch = int(row[columns["patch"]])
        ra = float(row[columns["ra"]])
        dec = float(row[columns["dec"]])
        for band in BANDS:
            info = manifest.get((tract, patch, band))
            if info is None:
                raise ValueError(f"missing complete coadd for {(tract, patch, band)}")
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
    return {
        "coadd_products": len(wcs_cache),
        "center_checks": checked,
        "max_center_axis_offset_pix": max_axis_offset,
        "max_center_radial_offset_pix": max_radial_offset,
        "max_center_radial_offset_arcsec": max_radial_offset * PIXEL_SCALE_ARCSEC,
        "fits_units": [list(item) for item in sorted(units)],
    }


def _validate_parquet(scratch_dir: str, expected_ids: set[str]) -> dict:
    paths = sorted(Path(scratch_dir).glob("part-*.parquet"))
    if not paths:
        raise ValueError(f"no Parquet shards in {scratch_dir}")
    found_ids = set()
    checked_rows = 0
    min_clean_fraction = 1.0
    max_clean_fraction = 0.0
    psf_valid_counts = np.zeros(len(BANDS), dtype=np.int64)
    for path in paths:
        table = pq.read_table(path)
        for index in range(table.num_rows):
            object_id = table["object_id"][index].as_py()
            if object_id in found_ids:
                raise ValueError(f"duplicate object_id in shards: {object_id}")
            found_ids.add(object_id)
            image = table["image"][index].as_py()
            if image["band"] != list(BANDS):
                raise ValueError(f"unexpected bands for {object_id}: {image['band']}")
            flux = np.asarray(image["flux"], dtype=np.float32)
            ivar = np.asarray(image["ivar"], dtype=np.float32)
            mask = np.asarray(image["mask"], dtype=bool)
            mask_bits = np.asarray(image["mask_bits"], dtype=np.int32)
            psf_image = np.asarray(image["psf_image"], dtype=np.float32)
            psf_image_valid = np.asarray(image["psf_image_valid"], dtype=bool)
            expected_shape = (len(BANDS), IMAGE_SIZE, IMAGE_SIZE)
            if any(array.shape != expected_shape for array in (flux, ivar, mask, mask_bits)):
                raise ValueError(f"invalid image shape for {object_id}")
            if psf_image.shape != (len(BANDS), PSF_SIZE, PSF_SIZE):
                raise ValueError(f"invalid PSF image shape for {object_id}")
            if psf_image_valid.shape != (len(BANDS),):
                raise ValueError(f"invalid PSF validity shape for {object_id}")
            if not np.isfinite(psf_image).all():
                raise ValueError(f"non-finite PSF image for {object_id}")
            psf_sums = psf_image.sum(axis=(-2, -1), dtype=np.float64)
            if not np.allclose(psf_sums[psf_image_valid], 1.0, rtol=1e-5, atol=1e-5):
                raise ValueError(f"PSF image is not normalized for {object_id}: {psf_sums}")
            if np.any(psf_sums[~psf_image_valid] != 0):
                raise ValueError(f"invalid PSF image is not zero-filled for {object_id}")
            psf_valid_counts += psf_image_valid
            if not np.isfinite(flux).all() or not np.isfinite(ivar).all():
                raise ValueError(f"non-finite flux/ivar for {object_id}")
            if (ivar < 0).any() or (ivar[~mask] != 0).any():
                raise ValueError(f"invalid ivar/mask relation for {object_id}")
            if not all(image["band_present"]):
                raise ValueError(f"missing band for {object_id}")
            psf = np.asarray(image["psf_fwhm"], dtype=np.float32)
            scale = np.asarray(image["scale"], dtype=np.float32)
            if not np.isfinite(psf).all() or (psf <= 0).any():
                raise ValueError(f"missing or invalid PSF FWHM for {object_id}")
            if not np.allclose(scale, PIXEL_SCALE_ARCSEC):
                raise ValueError(f"invalid pixel scale for {object_id}: {scale}")
            if not all(image["dataset_id"]) or not all(image["sha256"]):
                raise ValueError(f"missing provenance for {object_id}")
            if image["flux_unit"] != ["nJy"] * len(BANDS):
                raise ValueError(f"invalid flux units for {object_id}")
            if image["ivar_unit"] != ["nJy^-2"] * len(BANDS):
                raise ValueError(f"invalid ivar units for {object_id}")
            clean_fraction = float(mask.mean())
            min_clean_fraction = min(min_clean_fraction, clean_fraction)
            max_clean_fraction = max(max_clean_fraction, clean_fraction)
            checked_rows += 1
    if found_ids != expected_ids:
        missing = sorted(expected_ids - found_ids)[:10]
        extra = sorted(found_ids - expected_ids)[:10]
        raise ValueError(f"Parquet/catalog object IDs differ; missing={missing}, extra={extra}")
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
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument("--hats-root", required=True)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    try:
        catalog = read_catalog(args.catalog)
        columns = validate_catalog(catalog)
        object_ids = [str(value) for value in catalog[columns["object_id"]]]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("parent catalog contains duplicate object IDs")
        manifest = load_manifest(args.manifest, args.verify_checksums)
        report = {
            "status": "PASS",
            "catalog_rows": len(catalog),
            **_validate_coadds(catalog, columns, manifest),
            **_validate_parquet(args.scratch_dir, set(object_ids)),
        }
        hats_rows, properties_path = _hats_row_count(args.hats_root)
        if hats_rows != len(catalog):
            raise ValueError(f"HATS/catalog row mismatch: {hats_rows} != {len(catalog)}")
        report["hats_rows"] = hats_rows
        report["hats_properties"] = properties_path
        report_path = Path(args.report or Path(args.scratch_dir) / "validation_report.json")
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"PASS: {report_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
