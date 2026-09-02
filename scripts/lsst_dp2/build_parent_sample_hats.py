"""Build the MMU ``lsst_dp2`` image catalog from mirrored DP2 deep coadds.

Inputs are a DP2 Object parent catalog, the SQLite download manifest produced
by :mod:`scripts.lsst_dp2.download_coadds`, and its immutable FITS mirror.
Each output row contains six aligned 160x160 channels in the standard MMU
``image`` struct. Missing bands are explicitly padded and marked absent.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from astropy.table import Table
from filelock import FileLock

from mmu.hats_configs import MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats_from_parquet_dir
from scripts.lsst_dp2.coadd import (
    CORE_REQUIRED_MASK_PLANES,
    PSF_SIZE,
    clean_mask_from_bits,
    make_stamp,
    open_maskedimage,
    psf_kernel_at,
)
from scripts.lsst_dp2.common import (
    BANDS,
    CATALOG_NAME,
    DEFAULT_REJECT_MASK_PLANES,
    IMAGE_SIZE,
    PIXEL_SCALE_ARCSEC,
    mask_plane_mapping,
    read_catalog,
    sha256_file,
    validate_catalog,
)
from scripts.lsst_dp2.schema import (
    OUTPUT_SCHEMA_VERSION,
    PHOTOMETRY_SUFFIXES,
    build_image_struct,
    build_table,
)

__all__ = [
    "CORE_REQUIRED_MASK_PLANES",
    "OUTPUT_SCHEMA_VERSION",
    "PHOTOMETRY_SUFFIXES",
    "PSF_SIZE",
    "build_image_struct",
    "build_table",
    "clean_mask_from_bits",
    "make_stamp",
    "mask_plane_mapping",
    "open_maskedimage",
    "psf_kernel_at",
]


def _finite_float(value, default: float = float("nan")) -> float:
    try:
        if np.ma.is_masked(value):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _catalog_column_lookup(catalog: Table) -> dict[str, str]:
    return {name.casefold(): name for name in catalog.colnames}


def catalog_psf_fwhm(
    catalog: Table,
    row,
    band: str,
    column_lookup: dict[str, str] | None = None,
) -> float | None:
    """Compute catalog PSF FWHM from determinant-radius moments, in arcsec."""
    lookup = column_lookup or _catalog_column_lookup(catalog)
    columns = [
        lookup.get(f"{band}_{moment}psf".casefold()) for moment in ("ixx", "iyy", "ixy")
    ]
    if any(column is None for column in columns):
        return None
    ixx, iyy, ixy = (_finite_float(row[column]) for column in columns)
    determinant = ixx * iyy - ixy * ixy
    if not np.isfinite(determinant) or determinant <= 0:
        return None
    return float(2.354820045 * determinant**0.25 * PIXEL_SCALE_ARCSEC)


def patch_psf_fwhm(catalog: Table, rows: Table) -> dict[str, float | None]:
    """Return robust per-band PSF fallbacks from valid Object moments in a patch."""
    lookup = _catalog_column_lookup(catalog)
    result = {}
    for band in BANDS:
        values = [catalog_psf_fwhm(catalog, row, band, lookup) for row in rows]
        valid = [value for value in values if value is not None and value > 0]
        result[band] = float(np.median(valid)) if valid else None
    return result


def load_manifest(
    path: str, verify_checksums: bool = False
) -> dict[tuple[int, int, str], dict]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT tract, patch, band, output_path, sha256, dataset_id,
               s_resolution, mask_planes, status
        FROM coadds
        """
    ).fetchall()
    con.close()
    result = {}
    for row in rows:
        item = dict(row)
        if item["status"] != "complete" or not os.path.isfile(item["output_path"]):
            continue
        if verify_checksums and sha256_file(item["output_path"]) != item["sha256"]:
            raise ValueError(f"checksum mismatch: {item['output_path']}")
        result[(int(item["tract"]), int(item["patch"]), str(item["band"]))] = item
    return result


def _row_value(row, column: str | None, default):
    if column is None:
        return default
    value = row[column]
    if np.ma.is_masked(value):
        return default
    return value


def _make_records(
    catalog: Table,
    rows: Table,
    columns: dict[str, str | None],
    coadds: dict[str, dict | None],
    manifest_rows: dict[str, dict | None],
    reject_planes: tuple[str, ...],
    patch_psf: dict[str, float | None],
    catalog_columns: dict[str, str],
) -> list[dict]:
    records = []
    n = len(rows)
    for row in rows:
        flux = np.zeros((len(BANDS), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        ivar = np.zeros_like(flux)
        mask = np.zeros(flux.shape, dtype=bool)
        mask_bits = np.zeros(flux.shape, dtype=np.int32)
        psf_image = np.zeros((len(BANDS), PSF_SIZE, PSF_SIZE), dtype=np.float32)
        psf_image_valid = np.zeros(len(BANDS), dtype=bool)
        psf_fwhm = np.zeros(len(BANDS), dtype=np.float32)
        band_present = np.zeros(len(BANDS), dtype=bool)
        psf_source = ["missing"] * len(BANDS)
        dataset_ids = [""] * len(BANDS)
        checksums = [""] * len(BANDS)
        plane_maps = ["{}"] * len(BANDS)
        ra = float(row[columns["ra"]])
        dec = float(row[columns["dec"]])

        for band_index, band in enumerate(BANDS):
            coadd = coadds[band]
            info = manifest_rows[band]
            if coadd is None or info is None:
                continue
            stamp = make_stamp(coadd, ra, dec, reject_planes)
            (
                flux[band_index],
                ivar[band_index],
                mask[band_index],
                mask_bits[band_index],
            ) = stamp
            band_present[band_index] = True
            dataset_ids[band_index] = info.get("dataset_id") or ""
            checksums[band_index] = info.get("sha256") or ""
            plane_maps[band_index] = json.dumps(coadd["mask_mapping"], sort_keys=True)
            psf_image[band_index], psf_image_valid[band_index] = psf_kernel_at(
                coadd, ra, dec
            )
            catalog_psf = catalog_psf_fwhm(catalog, row, band, catalog_columns)
            if catalog_psf is not None:
                psf_fwhm[band_index] = catalog_psf
                psf_source[band_index] = "dp2.Object moments"
            elif patch_psf[band] is not None:
                psf_fwhm[band_index] = patch_psf[band]
                psf_source[band_index] = "patch median dp2.Object moments"
            elif (
                info.get("s_resolution") is not None
                and np.isfinite(info["s_resolution"])
                and float(info["s_resolution"]) > 0
            ):
                psf_fwhm[band_index] = float(info["s_resolution"])
                psf_source[band_index] = "SIA s_resolution"

        photometry = {}
        for band in BANDS:
            for suffix in PHOTOMETRY_SUFFIXES:
                name = f"{band}_{suffix}"
                column = catalog_columns.get(name.casefold())
                photometry[name] = (
                    _finite_float(row[column]) if column else float("nan")
                )
        records.append(
            {
                "ra": ra,
                "dec": dec,
                "object_id": str(_row_value(row, columns["object_id"], "")),
                "tract": int(row[columns["tract"]]),
                "patch": int(row[columns["patch"]]),
                "ref_band": str(_row_value(row, columns["ref_band"], "")),
                "ref_extendedness": _finite_float(
                    _row_value(row, columns["ref_extendedness"], float("nan"))
                ),
                "detect_is_isolated": bool(
                    _row_value(row, columns["detect_is_isolated"], False)
                ),
                "flux": flux,
                "ivar": ivar,
                "mask": mask,
                "mask_bits": mask_bits,
                "psf_image": psf_image,
                "psf_image_valid": psf_image_valid,
                "psf_fwhm": psf_fwhm,
                "scale": np.full(len(BANDS), PIXEL_SCALE_ARCSEC, dtype=np.float32),
                "band_present": band_present,
                "psf_source": psf_source,
                "dataset_id": dataset_ids,
                "sha256": checksums,
                "mask_plane_map": plane_maps,
                "photometry": photometry,
            }
        )
    assert len(records) == n
    return records


def process_patch(
    catalog: Table,
    rows: Table,
    columns: dict[str, str | None],
    manifest: dict[tuple[int, int, str], dict],
    scratch_dir: str,
    objects_per_shard: int,
    reject_planes: tuple[str, ...],
    require_all_bands: bool,
) -> tuple[int, int]:
    Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    tract = int(rows[columns["tract"]][0])
    patch = int(rows[columns["patch"]][0])
    infos = {band: manifest.get((tract, patch, band)) for band in BANDS}
    missing = [band for band, info in infos.items() if info is None]
    if missing and require_all_bands:
        raise RuntimeError(f"{tract}/{patch} is missing bands {missing}")

    written = 0
    patch_psf = patch_psf_fwhm(catalog, rows)
    catalog_columns = _catalog_column_lookup(catalog)
    with contextlib.ExitStack() as stack:
        coadds = {
            band: stack.enter_context(open_maskedimage(info["output_path"]))
            if info
            else None
            for band, info in infos.items()
        }
        for start in range(0, len(rows), objects_per_shard):
            stop = min(start + objects_per_shard, len(rows))
            path = Path(scratch_dir) / f"part-{tract}-{patch}-{start:08d}.parquet"
            if path.exists():
                actual_rows = pq.read_metadata(path).num_rows
                if actual_rows != stop - start:
                    raise RuntimeError(
                        f"existing shard has {actual_rows} rows, expected {stop - start}: {path}"
                    )
                written += stop - start
                continue
            records = _make_records(
                catalog,
                rows[start:stop],
                columns,
                coadds,
                infos,
                reject_planes,
                patch_psf,
                catalog_columns,
            )
            table = build_table(records)
            temp = path.with_suffix(".parquet.tmp")
            pq.write_table(table, temp, compression="zstd")
            os.replace(temp, path)
            written += table.num_rows
    return len(rows), written


def grouped_rows(catalog: Table, columns: dict[str, str | None]) -> list[Table]:
    order = np.lexsort(
        (
            np.asarray(catalog[columns["patch"]], dtype=np.int64),
            np.asarray(catalog[columns["tract"]], dtype=np.int64),
        )
    )
    sorted_catalog = catalog[order]
    tracts = np.asarray(sorted_catalog[columns["tract"]], dtype=np.int64)
    patches = np.asarray(sorted_catalog[columns["patch"]], dtype=np.int64)
    starts = np.r_[
        0,
        np.flatnonzero((tracts[1:] != tracts[:-1]) | (patches[1:] != patches[:-1])) + 1,
    ]
    ends = np.r_[starts[1:], len(sorted_catalog)]
    return [sorted_catalog[start:end] for start, end in zip(starts, ends)]


def ensure_build_contract(
    scratch_dir: str,
    catalog_path: str,
    manifest_path: str,
    objects_per_shard: int,
    reject_planes: tuple[str, ...],
) -> None:
    """Refuse to mix restart shards produced from incompatible inputs."""
    con = sqlite3.connect(manifest_path)
    products = con.execute(
        """
        SELECT tract, patch, band, status, COALESCE(sha256, ''),
               COALESCE(dataset_id, '')
        FROM coadds
        ORDER BY tract, patch, band
        """
    ).fetchall()
    con.close()
    manifest_signature = hashlib.sha256(
        json.dumps(products, separators=(",", ":")).encode()
    ).hexdigest()
    contract = {
        "catalog_path": str(Path(catalog_path).resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_signature": manifest_signature,
        "bands": list(BANDS),
        "image_size": IMAGE_SIZE,
        "pixel_scale_arcsec": PIXEL_SCALE_ARCSEC,
        "objects_per_shard": objects_per_shard,
        "reject_mask_planes": list(reject_planes),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "pipeline_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("build_parent_sample_hats.py", "coadd.py", "schema.py")
        },
    }
    path = Path(scratch_dir) / "build_contract.json"
    with FileLock(str(path) + ".lock"):
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != contract:
                raise RuntimeError(
                    "scratch build contract differs from the requested inputs or code; "
                    f"start a clean run directory: {path}"
                )
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, help="input DP2 Object Parquet")
    parser.add_argument("--manifest", required=True, help="completed download manifest")
    parser.add_argument(
        "--scratch-dir",
        required=True,
        help="restartable intermediate Parquet directory",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
        help="outer output directory passed to the MMU HATS writer",
    )
    parser.add_argument(
        "--objects-per-shard",
        type=int,
        default=64,
        help="rows per intermediate Parquet file",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=8192,
        help="HATS import partition threshold",
    )
    parser.add_argument(
        "--ingest-workers", type=int, default=8, help="Dask workers for HATS ingestion"
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="number of scatter jobs"
    )
    parser.add_argument(
        "--shard-idx", type=int, default=0, help="zero-based scatter job index"
    )
    parser.add_argument(
        "--skip-ingest", action="store_true", help="write Parquet only (scatter stage)"
    )
    parser.add_argument(
        "--only-ingest",
        action="store_true",
        help="ingest existing Parquet only (gather stage)",
    )
    parser.add_argument(
        "--require-all-bands",
        action="store_true",
        help="fail if any patch-band is unavailable",
    )
    parser.add_argument(
        "--verify-checksums",
        action="store_true",
        help="rehash mirrored FITS before processing",
    )
    parser.add_argument(
        "--reject-mask-planes",
        default=",".join(DEFAULT_REJECT_MASK_PLANES),
        help="comma-separated Rubin mask planes rejected from the valid-pixel mask",
    )
    args = parser.parse_args(argv)

    if args.num_shards <= 0 or args.objects_per_shard <= 0 or args.ingest_workers <= 0:
        parser.error(
            "--num-shards, --objects-per-shard, and --ingest-workers must be positive"
        )
    if args.pixel_threshold <= 0:
        parser.error("--pixel-threshold must be positive")
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        print("ERROR: invalid shard index", file=sys.stderr)
        return 2
    if args.only_ingest and args.skip_ingest:
        print(
            "ERROR: --only-ingest and --skip-ingest are mutually exclusive",
            file=sys.stderr,
        )
        return 2
    Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)
    reject_planes = tuple(
        name.strip().upper()
        for name in args.reject_mask_planes.split(",")
        if name.strip()
    )

    try:
        ensure_build_contract(
            args.scratch_dir,
            args.catalog,
            args.manifest,
            args.objects_per_shard,
            reject_planes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if not args.only_ingest:
        try:
            catalog = read_catalog(args.catalog)
            columns = validate_catalog(catalog)
            manifest = load_manifest(args.manifest, args.verify_checksums)
            groups = grouped_rows(catalog, columns)[args.shard_idx :: args.num_shards]
            total = 0
            for index, rows in enumerate(groups, 1):
                expected, written = process_patch(
                    catalog,
                    rows,
                    columns,
                    manifest,
                    args.scratch_dir,
                    args.objects_per_shard,
                    reject_planes,
                    args.require_all_bands,
                )
                if expected != written:
                    raise RuntimeError(f"wrote {written}/{expected} rows for one patch")
                total += written
                print(
                    f"[shard {args.shard_idx}] [{index}/{len(groups)}] rows={written} total={total}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    if args.skip_ingest:
        return 0

    try:
        output = write_hats_from_parquet_dir(
            args.scratch_dir,
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=args.pixel_threshold,
            n_workers=args.ingest_workers,
            debug=False,
        )
        print(f"Done: {output}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
