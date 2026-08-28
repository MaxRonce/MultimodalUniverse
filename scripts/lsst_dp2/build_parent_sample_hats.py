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
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from filelock import FileLock

from mmu.hats_configs import MMU_V2_HATS_ROOT
from mmu.hats_import import write_hats_from_parquet_dir
from scripts.lsst_dp2.common import (
    BANDS,
    CATALOG_NAME,
    DEFAULT_REJECT_MASK_PLANES,
    IMAGE_SIZE,
    PIXEL_SCALE_ARCSEC,
    SKYMAP,
    mask_plane_mapping,
    native_array,
    read_catalog,
    sha256_file,
    validate_catalog,
)

PHOTOMETRY_SUFFIXES = ("psfFlux", "psfFluxErr", "cModelFlux", "cModelFluxErr")
PSF_SIZE = 35
OUTPUT_SCHEMA_VERSION = 3
CORE_REQUIRED_MASK_PLANES = ("SAT", "NO_DATA")
MASK_PLANE_ALIASES = {
    "SAT": ("SAT", "SATURATED"),
    "INTRP": ("INTRP", "INTERPOLATED"),
    "CR": ("CR", "COSMIC_RAY"),
    "EDGE": ("EDGE", "DETECTION_EDGE"),
}


def _find_hdu(hdul: fits.HDUList, names: set[str]):
    for hdu in hdul:
        if str(hdu.header.get("EXTNAME", "")).upper() in names:
            return hdu
    raise ValueError(f"missing FITS extension in {sorted(names)}")


def _read_archive_json(hdul: fits.HDUList) -> dict:
    json_hdu = _find_hdu(hdul, {"JSON"})
    return json.loads(bytes(json_hdu.data["JSON"][0]))


def clean_mask_from_bits(
    mask_bits: np.ndarray,
    mapping: dict[str, int],
    reject_planes: tuple[str, ...] = DEFAULT_REJECT_MASK_PLANES,
) -> np.ndarray:
    def resolve(name: str) -> str | None:
        return next(
            (candidate for candidate in MASK_PLANE_ALIASES.get(name, (name,))
             if candidate in mapping),
            None,
        )

    missing_core = [name for name in CORE_REQUIRED_MASK_PLANES if resolve(name) is None]
    if missing_core:
        raise ValueError(f"mask metadata lacks required planes: {missing_core}")
    clean = np.ones(mask_bits.shape, dtype=bool)
    for name in reject_planes:
        native_name = resolve(name)
        if native_name is not None:
            clean &= (mask_bits & (1 << mapping[native_name])) == 0
    return clean


@contextlib.contextmanager
def open_maskedimage(path: str):
    """Open one mirrored SIA product and expose pixels and the cell PSF model."""
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        image_hdu = _find_hdu(hdul, {"IMAGE", "SCI", "SCIENCE"})
        mask_hdu = _find_hdu(hdul, {"MASK"})
        variance_hdu = _find_hdu(hdul, {"VARIANCE", "VAR"})
        psf_hdu = _find_hdu(hdul, {"PSF"})
        archive = _read_archive_json(hdul)
        psf_metadata = archive.get("psf", {})
        grid = psf_metadata.get("bounds", {}).get("grid", {})
        grid_bbox = grid.get("bbox", {})
        cell_shape = tuple(grid.get("cell_shape", ()))
        image_yx0 = tuple(archive.get("image", {}).get("yx0", ()))
        psf_array = native_array(np.asarray(psf_hdu.data, dtype=np.float32))
        if psf_array.ndim != 4 or psf_array.shape[-2:] != (PSF_SIZE, PSF_SIZE):
            raise ValueError(f"unexpected cell PSF shape in {path}: {psf_array.shape}")
        if len(cell_shape) != 2 or len(image_yx0) != 2:
            raise ValueError(f"incomplete cell PSF metadata in {path}")
        mapping = mask_plane_mapping(hdul[0].header, image_hdu.header, mask_hdu.header)
        yield {
            "image": image_hdu.data,
            "variance": variance_hdu.data,
            "mask_bits": mask_hdu.data,
            "wcs": WCS(image_hdu.header),
            "mask_mapping": mapping,
            "psf_array": psf_array,
            "psf_grid_start_yx": (
                int(grid_bbox["y"]["start"]), int(grid_bbox["x"]["start"])
            ),
            "psf_cell_shape_yx": (int(cell_shape[0]), int(cell_shape[1])),
            "image_yx0": (int(image_yx0[0]), int(image_yx0[1])),
        }


def psf_kernel_at(coadd: dict, ra: float, dec: float) -> tuple[np.ndarray, bool]:
    """Return the local normalized PSF kernel and whether that cell is valid."""
    x, y = coadd["wcs"].world_to_pixel_values(ra, dec)
    absolute_y = y + coadd["image_yx0"][0]
    absolute_x = x + coadd["image_yx0"][1]
    grid_y0, grid_x0 = coadd["psf_grid_start_yx"]
    cell_y, cell_x = coadd["psf_cell_shape_yx"]
    iy = int(np.floor((absolute_y - grid_y0) / cell_y))
    ix = int(np.floor((absolute_x - grid_x0) / cell_x))
    psf_array = coadd["psf_array"]
    if not (0 <= iy < psf_array.shape[0] and 0 <= ix < psf_array.shape[1]):
        raise ValueError(f"source is outside PSF grid: cell={(iy, ix)}")
    kernel = np.asarray(psf_array[iy, ix], dtype=np.float32).copy()
    if not np.isfinite(kernel).all():
        return np.zeros((PSF_SIZE, PSF_SIZE), dtype=np.float32), False
    total = float(np.sum(kernel, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        return np.zeros((PSF_SIZE, PSF_SIZE), dtype=np.float32), False
    kernel /= total
    return kernel, True


def make_stamp(
    coadd: dict,
    ra: float,
    dec: float,
    reject_planes: tuple[str, ...] = DEFAULT_REJECT_MASK_PLANES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract flux, ivar, clean mask, and raw mask bits for one source."""
    position = SkyCoord(ra=ra, dec=dec, unit="deg")
    shape = (IMAGE_SIZE, IMAGE_SIZE)
    image = Cutout2D(
        coadd["image"], position, shape, wcs=coadd["wcs"],
        mode="partial", fill_value=np.nan,
    ).data
    variance = Cutout2D(
        coadd["variance"], position, shape, wcs=coadd["wcs"],
        mode="partial", fill_value=np.nan,
    ).data
    bits = Cutout2D(
        coadd["mask_bits"], position, shape, wcs=coadd["wcs"],
        mode="partial", fill_value=0,
    ).data
    coverage = Cutout2D(
        np.ones(coadd["image"].shape, dtype=np.uint8), position, shape,
        wcs=coadd["wcs"], mode="partial", fill_value=0,
    ).data.astype(bool)

    bits = native_array(np.asarray(bits, dtype=np.int32))
    clean = coverage & clean_mask_from_bits(bits, coadd["mask_mapping"], reject_planes)
    finite = np.isfinite(image) & np.isfinite(variance) & (variance > 0)
    valid = clean & finite
    flux = np.nan_to_num(
        native_array(image).astype(np.float32),
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    ivar = np.zeros(shape, dtype=np.float32)
    np.divide(1.0, variance, out=ivar, where=valid)
    return flux, ivar, valid, bits


def _finite_float(value, default: float = float("nan")) -> float:
    try:
        if np.ma.is_masked(value):
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _find_column_casefold(catalog: Table, target: str) -> str | None:
    lookup = {name.casefold(): name for name in catalog.colnames}
    return lookup.get(target.casefold())


def catalog_psf_fwhm(catalog: Table, row, band: str) -> float | None:
    """Compute catalog PSF FWHM from determinant-radius moments, in arcsec."""
    columns = [
        _find_column_casefold(catalog, f"{band}_ixxPSF"),
        _find_column_casefold(catalog, f"{band}_iyyPSF"),
        _find_column_casefold(catalog, f"{band}_ixyPSF"),
    ]
    if any(column is None for column in columns):
        return None
    ixx, iyy, ixy = (_finite_float(row[column]) for column in columns)
    determinant = ixx * iyy - ixy * ixy
    if not np.isfinite(determinant) or determinant <= 0:
        return None
    return float(2.354820045 * determinant ** 0.25 * PIXEL_SCALE_ARCSEC)


def patch_psf_fwhm(catalog: Table, rows: Table) -> dict[str, float | None]:
    """Return robust per-band PSF fallbacks from valid Object moments in a patch."""
    result = {}
    for band in BANDS:
        values = [catalog_psf_fwhm(catalog, row, band) for row in rows]
        valid = [value for value in values if value is not None and value > 0]
        result[band] = float(np.median(valid)) if valid else None
    return result


def load_manifest(path: str, verify_checksums: bool = False) -> dict[tuple[int, int, str], dict]:
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


def _ndarray_to_nested_array(array: np.ndarray, value_type: pa.DataType) -> pa.Array:
    arr = np.ascontiguousarray(array)
    nested: pa.Array = pa.array(arr.reshape(-1), type=value_type)
    for dim in reversed(arr.shape):
        offsets = np.arange(0, len(nested) + 1, dim, dtype=np.int32)
        nested = pa.ListArray.from_arrays(offsets, nested)
    return nested


def _nested_column(arrays: list[np.ndarray], value_type: pa.DataType) -> pa.Array:
    return pa.concat_arrays([_ndarray_to_nested_array(array, value_type) for array in arrays])


def build_image_struct(records: list[dict]) -> pa.StructArray:
    return pa.StructArray.from_arrays(
        [
            pa.array([list(BANDS)] * len(records), type=pa.list_(pa.string())),
            _nested_column([r["flux"] for r in records], pa.float32()),
            _nested_column([r["ivar"] for r in records], pa.float32()),
            _nested_column([r["mask"] for r in records], pa.bool_()),
            _nested_column([r["mask_bits"] for r in records], pa.int32()),
            _nested_column([r["psf_image"] for r in records], pa.float32()),
            _nested_column([r["psf_image_valid"] for r in records], pa.bool_()),
            _nested_column([r["psf_fwhm"] for r in records], pa.float32()),
            _nested_column([r["scale"] for r in records], pa.float32()),
            _nested_column([r["band_present"] for r in records], pa.bool_()),
            pa.array([r["psf_source"] for r in records], type=pa.list_(pa.string())),
            pa.array([r["dataset_id"] for r in records], type=pa.list_(pa.string())),
            pa.array([r["sha256"] for r in records], type=pa.list_(pa.string())),
            pa.array([r["mask_plane_map"] for r in records], type=pa.list_(pa.string())),
            pa.array([["nJy"] * len(BANDS)] * len(records), type=pa.list_(pa.string())),
            pa.array([["nJy^-2"] * len(BANDS)] * len(records), type=pa.list_(pa.string())),
        ],
        names=[
            "band", "flux", "ivar", "mask", "mask_bits", "psf_image",
            "psf_image_valid", "psf_fwhm", "scale",
            "band_present", "psf_source", "dataset_id", "sha256", "mask_plane_map",
            "flux_unit", "ivar_unit",
        ],
    )


def build_table(records: list[dict]) -> pa.Table:
    columns: dict[str, pa.Array] = {
        "ra": pa.array([r["ra"] for r in records], type=pa.float64()),
        "dec": pa.array([r["dec"] for r in records], type=pa.float64()),
        "object_id": pa.array([r["object_id"] for r in records], type=pa.string()),
        "tract": pa.array([r["tract"] for r in records], type=pa.int32()),
        "patch": pa.array([r["patch"] for r in records], type=pa.int32()),
        "ref_band": pa.array([r["ref_band"] for r in records], type=pa.string()),
        "ref_extendedness": pa.array(
            [r["ref_extendedness"] for r in records], type=pa.float32()
        ),
        "detect_is_isolated": pa.array(
            [r["detect_is_isolated"] for r in records], type=pa.bool_()
        ),
        "image": build_image_struct(records),
        "dp2_collection": pa.array(["dp2"] * len(records), type=pa.string()),
        "skymap": pa.array([SKYMAP] * len(records), type=pa.string()),
    }
    for band in BANDS:
        for suffix in PHOTOMETRY_SUFFIXES:
            name = f"{band}_{suffix}"
            columns[name] = pa.array([r["photometry"][name] for r in records], type=pa.float32())
    return pa.table(columns)


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
) -> list[dict]:
    records = []
    n = len(rows)
    patch_psf = patch_psf_fwhm(catalog, rows)
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
            flux[band_index], ivar[band_index], mask[band_index], mask_bits[band_index] = stamp
            band_present[band_index] = True
            dataset_ids[band_index] = info.get("dataset_id") or ""
            checksums[band_index] = info.get("sha256") or ""
            plane_maps[band_index] = json.dumps(coadd["mask_mapping"], sort_keys=True)
            psf_image[band_index], psf_image_valid[band_index] = psf_kernel_at(
                coadd, ra, dec
            )
            catalog_psf = catalog_psf_fwhm(catalog, row, band)
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
                column = _find_column_casefold(catalog, name)
                photometry[name] = _finite_float(row[column]) if column else float("nan")
        records.append({
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
        })
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
    with contextlib.ExitStack() as stack:
        coadds = {
            band: stack.enter_context(open_maskedimage(info["output_path"])) if info else None
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
                catalog, rows[start:stop], columns, coadds, infos, reject_planes
            )
            table = build_table(records)
            temp = path.with_suffix(".parquet.tmp")
            pq.write_table(table, temp, compression="zstd")
            os.replace(temp, path)
            written += table.num_rows
    return len(rows), written


def grouped_rows(catalog: Table, columns: dict[str, str | None]) -> list[Table]:
    order = np.lexsort((
        np.asarray(catalog[columns["patch"]], dtype=np.int64),
        np.asarray(catalog[columns["tract"]], dtype=np.int64),
    ))
    sorted_catalog = catalog[order]
    tracts = np.asarray(sorted_catalog[columns["tract"]], dtype=np.int64)
    patches = np.asarray(sorted_catalog[columns["patch"]], dtype=np.int64)
    starts = np.r_[0, np.flatnonzero((tracts[1:] != tracts[:-1]) | (patches[1:] != patches[:-1])) + 1]
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
    }
    path = Path(scratch_dir) / "build_contract.json"
    with FileLock(str(path) + ".lock"):
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != contract:
                raise RuntimeError(
                    f"scratch build contract differs from requested build: {path}"
                )
            return
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--scratch-dir", required=True)
    parser.add_argument(
        "--output-root", default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME)
    )
    parser.add_argument("--objects-per-shard", type=int, default=64)
    parser.add_argument("--pixel-threshold", type=int, default=8192)
    parser.add_argument("--ingest-workers", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-idx", type=int, default=0)
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--only-ingest", action="store_true")
    parser.add_argument("--require-all-bands", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument(
        "--reject-mask-planes",
        default=",".join(DEFAULT_REJECT_MASK_PLANES),
    )
    args = parser.parse_args(argv)

    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        print("ERROR: invalid shard index", file=sys.stderr)
        return 2
    if args.only_ingest and args.skip_ingest:
        print("ERROR: --only-ingest and --skip-ingest are mutually exclusive", file=sys.stderr)
        return 2
    Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)
    reject_planes = tuple(
        name.strip().upper() for name in args.reject_mask_planes.split(",") if name.strip()
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
            groups = grouped_rows(catalog, columns)[args.shard_idx::args.num_shards]
            total = 0
            for index, rows in enumerate(groups, 1):
                expected, written = process_patch(
                    catalog, rows, columns, manifest, args.scratch_dir,
                    args.objects_per_shard, reject_planes, args.require_all_bands,
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
