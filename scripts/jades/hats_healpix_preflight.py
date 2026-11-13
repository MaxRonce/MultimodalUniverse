"""JADES HATS sizing audit using the exact checkpointed source population.

Read the 484k coordinates, compute the *actual* densest pixel at candidate
HEALPix orders, and estimate the dense histogram footprint used by hats-import.
Run this before committing to terabytes of splitting/reducing work.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq
from hats.pixel_math import healpix_shim as hp


COLUMNS = ["field", "object_id", "ra", "dec"]
FIELDS = ("GOODS-S", "GOODS-N")
CACHE_VERSION = 1


def snapshot(scratch: Path, workers: int) -> dict:
    """Validate completed checkpoints and inventory files without opening Parquet.

    File size/mtime and checkpoint contents invalidate the cache on normal file
    replacement or extraction changes. This is not a content-integrity audit;
    use --refresh-cache after storage repair preserving both size and mtime.
    """
    checkpoints = sorted(scratch.rglob("*.progress.json"))
    if not checkpoints:
        raise ValueError("No extraction checkpoints found in --scratch-dir")
    states, expected, groups = {}, {}, set()
    shard_totals = set()
    field_counts = dict.fromkeys(FIELDS, 0)
    for checkpoint in checkpoints:
        field = checkpoint.name.removesuffix(".progress.json")
        if field not in FIELDS:
            raise ValueError(f"Unknown checkpoint field: {checkpoint}")
        state = json.loads(checkpoint.read_text())
        for key in ("count", "next_index", "next_shard"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f"Invalid {key} in {checkpoint}")
        if state["next_index"] < state["count"] or not state.get("fingerprint"):
            raise ValueError(f"Invalid extraction state in {checkpoint}")
        total, index = state.get("num_shards", 1), state.get("shard_index", 0)
        if (type(total) is not int or type(index) is not int
                or total < 1 or not 0 <= index < total):
            raise ValueError(f"Invalid shard identity in {checkpoint}")
        directory = scratch if total == 1 else scratch / f"shard_{index:04d}_of_{total:04d}"
        if checkpoint.parent != directory or (index, field) in groups:
            raise ValueError(f"Wrong or repeated shard directory: {checkpoint}")
        done = checkpoint.with_name(f"{field}.done")
        if not done.is_file() or done.read_text().strip() != str(state["count"]):
            raise ValueError(f"Incomplete or inconsistent extraction: {checkpoint}")
        groups.add((index, field))
        shard_totals.add(total)
        key = checkpoint.relative_to(scratch).as_posix()
        states[key] = state
        field_counts[field] += state["count"]
        for part in range(state["next_shard"]):
            path = directory / f"{field}_{part:05d}.parquet"
            expected[path.relative_to(scratch).as_posix()] = key
    if len(shard_totals) != 1 or groups != {
            (i, field) for i in range(next(iter(shard_totals))) for field in FIELDS}:
        raise ValueError("Missing field/shard checkpoints or mixed shard configurations")
    found = {p.relative_to(scratch).as_posix() for p in scratch.rglob("*.parquet")}
    if found != set(expected):
        raise ValueError(f"Missing/extra Parquet files: {len(set(expected) - found)} missing, "
                         f"{len(found - set(expected))} extra")

    def signature(name):
        stat = (scratch / name).stat()
        return [name, stat.st_size, stat.st_mtime_ns, expected[name]]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(signature, sorted(expected)))
    return {"version": CACHE_VERSION, "scratch": str(scratch.resolve()),
            "checkpoints": states, "files": files, "field_counts": field_counts}


def validate_coordinates(columns: dict, source: dict, expected_rows: int) -> None:
    n = len(columns["ra"])
    if n != expected_rows or any(len(v) != n for v in columns.values()):
        raise ValueError(f"Coordinate row count {n} differs from expected {expected_rows}")
    ra, dec = columns["ra"], columns["dec"]
    if (not np.isfinite(ra).all() or not np.isfinite(dec).all()
            or ((ra < 0) | (ra >= 360)).any() or ((dec < -90) | (dec > 90)).any()):
        raise ValueError("Invalid or non-finite RA/Dec coordinates")
    ids = columns["object_id"]
    if len(np.unique(ids)) != n:
        raise ValueError("Duplicate extracted object_id")
    for field, count in source["field_counts"].items():
        mask = columns["field"] == field
        if int(mask.sum()) != count:
            raise ValueError(f"{field}: coordinate count differs from checkpoints")
        if any(not re.fullmatch(re.escape(field) + r":-?\d+", value) for value in ids[mask]):
            raise ValueError(f"{field}: inconsistent object_id prefix or source ID")
    if not np.isin(columns["field"], FIELDS).all():
        raise ValueError("Unknown coordinate field")


def read_coordinates(path: Path) -> dict:
    # Dataset scanners pre-buffer/coalesce ranges by default. Open one file at a
    # time and explicitly disable that behaviour for image-heavy row groups.
    # Also avoid dataset partition discovery, schema inference and nested pools.
    with pq.ParquetFile(path, pre_buffer=False, buffer_size=0) as parquet:
        table = parquet.read(columns=COLUMNS, use_threads=False)
        if table.column_names != COLUMNS or any(table[c].null_count for c in COLUMNS):
            raise ValueError(f"{path}: missing/null scalar coordinate columns")
        return {c: np.asarray(table[c].to_pylist(), dtype=str) if c in COLUMNS[:2]
                else table[c].to_numpy().astype(np.float64) for c in COLUMNS}


def load_coordinates(scratch: Path, cache: Path, workers: int, expected_rows: int,
                     refresh: bool = False) -> dict:
    print("Checking completed extraction and file size/mtime inventory...", flush=True)
    source = snapshot(scratch, workers)
    if sum(source["field_counts"].values()) != expected_rows:
        raise ValueError(f"Checkpoint counts {source['field_counts']} differ from {expected_rows}")
    print(f"Checkpoint field counts: {source['field_counts']}", flush=True)
    if cache.is_file() and not refresh:
        with np.load(cache, allow_pickle=False) as saved:
            if json.loads(str(saved["source"])) == source:
                columns = {c: saved[c] for c in COLUMNS}
                validate_coordinates(columns, source, expected_rows)
                print(f"Using validated coordinate cache: {cache} (no Parquet opened)", flush=True)
                return columns
        print("Coordinate cache is stale; rebuilding", flush=True)
    started = time.monotonic()
    print(f"Reading four scalar columns from {len(source['files'])} Parquet files; "
          f"workers={workers}, pre_buffer=False", flush=True)
    pieces = {c: [] for c in COLUMNS}
    counts = dict.fromkeys(source["checkpoints"], 0)
    # Limit pending work as well as worker count: don't queue 7,600 open/read jobs.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(source["files"]), workers):
            batch = source["files"][start:start + workers]
            paths = [scratch / entry[0] for entry in batch]
            for entry, columns in zip(batch, pool.map(read_coordinates, paths)):
                field = Path(entry[3]).name.removesuffix(".progress.json")
                if not (columns["field"] == field).all():
                    raise ValueError(f"{entry[0]}: field differs from checkpoint")
                counts[entry[3]] += len(columns["ra"])
                for c in COLUMNS:
                    pieces[c].append(columns[c])
            completed = start + len(batch)
            if start == 0 or completed // 100 != start // 100 or completed == len(source["files"]):
                print(f"  {completed}/{len(source['files'])} files, "
                      f"{sum(counts.values()):,} rows, {time.monotonic() - started:.1f}s", flush=True)
    for key, state in source["checkpoints"].items():
        if counts[key] != state["count"]:
            raise ValueError(f"{key}: Parquet rows differ from checkpoint")
    columns = {c: np.concatenate(pieces[c]) for c in COLUMNS}
    validate_coordinates(columns, source, expected_rows)
    if snapshot(scratch, workers) != source:
        raise ValueError("Extraction changed during coordinate scan; cache was not written")
    cache.parent.mkdir(parents=True, exist_ok=True)
    # An interruption cannot leave a published partial cache. Unique temp names
    # also allow concurrent read-only preflights without clobbering each other.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache.parent, suffix=".npz.tmp", delete=False) as out:
            temporary = Path(out.name)
            np.savez_compressed(out, source=json.dumps(source, sort_keys=True), **columns)
        os.replace(temporary, cache)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved exact selected coordinates: {cache}", flush=True)
    return columns


def validate_catalogs(columns: dict, raw_root: Path, manifest: Path | None) -> None:
    """Match selected IDs to SIZE, never substitute the whole FITS population."""
    from astropy.io import fits
    from scripts.jades.audit_selection import resolve_catalogs

    entries = dict(resolve_catalogs(raw_root, manifest))
    if set(entries) != set(FIELDS):
        raise ValueError("Expected GOODS-S and GOODS-N catalogs")
    for field, path in entries.items():
        mask = columns["field"] == field
        selected = np.array([int(v.rsplit(":", 1)[1]) for v in columns["object_id"][mask]],
                            dtype=np.int64)
        with fits.open(path, memmap=True) as hdus:
            size = hdus["SIZE"].data
            ids = np.asarray(size["ID"], dtype=np.int64)
            order = np.argsort(ids)
            ordered = ids[order]
            if len(np.unique(ids)) != len(ids):
                raise ValueError(f"{field}: duplicate catalog IDs")
            positions = np.searchsorted(ordered, selected)
            if (positions >= len(ids)).any() or not np.array_equal(ordered[positions], selected):
                raise ValueError(f"{field}: extracted ID absent from catalog")
            indices = order[positions]
            for col in ("ra", "dec"):
                if not np.array_equal(columns[col][mask], np.asarray(size[col.upper()])[indices]):
                    raise ValueError(f"{field}: extracted {col} differs from catalog SIZE")
            print(f"{field}: {len(selected):,} exact selected IDs/coordinates verified; "
                  f"{len(ids) - len(selected):,} catalog rows excluded", flush=True)


def pixel_occupancies(ra: np.ndarray, dec: np.ndarray, order: int) -> np.ndarray:
    """Count sources per occupied NESTED HEALPix pixel without a sky-wide array."""
    pixels = hp.radec2pix(order, ra, dec)
    _, counts = np.unique(pixels, return_counts=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    parser.add_argument("--orders", nargs="+", type=int, default=[10, 11, 12, 13])
    parser.add_argument("--highest-order", type=int, default=13)
    parser.add_argument("--pixel-threshold", type=int, default=1024)
    parser.add_argument("--expected-rows", type=int, default=484219)
    parser.add_argument("--io-workers", type=int, default=8,
                        help="Bounded parallel scalar reads/stat calls (default: 8)")
    parser.add_argument("--coordinate-cache", type=Path,
                        help="Default: <scratch-dir sibling>.preflight-coordinates.npz")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--raw-root", type=Path,
                        help="Also verify selected IDs and exact coordinates against FITS SIZE")
    parser.add_argument("--manifest", type=Path, help="Catalog manifest for --raw-root")
    args = parser.parse_args()
    if args.io_workers < 1 or args.expected_rows < 1 or args.pixel_threshold < 1:
        parser.error("--io-workers, --expected-rows and --pixel-threshold must be positive")
    if any(not 0 <= order <= 29 for order in args.orders + [args.highest_order]):
        parser.error("HEALPix orders must be between 0 and 29")
    if args.manifest and not args.raw_root:
        parser.error("--manifest requires --raw-root")
    scratch = args.scratch_dir.resolve()
    cache = args.coordinate_cache or scratch.with_name(scratch.name + ".preflight-coordinates.npz")
    if cache.suffix.lower() != ".npz":
        parser.error("--coordinate-cache must have an .npz extension")
    coordinates = load_coordinates(scratch, cache, args.io_workers, args.expected_rows,
                                   args.refresh_cache)
    if args.raw_root:
        validate_catalogs(coordinates, args.raw_root, args.manifest)
    ra, dec = coordinates["ra"], coordinates["dec"]
    selection_hash = hashlib.sha256("\n".join(sorted(coordinates["object_id"])).encode()).hexdigest()
    print(f"Rows: {len(ra):,} (expected: {args.expected_rows:,}); "
          f"selected ID SHA256: {selection_hash}", flush=True)

    print("Order | Histogram GiB | Occupied pixels | Median rows | P99 rows | Max rows")
    for order in sorted(set(args.orders + [args.highest_order])):
        counts = pixel_occupancies(ra, dec, order)
        histogram_gib = (12 * (4 ** order) * 8) / (1024 ** 3)
        print(f"{order:>5} | {histogram_gib:>13.2f} | {len(counts):>15} | "
              f"{np.median(counts):>11.0f} | {np.percentile(counts, 99):>8.0f} | "
              f"{counts.max():>8}", flush=True)
        if order == args.highest_order and counts.max() > args.pixel_threshold:
            raise ValueError(
                f"Order {order} has a pixel with {counts.max()} galaxies, exceeding "
                f"--pixel-threshold {args.pixel_threshold}. Raise the threshold, or "
                "use a finer order with sufficient worker RAM."
            )

    print("PASS: row count, coordinates, and maximum pixel occupancy", flush=True)


if __name__ == "__main__":
    main()
