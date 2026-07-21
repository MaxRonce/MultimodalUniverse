"""Convert ZTF DR23 light curves from IPAC HATS to MMU-compatible HATS.

The public ZTF DR23 light-curve product is already HATS, but its schema is the
IPAC/ZTF schema:

    objectid, filterid, fieldid, rcid, objra, objdec, nepochs,
    lightcurve: struct<hmjd, mag, magerr, clrcoeff, catflags>

where the light-curve fields are nested lists. This script preserves the
native row unit of that product: one output row is one ZTF light-curve-series
row, usually one ``objectid/filterid`` series. It deliberately does not group
rows by ``objectid`` because ``objectid`` has not been validated as a
cross-band astrophysical-object identifier in this HATS product.

No observations or series are filtered out. Quality-cut information is written
as per-epoch masks and per-series counters/flags so training jobs can select a
view later without losing raw/provenance data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mmu.hats_configs import MMU_V2_HATS_ROOT


CATALOG_NAME = "ztf"
QUALITY_POLICY_VERSION = "ztf_aion_relaxed_ge30_v1"
DEFAULT_ZTF_INPUT = (
    "/lustre/fsn1/projects/rech/jrx/urx63nr/"
    "ztf_dr23_lc_hats_full/ztf_dr23_lc-hats"
)
DEFAULT_LEGACY_SCRATCH_ROOT = "/mnt/ceph/users/polymathic/MultimodalUniverse_v2_hats_scratch"


def default_scratch_dir(catalog_name: str) -> str:
    """Return the legacy hats-import scratch path without importing hats-import."""
    return os.path.join(DEFAULT_LEGACY_SCRATCH_ROOT, f"{catalog_name}_{os.getpid()}")

BAND_LABELS: dict[int, str] = {
    1: "ztf_g",
    2: "ztf_r",
    3: "ztf_i",
}
BAND_SUFFIXES: dict[int, str] = {
    1: "ztf_g",
    2: "ztf_r",
    3: "ztf_i",
}
RELAXED_THRESHOLDS = (3, 5, 10, 20, 30, 50)
STRICT_THRESHOLDS = RELAXED_THRESHOLDS


@dataclass(frozen=True)
class ConvertConfig:
    object_col: str = "objectid"
    filter_col: str = "filterid"
    field_col: str = "fieldid"
    rcid_col: str = "rcid"
    ra_col: str = "objra"
    dec_col: str = "objdec"
    nepochs_col: str = "nepochs"
    healpix_col: str = "_healpix_29"
    lightcurve_col: str = "lightcurve"
    time_col: str = "hmjd"
    mag_col: str = "mag"
    magerr_col: str = "magerr"
    clrcoeff_col: str = "clrcoeff"
    catflags_col: str = "catflags"
    bad_epoch_bit: int = 32768
    bright_mag_threshold: float = 15.5
    include_quality_flags: bool = True
    include_band_list: bool = True
    aion_relaxed_threshold: int = 30
    arrow_threads: int = 1


def quality_policy_version(relaxed_threshold: int) -> str:
    return f"ztf_aion_relaxed_ge{int(relaxed_threshold)}_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        "--input_dir",
        "--raw-root",
        dest="input_dir",
        default=os.environ.get("ZTF_DR23_HATS_ROOT", DEFAULT_ZTF_INPUT),
        help="ZTF DR23 HATS catalog root or dataset/ directory.",
    )
    parser.add_argument(
        "--output-root",
        "--out-dir",
        "--out_dir",
        dest="output_root",
        default=os.path.join(MMU_V2_HATS_ROOT, CATALOG_NAME),
        help="Output root under which ztf/ztf will be written.",
    )
    parser.add_argument(
        "--catalog",
        choices=["auto", "primary", "margin", "all"],
        default="auto",
        help="Which HATS catalog to convert when input-dir is a multi-catalog root.",
    )
    parser.add_argument("--glob", default="**/*.parquet")
    parser.add_argument("--max-files", "--max_files", dest="max_files", type=int)
    parser.add_argument("--file-sample", "--file_sample", dest="file_sample",
                        choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size",
                        type=int, default=10_000)
    parser.add_argument("--num-processes", "--workers", dest="num_processes",
                        type=int, default=_default_workers())
    parser.add_argument("--arrow-threads", "--arrow_threads", dest="arrow_threads",
                        type=int, default=1)
    parser.add_argument("--scratch-dir", "--scratch_dir", dest="scratch_dir",
                        default=None)
    parser.add_argument("--pixel-threshold", "--pixel_threshold",
                        dest="pixel_threshold", type=int, default=100_000)
    parser.add_argument("--ingest-workers", "--ingest_workers",
                        dest="ingest_workers", type=int, default=8)
    parser.add_argument("--ingest-chunksize", "--ingest_chunksize",
                        dest="ingest_chunksize", type=int, default=50_000)
    parser.add_argument("--num-shards", "--num_shards", dest="num_shards",
                        type=int, default=1)
    parser.add_argument("--shard-idx", "--shard_idx", dest="shard_idx",
                        type=int, default=0)
    parser.add_argument("--skip-ingest", "--skip_ingest", dest="skip_ingest",
                        action="store_true")
    parser.add_argument("--only-ingest", "--only_ingest", dest="only_ingest",
                        action="store_true")
    parser.add_argument("--direct-hats", "--direct_hats", "--preserve-hats-partitions",
                        "--preserve_hats_partitions", dest="direct_hats",
                        action="store_true",
                        help=(
                            "Write converted files directly into the same HATS "
                            "partition paths as the input. This avoids hats-import "
                            "spatial split/reduce and is the production path for "
                            "distributed ZTF conversion."
                        ))
    parser.add_argument("--finalize-direct-hats", "--finalize_direct_hats",
                        dest="finalize_direct_hats", action="store_true",
                        help=(
                            "Finalize metadata for a --direct-hats output after "
                            "all shards have completed."
                        ))
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed per-input converted parquet files.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite converted parquet files if they exist.")
    parser.add_argument("--no-quality-flags", dest="include_quality_flags",
                        action="store_false",
                        help="Do not write per-epoch quality masks; counters remain.")
    parser.add_argument("--no-band-list", dest="include_band_list",
                        action="store_false",
                        help="Do not repeat band labels inside lightcurve.band.")
    parser.add_argument("--aion-relaxed-threshold", type=int, default=30,
                        help="Relaxed-epoch threshold used for aion_training_candidate.")
    parser.add_argument("--bad-epoch-bit", type=int, default=32768)
    parser.add_argument("--bright-mag-threshold", type=float, default=15.5)
    parser.add_argument("--ra-center", type=float, default=None,
                        help="Accepted for Snakemake compatibility; ignored.")
    parser.add_argument("--dec-center", type=float, default=None,
                        help="Accepted for Snakemake compatibility; ignored.")
    parser.add_argument("--radius", type=float, default=None,
                        help="Accepted for Snakemake compatibility; ignored.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.skip_ingest and args.only_ingest:
        print("ERROR: --skip-ingest and --only-ingest are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
        print(f"ERROR: shard-idx {args.shard_idx} outside [0, {args.num_shards})",
              file=sys.stderr)
        return 2
    if args.aion_relaxed_threshold not in RELAXED_THRESHOLDS:
        print(
            f"ERROR: --aion-relaxed-threshold must be one of {RELAXED_THRESHOLDS}",
            file=sys.stderr,
        )
        return 2
    if args.ra_center is not None or args.dec_center is not None or args.radius is not None:
        print(
            "WARNING: ztf conversion ignores cone-cut arguments; convert the full "
            "raw/provenance-preserving catalog and filter training views later.",
            file=sys.stderr,
            flush=True,
        )

    direct_mode = bool(args.direct_hats or args.finalize_direct_hats)
    if direct_mode and args.only_ingest:
        print(
            "ERROR: --only-ingest is only valid for the hats-import path; "
            "use --finalize-direct-hats for direct HATS output.",
            file=sys.stderr,
        )
        return 2

    _set_thread_env(args.arrow_threads)
    sharded_mode = args.num_shards > 1 or args.skip_ingest or args.only_ingest
    if sharded_mode and args.scratch_dir is None and not direct_mode:
        print("ERROR: --scratch-dir is required in sharded mode", file=sys.stderr)
        return 2

    scratch = args.scratch_dir or (
        direct_manifest_dir(args.output_root) if direct_mode
        else default_scratch_dir(CATALOG_NAME)
    )
    os.makedirs(scratch, exist_ok=True)

    cfg = ConvertConfig(
        bad_epoch_bit=int(args.bad_epoch_bit),
        bright_mag_threshold=float(args.bright_mag_threshold),
        include_quality_flags=bool(args.include_quality_flags),
        include_band_list=bool(args.include_band_list),
        aion_relaxed_threshold=int(args.aion_relaxed_threshold),
        arrow_threads=max(1, int(args.arrow_threads)),
    )

    manifest: dict[str, Any] | None = None
    try:
        if args.finalize_direct_hats and not args.direct_hats:
            scan_dir, catalog_warning = resolve_scan_dir(Path(args.input_dir), args.catalog)
            if catalog_warning:
                print(f"WARNING: {catalog_warning}", file=sys.stderr, flush=True)
            expected_files = select_files(
                discover_parquet_files(scan_dir, args.glob),
                args.max_files,
                args.file_sample,
                args.seed,
            )
            catalog_dir = finalize_direct_hats_output(
                scan_dir=scan_dir,
                output_root=Path(args.output_root),
                manifest_dir=Path(scratch),
                glob_pattern=args.glob,
                expected_file_count=len(expected_files),
            )
            print(f"Done: {catalog_dir}")
            return 0

        if args.direct_hats:
            scan_dir, catalog_warning = resolve_scan_dir(Path(args.input_dir), args.catalog)
            files = discover_parquet_files(scan_dir, args.glob)
            files = select_files(files, args.max_files, args.file_sample, args.seed)
            if not files:
                print(
                    f"ERROR: no parquet files under {scan_dir} with glob {args.glob!r}",
                    file=sys.stderr,
                )
                return 1
            duplicate_outputs = duplicate_direct_hats_outputs(files, scan_dir)
            if duplicate_outputs:
                sample = "\n".join(f"  - {path}" for path in duplicate_outputs[:8])
                print(
                    "ERROR: multiple input parquet files map to the same direct "
                    "HATS output pixel. Direct mode requires one input file per "
                    f"HATS pixel. Examples:\n{sample}",
                    file=sys.stderr,
                )
                return 2
            indexed_files = list(enumerate(files))
            if args.num_shards > 1:
                indexed_files = indexed_files[args.shard_idx::args.num_shards]
            if not indexed_files:
                print(f"[shard {args.shard_idx}] no files assigned; nothing to do")
                return 0

            output_dataset = direct_dataset_dir(args.output_root)
            print(
                f"[shard {args.shard_idx}/{args.num_shards}] direct HATS conversion "
                f"of {len(indexed_files):,} file(s) from {scan_dir} to {output_dataset}",
                flush=True,
            )
            if catalog_warning:
                print(f"WARNING: {catalog_warning}", file=sys.stderr, flush=True)

            manifest = convert_files_to_direct_hats(
                indexed_files,
                scan_dir,
                output_dataset,
                cfg,
                batch_size=int(args.batch_size),
                num_processes=int(args.num_processes),
                resume=bool(args.resume),
                force=bool(args.force),
                shard_idx=int(args.shard_idx),
            )
            manifest.update(
                {
                    "input_dir": str(args.input_dir),
                    "scan_dir": str(scan_dir),
                    "output_dataset_dir": str(output_dataset),
                    "manifest_dir": str(scratch),
                    "catalog_warning": catalog_warning,
                    "config": asdict(cfg),
                    "num_shards": int(args.num_shards),
                    "shard_idx": int(args.shard_idx),
                    "mode": "direct_hats",
                }
            )
            manifest_path = Path(scratch) / f"manifest-direct-shard-{args.shard_idx:03d}.json"
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            print(
                f"[shard {args.shard_idx}] DIRECT BUILD DONE: "
                f"{manifest['files_written']:,} written, "
                f"{manifest['files_skipped']:,} skipped, "
                f"{manifest['rows']:,} series rows, "
                f"{manifest['observations']:,} observations",
                flush=True,
            )
            if args.finalize_direct_hats or (args.num_shards == 1 and not args.skip_ingest):
                catalog_dir = finalize_direct_hats_output(
                    scan_dir=scan_dir,
                    output_root=Path(args.output_root),
                    manifest_dir=Path(scratch),
                    glob_pattern=args.glob,
                    expected_file_count=len(files),
                )
                print(f"Done: {catalog_dir}")
            else:
                print(
                    "Direct HATS shards do not need hats-import. After all array "
                    "tasks finish, run once with --finalize-direct-hats.",
                    flush=True,
                )
            return 0

        if not args.only_ingest:
            scan_dir, catalog_warning = resolve_scan_dir(Path(args.input_dir), args.catalog)
            files = discover_parquet_files(scan_dir, args.glob)
            files = select_files(files, args.max_files, args.file_sample, args.seed)
            if not files:
                print(
                    f"ERROR: no parquet files under {scan_dir} with glob {args.glob!r}",
                    file=sys.stderr,
                )
                return 1
            indexed_files = list(enumerate(files))
            if args.num_shards > 1:
                indexed_files = indexed_files[args.shard_idx::args.num_shards]
            if not indexed_files:
                print(f"[shard {args.shard_idx}] no files assigned; nothing to do")
                return 0

            print(
                f"[shard {args.shard_idx}/{args.num_shards}] converting "
                f"{len(indexed_files):,} ZTF parquet file(s) from {scan_dir} to {scratch}",
                flush=True,
            )
            if catalog_warning:
                print(f"WARNING: {catalog_warning}", file=sys.stderr, flush=True)

            manifest = convert_files_to_parquet_dir(
                indexed_files,
                scratch,
                cfg,
                batch_size=int(args.batch_size),
                num_processes=int(args.num_processes),
                resume=bool(args.resume),
                force=bool(args.force),
                shard_idx=int(args.shard_idx),
            )
            manifest.update(
                {
                    "input_dir": str(args.input_dir),
                    "scan_dir": str(scan_dir),
                    "scratch_dir": str(scratch),
                    "catalog_warning": catalog_warning,
                    "config": asdict(cfg),
                    "num_shards": int(args.num_shards),
                    "shard_idx": int(args.shard_idx),
                }
            )
            manifest_path = Path(scratch) / f"manifest-shard-{args.shard_idx:03d}.json"
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            print(
                f"[shard {args.shard_idx}] BUILD DONE: "
                f"{manifest['files_written']:,} files, "
                f"{manifest['rows']:,} series rows, "
                f"{manifest['observations']:,} observations",
                flush=True,
            )

        if args.skip_ingest:
            return 0

        print(
            f"Ingesting {scratch} -> HATS catalog "
            f"(workers={args.ingest_workers}, chunksize={args.ingest_chunksize})",
            flush=True,
        )
        from mmu.hats_import import write_hats_from_parquet_dir

        catalog_dir = write_hats_from_parquet_dir(
            scratch,
            output_path=args.output_root,
            catalog_name=CATALOG_NAME,
            pixel_threshold=int(args.pixel_threshold),
            chunksize=int(args.ingest_chunksize),
            n_workers=int(args.ingest_workers),
            debug=False,
        )
        write_quality_policy(Path(catalog_dir), asdict(cfg))
        print(f"Done: {catalog_dir}")
    finally:
        if args.scratch_dir is None and not sharded_mode and not direct_mode:
            shutil.rmtree(scratch, ignore_errors=True)
    return 0


def _default_workers() -> int:
    value = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("OMP_NUM_THREADS")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass
    return max(1, min(os.cpu_count() or 1, 16))


def _set_thread_env(arrow_threads: int) -> None:
    threads = str(max(1, int(arrow_threads)))
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(key, "1")
    try:
        pa.set_cpu_count(int(threads))
    except Exception:
        pass


def resolve_scan_dir(input_dir: Path, catalog: str = "auto") -> tuple[Path, str | None]:
    root = input_dir.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if root.name == "dataset" or any(root.glob("Norder=*")):
        return root, _catalog_warning(root)
    if (root / "dataset").is_dir():
        return root / "dataset", _catalog_warning(root / "dataset")

    candidates = _candidate_dataset_dirs(root)
    if catalog == "all":
        return root, "Scanning all parquet files under input_dir; verify catalogs are not mixed."

    def kind(path: Path) -> str:
        text = str(path)
        if "margin" in text:
            return "margin"
        if "index" in text:
            return "index"
        return "primary"

    if catalog in {"primary", "margin"}:
        selected = [path for path in candidates if kind(path) == catalog]
        if len(selected) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {catalog} HATS dataset under {root}, "
                f"found {len(selected)}."
            )
        return selected[0], _catalog_warning(selected[0])

    primary = [path for path in candidates if kind(path) == "primary"]
    if len(primary) == 1:
        return primary[0], None
    non_index = [path for path in candidates if kind(path) != "index"]
    if len(non_index) == 1:
        return non_index[0], _catalog_warning(non_index[0])
    if candidates:
        sample = "\n".join(f"  - {path}" for path in candidates[:8])
        raise ValueError(
            "Multiple HATS dataset directories found; pass --catalog or a more "
            f"specific --input-dir:\n{sample}"
        )
    return root, "No HATS dataset directory was auto-detected; scanning input-dir directly."


def _candidate_dataset_dirs(root: Path, max_depth: int = 7) -> list[Path]:
    candidates: list[Path] = []
    stack = [(root, 0)]
    while stack:
        path, depth = stack.pop()
        dataset = path / "dataset"
        if dataset.is_dir():
            candidates.append(dataset)
            continue
        if depth >= max_depth:
            continue
        try:
            children = [child for child in path.iterdir() if child.is_dir()]
        except OSError:
            continue
        for child in children:
            if child.name.startswith(("Norder=", "Dir=")):
                continue
            stack.append((child, depth + 1))
    return sorted(candidates)


def _catalog_warning(path: Path) -> str | None:
    text = str(path)
    if "margin" in text:
        return (
            "Input path appears to be a HATS margin catalog. For production "
            "MMU ZTF, convert the primary ztf_dr23_lc-hats/dataset catalog; "
            "hats-import will create the MMU margin cache."
        )
    if "index" in text:
        return "Input path appears to be a HATS index catalog, not the light-curve table."
    return None


def discover_parquet_files(scan_dir: Path, pattern: str = "**/*.parquet") -> list[Path]:
    return sorted(
        path for path in scan_dir.glob(pattern)
        if path.is_file() and path.suffix == ".parquet" and not path.name.startswith("_")
    )


def select_files(
    files: list[Path],
    max_files: int | None,
    mode: str = "first",
    seed: int = 42,
) -> list[Path]:
    if max_files is None or max_files >= len(files):
        return files
    if max_files <= 0:
        raise ValueError("--max-files must be positive")
    if mode == "first":
        return files[:max_files]
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(files), size=max_files, replace=False))
    return [files[int(i)] for i in idx]


def input_catalog_dir_from_scan_dir(scan_dir: str | Path) -> Path:
    path = Path(scan_dir)
    return path.parent if path.name == "dataset" else path


def finalize_direct_hats_output(
    *,
    scan_dir: str | Path,
    output_root: str | Path,
    manifest_dir: str | Path,
    glob_pattern: str = "**/*.parquet",
    expected_file_count: int | None = None,
) -> Path:
    input_catalog = input_catalog_dir_from_scan_dir(scan_dir)
    catalog_dir = direct_catalog_dir(output_root)
    dataset_dir = catalog_dir / "dataset"
    output_files = discover_parquet_files(dataset_dir, glob_pattern)
    if not output_files:
        raise FileNotFoundError(f"No direct HATS parquet files found under {dataset_dir}")

    stats = direct_manifest_stats(manifest_dir)
    config = direct_manifest_config(manifest_dir)
    metadata_rows, metadata_max_rows = parquet_file_row_stats(output_files)
    if stats["rows"] == 0:
        stats["rows"] = metadata_rows
    if stats["max_rows"] == 0:
        stats["max_rows"] = metadata_max_rows
    if expected_file_count is not None and len(output_files) != expected_file_count:
        print(
            f"WARNING: direct HATS output has {len(output_files):,} parquet files, "
            f"expected {expected_file_count:,}. Finalizing anyway.",
            file=sys.stderr,
            flush=True,
        )
    if stats["rows"] != metadata_rows:
        print(
            f"WARNING: direct manifest rows ({stats['rows']:,}) differ from "
            f"parquet metadata rows ({metadata_rows:,}). Using parquet metadata.",
            file=sys.stderr,
            flush=True,
        )
        stats["rows"] = metadata_rows
    if metadata_max_rows:
        stats["max_rows"] = metadata_max_rows

    catalog_dir.mkdir(parents=True, exist_ok=True)
    copy_or_create_partition_info(input_catalog, catalog_dir, output_files, dataset_dir)
    copy_hats_sidecars(input_catalog, catalog_dir)
    write_direct_parquet_metadata(dataset_dir, output_files)
    write_direct_properties(input_catalog, output_root, stats, output_files)
    write_quality_policy(catalog_dir, config)
    return catalog_dir


def direct_manifest_stats(manifest_dir: str | Path) -> dict[str, int]:
    stats = {
        "files_seen": 0,
        "files_written": 0,
        "files_skipped": 0,
        "rows": 0,
        "observations": 0,
        "max_rows": 0,
    }
    for path in sorted(Path(manifest_dir).glob("manifest-direct-shard-*.json")):
        with open(path) as fh:
            manifest = json.load(fh)
        for key in ["files_seen", "files_written", "files_skipped", "rows", "observations"]:
            stats[key] += int(manifest.get(key, 0))
        stats["max_rows"] = max(stats["max_rows"], int(manifest.get("max_rows", 0)))
    return stats


def direct_manifest_config(manifest_dir: str | Path) -> dict[str, Any]:
    configs: list[dict[str, Any]] = []
    for path in sorted(Path(manifest_dir).glob("manifest-direct-shard-*.json")):
        with open(path) as fh:
            manifest = json.load(fh)
        if isinstance(manifest.get("config"), dict):
            configs.append(manifest["config"])
    if not configs:
        return asdict(ConvertConfig())
    canonical = json.dumps(configs[0], sort_keys=True)
    if any(json.dumps(config, sort_keys=True) != canonical for config in configs[1:]):
        raise ValueError("Direct HATS shard manifests use inconsistent conversion configurations.")
    return configs[0]


def write_quality_policy(catalog_dir: Path, config: dict[str, Any]) -> None:
    threshold = int(config.get("aion_relaxed_threshold", 30))
    payload = {
        "version": quality_policy_version(threshold),
        "bands": list(BAND_LABELS.values()),
        "candidate_mask": "relaxed_mask",
        "min_quality_epochs": threshold,
        "bad_epoch_bit": int(config.get("bad_epoch_bit", 32768)),
        "bright_mag_threshold": float(config.get("bright_mag_threshold", 15.5)),
    }
    (catalog_dir / "aion_quality_policy.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def parquet_file_row_stats(files: list[Path]) -> tuple[int, int]:
    total_rows = 0
    max_rows = 0
    for path in files:
        metadata = pq.read_metadata(path)
        rows = int(metadata.num_rows)
        total_rows += rows
        max_rows = max(max_rows, rows)
    return total_rows, max_rows


def copy_hats_sidecars(input_catalog: Path, output_catalog: Path) -> None:
    for name in ["point_map.fits", "skymap.fits"]:
        src = input_catalog / name
        if src.is_file():
            shutil.copy2(src, output_catalog / name)


def copy_or_create_partition_info(
    input_catalog: Path,
    output_catalog: Path,
    output_files: list[Path],
    dataset_dir: Path,
) -> None:
    src = input_catalog / "partition_info.csv"
    dst = output_catalog / "partition_info.csv"
    if src.is_file():
        shutil.copy2(src, dst)
        return

    partitions = sorted({
        parse_hats_partition(path.relative_to(dataset_dir))
        for path in output_files
    })
    with open(dst, "w") as fh:
        fh.write("Norder,Npix\n")
        for norder, npix in partitions:
            fh.write(f"{norder},{npix}\n")


def parse_hats_partition(relative_path: Path) -> tuple[int, int]:
    norder: int | None = None
    npix: int | None = None
    for part in relative_path.parts:
        if part.startswith("Norder="):
            norder = int(part.split("=", 1)[1])
        elif part.startswith("Npix="):
            value = part.split("=", 1)[1]
            if value.endswith(".parquet"):
                value = value[: -len(".parquet")]
            npix = int(value)
    if norder is None or npix is None:
        raise ValueError(f"Cannot parse HATS partition from {relative_path}")
    return norder, npix


def write_direct_parquet_metadata(dataset_dir: Path, files: list[Path]) -> None:
    first_schema = pq.read_schema(files[0])
    pq.write_metadata(first_schema, dataset_dir / "_common_metadata")
    merged_metadata: pq.FileMetaData | None = None
    for path in files:
        metadata = pq.read_metadata(path)
        metadata.set_file_path(path.relative_to(dataset_dir).as_posix())
        if merged_metadata is None:
            merged_metadata = metadata
        else:
            merged_metadata.append_row_groups(metadata)
    if merged_metadata is not None:
        merged_metadata.write_metadata_file(dataset_dir / "_metadata")


def write_direct_properties(
    input_catalog: Path,
    output_root: str | Path,
    stats: dict[str, int],
    output_files: list[Path],
) -> None:
    output_collection = Path(output_root) / CATALOG_NAME
    output_catalog = output_collection / CATALOG_NAME
    output_collection.mkdir(parents=True, exist_ok=True)
    output_catalog.mkdir(parents=True, exist_ok=True)

    input_props = read_hats_properties(input_catalog / "hats.properties")
    if not input_props:
        input_props = read_hats_properties(input_catalog / "properties")
    creation_date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MUTC")
    est_size = max(1, int(sum(path.stat().st_size for path in output_files) / max(1, stats["rows"])))
    catalog_props = {
        "obs_collection": CATALOG_NAME,
        "dataproduct_type": "object",
        "hats_nrows": str(int(stats["rows"])),
        "hats_col_ra": "ra",
        "hats_col_dec": "dec",
        "hats_col_healpix": "_healpix_29",
        "hats_col_healpix_order": "29",
        "hats_npix_suffix": ".parquet",
        "hats_skymap_order": input_props.get("hats_skymap_order", "8"),
        "hats_max_rows": str(max(1, int(stats["max_rows"]))),
        "hats_estsize": str(est_size),
        "hats_builder": "MultimodalUniverse ZTF direct HATS converter",
        "hats_creation_date": creation_date,
        "hats_release_date": input_props.get("hats_release_date", "2026-06-30"),
        "hats_version": input_props.get("hats_version", "v1.0"),
    }
    collection_props = {
        "obs_collection": CATALOG_NAME,
        "hats_primary_table_url": CATALOG_NAME,
        "hats_builder": catalog_props["hats_builder"],
        "hats_creation_date": creation_date,
        "hats_estsize": str(est_size),
        "hats_release_date": catalog_props["hats_release_date"],
        "hats_version": catalog_props["hats_version"],
    }
    write_hats_properties(output_catalog / "hats.properties", "#HATS catalog", catalog_props)
    write_hats_properties(output_catalog / "properties", "#HATS catalog", catalog_props)
    write_hats_properties(output_collection / "collection.properties", "#HATS Collection", collection_props)


def read_hats_properties(path: Path) -> dict[str, str]:
    props: dict[str, str] = {}
    if not path.is_file():
        return props
    with open(path) as fh:
        for line in fh:
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            props[key] = value
    return props


def write_hats_properties(path: Path, header: str, props: dict[str, str]) -> None:
    with open(path, "w") as fh:
        fh.write(f"{header}\n")
        for key, value in props.items():
            fh.write(f"{key}={value}\n")


def convert_files_to_parquet_dir(
    files: list[Path] | list[tuple[int, Path]],
    parquet_dir: str | Path,
    cfg: ConvertConfig,
    *,
    batch_size: int = 10_000,
    num_processes: int = 1,
    resume: bool = False,
    force: bool = False,
    shard_idx: int = 0,
) -> dict[str, int]:
    out_dir = Path(parquet_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indexed_files = normalize_indexed_files(files)
    tasks = [
        (i, str(path), str(out_dir), cfg, int(batch_size), bool(resume), bool(force))
        for i, path in indexed_files
    ]
    rows = 0
    observations = 0
    files_written = 0
    files_skipped = 0

    if num_processes > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            task_iter = iter(tasks)
            futures = set()
            for _ in range(min(len(tasks), max(1, num_processes * 2))):
                futures.add(pool.submit(_process_file_task, next(task_iter)))
            completed = 0
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    result = fut.result()
                    completed += 1
                    _raise_if_error(result)
                    rows += int(result["rows"])
                    observations += int(result["observations"])
                    files_written += int(result["written"])
                    files_skipped += int(result["skipped"])
                    if completed % 25 == 0 or completed == len(tasks):
                        print(
                            f"[shard {shard_idx}] converted {completed:,}/{len(tasks):,} "
                            f"files ({rows:,} rows, {observations:,} obs)",
                            flush=True,
                        )
                    try:
                        futures.add(pool.submit(_process_file_task, next(task_iter)))
                    except StopIteration:
                        pass
    else:
        for completed, task in enumerate(tasks, 1):
            result = _process_file_task(task)
            _raise_if_error(result)
            rows += int(result["rows"])
            observations += int(result["observations"])
            files_written += int(result["written"])
            files_skipped += int(result["skipped"])
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"[shard {shard_idx}] converted {completed:,}/{len(tasks):,} "
                    f"files ({rows:,} rows, {observations:,} obs)",
                    flush=True,
                )

    return {
        "files_seen": len(files),
        "files_written": files_written,
        "files_skipped": files_skipped,
        "rows": rows,
        "observations": observations,
    }


def direct_catalog_dir(output_root: str | Path) -> Path:
    return Path(output_root) / CATALOG_NAME / CATALOG_NAME


def direct_dataset_dir(output_root: str | Path) -> Path:
    return direct_catalog_dir(output_root) / "dataset"


def direct_manifest_dir(output_root: str | Path) -> Path:
    return Path(output_root) / CATALOG_NAME / "_direct_manifests"


def convert_files_to_direct_hats(
    files: list[Path] | list[tuple[int, Path]],
    input_dataset_dir: str | Path,
    output_dataset_dir: str | Path,
    cfg: ConvertConfig,
    *,
    batch_size: int = 10_000,
    num_processes: int = 1,
    resume: bool = False,
    force: bool = False,
    shard_idx: int = 0,
) -> dict[str, Any]:
    input_root = Path(input_dataset_dir)
    output_root = Path(output_dataset_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    indexed_files = normalize_indexed_files(files)
    tasks = [
        (
            i,
            str(path),
            path.relative_to(input_root).as_posix(),
            direct_hats_output_relative_path(path.relative_to(input_root)).as_posix(),
            str(output_root),
            cfg,
            int(batch_size),
            bool(resume),
            bool(force),
        )
        for i, path in indexed_files
    ]
    rows = 0
    observations = 0
    files_written = 0
    files_skipped = 0
    max_rows = 0
    file_records: list[dict[str, Any]] = []

    if num_processes > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            task_iter = iter(tasks)
            futures = set()
            for _ in range(min(len(tasks), max(1, num_processes * 2))):
                futures.add(pool.submit(_process_direct_hats_file_task, next(task_iter)))
            completed = 0
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    result = fut.result()
                    completed += 1
                    _raise_if_error(result)
                    file_records.append(result)
                    rows += int(result["rows"])
                    observations += int(result["observations"])
                    files_written += int(result["written"])
                    files_skipped += int(result["skipped"])
                    max_rows = max(max_rows, int(result["rows"]))
                    if completed % 25 == 0 or completed == len(tasks):
                        print(
                            f"[shard {shard_idx}] direct-converted "
                            f"{completed:,}/{len(tasks):,} files "
                            f"({rows:,} rows, {observations:,} obs)",
                            flush=True,
                        )
                    try:
                        futures.add(pool.submit(_process_direct_hats_file_task, next(task_iter)))
                    except StopIteration:
                        pass
    else:
        for completed, task in enumerate(tasks, 1):
            result = _process_direct_hats_file_task(task)
            _raise_if_error(result)
            file_records.append(result)
            rows += int(result["rows"])
            observations += int(result["observations"])
            files_written += int(result["written"])
            files_skipped += int(result["skipped"])
            max_rows = max(max_rows, int(result["rows"]))
            if completed % 25 == 0 or completed == len(tasks):
                print(
                    f"[shard {shard_idx}] direct-converted {completed:,}/{len(tasks):,} "
                    f"files ({rows:,} rows, {observations:,} obs)",
                    flush=True,
                )

    file_records.sort(key=lambda item: int(item["file_index"]))
    return {
        "files_seen": len(files),
        "files_written": files_written,
        "files_skipped": files_skipped,
        "rows": rows,
        "observations": observations,
        "max_rows": max_rows,
        "files": file_records,
    }


def direct_hats_output_relative_path(input_relative: Path) -> Path:
    """Map an input ZTF/IPAC HATS file path to the MMU HATS file path.

    The ZTF/IPAC product stores one parquet under a partition directory, e.g.
    ``Npix=562/part0.snappy.parquet``. HATS/LSDB expects the MMU output file
    itself to be the pixel leaf, e.g. ``Npix=562.parquet``. This preserves the
    HEALPix partition assignment without running a global spatial repartition.
    """
    parts = list(input_relative.parts)
    for idx, part in enumerate(parts):
        if not part.startswith("Npix="):
            continue
        if part.endswith(".parquet"):
            return input_relative
        if idx == len(parts) - 2 and parts[-1].endswith(".parquet"):
            return Path(*parts[:idx], f"{part}.parquet")
    return input_relative


def duplicate_direct_hats_outputs(files: list[Path], input_dataset_dir: str | Path) -> list[str]:
    input_root = Path(input_dataset_dir)
    seen: set[str] = set()
    duplicates: list[str] = []
    for path in files:
        output_relative = direct_hats_output_relative_path(path.relative_to(input_root)).as_posix()
        if output_relative in seen:
            duplicates.append(output_relative)
        else:
            seen.add(output_relative)
    return duplicates


def normalize_indexed_files(
    files: list[Path] | list[tuple[int, Path]],
) -> list[tuple[int, Path]]:
    if not files:
        return []
    first = files[0]
    if isinstance(first, tuple):
        return [(int(idx), Path(path)) for idx, path in files]  # type: ignore[misc]
    return [(idx, Path(path)) for idx, path in enumerate(files)]  # type: ignore[arg-type]


def _raise_if_error(result: dict[str, Any]) -> None:
    if result.get("error"):
        raise RuntimeError(f"{result['path']}: {result['error']}")


def _process_file_task(task: tuple[int, str, str, ConvertConfig, int, bool, bool]) -> dict[str, Any]:
    file_index, path_str, out_dir_str, cfg, batch_size, resume, force = task
    _set_thread_env(cfg.arrow_threads)
    path = Path(path_str)
    out_path = Path(out_dir_str) / f"part-{file_index:08d}.parquet"
    if out_path.exists() and resume and not force:
        meta = pq.read_metadata(out_path)
        n_total = pq.read_table(out_path, columns=["n_total"])
        return {
            "path": str(path),
            "output": str(out_path),
            "rows": int(meta.num_rows),
            "observations": int(n_total.column("n_total").to_numpy(zero_copy_only=False).sum()),
            "written": 0,
            "skipped": 1,
            "error": None,
        }
    if out_path.exists():
        out_path.unlink()

    writer: pq.ParquetWriter | None = None
    rows = 0
    observations = 0
    try:
        pf = pq.ParquetFile(path)
        validate_schema(pf.schema_arrow, cfg, path)
        columns = input_columns(pf.schema_arrow, cfg)
        for batch in pf.iter_batches(
            batch_size=batch_size,
            columns=columns,
            use_threads=cfg.arrow_threads > 1,
        ):
            table = convert_batch(batch, cfg)
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            rows += table.num_rows
            observations += int(table.column("n_total").to_numpy(zero_copy_only=False).sum())
        if writer is None:
            empty = empty_output_table(cfg)
            writer = pq.ParquetWriter(out_path, empty.schema, compression="snappy")
            writer.write_table(empty)
    except BaseException as exc:  # noqa: BLE001
        if writer is not None:
            writer.close()
        if out_path.exists():
            out_path.unlink()
        return {
            "path": str(path),
            "output": str(out_path),
            "rows": rows,
            "observations": observations,
            "written": 0,
            "skipped": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if writer is not None:
            writer.close()

    return {
        "path": str(path),
        "output": str(out_path),
        "rows": rows,
        "observations": observations,
        "written": 1,
        "skipped": 0,
        "error": None,
    }


def _process_direct_hats_file_task(
    task: tuple[int, str, str, str, str, ConvertConfig, int, bool, bool],
) -> dict[str, Any]:
    (
        file_index,
        path_str,
        input_relative_str,
        output_relative_str,
        output_dataset_str,
        cfg,
        batch_size,
        resume,
        force,
    ) = task
    _set_thread_env(cfg.arrow_threads)
    path = Path(path_str)
    input_relative_path = Path(input_relative_str)
    output_relative_path = Path(output_relative_str)
    out_path = Path(output_dataset_str) / output_relative_path
    if out_path.exists() and resume and not force:
        try:
            if direct_output_matches_config(out_path, cfg):
                meta = pq.read_metadata(out_path)
                n_total = pq.read_table(out_path, columns=["n_total"])
                return {
                    "file_index": file_index,
                    "path": str(path),
                    "input_relative_path": input_relative_path.as_posix(),
                    "relative_path": output_relative_path.as_posix(),
                    "output": str(out_path),
                    "rows": int(meta.num_rows),
                    "observations": int(n_total.column("n_total").to_numpy(zero_copy_only=False).sum()),
                    "written": 0,
                    "skipped": 1,
                    "error": None,
                }
            out_path.unlink(missing_ok=True)
        except BaseException:
            out_path.unlink(missing_ok=True)
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    rows = 0
    observations = 0
    try:
        pf = pq.ParquetFile(path)
        validate_schema(pf.schema_arrow, cfg, path)
        if cfg.healpix_col not in pf.schema_arrow.names:
            raise ValueError(
                f"{path} is missing {cfg.healpix_col}; direct HATS mode "
                "requires the input HATS healpix column."
            )
        columns = input_columns(pf.schema_arrow, cfg, include_healpix=True)
        for batch in pf.iter_batches(
            batch_size=batch_size,
            columns=columns,
            use_threads=cfg.arrow_threads > 1,
        ):
            table = convert_batch(batch, cfg, include_healpix=True)
            if table.num_rows == 0:
                continue
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="snappy")
            writer.write_table(table)
            rows += table.num_rows
            observations += int(table.column("n_total").to_numpy(zero_copy_only=False).sum())
        if writer is None:
            empty = empty_output_table(cfg, include_healpix=True)
            writer = pq.ParquetWriter(out_path, empty.schema, compression="snappy")
            writer.write_table(empty)
    except BaseException as exc:  # noqa: BLE001
        if writer is not None:
            writer.close()
        if out_path.exists():
            out_path.unlink()
        return {
            "file_index": file_index,
            "path": str(path),
            "input_relative_path": input_relative_path.as_posix(),
            "relative_path": output_relative_path.as_posix(),
            "output": str(out_path),
            "rows": rows,
            "observations": observations,
            "written": 0,
            "skipped": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if writer is not None:
            writer.close()

    return {
        "file_index": file_index,
        "path": str(path),
        "input_relative_path": input_relative_path.as_posix(),
        "relative_path": output_relative_path.as_posix(),
        "output": str(out_path),
        "rows": rows,
        "observations": observations,
        "written": 1,
        "skipped": 0,
        "error": None,
    }


def direct_output_matches_config(path: Path, cfg: ConvertConfig) -> bool:
    parquet = pq.ParquetFile(path)
    names = set(parquet.schema_arrow.names)
    if "quality_policy_version" not in names:
        return False
    if cfg.include_quality_flags:
        lightcurve_names = set(nested_lightcurve_field_names(parquet.schema_arrow.field("lightcurve").type))
        if "relaxed_mask" not in lightcurve_names:
            return False
    table = parquet.read_row_group(0, columns=["quality_policy_version"])
    if table.num_rows == 0:
        return True
    return table.column("quality_policy_version")[0].as_py() == quality_policy_version(
        cfg.aion_relaxed_threshold
    )


def input_columns(
    schema: pa.Schema,
    cfg: ConvertConfig,
    *,
    include_healpix: bool = False,
) -> list[str]:
    names = set(schema.names)
    required = [
        cfg.object_col,
        cfg.filter_col,
        cfg.ra_col,
        cfg.dec_col,
        cfg.nepochs_col,
        cfg.lightcurve_col,
    ]
    optional = [cfg.field_col, cfg.rcid_col]
    if include_healpix:
        required.insert(0, cfg.healpix_col)
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing required ZTF columns {missing}")
    return [name for name in required + optional if name in names]


def validate_schema(schema: pa.Schema, cfg: ConvertConfig, path: Path | str) -> None:
    names = set(schema.names)
    missing_top = [
        name for name in [
            cfg.object_col,
            cfg.filter_col,
            cfg.ra_col,
            cfg.dec_col,
            cfg.nepochs_col,
            cfg.lightcurve_col,
        ]
        if name not in names
    ]
    if missing_top:
        raise ValueError(f"{path} is missing top-level columns {missing_top}")
    lc_names = set(nested_lightcurve_field_names(schema.field(cfg.lightcurve_col).type))
    missing_lc = [
        name for name in [cfg.time_col, cfg.mag_col, cfg.magerr_col, cfg.catflags_col]
        if name not in lc_names
    ]
    if missing_lc:
        raise ValueError(f"{path} is missing lightcurve fields {missing_lc}")


def nested_lightcurve_field_names(dtype: pa.DataType) -> list[str]:
    dtype = storage_type(dtype)
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        value_type = storage_type(dtype.value_type)
        if pa.types.is_struct(value_type):
            return [field.name for field in value_type]
        return []
    if pa.types.is_struct(dtype):
        return [field.name for field in dtype]
    return []


def convert_batch(
    batch: pa.RecordBatch,
    cfg: ConvertConfig,
    *,
    include_healpix: bool = False,
) -> pa.Table:
    healpix_array = maybe_batch_column(batch, cfg.healpix_col) if include_healpix else None
    object_array = batch_column(batch, cfg.object_col)
    filter_array = batch_column(batch, cfg.filter_col)
    ra_array = batch_column(batch, cfg.ra_col)
    dec_array = batch_column(batch, cfg.dec_col)
    nepochs_array = batch_column(batch, cfg.nepochs_col)
    field_array = maybe_batch_column(batch, cfg.field_col)
    rcid_array = maybe_batch_column(batch, cfg.rcid_col)
    lightcurve = unwrap_extension_array(batch_column(batch, cfg.lightcurve_col))

    list_fields, lengths = lightcurve_list_fields(lightcurve, cfg)
    total = int(lengths.sum())

    hmjd_flat = array_to_float_numpy(list_fields[cfg.time_col].flatten(), dtype=np.float64)
    mag_flat = array_to_float_numpy(list_fields[cfg.mag_col].flatten(), dtype=np.float64)
    magerr_flat = array_to_float_numpy(list_fields[cfg.magerr_col].flatten(), dtype=np.float64)
    cat_flat = array_to_float_numpy(list_fields[cfg.catflags_col].flatten(), dtype=np.float64)
    filter_np = array_to_int_numpy(filter_array, dtype=np.int64)

    valid_filter_parent = np.isin(filter_np, list(BAND_LABELS))
    valid_filter_epoch = np.repeat(valid_filter_parent, lengths)
    band_epoch_id = np.repeat(filter_np, lengths)
    masks = quality_masks_from_arrays(
        hmjd_flat,
        mag_flat,
        magerr_flat,
        cat_flat,
        valid_filter_epoch,
        cfg,
    )

    n_finite = sum_by_parent(masks["finite"], lengths)
    n_poserr = sum_by_parent(masks["poserr"], lengths)
    n_relaxed = sum_by_parent(masks["relaxed"], lengths)
    n_strict = sum_by_parent(masks["strict"], lengths)
    n_bright = sum_by_parent(masks["bright"], lengths)

    labels = band_labels_for_filter(filter_np)
    columns: dict[str, pa.Array] = {}
    if include_healpix:
        if healpix_array is None:
            raise KeyError(cfg.healpix_col)
        columns[cfg.healpix_col] = pc.cast(healpix_array, pa.int64())
    columns.update({
        "ra": pc.cast(ra_array, pa.float64()),
        "dec": pc.cast(dec_array, pa.float64()),
        "object_id": object_id_strings(object_array),
        "ztf_objectid": pc.cast(object_array, pa.int64()),
        "filterid": pc.cast(filter_array, pa.int16()),
        "band": pa.array(labels, type=pa.string()),
        "nepochs": pc.cast(nepochs_array, pa.int64()),
        "n_total": pa.array(lengths, type=pa.int64()),
        "n_finite": pa.array(n_finite, type=pa.int64()),
        "n_poserr": pa.array(n_poserr, type=pa.int64()),
        "n_relaxed": pa.array(n_relaxed, type=pa.int64()),
        "n_strict": pa.array(n_strict, type=pa.int64()),
        "n_bright_relaxed": pa.array(n_bright, type=pa.int64()),
    })
    columns["fieldid"] = (
        pc.cast(field_array, pa.int32()) if field_array is not None
        else pa.nulls(len(lengths), type=pa.int32())
    )
    columns["rcid"] = (
        pc.cast(rcid_array, pa.int16()) if rcid_array is not None
        else pa.nulls(len(lengths), type=pa.int16())
    )

    for band_id, suffix in BAND_SUFFIXES.items():
        is_band = filter_np == band_id
        columns[f"n_total_{suffix}"] = pa.array(np.where(is_band, lengths, 0), type=pa.int64())
        columns[f"n_finite_{suffix}"] = pa.array(np.where(is_band, n_finite, 0), type=pa.int64())
        columns[f"n_poserr_{suffix}"] = pa.array(np.where(is_band, n_poserr, 0), type=pa.int64())
        columns[f"n_relaxed_{suffix}"] = pa.array(np.where(is_band, n_relaxed, 0), type=pa.int64())
        columns[f"n_strict_{suffix}"] = pa.array(np.where(is_band, n_strict, 0), type=pa.int64())
        columns[f"n_bright_relaxed_{suffix}"] = pa.array(np.where(is_band, n_bright, 0), type=pa.int64())

    for threshold in RELAXED_THRESHOLDS:
        columns[f"pass_relaxed_ge{threshold}"] = pa.array(n_relaxed >= threshold, type=pa.bool_())
    for threshold in STRICT_THRESHOLDS:
        columns[f"pass_strict_ge{threshold}"] = pa.array(n_strict >= threshold, type=pa.bool_())
    columns["aion_training_candidate"] = pa.array(
        n_relaxed >= cfg.aion_relaxed_threshold,
        type=pa.bool_(),
    )
    columns["aion_training_candidate_v1"] = columns["aion_training_candidate"]
    columns["quality_policy_version"] = pa.array(
        [quality_policy_version(cfg.aion_relaxed_threshold)] * len(lengths),
        type=pa.string(),
    )

    columns["lightcurve"] = build_lightcurve_struct(
        list_fields,
        lengths,
        labels,
        masks,
        band_epoch_id,
        total,
        cfg,
    )
    return pa.table(columns)


def lightcurve_list_fields(
    lightcurve: pa.Array,
    cfg: ConvertConfig,
) -> tuple[dict[str, pa.Array], np.ndarray]:
    dtype = storage_type(lightcurve.type)
    if pa.types.is_struct(dtype):
        first = unwrap_extension_array(lightcurve.field(cfg.time_col))
        lengths = list_lengths(first)
        fields: dict[str, pa.Array] = {}
        for name in [cfg.time_col, cfg.mag_col, cfg.magerr_col, cfg.catflags_col, cfg.clrcoeff_col]:
            if dtype.get_field_index(name) < 0:
                continue
            array = unwrap_extension_array(lightcurve.field(name))
            ensure_list_lengths(name, array, lengths)
            fields[name] = array
        return fields, lengths

    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        lengths = list_lengths(lightcurve)
        flat = unwrap_extension_array(lightcurve.flatten())
        if not pa.types.is_struct(storage_type(flat.type)):
            raise TypeError(f"Expected list<struct<...>> lightcurve, got {lightcurve.type}")
        fields = {}
        for name in [cfg.time_col, cfg.mag_col, cfg.magerr_col, cfg.catflags_col, cfg.clrcoeff_col]:
            if storage_type(flat.type).get_field_index(name) < 0:
                continue
            fields[name] = rebuild_lists_from_flat(flat.field(name), lengths)
        return fields, lengths

    raise TypeError(f"Unsupported lightcurve type: {lightcurve.type}")


def ensure_list_lengths(name: str, array: pa.Array, expected: np.ndarray) -> None:
    dtype = storage_type(array.type)
    if not (pa.types.is_list(dtype) or pa.types.is_large_list(dtype)):
        raise TypeError(f"Expected lightcurve.{name} to be a list, got {array.type}")
    actual = list_lengths(array)
    if not np.array_equal(actual, expected):
        raise ValueError(f"lightcurve.{name} has inconsistent list lengths")


def rebuild_lists_from_flat(values: pa.Array, lengths: np.ndarray) -> pa.Array:
    return pa.ListArray.from_arrays(offsets_from_lengths(lengths), values)


def quality_masks_from_arrays(
    hmjd: np.ndarray,
    mag: np.ndarray,
    magerr: np.ndarray,
    cat: np.ndarray,
    valid_filter: np.ndarray,
    cfg: ConvertConfig,
) -> dict[str, np.ndarray]:
    finite = np.isfinite(hmjd) & np.isfinite(mag) & np.isfinite(magerr) & valid_filter
    poserr = finite & (magerr > 0)
    has_cat = np.isfinite(cat)
    cat_int = np.where(has_cat, cat, 0).astype(np.int64, copy=False)
    relaxed = poserr & has_cat & ((cat_int & int(cfg.bad_epoch_bit)) == 0)
    strict = relaxed & (cat_int == 0)
    bright = relaxed & (mag < float(cfg.bright_mag_threshold))
    return {
        "finite": finite,
        "poserr": poserr,
        "relaxed": relaxed,
        "strict": strict,
        "bright": bright,
        "cat_int": cat_int,
    }


def build_lightcurve_struct(
    list_fields: dict[str, pa.Array],
    lengths: np.ndarray,
    labels: list[str | None],
    masks: dict[str, np.ndarray],
    band_epoch_id: np.ndarray,
    total: int,
    cfg: ConvertConfig,
) -> pa.StructArray:
    arrays: list[pa.Array] = [
        pc.cast(list_fields[cfg.time_col], pa.list_(pa.float64())),
    ]
    names = ["time"]

    if cfg.include_band_list:
        arrays.append(constant_string_lists(labels, lengths))
        names.append("band")
        arrays.append(list_from_flat_numpy(band_epoch_id.astype(np.int16, copy=False), lengths, pa.int16()))
        names.append("band_id")

    arrays.extend(
        [
            pc.cast(list_fields[cfg.mag_col], pa.list_(pa.float32())),
            pc.cast(list_fields[cfg.magerr_col], pa.list_(pa.float32())),
        ]
    )
    names.extend(["mag", "mag_err"])
    if cfg.clrcoeff_col in list_fields:
        arrays.append(pc.cast(list_fields[cfg.clrcoeff_col], pa.list_(pa.float32())))
        names.append("clrcoeff")
    arrays.append(pc.cast(list_fields[cfg.catflags_col], pa.list_(pa.int32())))
    names.append("catflags")

    if cfg.include_quality_flags:
        offsets = offsets_from_lengths(lengths)
        for mask_name, field_name in [
            ("finite", "finite_mask"),
            ("poserr", "poserr_mask"),
            ("relaxed", "relaxed_mask"),
            ("strict", "strict_mask"),
            ("bright", "bright_mask"),
        ]:
            if total:
                values = pa.array(masks[mask_name].astype(bool, copy=False), type=pa.bool_())
            else:
                values = pa.array([], type=pa.bool_())
            arrays.append(pa.ListArray.from_arrays(offsets, values))
            names.append(field_name)

    return pa.StructArray.from_arrays(arrays, names=names)


def empty_output_table(cfg: ConvertConfig, *, include_healpix: bool = False) -> pa.Table:
    fields = []
    if include_healpix:
        fields.append(pa.field(cfg.healpix_col, pa.int64()))
    fields.extend([
        pa.field("ra", pa.float64()),
        pa.field("dec", pa.float64()),
        pa.field("object_id", pa.string()),
        pa.field("ztf_objectid", pa.int64()),
        pa.field("filterid", pa.int16()),
        pa.field("band", pa.string()),
        pa.field("nepochs", pa.int64()),
        pa.field("n_total", pa.int64()),
        pa.field("n_finite", pa.int64()),
        pa.field("n_poserr", pa.int64()),
        pa.field("n_relaxed", pa.int64()),
        pa.field("n_strict", pa.int64()),
        pa.field("n_bright_relaxed", pa.int64()),
        pa.field("fieldid", pa.int32()),
        pa.field("rcid", pa.int16()),
    ])
    for suffix in BAND_SUFFIXES.values():
        for prefix in ["total", "finite", "poserr", "relaxed", "strict", "bright_relaxed"]:
            fields.append(pa.field(f"n_{prefix}_{suffix}", pa.int64()))
    for threshold in RELAXED_THRESHOLDS:
        fields.append(pa.field(f"pass_relaxed_ge{threshold}", pa.bool_()))
    for threshold in STRICT_THRESHOLDS:
        fields.append(pa.field(f"pass_strict_ge{threshold}", pa.bool_()))
    fields.extend([
        pa.field("aion_training_candidate", pa.bool_()),
        pa.field("aion_training_candidate_v1", pa.bool_()),
        pa.field("quality_policy_version", pa.string()),
    ])

    lc_fields = [pa.field("time", pa.list_(pa.float64()))]
    if cfg.include_band_list:
        lc_fields.extend([
            pa.field("band", pa.list_(pa.string())),
            pa.field("band_id", pa.list_(pa.int16())),
        ])
    lc_fields.extend([
        pa.field("mag", pa.list_(pa.float32())),
        pa.field("mag_err", pa.list_(pa.float32())),
        pa.field("clrcoeff", pa.list_(pa.float32())),
        pa.field("catflags", pa.list_(pa.int32())),
    ])
    if cfg.include_quality_flags:
        lc_fields.extend([
            pa.field("finite_mask", pa.list_(pa.bool_())),
            pa.field("poserr_mask", pa.list_(pa.bool_())),
            pa.field("relaxed_mask", pa.list_(pa.bool_())),
            pa.field("strict_mask", pa.list_(pa.bool_())),
            pa.field("bright_mask", pa.list_(pa.bool_())),
        ])
    fields.append(pa.field("lightcurve", pa.struct(lc_fields)))
    return pa.Table.from_arrays([pa.array([], type=f.type) for f in fields], schema=pa.schema(fields))


def batch_column(batch: pa.RecordBatch, name: str) -> pa.Array:
    idx = batch.schema.get_field_index(name)
    if idx < 0:
        raise KeyError(name)
    return batch.column(idx)


def maybe_batch_column(batch: pa.RecordBatch, name: str) -> pa.Array | None:
    idx = batch.schema.get_field_index(name)
    if idx < 0:
        return None
    return batch.column(idx)


def storage_type(dtype: pa.DataType) -> pa.DataType:
    return dtype.storage_type if isinstance(dtype, pa.ExtensionType) else dtype


def unwrap_extension_array(array: pa.Array) -> pa.Array:
    return array.storage if isinstance(array, pa.ExtensionArray) else array


def list_lengths(array: pa.Array) -> np.ndarray:
    lengths = pc.fill_null(array.value_lengths(), 0)
    return np.asarray(lengths.to_numpy(zero_copy_only=False), dtype=np.int64)


def offsets_from_lengths(lengths: np.ndarray) -> pa.Array:
    offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    if offsets[-1] > np.iinfo(np.int32).max:
        raise ValueError(
            "One conversion batch contains more than 2^31-1 epochs; "
            "reduce --batch-size."
        )
    return pa.array(offsets.astype(np.int32, copy=False), type=pa.int32())


def array_to_float_numpy(array: pa.Array, *, dtype: np.dtype) -> np.ndarray:
    array = unwrap_extension_array(array)
    try:
        out = array.to_numpy(zero_copy_only=False)
    except Exception:
        out = array.to_pandas().to_numpy()
    if out.dtype.kind in {"O", "U", "S"}:
        out = np.array([np.nan if x is None else x for x in out], dtype=dtype)
    else:
        out = out.astype(dtype, copy=False)
    return out


def array_to_int_numpy(array: pa.Array, *, dtype: np.dtype) -> np.ndarray:
    array = unwrap_extension_array(array)
    try:
        out = array.to_numpy(zero_copy_only=False)
    except Exception:
        out = array.to_pandas().to_numpy()
    if out.dtype.kind in {"O", "U", "S"}:
        out = np.array([0 if x is None else x for x in out], dtype=dtype)
    else:
        out = out.astype(dtype, copy=False)
    return out


def object_id_strings(object_array: pa.Array) -> pa.Array:
    values = object_array.to_pylist()
    return pa.array([None if value is None else str(int(value)) for value in values], type=pa.string())


def band_labels_for_filter(filter_ids: np.ndarray) -> list[str | None]:
    labels: list[str | None] = []
    for value in filter_ids:
        if value in BAND_LABELS:
            labels.append(BAND_LABELS[int(value)])
        elif value == 0:
            labels.append(None)
        else:
            labels.append(f"filter{int(value)}")
    return labels


def constant_string_lists(labels: list[str | None], lengths: np.ndarray) -> pa.Array:
    offsets = offsets_from_lengths(lengths)
    if int(lengths.sum()) == 0:
        return pa.ListArray.from_arrays(offsets, pa.array([], type=pa.string()))
    values = np.repeat(np.asarray(labels, dtype=object), lengths)
    return pa.ListArray.from_arrays(offsets, pa.array(values.tolist(), type=pa.string()))


def list_from_flat_numpy(values: np.ndarray, lengths: np.ndarray, value_type: pa.DataType) -> pa.Array:
    offsets = offsets_from_lengths(lengths)
    if len(values) == 0:
        return pa.ListArray.from_arrays(offsets, pa.array([], type=value_type))
    return pa.ListArray.from_arrays(offsets, pa.array(values, type=value_type))


def sum_by_parent(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    out = np.zeros(len(lengths), dtype=np.int64)
    nonzero = lengths > 0
    if not nonzero.any():
        return out
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))[nonzero]
    out[nonzero] = np.add.reduceat(values.astype(np.int64, copy=False), starts)
    return out


if __name__ == "__main__":
    sys.exit(main())
