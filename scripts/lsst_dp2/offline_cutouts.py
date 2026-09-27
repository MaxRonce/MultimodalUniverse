"""Snapshot ready patches, scatter offline MMU extraction, then gather HATS.

No network access or token is needed. Inputs are frozen in a new run; the FITS
mirror is referenced, not copied. Each array task owns whole patches and its
own catalog, manifest, Parquet directory, and validation receipt.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from filelock import FileLock

from scripts.lsst_dp2 import build_parent_sample_hats as build
from scripts.lsst_dp2 import validate_parent_sample as validate
from scripts.lsst_dp2.common import BANDS, read_catalog, sha256_file, validate_catalog

MANIFEST_COLUMNS = (
    "tract",
    "patch",
    "band",
    "output_path",
    "sha256",
    "dataset_id",
    "s_resolution",
    "mask_planes",
    "status",
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _software() -> dict:
    here = Path(__file__).parent
    paths = [
        here / name
        for name in (
            "offline_cutouts.py",
            "build_parent_sample_hats.py",
            "coadd.py",
            "common.py",
            "schema.py",
            "validate_parent_sample.py",
        )
    ]
    paths.append(here.parents[1] / "mmu/hats_import.py")
    return {
        "files": {path.name: sha256_file(path) for path in paths},
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "astropy",
                "pyarrow",
                "hats-import",
                "hats",
                "distributed",
            )
        },
    }


def ready_patches(rows: list[dict]) -> set[tuple[int, int]]:
    """Unresolved products are never silently reclassified as unavailable."""
    states = defaultdict(dict)
    for row in rows:
        key = (row["tract"], row["patch"])
        if row["band"] in states[key]:
            raise ValueError(f"duplicate patch-band in manifest: {key}/{row['band']}")
        states[key][row["band"]] = row["status"]
    return {
        key
        for key, bands in states.items()
        if set(bands) == set(BANDS)
        and set(bands.values()) <= {"complete", "unavailable"}
        and "complete" in bands.values()
    }


def balance_patches(
    counts: dict[tuple[int, int], int], shards: int
) -> list[list[tuple[int, int]]]:
    """Deterministic largest-first scheduling; never split a patch across jobs."""
    if shards < 1 or not counts:
        raise ValueError("positive shard count and at least one ready patch required")
    groups = [[] for _ in range(min(shards, len(counts)))]
    heap = [(0, index) for index in range(len(groups))]
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        load, index = heapq.heappop(heap)
        groups[index].append(key)
        heapq.heappush(heap, (load + count, index))
    return groups


def _write_manifest(path: Path, rows: list[dict]) -> None:
    with sqlite3.connect(path) as con:
        con.execute("""CREATE TABLE coadds (
            tract INTEGER, patch INTEGER, band TEXT, output_path TEXT,
            sha256 TEXT, dataset_id TEXT, s_resolution REAL, mask_planes TEXT,
            status TEXT, PRIMARY KEY (tract, patch, band))""")
        con.executemany(
            "INSERT INTO coadds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(row[name] for name in MANIFEST_COLUMNS) for row in rows],
        )


def load_plan(root: Path) -> dict:
    plan = json.loads((root / "inputs/plan.json").read_text())
    if plan["software"] != _software():
        raise ValueError(
            "processing code or packages changed; restore them or use a new output run"
        )
    return plan


def prepare(source: Path, root: Path, shards: int) -> dict:
    source, root = source.resolve(), root.resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError(
            "source and output must be separate, non-nested run directories"
        )
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".prepare.lock"), timeout=0):
        if (root / "inputs").exists():
            plan = load_plan(root)
            if plan["source_run"] != str(source) or plan["requested_shards"] != shards:
                raise ValueError(
                    "existing snapshot has different source or shard count"
                )
            print(f"Reusing frozen snapshot: {plan['objects']:,} objects", flush=True)
            return plan

        catalog_path = source / "catalog/objects.parquet"
        # One SELECT provides a consistent SQLite snapshot, even during downloads.
        with sqlite3.connect(
            (source / "download_manifest.sqlite").as_uri() + "?mode=ro", uri=True
        ) as con:
            con.row_factory = sqlite3.Row
            rows = [
                dict(row)
                for row in con.execute(
                    f"SELECT {', '.join(MANIFEST_COLUMNS)} FROM coadds ORDER BY tract, patch, band"
                )
            ]
        catalog = read_catalog(catalog_path)
        columns = validate_catalog(catalog)
        if any(
            np.ma.getmaskarray(catalog[columns[name]]).any()
            for name in ("object_id", "tract", "patch", "ra", "dec")
        ):
            raise ValueError("source catalog has null identifiers or coordinates")
        ids = np.asarray(catalog[columns["object_id"]])
        if np.ma.getmaskarray(catalog[columns["object_id"]]).any() or len(
            np.unique(ids)
        ) != len(ids):
            raise ValueError("source catalog has null or duplicate object IDs")
        keys, inverse, counts = np.unique(
            np.column_stack(
                [
                    np.asarray(catalog[columns["tract"]]),
                    np.asarray(catalog[columns["patch"]]),
                ]
            ),
            axis=0,
            return_inverse=True,
            return_counts=True,
        )
        keys = [tuple(map(int, key)) for key in keys]
        ready = ready_patches(rows) & set(keys)
        source_states = defaultdict(dict)
        for row in rows:
            source_states[(row["tract"], row["patch"])][row["band"]] = row["status"]
        groups = balance_patches(
            {key: int(count) for key, count in zip(keys, counts) if key in ready},
            shards,
        )
        by_patch = defaultdict(list)
        for row in rows:
            if (row["tract"], row["patch"]) in ready:
                if row["status"] == "complete":
                    # Older local mirrors used paths relative to the repo cwd.
                    path = Path(row["output_path"]).resolve()
                    if not path.is_file() or not row["sha256"] or not row["dataset_id"]:
                        raise ValueError(f"invalid complete product: {row}")
                    row["output_path"] = str(path)
                by_patch[(row["tract"], row["patch"])].append(row)
        assignments = {
            key: index for index, group in enumerate(groups) for key in group
        }
        row_shards = np.array([assignments.get(key, -1) for key in keys])[inverse]
        selected_rows = int((row_shards >= 0).sum())
        plan = {
            "version": 1,
            "source_run": str(source),
            "software": _software(),
            "source_catalog_sha256": sha256_file(catalog_path),
            "source_objects": len(catalog),
            "objects": selected_rows,
            "excluded_objects": len(catalog) - selected_rows,
            "patches": len(ready),
            "excluded_patches": len(keys) - len(ready),
            "excluded_patch_details": [
                {
                    "tract": key[0],
                    "patch": key[1],
                    "objects": int(count),
                    "states": source_states[key],
                }
                for key, count in zip(keys, counts)
                if key not in ready
            ],
            "requested_shards": shards,
            "objects_per_file": 64,
            "bands": list(BANDS),
            "image_shape": [6, 160, 160],
            "manifest_status_counts": dict(
                Counter(row["status"] for key in ready for row in by_patch[key])
            ),
            "shards": [],
        }
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
            staging = Path(temporary) / "inputs"
            staging.mkdir()
            selection = catalog_path.with_suffix(".parquet.selection.json")
            if selection.exists():
                (staging / "source_selection.json").write_bytes(selection.read_bytes())
            for index, group in enumerate(groups):
                directory = staging / f"{index:05d}"
                directory.mkdir()
                subset = catalog[row_shards == index]
                subset.write(directory / "objects.parquet", format="parquet")
                _write_manifest(
                    directory / "manifest.sqlite",
                    [row for key in sorted(group) for row in by_patch[key]],
                )
                plan["shards"].append(
                    {
                        "index": index,
                        "objects": len(subset),
                        "patches": len(group),
                        "catalog_sha256": sha256_file(directory / "objects.parquet"),
                        "manifest_sha256": sha256_file(directory / "manifest.sqlite"),
                    }
                )
                if (index + 1) % 16 == 0 or index + 1 == len(groups):
                    print(
                        f"[prepare] {index + 1}/{len(groups)} frozen shards", flush=True
                    )
            _write_json(staging / "plan.json", plan)
            os.replace(staging, root / "inputs")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in plan.items()
                    if key not in {"software", "shards"}
                },
                indent=2,
            ),
            flush=True,
        )
        return plan


def _shard_inputs(root: Path, shard: dict) -> tuple[Path, Path]:
    directory = root / "inputs" / f"{shard['index']:05d}"
    catalog, manifest = directory / "objects.parquet", directory / "manifest.sqlite"
    for path, key in ((catalog, "catalog_sha256"), (manifest, "manifest_sha256")):
        if sha256_file(path) != shard[key]:
            raise ValueError(f"frozen input changed: {path}")
    return catalog, manifest


def _file_inventory(directory: Path, verify: bool = False) -> list[dict]:
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path) if verify else None,
        }
        for path in sorted(directory.glob("part-*.parquet"))
    ]


def extract(root: Path, index: int, workers: int) -> dict | None:
    plan = load_plan(root)
    if index >= len(plan["shards"]):
        print("No patches assigned to this array index", flush=True)
        return None
    shard = plan["shards"][index]
    catalog_path, manifest_path = _shard_inputs(root, shard)
    work = root / "shards" / f"{index:05d}"
    work.mkdir(parents=True, exist_ok=True)
    parquet = work / "parquet"
    plan_sha = sha256_file(root / "inputs/plan.json")
    with FileLock(str(work / ".extract.lock"), timeout=0):
        receipt = work / "validation.json"
        if receipt.exists():
            report = json.loads(receipt.read_text())
            if report["plan_sha256"] != plan_sha or report["files"] != _file_inventory(
                parquet, verify=True
            ):
                raise ValueError("completed shard was modified; do not mix artifacts")
            print(
                f"Verified completed shard {index}: {report['parquet_rows']} objects",
                flush=True,
            )
            return report
        started = time.monotonic()
        result = build.main(
            [
                "--catalog",
                str(catalog_path),
                "--manifest",
                str(manifest_path),
                "--scratch-dir",
                str(parquet),
                "--skip-ingest",
                "--verify-checksums",
                "--patch-workers",
                str(workers),
                "--objects-per-shard",
                str(plan["objects_per_file"]),
            ]
        )
        if result:
            raise RuntimeError(f"extraction failed for shard {index}")
        extraction_seconds = time.monotonic() - started
        catalog = read_catalog(catalog_path)
        columns = validate_catalog(catalog)
        manifest = build.load_manifest(str(manifest_path))
        ids = [str(value) for value in catalog[columns["object_id"]]]
        presence = {
            oid: np.array(
                [
                    (int(row[columns["tract"]]), int(row[columns["patch"]]), band)
                    in manifest
                    for band in BANDS
                ]
            )
            for oid, row in zip(ids, catalog)
        }
        report = {
            "status": "PASS",
            "scope": "intermediate_parquet",
            "plan_sha256": plan_sha,
            "index": index,
            "catalog_rows": len(catalog),
            **validate._validate_coadds(catalog, columns, manifest),
            **validate._validate_parquet(str(parquet), set(ids), presence),
            "files": _file_inventory(parquet, verify=True),
            "extraction_seconds": extraction_seconds,
            "total_seconds": time.monotonic() - started,
        }
        _write_json(receipt, report)
        print(f"PASS shard {index}: {len(catalog)} objects", flush=True)
        return report


def gather(root: Path, workers: int, pixel_threshold: int) -> dict:
    """Ingest only validated shards; verify final IDs and one image per partition."""
    plan = load_plan(root)
    plan_sha = sha256_file(root / "inputs/plan.json")
    with FileLock(str(root / ".gather.lock"), timeout=0):
        gather_started = time.monotonic()
        final = root / "validation_report.json"
        if final.exists():
            report = json.loads(final.read_text())
            if report["plan_sha256"] != plan_sha:
                raise ValueError("final report belongs to a different snapshot")
            print(f"Already gathered: {report['hats_path']}", flush=True)
            return report
        expected_ids, reports = set(), []
        links = root / "ingest_input"
        links.mkdir(exist_ok=True)
        expected_links = set()
        # Hash intermediate files in parallel before passing them to HATS. This
        # catches same-size corruption since the per-shard validation receipt.
        directories = [
            root / "shards" / f"{shard['index']:05d}" / "parquet"
            for shard in plan["shards"]
        ]
        print(
            f"[gather 1/3] verifying checksums of {len(directories)} shards", flush=True
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            inventories = list(
                pool.map(
                    lambda directory: _file_inventory(directory, verify=True),
                    directories,
                )
            )
        for shard, actual in zip(plan["shards"], inventories):
            catalog, _ = _shard_inputs(root, shard)
            id_name = (
                "objectId"
                if "objectId" in pq.read_schema(catalog).names
                else "object_id"
            )
            ids = set(
                map(str, pq.read_table(catalog, columns=[id_name])[id_name].to_pylist())
            )
            if len(ids) != shard["objects"] or expected_ids & ids:
                raise ValueError("snapshot shards have missing or duplicate object IDs")
            expected_ids.update(ids)
            work = root / "shards" / f"{shard['index']:05d}"
            report = json.loads((work / "validation.json").read_text())
            if (
                report["status"] != "PASS"
                or report["plan_sha256"] != plan_sha
                or report["parquet_rows"] != shard["objects"]
            ):
                raise ValueError(f"unvalidated shard: {work}")
            if report["files"] != actual:
                raise ValueError(f"validated shard files changed: {work}")
            for item in actual:
                path = (work / "parquet" / item["name"]).resolve()
                link = links / item["name"]
                expected_links.add(link.name)
                if link.is_symlink():
                    if link.resolve() != path:
                        raise ValueError(f"wrong ingest link: {link}")
                else:
                    link.symlink_to(path)
            reports.append(report)
            if len(reports) % 16 == 0 or len(reports) == len(plan["shards"]):
                print(
                    f"[gather] verified {len(reports)}/{len(plan['shards'])} receipts",
                    flush=True,
                )
        if {path.name for path in links.iterdir()} != expected_links:
            raise ValueError("unexpected files in ingest_input")
        if len(expected_ids) != plan["objects"]:
            raise ValueError("snapshot object count mismatch")
        # Limit each worker's decoded image payload; the common MMU writer also
        # creates the standard 10 arcsec margin and collection metadata.
        from dask.distributed import Client, LocalCluster
        from distributed.system import MEMORY_LIMIT

        from mmu.hats_import import write_hats_from_parquet_dir

        started = time.monotonic()
        # Dask's per-worker "auto" uses CPU count, not the requested number of
        # workers. Divide the node/cgroup budget explicitly and leave 30% free.
        worker_memory = int(0.7 * MEMORY_LIMIT / workers)
        print(
            f"[gather 2/3] HATS ingestion: {workers} workers, {worker_memory / 2**30:.1f} GiB each",
            flush=True,
        )
        with (
            LocalCluster(
                n_workers=workers,
                threads_per_worker=1,
                memory_limit=worker_memory,
                dashboard_address=None,
            ) as cluster,
            Client(cluster) as client,
        ):
            hats_path = Path(
                write_hats_from_parquet_dir(
                    str(links),
                    output_path=str(root / "hats/lsst_dp2"),
                    catalog_name="lsst_dp2",
                    pixel_threshold=pixel_threshold,
                    chunksize=64,
                    n_workers=workers,
                    client=client,
                )
            )
        hats_rows, properties = validate._hats_row_count(str(root / "hats"))
        found_ids, sampled = set(), 0
        partitions = sorted((hats_path / "dataset").rglob("*.parquet"))
        print(
            f"[gather 3/3] checking IDs and reading images in {len(partitions)} HATS partitions",
            flush=True,
        )
        for partition_index, path in enumerate(partitions, 1):
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                columns=["object_id"], batch_size=8192
            ):
                values = batch["object_id"].to_pylist()
                if len(set(values)) != len(values) or found_ids.intersection(values):
                    raise ValueError(f"duplicate HATS IDs: {path}")
                found_ids.update(values)
            first = next(
                parquet_file.iter_batches(columns=["object_id", "image"], batch_size=1),
                None,
            )
            if first is not None:
                validate._validate_image(
                    first["object_id"][0].as_py(),
                    validate.image_from_arrow(first["image"][0]),
                )
                sampled += 1
            if partition_index % 100 == 0 or partition_index == len(partitions):
                print(
                    f"[HATS readback] {partition_index}/{len(partitions)} partitions; {len(found_ids)} IDs",
                    flush=True,
                )
        if found_ids != expected_ids or hats_rows != len(expected_ids):
            raise ValueError("HATS IDs or metadata count differ from snapshot")
        report = {
            "status": "PASS",
            "plan_sha256": plan_sha,
            "hats_path": str(hats_path),
            "catalog_rows": plan["objects"],
            "parquet_rows": sum(r["parquet_rows"] for r in reports),
            "hats_rows": hats_rows,
            "unique_object_ids": len(found_ids),
            "hats_properties": properties,
            "hats_image_records_checked": sampled,
            "validation_scope": "all intermediate images; all final IDs; one final image per partition",
            "manifest_status_counts": plan["manifest_status_counts"],
            "min_clean_fraction": min(r["min_clean_fraction"] for r in reports),
            "max_clean_fraction": max(r["max_clean_fraction"] for r in reports),
            "gather_seconds": time.monotonic() - gather_started,
            "hats_and_readback_seconds": time.monotonic() - started,
            "sum_shard_seconds": sum(r["total_seconds"] for r in reports),
            "max_shard_seconds": max(r["total_seconds"] for r in reports),
            "scientific_quality_certified": False,
        }
        for key in (
            "center_checks",
            "coadd_products",
            "full_band_rows",
            "no_clean_pixel_rows",
        ):
            report[key] = sum(r[key] for r in reports)
        for key in (
            "max_center_axis_offset_pix",
            "max_center_radial_offset_pix",
            "max_center_radial_offset_arcsec",
        ):
            report[key] = max(r[key] for r in reports)
        report["fits_units"] = [
            list(units)
            for units in sorted(
                {tuple(units) for r in reports for units in r["fits_units"]}
            )
        ]
        for key in (
            "band_present_counts",
            "psf_valid_counts",
            "psf_fwhm_valid_counts",
            "present_band_no_clean_pixel_counts",
            "central_5x5_valid_counts",
        ):
            report[key] = {band: sum(r[key][band] for r in reports) for band in BANDS}
        for key in ("band_present", "psf_valid", "psf_fwhm_valid"):
            report[f"{key}_fractions"] = {
                band: report[f"{key}_counts"][band] / hats_rows for band in BANDS
            }
        _write_json(final, report)
        print(json.dumps(report, indent=2), flush=True)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source-run", type=Path, required=True)
    prep.add_argument("--shards", type=int, default=128)
    scatter = sub.add_parser("extract")
    scatter.add_argument("--index", type=int, required=True)
    scatter.add_argument("--workers", type=int, default=4)
    collect = sub.add_parser("gather")
    collect.add_argument("--workers", type=int, default=16)
    collect.add_argument("--pixel-threshold", type=int, default=256)
    for command in (prep, scatter, collect):
        command.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args(argv)
    for name in ("shards", "workers", "pixel_threshold"):
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if hasattr(args, "index") and args.index < 0:
        parser.error("index must be nonnegative")
    try:
        root = args.output_run.resolve()
        if args.stage == "prepare":
            prepare(args.source_run, root, args.shards)
        elif args.stage == "extract":
            extract(root, args.index, args.workers)
        else:
            gather(root, args.workers, args.pixel_threshold)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
