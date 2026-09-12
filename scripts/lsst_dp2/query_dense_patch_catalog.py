"""Build a DP2 parent catalog from the densest patches for a fixed selection.

The command first counts eligible objects per ``(tract, patch)`` with one
aggregate TAP query. It then selects the highest-density patches and retrieves
all eligible objects in those patches in restartable TAP batches. This is meant
for network-efficient qualification samples: each mirrored coadd is reused for
as many selected objects as the catalog contains.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyvo
import requests
from astropy.table import Column, Table

from scripts.lsst_dp2.query_catalog import (
    DEFAULT_TAP_URL,
    _authenticated_tap,
    discover_columns,
    select_columns,
    write_catalog,
)

DEFAULT_WHERE = (
    "i_cModelMag < 21 "
    "AND i_extendedness = 1 "
    "AND griz_model_extendedness >= 0.8 "
    "AND sersic_no_data_flag = 0 "
    "AND sersic_unknown_flag = 0 "
    "AND sersic_reff_major > 0.6"
)


def patch_predicate(patches: list[tuple[int, int]]) -> str:
    """Return a compact deterministic ADQL predicate for tract/patch pairs."""
    by_tract: dict[int, list[int]] = defaultdict(list)
    for tract, patch in patches:
        by_tract[int(tract)].append(int(patch))
    clauses = []
    for tract in sorted(by_tract):
        patch_values = ", ".join(str(value) for value in sorted(by_tract[tract]))
        clauses.append(f"(tract = {tract} AND patch IN ({patch_values}))")
    if not clauses:
        raise ValueError("at least one patch is required")
    return "(" + " OR ".join(clauses) + ")"


def rank_inventory(table: Table, patch_limit: int) -> Table:
    """Sort inventory rows by decreasing population with stable tie breaking."""
    names = {name.lower(): name for name in table.colnames}
    required = {"tract", "patch", "n_objects"}
    missing = sorted(required - names.keys())
    if missing:
        raise ValueError(f"inventory is missing columns: {missing}")
    counts = np.asarray(table[names["n_objects"]], dtype=np.int64)
    tracts = np.asarray(table[names["tract"]], dtype=np.int64)
    patches = np.asarray(table[names["patch"]], dtype=np.int64)
    if len(table) == 0 or np.any(counts <= 0):
        raise ValueError("inventory must contain positive object counts")
    order = np.lexsort((patches, tracts, -counts))
    selected = table[order[: min(patch_limit, len(order))]]
    selected.add_column(Column(np.arange(1, len(selected) + 1)), name="density_rank")
    return selected


def retry_dal_call(operation, label: str, attempts: int, base_seconds: float):
    """Retry transient TAP/proxy failures with capped exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except pyvo.dal.DALQueryError:
            raise
        except (pyvo.dal.DALAccessError, requests.RequestException, OSError) as exc:
            if attempt == attempts:
                raise
            delay = min(300.0, base_seconds * 2 ** (attempt - 1))
            print(
                f"RETRY {label} after {type(exc).__name__}: {exc}; "
                f"attempt={attempt + 1}/{attempts} delay={delay:g}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def run_async_query(
    service,
    query: str,
    maxrec: int,
    attempts: int,
    base_seconds: float,
) -> Table:
    """Run one TAP job and reject truncated results."""

    def run_once():
        job = service.submit_job(query, maxrec=maxrec)
        print(f"TAP job: {job.url}", flush=True)
        try:
            job.run()
            job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
            job.raise_if_error()
            if job.phase != "COMPLETED":
                raise RuntimeError(f"TAP job ended in phase {job.phase}")
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=pyvo.dal.DALOverflowWarning)
                return job.fetch_result().to_table()
        finally:
            try:
                job.delete()
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not delete TAP job: {exc}", file=sys.stderr)

    return retry_dal_call(
        run_once,
        "TAP query",
        attempts=attempts,
        base_seconds=base_seconds,
    )


def _write_plan(path: Path, plan: dict) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != plan:
            raise RuntimeError(
                f"existing query plan differs from this request: {path}; "
                "use a new output path"
            )
        return
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--patch-limit", type=int, default=6000)
    parser.add_argument("--batch-patches", type=int, default=200)
    parser.add_argument("--where", default=DEFAULT_WHERE)
    parser.add_argument("--tap-url", default=DEFAULT_TAP_URL)
    parser.add_argument("--token-env", default="RSP_TOKEN")
    parser.add_argument("--tap-attempts", type=int, default=8)
    parser.add_argument("--retry-base-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    if (
        args.patch_limit <= 0
        or args.batch_patches <= 0
        or args.tap_attempts <= 0
        or args.retry_base_seconds <= 0
    ):
        parser.error("limits, attempts, and retry delay must be positive")
    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"environment variable {args.token_env} is not set")

    root = args.output.parent
    work = root / f".{args.output.stem}.dense_query"
    batches = work / "batches"
    root.mkdir(parents=True, exist_ok=True)
    batches.mkdir(parents=True, exist_ok=True)
    plan = {
        "where": args.where,
        "patch_limit": args.patch_limit,
        "batch_patches": args.batch_patches,
        "tap_url": args.tap_url,
        "tap_attempts": args.tap_attempts,
        "retry_base_seconds": args.retry_base_seconds,
    }
    _write_plan(work / "plan.json", plan)

    service = _authenticated_tap(args.tap_url, token)
    try:
        available = retry_dal_call(
            lambda: discover_columns(service),
            "schema discovery",
            attempts=args.tap_attempts,
            base_seconds=args.retry_base_seconds,
        )
        columns = select_columns(available)
        inventory_path = work / "patch_inventory.parquet"
        if inventory_path.exists():
            inventory = Table.read(inventory_path, format="parquet")
            print(f"Reusing patch inventory: {inventory_path}", flush=True)
        else:
            query = (
                "SELECT tract, patch, COUNT(*) AS n_objects\n"
                "FROM dp2.Object AS obj\n"
                f"WHERE {args.where}\n"
                "GROUP BY tract, patch"
            )
            (work / "inventory.adql").write_text(query + "\n", encoding="ascii")
            inventory = run_async_query(
                service,
                query,
                maxrec=2_000_000,
                attempts=args.tap_attempts,
                base_seconds=args.retry_base_seconds,
            )
            write_catalog(inventory, str(inventory_path))
            print(f"Wrote {len(inventory)} patch inventory rows", flush=True)

        selected = rank_inventory(inventory, args.patch_limit)
        write_catalog(selected, str(work / "selected_patches.parquet"))
        names = {name.lower(): name for name in selected.colnames}
        patch_pairs = list(
            zip(
                np.asarray(selected[names["tract"]], dtype=np.int64).tolist(),
                np.asarray(selected[names["patch"]], dtype=np.int64).tolist(),
                strict=True,
            )
        )
        expected_rows = int(
            np.asarray(selected[names["n_objects"]], dtype=np.int64).sum()
        )
        rank = {pair: index + 1 for index, pair in enumerate(patch_pairs)}
        population = {
            pair: int(value)
            for pair, value in zip(
                patch_pairs,
                np.asarray(selected[names["n_objects"]], dtype=np.int64),
                strict=True,
            )
        }

        batch_paths = []
        for start in range(0, len(patch_pairs), args.batch_patches):
            subset = patch_pairs[start : start + args.batch_patches]
            batch_index = start // args.batch_patches
            path = batches / f"batch-{batch_index:05d}.parquet"
            batch_paths.append(path)
            if path.exists():
                print(f"Reusing [{batch_index + 1}] {path.name}", flush=True)
                continue
            predicate = patch_predicate(subset)
            query = (
                f"SELECT {', '.join(columns)}\n"
                "FROM dp2.Object AS obj\n"
                f"WHERE {predicate} AND ({args.where})"
            )
            query_path = path.with_suffix(".adql")
            query_path.write_text(query + "\n", encoding="ascii")
            batch_expected = sum(population[pair] for pair in subset)
            table = run_async_query(
                service,
                query,
                maxrec=batch_expected + 1,
                attempts=args.tap_attempts,
                base_seconds=args.retry_base_seconds,
            )
            if len(table) != batch_expected:
                raise RuntimeError(
                    f"batch {batch_index} returned {len(table)}/{batch_expected} rows"
                )
            table.add_column(
                Column(
                    [rank[(int(row["tract"]), int(row["patch"]))] for row in table],
                    dtype=np.int32,
                ),
                name="dense_patch_rank",
            )
            table.add_column(
                Column(
                    [
                        population[(int(row["tract"]), int(row["patch"]))]
                        for row in table
                    ],
                    dtype=np.int32,
                ),
                name="dense_patch_n_objects",
            )
            write_catalog(table, str(path))
            print(
                f"Wrote batch {batch_index + 1}/{len(range(0, len(patch_pairs), args.batch_patches))}: "
                f"{len(table)} objects",
                flush=True,
            )

        arrow_tables = [pq.read_table(path) for path in batch_paths]
        catalog = pa.concat_tables(arrow_tables, promote_options="default")
        if catalog.num_rows != expected_rows:
            raise RuntimeError(
                f"combined catalog has {catalog.num_rows}/{expected_rows} rows"
            )
        ids = catalog.column("objectId").combine_chunks()
        if len(set(ids.to_pylist())) != catalog.num_rows:
            raise RuntimeError("combined catalog contains duplicate objectId values")
        sort_indices = pc.sort_indices(
            catalog, sort_keys=[("dense_patch_rank", "ascending"), ("objectId", "ascending")]
        )
        catalog = pc.take(catalog, sort_indices)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        pq.write_table(catalog, temporary)
        os.replace(temporary, args.output)

        counts = np.asarray(selected[names["n_objects"]], dtype=np.int64)
        report = {
            **plan,
            "status": "PASS",
            "objects": expected_rows,
            "patches": len(patch_pairs),
            "objects_per_patch_mean": float(counts.mean()),
            "objects_per_patch_min": int(counts.min()),
            "objects_per_patch_max": int(counts.max()),
            "download_tasks_six_bands": 6 * len(patch_pairs),
            "request_budget_hours_three_requests_at_59_per_minute": (
                6 * len(patch_pairs) * 3 / 59 / 60
            ),
            "estimated_coadds_gib_at_31_mib_each": (
                6 * len(patch_pairs) * 31 / 1024
            ),
        }
        report_path = Path(str(args.output) + ".selection.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Wrote {expected_rows} objects to {args.output}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
