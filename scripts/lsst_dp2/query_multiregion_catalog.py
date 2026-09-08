"""Query and deterministically sample an equal number of DP2 objects per region."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.table import Column, Table, vstack

from scripts.lsst_dp2.query_catalog import (
    DEFAULT_TAP_URL,
    _authenticated_tap,
    build_query,
    discover_columns,
    select_columns,
    write_catalog,
)
from scripts.lsst_dp2.stratify_catalog import stable_score

DEFAULT_WHERE = (
    "i_cModelMag < 22 "
    "AND i_extendedness = 1 "
    "AND griz_model_extendedness >= 0.8 "
    "AND sersic_no_data_flag = 0 "
    "AND sersic_unknown_flag = 0"
)


@dataclass(frozen=True)
class Region:
    name: str
    ra: float
    dec: float
    radius_deg: float


DEFAULT_REGIONS = (
    Region("DDF_ELAIS_S1", 9.5, -44.0, 1.0),
    Region("DDF_ECDFS", 53.0, -28.1, 1.0),
    Region("DDF_EDFS_A", 59.2, -49.2, 1.0),
    Region("DDF_COSMOS", 150.1, 2.1, 1.0),
    Region("RUBIN_SV_225_-40", 225.0, -39.5, 1.0),
)


def parse_region(value: str) -> Region:
    """Parse ``NAME,RA,DEC,RADIUS_DEG``."""
    items = [item.strip() for item in value.split(",")]
    if len(items) != 4 or not items[0]:
        raise ValueError("region must be NAME,RA,DEC,RADIUS_DEG")
    region = Region(items[0], float(items[1]), float(items[2]), float(items[3]))
    if not 0 <= region.ra < 360 or not -90 <= region.dec <= 90:
        raise ValueError(f"invalid region coordinates: {region}")
    if region.radius_deg <= 0:
        raise ValueError(f"invalid region radius: {region.radius_deg}")
    return region


def region_predicate(region: Region) -> str:
    return (
        "CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
        f"CIRCLE('ICRS', {region.ra}, {region.dec}, {region.radius_deg})) = 1"
    )


def select_region_rows(
    table: Table,
    region: Region,
    per_region: int,
    seed: int,
    used_object_ids: set[str],
) -> tuple[Table, int]:
    """Select deterministic unique rows for one region."""
    object_ids = [str(value) for value in table["objectId"]]
    ranked = sorted(
        range(len(table)),
        key=lambda index: stable_score(object_ids[index], seed),
    )
    chosen = []
    overlap_skipped = 0
    for index in ranked:
        object_id = object_ids[index]
        if object_id in used_object_ids:
            overlap_skipped += 1
            continue
        chosen.append(index)
        used_object_ids.add(object_id)
        if len(chosen) == per_region:
            break
    if len(chosen) != per_region:
        raise RuntimeError(
            f"{region.name} has only {len(chosen)} unique eligible rows; "
            f"requested {per_region}"
        )
    selected = table[chosen]
    selected.add_column(
        Column(np.full(len(selected), region.name, dtype=f"U{len(region.name)}")),
        name="dp2_region",
    )
    return selected, overlap_skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-region", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--where", default=DEFAULT_WHERE)
    parser.add_argument(
        "--region",
        action="append",
        type=parse_region,
        help="repeat NAME,RA,DEC,RADIUS_DEG; defaults to five DP2 regions",
    )
    parser.add_argument("--tap-url", default=DEFAULT_TAP_URL)
    parser.add_argument("--token-env", default="RSP_TOKEN")
    args = parser.parse_args(argv)
    if args.per_region <= 0:
        parser.error("--per-region must be positive")
    regions = tuple(args.region or DEFAULT_REGIONS)
    if len({region.name for region in regions}) != len(regions):
        parser.error("region names must be unique")

    token = os.environ.get(args.token_env)
    if not token:
        print(f"ERROR: environment variable {args.token_env} is not set", file=sys.stderr)
        return 2

    try:
        service = _authenticated_tap(args.tap_url, token)
        columns = select_columns(discover_columns(service))
        selected_tables = []
        used_object_ids: set[str] = set()
        report_regions = []
        queries = {}
        for index, region in enumerate(regions, 1):
            query = build_query(columns, region_predicate(region), args.where)
            queries[region.name] = query
            print(f"[{index}/{len(regions)}] querying {region.name}", flush=True)
            job = service.submit_job(query)
            try:
                job.run()
                job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
                if job.phase != "COMPLETED":
                    raise RuntimeError(
                        f"TAP job for {region.name} ended in phase {job.phase}"
                    )
                pool = job.fetch_result().to_table()
            finally:
                try:
                    job.delete()
                except Exception as cleanup_error:  # noqa: BLE001
                    print(
                        f"WARNING: could not delete TAP job for {region.name}: "
                        f"{cleanup_error}",
                        file=sys.stderr,
                    )
            selected, overlap_skipped = select_region_rows(
                pool,
                region,
                args.per_region,
                args.seed,
                used_object_ids,
            )
            selected_tables.append(selected)
            report_regions.append(
                {
                    **asdict(region),
                    "eligible_rows": len(pool),
                    "selected_rows": len(selected),
                    "overlap_rows_skipped": overlap_skipped,
                }
            )
            print(
                f"[{index}/{len(regions)}] {region.name}: "
                f"eligible={len(pool)} selected={len(selected)}",
                flush=True,
            )

        output_table = vstack(selected_tables, metadata_conflicts="silent")
        output_table.sort(["tract", "patch", "objectId"])
        expected_rows = args.per_region * len(regions)
        if len(output_table) != expected_rows:
            raise RuntimeError(f"selected {len(output_table)}/{expected_rows} rows")
        output_ids = [str(value) for value in output_table["objectId"]]
        if len(set(output_ids)) != len(output_ids):
            raise RuntimeError("multi-region sample contains duplicate objectId values")

        write_catalog(output_table, args.output)
        report = {
            "status": "PASS",
            "output_rows": len(output_table),
            "unique_object_ids": len(set(output_ids)),
            "per_region": args.per_region,
            "seed": args.seed,
            "where": args.where,
            "regions": report_regions,
            "queries": queries,
        }
        report_path = Path(args.output + ".selection.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Wrote {len(output_table)} objects to {args.output}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
