"""Query a spatially bounded DP2 Object parent catalog through TAP.

The command discovers the live DP2 schema before constructing the query, so
optional PSF-moment and quality columns are requested only when they exist.
Authentication uses ``RSP_TOKEN`` unless ``--token-env`` selects another
environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from astropy.table import Table

from scripts.lsst_dp2.common import BANDS

DEFAULT_TAP_URL = "https://data.lsst.cloud/api/tap"


def _authenticated_tap(url: str, token: str):
    try:
        import pyvo
        import requests
    except ImportError as exc:  # pragma: no cover - exercised in Rubin environment
        raise RuntimeError("install pyvo and requests to query DP2") from exc
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return pyvo.dal.TAPService(url, session=session)


def discover_columns(service, table_name: str = "dp2.Object") -> set[str]:
    query = f"""
        SELECT column_name
        FROM TAP_SCHEMA.columns
        WHERE table_name = '{table_name}'
    """
    return {str(value) for value in service.search(query).to_table()["column_name"]}


def select_columns(available: set[str]) -> list[str]:
    required = ["objectId", "coord_ra", "coord_dec", "tract", "patch"]
    missing = [name for name in required if name not in available]
    if missing:
        raise ValueError(f"DP2 Object schema is missing required columns: {missing}")

    wanted = required + ["refBand", "refExtendedness", "detect_isIsolated"]
    for band in BANDS:
        wanted.extend([
            f"{band}_psfFlux", f"{band}_psfFluxErr",
            f"{band}_cModelFlux", f"{band}_cModelFluxErr",
            f"{band}_ixx", f"{band}_iyy", f"{band}_ixy",
            f"{band}_ixxPSF", f"{band}_iyyPSF", f"{band}_ixyPSF",
            f"{band}_pixelFlags_inexact_psfCenter",
        ])
    return [name for name in wanted if name in available]


def spatial_predicate(args: argparse.Namespace) -> str:
    if args.polygon:
        values = [float(value) for value in args.polygon.replace(",", " ").split()]
        if len(values) < 6 or len(values) % 2:
            raise ValueError("--polygon requires at least three RA/Dec pairs")
        coords = ", ".join(str(value) for value in values)
        return (
            "CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
            f"POLYGON('ICRS', {coords})) = 1"
        )
    if args.ra is None or args.dec is None or args.radius_deg is None:
        raise ValueError("provide either --polygon or --ra/--dec/--radius-deg")
    return (
        "CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
        f"CIRCLE('ICRS', {args.ra}, {args.dec}, {args.radius_deg})) = 1"
    )


def build_query(
    columns: list[str],
    predicate: str,
    where: str | None,
    limit: int | None = None,
) -> str:
    clauses = [predicate]
    if where:
        clauses.append(f"({where})")
    top = f"TOP {limit} " if limit is not None else ""
    return (
        f"SELECT {top}{', '.join(columns)}\n"
        "FROM dp2.Object AS obj\n"
        f"WHERE {' AND '.join(clauses)}"
    )


def write_catalog(table: Table, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.write(path, format="parquet", overwrite=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ra", type=float)
    parser.add_argument("--dec", type=float)
    parser.add_argument("--radius-deg", type=float)
    parser.add_argument("--polygon", help="Whitespace/comma-separated RA Dec vertex pairs")
    parser.add_argument("--where", help="Additional ADQL selection applied after the spatial cut")
    parser.add_argument("--limit", type=int, help="Maximum number of catalog rows")
    parser.add_argument("--tap-url", default=DEFAULT_TAP_URL)
    parser.add_argument("--token-env", default="RSP_TOKEN")
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    token = os.environ.get(args.token_env)
    if not token:
        print(f"ERROR: environment variable {args.token_env} is not set", file=sys.stderr)
        return 2
    try:
        service = _authenticated_tap(args.tap_url, token)
        columns = select_columns(discover_columns(service))
        query = build_query(columns, spatial_predicate(args), args.where, args.limit)
        job = service.submit_job(query)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
        if job.phase != "COMPLETED":
            raise RuntimeError(f"TAP job ended in phase {job.phase}")
        table = job.fetch_result().to_table()
        write_catalog(table, args.output)
        Path(args.output + ".adql").write_text(query + "\n", encoding="ascii")
        print(f"Wrote {len(table)} objects to {args.output}", flush=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
