"""Mirror complete DP2 deep-coadd exposures for catalogued patches.

One SQLite manifest row represents one immutable ``(tract, patch, band)``
task. Downloads use SIA for exact product discovery and the DataLink ``#this``
record for the complete image, variance, integer mask, PSF, and provenance.
Completed files are checksummed and retries resume without re-downloading them.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from astropy.io import fits

from scripts.lsst_dp2.common import (
    BANDS,
    EFFECTIVE_WAVELENGTH_M,
    coadd_path,
    mask_plane_mapping,
    read_catalog,
    sha256_file,
    validate_catalog,
)

DEFAULT_SIA_URL = "https://data.lsst.cloud/api/sia/dp2/query"
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class Task:
    tract: int
    patch: int
    band: str
    ra: float
    dec: float
    output_path: str
    attempts: int


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def connect_manifest(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS coadds (
            tract INTEGER NOT NULL,
            patch INTEGER NOT NULL,
            band TEXT NOT NULL,
            ra REAL NOT NULL,
            dec REAL NOT NULL,
            output_path TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            bytes INTEGER,
            sha256 TEXT,
            dataset_id TEXT,
            datalink_url TEXT,
            access_url TEXT,
            s_resolution REAL,
            mask_planes TEXT,
            error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (tract, patch, band)
        )
        """
    )
    columns = {row[1] for row in con.execute("PRAGMA table_info(coadds)")}
    if "datalink_url" not in columns:
        con.execute("ALTER TABLE coadds ADD COLUMN datalink_url TEXT")
    con.commit()
    return con


def initialize_tasks(con: sqlite3.Connection, catalog_path: str, mirror_root: str) -> int:
    catalog = read_catalog(catalog_path)
    columns = validate_catalog(catalog)
    tract_col = columns["tract"]
    patch_col = columns["patch"]
    ra_col = columns["ra"]
    dec_col = columns["dec"]
    seen: set[tuple[int, int]] = set()
    rows = []
    for row in catalog:
        key = (int(row[tract_col]), int(row[patch_col]))
        if key in seen:
            continue
        seen.add(key)
        for band in BANDS:
            rows.append((
                key[0], key[1], band, float(row[ra_col]), float(row[dec_col]),
                str(coadd_path(mirror_root, key[0], key[1], band)), utcnow(),
            ))
    con.executemany(
        """
        INSERT INTO coadds (tract, patch, band, ra, dec, output_path, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tract, patch, band) DO UPDATE SET
            ra=excluded.ra, dec=excluded.dec, output_path=excluded.output_path
        """,
        rows,
    )
    con.commit()
    return len(rows)


def pending_tasks(con: sqlite3.Connection, max_attempts: int) -> list[Task]:
    completed = con.execute(
        "SELECT tract, patch, band, output_path, bytes, mask_planes "
        "FROM coadds WHERE status='complete'"
    ).fetchall()
    for tract, patch, band, output_path, expected_bytes, planes in completed:
        path = Path(output_path)
        if not path.is_file() or (
            expected_bytes is not None and path.stat().st_size != expected_bytes
        ):
            con.execute(
                "UPDATE coadds SET status='pending', error='completed file missing or truncated', "
                "updated_at=? WHERE tract=? AND patch=? AND band=?",
                (utcnow(), tract, patch, band),
            )
        elif not planes:
            con.execute(
                "UPDATE coadds SET mask_planes=?, updated_at=? "
                "WHERE tract=? AND patch=? AND band=?",
                (",".join(mask_plane_names(output_path)), utcnow(), tract, patch, band),
            )
    con.commit()
    rows = con.execute(
        """
        SELECT tract, patch, band, ra, dec, output_path, attempts
        FROM coadds
        WHERE status != 'complete' AND attempts < ?
        ORDER BY tract, patch, band
        """,
        (max_attempts,),
    ).fetchall()
    return [Task(*row) for row in rows]


def select_sia_record(table, tract: int, patch: int, band: str):
    """Select the exact DP2 image rather than trusting SIA result order."""
    matches = []
    for index, row in enumerate(table):
        row_tract = int(row["lsst_tract"])
        row_patch = int(row["lsst_patch"])
        row_band = str(row["lsst_band"]).strip()
        if (row_tract, row_patch, row_band) == (tract, patch, band):
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one SIA result for {(tract, patch, band)}, found {len(matches)}"
        )
    return matches[0]


def mask_plane_names(path: str) -> list[str]:
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if str(hdu.header.get("EXTNAME", "")).upper() == "MASK":
                return sorted(mask_plane_mapping(hdu.header))
    return []


def validate_maskedimage(path: str) -> list[str]:
    names: set[str] = set()
    with fits.open(path, memmap=False) as hdul:
        hdul.verify("exception")
        for hdu in hdul:
            extname = str(hdu.header.get("EXTNAME", "")).upper()
            if extname:
                names.add(extname)
    aliases = {
        "image": {"IMAGE", "SCI", "SCIENCE"},
        "mask": {"MASK"},
        "variance": {"VARIANCE", "VAR"},
    }
    missing = [label for label, choices in aliases.items() if not names.intersection(choices)]
    if missing:
        raise RuntimeError(f"downloaded FITS lacks planes {missing}; extensions={sorted(names)}")
    return mask_plane_names(path)


def select_full_product_url(datalink) -> str:
    """Return the unique DataLink primary-product URL (semantics ``#this``)."""
    urls = []
    for row in datalink:
        semantics = str(row["semantics"]).strip()
        if semantics == "#this" or semantics.endswith("#this"):
            url = str(row["access_url"]).strip()
            if url and url.lower() != "none":
                urls.append(url)
    if len(urls) != 1:
        raise RuntimeError(f"expected one DataLink #this product, found {len(urls)}")
    return urls[0]


def _clients(token: str, sia_url: str):
    if not hasattr(_THREAD_LOCAL, "sia"):
        try:
            import pyvo
            import requests
        except ImportError as exc:  # pragma: no cover - Rubin runtime only
            raise RuntimeError("install pyvo and requests to download DP2") from exc
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {token}"
        _THREAD_LOCAL.session = session
        _THREAD_LOCAL.sia = pyvo.dal.SIA2Service(
            sia_url, session=session, check_baseurl=False
        )
    return _THREAD_LOCAL.sia, _THREAD_LOCAL.session


def _optional_float(row, name: str) -> float | None:
    try:
        raw_value = row[name]
    except (KeyError, TypeError, ValueError):
        return None
    if np.ma.is_masked(raw_value):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def download_one(task: Task, token: str, sia_url: str) -> dict:
    from pyvo.dal.adhoc import DatalinkResults

    sia, session = _clients(token, sia_url)
    results = sia.search(
        pos=(task.ra, task.dec, 0.3),
        calib_level=3,
        dpsubtype="lsst.deep_coadd",
        band=EFFECTIVE_WAVELENGTH_M[task.band],
    )
    table = results.to_table()
    index = select_sia_record(table, task.tract, task.patch, task.band)
    row = table[index]
    datalink_url = str(row["access_url"])
    dataset_id = str(row["obs_publisher_did"])
    dl_result = DatalinkResults.from_result_url(datalink_url, session=session)
    access_url = select_full_product_url(dl_result)

    output = Path(task.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    try:
        with session.get(access_url, stream=True, timeout=(30, 600)) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with open(partial, "wb") as handle:
                shutil.copyfileobj(response.raw, handle, length=8 << 20)
                handle.flush()
                os.fsync(handle.fileno())
        planes = validate_maskedimage(str(partial))
        digest = sha256_file(partial)
        os.replace(partial, output)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "task": task,
        "bytes": output.stat().st_size,
        "sha256": digest,
        "dataset_id": dataset_id,
        "datalink_url": datalink_url,
        "access_url": access_url,
        "s_resolution": _optional_float(row, "s_resolution"),
        "mask_planes": ",".join(planes),
    }


def download_after_delay(
    task: Task,
    token: str,
    sia_url: str,
    retry_base_seconds: float,
) -> dict:
    if task.attempts:
        time.sleep(min(retry_base_seconds * (2 ** task.attempts), 60.0))
    print(
        f"START tract={task.tract} patch={task.patch} band={task.band} "
        f"attempt={task.attempts + 1}",
        flush=True,
    )
    return download_one(task, token, sia_url)


def print_status(con: sqlite3.Connection) -> None:
    counts = dict(
        con.execute(
            "SELECT status, COUNT(*) FROM coadds GROUP BY status ORDER BY status"
        ).fetchall()
    )
    print("Manifest status: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    failures = con.execute(
        """
        SELECT tract, patch, band, attempts, error
        FROM coadds
        WHERE status = 'failed'
        ORDER BY updated_at DESC
        LIMIT 20
        """
    ).fetchall()
    for tract, patch, band, attempts, error in failures:
        print(f"FAILED {tract}/{patch}/{band} attempts={attempts}: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--mirror-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--sia-url", default=DEFAULT_SIA_URL)
    parser.add_argument("--token-env", default="RSP_TOKEN")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args(argv)

    con = connect_manifest(args.manifest)
    initialized = initialize_tasks(con, args.catalog, args.mirror_root)
    if args.status_only:
        print(f"Manifest has {initialized} tasks")
        print_status(con)
        con.close()
        return 0

    token = os.environ.get(args.token_env)
    if not token:
        print(f"ERROR: environment variable {args.token_env} is not set", file=sys.stderr)
        con.close()
        return 2
    tasks = pending_tasks(con, args.max_attempts)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    print(f"Manifest has {initialized} tasks; {len(tasks)} scheduled", flush=True)

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for task in tasks:
            con.execute(
                "UPDATE coadds SET status='running', attempts=attempts+1, updated_at=? "
                "WHERE tract=? AND patch=? AND band=?",
                (utcnow(), task.tract, task.patch, task.band),
            )
            con.commit()
            futures[
                pool.submit(
                    download_after_delay,
                    task,
                    token,
                    args.sia_url,
                    args.retry_base_seconds,
                )
            ] = task

        for done, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                result = future.result()
                con.execute(
                    """
                    UPDATE coadds SET status='complete', bytes=?, sha256=?, dataset_id=?,
                        datalink_url=?, access_url=?, s_resolution=?, mask_planes=?,
                        error=NULL, updated_at=?
                    WHERE tract=? AND patch=? AND band=?
                    """,
                    (
                        result["bytes"], result["sha256"], result["dataset_id"],
                        result["datalink_url"], result["access_url"], result["s_resolution"],
                        result["mask_planes"], utcnow(), task.tract, task.patch, task.band,
                    ),
                )
                size_mib = result["bytes"] / (1024 * 1024)
                print(
                    f"DONE [{done}/{len(futures)}] tract={task.tract} "
                    f"patch={task.patch} band={task.band} size={size_mib:.1f} MiB",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                con.execute(
                    "UPDATE coadds SET status='failed', error=?, updated_at=? "
                    "WHERE tract=? AND patch=? AND band=?",
                    (f"{type(exc).__name__}: {exc}", utcnow(), task.tract, task.patch, task.band),
                )
                print(
                    f"FAILED [{done}/{len(futures)}] tract={task.tract} "
                    f"patch={task.patch} band={task.band}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            con.commit()
    print(f"Finished {len(futures)} tasks; failed={failed}", flush=True)
    con.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
