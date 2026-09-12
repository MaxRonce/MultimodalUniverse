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
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
MANIFEST_SCHEMA_VERSION = 2
_THREAD_LOCAL = threading.local()


class ProductUnavailableError(RuntimeError):
    """The requested patch-band has no DP2 SIA product."""


class RequestRateLimiter:
    """Serialize request starts to a process-wide maximum rate."""

    def __init__(self, requests_per_minute: float):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._next_request = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_request - now
            if delay > 0:
                time.sleep(delay)
                now = time.monotonic()
            self._next_request = max(now, self._next_request) + self.interval


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
            priority INTEGER NOT NULL DEFAULT 2147483647,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (tract, patch, band)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    columns = {row[1] for row in con.execute("PRAGMA table_info(coadds)")}
    if "datalink_url" not in columns:
        con.execute("ALTER TABLE coadds ADD COLUMN datalink_url TEXT")
    if "priority" not in columns:
        con.execute(
            "ALTER TABLE coadds ADD COLUMN priority INTEGER NOT NULL "
            "DEFAULT 2147483647"
        )
    con.commit()
    return con


def initialize_tasks(
    con: sqlite3.Connection, catalog_path: str, mirror_root: str
) -> int:
    """Bind a manifest to one catalog and populate its immutable task set."""
    catalog = read_catalog(catalog_path)
    columns = validate_catalog(catalog)
    tract_col = columns["tract"]
    patch_col = columns["patch"]
    ra_col = columns["ra"]
    dec_col = columns["dec"]
    priority_col = "dense_patch_rank" if "dense_patch_rank" in catalog.colnames else None
    seen: set[tuple[int, int]] = set()
    rows = []
    for row in catalog:
        key = (int(row[tract_col]), int(row[patch_col]))
        if key in seen:
            continue
        seen.add(key)
        priority = int(row[priority_col]) if priority_col else len(seen)
        for band in BANDS:
            rows.append(
                (
                    key[0],
                    key[1],
                    band,
                    float(row[ra_col]),
                    float(row[dec_col]),
                    str(coadd_path(mirror_root, key[0], key[1], band)),
                    priority,
                    utcnow(),
                )
            )
    expected_keys = {(row[0], row[1], row[2]) for row in rows}
    expected_paths = {(row[0], row[1], row[2]): row[5] for row in rows}
    metadata = dict(con.execute("SELECT key, value FROM metadata").fetchall())
    identity = {
        "schema_version": str(MANIFEST_SCHEMA_VERSION),
        "catalog_path": str(Path(catalog_path).resolve()),
        "catalog_sha256": sha256_file(catalog_path),
        "mirror_root": str(Path(mirror_root).resolve()),
    }

    if metadata:
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in identity.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "manifest identity differs from requested inputs; use a new manifest "
                f"or remove it explicitly: {mismatches}"
            )
    else:
        existing = {
            (int(tract), int(patch), str(band)): str(output_path)
            for tract, patch, band, output_path in con.execute(
                "SELECT tract, patch, band, output_path FROM coadds"
            )
        }
        if existing and (
            set(existing) != expected_keys
            or any(existing[key] != expected_paths[key] for key in existing)
        ):
            raise RuntimeError(
                "legacy manifest tasks differ from the requested catalog or mirror; "
                "use a new manifest or remove it explicitly"
            )

    con.executemany(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        identity.items(),
    )
    con.executemany(
        """
        INSERT INTO coadds (
            tract, patch, band, ra, dec, output_path, priority, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tract, patch, band) DO UPDATE SET
            ra=excluded.ra, dec=excluded.dec, output_path=excluded.output_path,
            priority=excluded.priority
        """,
        rows,
    )
    con.commit()
    return len(rows)


def reset_failed_tasks(con: sqlite3.Connection) -> int:
    """Reset attempt counters for incomplete tasks after an operator decision."""
    cursor = con.execute(
        """
        UPDATE coadds
        SET status='pending', attempts=0, error=NULL, updated_at=?
        WHERE status NOT IN ('complete', 'unavailable')
        """,
        (utcnow(),),
    )
    con.commit()
    return cursor.rowcount


def recover_interrupted_tasks(con: sqlite3.Connection) -> int:
    """Return tasks left running by an interrupted process to the pending queue."""
    cursor = con.execute(
        """
        UPDATE coadds
        SET status='pending', attempts=MAX(attempts - 1, 0),
            error='recovered after interrupted downloader', updated_at=?
        WHERE status='running'
        """,
        (utcnow(),),
    )
    con.commit()
    return cursor.rowcount


def migrate_known_unavailable_tasks(con: sqlite3.Connection) -> int:
    """Promote legacy zero-result failures to explicit product unavailability."""
    cursor = con.execute(
        """
        UPDATE coadds
        SET status='unavailable', updated_at=?
        WHERE status='failed'
          AND error LIKE '%expected one SIA result for % found 0'
        """,
        (utcnow(),),
    )
    con.commit()
    return cursor.rowcount


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
        WHERE status NOT IN ('complete', 'unavailable') AND attempts < ?
        ORDER BY priority, tract, patch, band
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
    if not matches:
        raise ProductUnavailableError(f"no SIA product for {(tract, patch, band)}")
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
        "psf": {"PSF"},
        "archive metadata": {"JSON"},
    }
    missing = [
        label for label, choices in aliases.items() if not names.intersection(choices)
    ]
    if missing:
        raise RuntimeError(
            f"downloaded FITS lacks planes {missing}; extensions={sorted(names)}"
        )
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


def download_one(
    task: Task,
    token: str,
    sia_url: str,
    rate_limiter: RequestRateLimiter,
) -> dict:
    from pyvo.dal.adhoc import DatalinkResults

    sia, session = _clients(token, sia_url)
    rate_limiter.wait()
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
    rate_limiter.wait()
    dl_result = DatalinkResults.from_result_url(datalink_url, session=session)
    access_url = select_full_product_url(dl_result)

    output = Path(task.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    try:
        rate_limiter.wait()
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
    rate_limiter: RequestRateLimiter,
) -> dict:
    if task.attempts:
        time.sleep(min(retry_base_seconds * (2**task.attempts), 60.0))
    print(
        f"START tract={task.tract} patch={task.patch} band={task.band} "
        f"attempt={task.attempts + 1}",
        flush=True,
    )
    return download_one(task, token, sia_url, rate_limiter)


def status_counts(con: sqlite3.Connection) -> dict[str, int]:
    return dict(
        con.execute(
            "SELECT status, COUNT(*) FROM coadds GROUP BY status ORDER BY status"
        ).fetchall()
    )


def print_status(con: sqlite3.Connection) -> dict[str, int]:
    counts = status_counts(con)
    print(
        "Manifest status: "
        + ", ".join(f"{key}={value}" for key, value in counts.items())
    )
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
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog", required=True, help="DP2 Object catalog in Parquet"
    )
    parser.add_argument(
        "--mirror-root", required=True, help="destination for full coadd FITS"
    )
    parser.add_argument("--manifest", required=True, help="restartable SQLite manifest")
    parser.add_argument(
        "--workers", type=int, default=4, help="concurrent SIA downloads"
    )
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=50.0,
        help="global request-start limit across all workers (default: 50)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=5,
        help="maximum attempts per task across command reruns",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=5.0,
        help="backoff base applied to tasks attempted by earlier runs",
    )
    parser.add_argument(
        "--sia-url", default=DEFAULT_SIA_URL, help="Rubin DP2 SIA endpoint"
    )
    parser.add_argument(
        "--token-env",
        default="RSP_TOKEN",
        help="environment variable holding the RSP token",
    )
    parser.add_argument(
        "--task-limit",
        "--limit",
        dest="task_limit",
        type=int,
        default=None,
        help="process at most this many patch-band tasks (debugging only)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="print manifest state without authentication",
    )
    parser.add_argument(
        "--reset-failed",
        action="store_true",
        help="reset attempts for every incomplete task before downloading",
    )
    args = parser.parse_args(argv)

    if (
        args.workers <= 0
        or args.max_attempts <= 0
        or args.requests_per_minute <= 0
    ):
        parser.error(
            "--workers, --max-attempts, and --requests-per-minute must be positive"
        )
    if args.task_limit is not None and args.task_limit <= 0:
        parser.error("--task-limit must be positive")

    try:
        con = connect_manifest(args.manifest)
        initialized = initialize_tasks(con, args.catalog, args.mirror_root)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.status_only:
        print(f"Manifest has {initialized} tasks")
        print_status(con)
        con.close()
        return 0
    recovered = recover_interrupted_tasks(con)
    migrated = migrate_known_unavailable_tasks(con)
    if recovered:
        print(f"Recovered {recovered} interrupted tasks", flush=True)
    if migrated:
        print(f"Marked {migrated} known missing products unavailable", flush=True)
    if args.reset_failed:
        print(f"Reset {reset_failed_tasks(con)} incomplete tasks", flush=True)

    token = os.environ.get(args.token_env)
    if not token:
        print(
            f"ERROR: environment variable {args.token_env} is not set", file=sys.stderr
        )
        con.close()
        return 2
    tasks = pending_tasks(con, args.max_attempts)
    if args.task_limit is not None:
        tasks = tasks[: args.task_limit]
    print(
        f"Manifest has {initialized} tasks; {len(tasks)} scheduled; "
        f"request_limit={args.requests_per_minute:g}/min",
        flush=True,
    )

    failed = 0
    unavailable = 0
    processed = 0
    rate_limiter = RequestRateLimiter(args.requests_per_minute)

    def submit(pool: ThreadPoolExecutor, task: Task) -> Future:
        con.execute(
            "UPDATE coadds SET status='running', attempts=attempts+1, updated_at=? "
            "WHERE tract=? AND patch=? AND band=?",
            (utcnow(), task.tract, task.patch, task.band),
        )
        con.commit()
        return pool.submit(
            download_after_delay,
            task,
            token,
            args.sia_url,
            args.retry_base_seconds,
            rate_limiter,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        task_iterator = iter(tasks)
        futures: dict[Future, Task] = {}
        for task in task_iterator:
            futures[submit(pool, task)] = task
            if len(futures) == args.workers:
                break

        while futures:
            completed, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                task = futures.pop(future)
                processed += 1
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
                            result["datalink_url"], result["access_url"],
                            result["s_resolution"], result["mask_planes"], utcnow(),
                            task.tract, task.patch, task.band,
                        ),
                    )
                    size_mib = result["bytes"] / (1024 * 1024)
                    print(
                        f"DONE [{processed}/{len(tasks)}] tract={task.tract} "
                        f"patch={task.patch} band={task.band} size={size_mib:.1f} MiB",
                        flush=True,
                    )
                except ProductUnavailableError as exc:
                    unavailable += 1
                    con.execute(
                        "UPDATE coadds SET status='unavailable', error=?, updated_at=? "
                        "WHERE tract=? AND patch=? AND band=?",
                        (str(exc), utcnow(), task.tract, task.patch, task.band),
                    )
                    print(
                        f"UNAVAILABLE [{processed}/{len(tasks)}] tract={task.tract} "
                        f"patch={task.patch} band={task.band}: {exc}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    con.execute(
                        "UPDATE coadds SET status='failed', error=?, updated_at=? "
                        "WHERE tract=? AND patch=? AND band=?",
                        (
                            f"{type(exc).__name__}: {exc}", utcnow(),
                            task.tract, task.patch, task.band,
                        ),
                    )
                    print(
                        f"FAILED [{processed}/{len(tasks)}] tract={task.tract} "
                        f"patch={task.patch} band={task.band}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                con.commit()
                try:
                    next_task = next(task_iterator)
                except StopIteration:
                    continue
                futures[submit(pool, next_task)] = next_task
    counts = status_counts(con)
    incomplete = sum(
        value for key, value in counts.items() if key not in {"complete", "unavailable"}
    )
    print(
        f"Finished {processed} tasks; failed={failed}; unavailable={unavailable}; "
        f"incomplete={incomplete}",
        flush=True,
    )
    con.close()
    if args.task_limit is not None and incomplete:
        print(
            f"Task-limited run left {incomplete} manifest tasks for the next job",
            flush=True,
        )
    return 1 if failed or (incomplete and args.task_limit is None) else 0


if __name__ == "__main__":
    sys.exit(main())
