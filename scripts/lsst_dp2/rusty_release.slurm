#!/bin/bash
#SBATCH --partition=gen
#SBATCH --job-name=lsst-dp2-release
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
# Build an immutable Rusty snapshot while downloads continue in separate runs.
set -euo pipefail
unset RSP_TOKEN

export MMU_REPO="${MMU_REPO:-$HOME/MultimodalUniverse}"
export MMU_PYTHON="${MMU_PYTHON:-$MMU_REPO/.venv/bin/python}"
export DP2_BASE="${DP2_BASE:-/mnt/ceph/users/polymathic/mronceray}"
export LSST_DP2_RELEASE_NAME="${LSST_DP2_RELEASE_NAME:-lsst_dp2_release_20261114}"
export LSST_DP2_MAX_EXTENSION="${LSST_DP2_MAX_EXTENSION:-4}"
export LSST_DP2_AUTO_PUBLISH="${LSST_DP2_AUTO_PUBLISH:-0}"
export LSST_DP2_PUBLISH_ROOT="${LSST_DP2_PUBLISH_ROOT:-/mnt/ceph/users/polymathic/MultimodalUniverse_v1.6/lsst_dp2}"
SHARDS="${LSST_DP2_RELEASE_SHARDS:-512}"
EXTRACT_JOBS="${LSST_DP2_EXTRACT_JOBS:-16}"
PATCH_WORKERS="${LSST_DP2_PATCH_WORKERS:-4}"
INGEST_WORKERS="${LSST_DP2_INGEST_WORKERS:-16}"
PIXEL_THRESHOLD="${LSST_DP2_PIXEL_THRESHOLD:-512}"

[[ "$LSST_DP2_RELEASE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid release name" >&2; exit 2; }
for name in SHARDS EXTRACT_JOBS PATCH_WORKERS INGEST_WORKERS PIXEL_THRESHOLD; do
  [[ "${!name}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid $name" >&2; exit 2; }
done
[[ "$LSST_DP2_MAX_EXTENSION" =~ ^[0-9]+$ ]] || exit 2
[[ "$LSST_DP2_AUTO_PUBLISH" =~ ^[01]$ ]] || exit 2
export LSST_DP2_RELEASE_RUN="$DP2_BASE/$LSST_DP2_RELEASE_NAME"
export LSST_DP2_RELEASE_SNAPSHOT="$DP2_BASE/${LSST_DP2_RELEASE_NAME}_inputs"
export LSST_DP2_RELEASE_SCRIPT="$0"
mkdir -p "$LSST_DP2_RELEASE_RUN/logs" "$LSST_DP2_RELEASE_RUN/tmp"
cd "$MMU_REPO"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 ARROW_NUM_THREADS=1
export TMPDIR="$LSST_DP2_RELEASE_RUN/tmp/${SLURM_JOB_ID:-preflight}"
export DASK_TEMPORARY_DIRECTORY="$TMPDIR/dask"
mkdir -p "$DASK_TEMPORARY_DIRECTORY"

# Exercise the masked-column round trip before allocating the expensive build.
"$MMU_PYTHON" -u - <<'PY'
import os
import tempfile
from pathlib import Path
import numpy as np
from astropy.table import Table, MaskedColumn, vstack
from scripts.lsst_dp2 import offline_cutouts
base = Path(os.environ["DP2_BASE"])
assert (base / "lsst_dp2_raw/catalog/objects.parquet").is_file()
assert (base / "lsst_dp2_raw/download_manifest.sqlite").is_file()
with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as d:
    tables = []
    for i, mask in enumerate(([False, True], [True, False])):
        table = Table({"objectId": np.array([2*i+1, 2*i+2], dtype=np.int64)})
        table["i_psfFlux"] = MaskedColumn([1.0, 2.0], mask=mask)
        p = Path(d) / f"batch-{i}.parquet"
        table.write(p, format="parquet")
        tables.append(Table.read(p, format="parquet"))
    combined = vstack(tables, join_type="exact", metadata_conflicts="silent")
    p = Path(d) / "combined.parquet"
    combined.write(p, format="parquet")
    check = Table.read(p, format="parquet")
    assert np.array_equal(check["objectId"], [1, 2, 3, 4])
    assert np.array_equal(np.ma.getmaskarray(check["i_psfFlux"]), [False, True, True, False])
print("PREFLIGHT PASS: imports, source paths, masked-column round trip", flush=True)
print("Output:", os.environ["LSST_DP2_RELEASE_RUN"], flush=True)
print("Auto publication:", os.environ["LSST_DP2_AUTO_PUBLISH"], flush=True)
PY
if [[ "${1:-}" == "--check" ]]; then exit 0; fi
[[ $# == 0 ]] || { echo "Usage: bash rusty_release.slurm [--check]" >&2; exit 2; }
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]] && (( EXTRACT_JOBS * PATCH_WORKERS > SLURM_CPUS_PER_TASK )); then
  echo "Extraction concurrency exceeds allocated CPUs" >&2
  exit 2
fi
exec 9>"$LSST_DP2_RELEASE_RUN/.release.lock"
flock -n 9 || { echo "Another build owns this release" >&2; exit 1; }

if [[ ! -f "$LSST_DP2_RELEASE_RUN/validation_report.json" ]]; then
  "$MMU_PYTHON" -u - <<'PY'
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from astropy.table import Table, vstack
from scripts.lsst_dp2.common import sha256_file
from scripts.lsst_dp2.offline_cutouts import MANIFEST_COLUMNS, _write_manifest

base = Path(os.environ["DP2_BASE"]).resolve(strict=True)
target = Path(os.environ["LSST_DP2_RELEASE_SNAPSHOT"])
if target.exists():
    assert (target / "catalog/objects.parquet").is_file()
    assert (target / "download_manifest.sqlite").is_file()
    assert (target / "catalog/objects.parquet.selection.json").is_file()
    print(f"REUSING FROZEN SNAPSHOT: {target}", flush=True)
else:
    sources = [base / "lsst_dp2_raw"] + [
        base / f"lsst_dp2_raw_ext{i:02d}"
        for i in range(1, int(os.environ["LSST_DP2_MAX_EXTENSION"]) + 1)
    ]
    tables, rows, provenance = [], [], []
    for index, source in enumerate(sources):
        catalog = source / "catalog/objects.parquet"
        manifest = source / "download_manifest.sqlite"
        selection = Path(str(catalog) + ".selection.json")
        if not all(p.is_file() for p in (catalog, manifest, selection)):
            if index == 0:
                raise RuntimeError(f"Missing initial run inputs: {source}")
            print(f"SKIP: run not ready for snapshot: {source}", flush=True)
            continue
        with sqlite3.connect(manifest.as_uri() + "?mode=ro", uri=True) as con:
            con.row_factory = sqlite3.Row
            con.execute("BEGIN")
            records = [dict(r) for r in con.execute(
                f"SELECT {', '.join(MANIFEST_COLUMNS)} FROM coadds ORDER BY tract, patch, band"
            )]
        if not records:
            if index == 0:
                raise RuntimeError("Initial manifest is empty")
            print(f"SKIP: manifest still initializing: {source}", flush=True)
            continue
        table = Table.read(catalog, format="parquet")
        meta = json.loads(selection.read_text())
        assert meta["status"] == "PASS" and len(table) == meta["objects"]
        states = dict(Counter(r["status"] for r in records))
        print(f"SNAPSHOT {source.name}: {len(table):,} candidates; {states}", flush=True)
        tables.append(table)
        rows.extend(records)
        provenance.append({
            "run": str(source), "selection": meta,
            "catalog_sha256": sha256_file(catalog),
            "manifest_status_counts": states,
        })
    assert len({p["selection"]["where"] for p in provenance}) == 1, "Different selections"
    table = vstack(tables, join_type="exact", metadata_conflicts="silent")
    assert not np.ma.getmaskarray(table["objectId"]).any(), "Masked identifiers"
    assert len(np.unique(table["objectId"])) == len(table), "Duplicate objects"
    keys = [(r["tract"], r["patch"], r["band"]) for r in rows]
    assert len(set(keys)) == len(keys), "Overlapping source patches"
    free = shutil.disk_usage(base).free
    print(f"Filesystem free: {free / 2**40:.2f} TiB (project quota is separate)", flush=True)
    with tempfile.TemporaryDirectory(prefix=".dp2-snapshot-", dir=base) as tmp:
        staging = Path(tmp) / "inputs"
        (staging / "catalog").mkdir(parents=True)
        p = staging / "catalog/objects.parquet"
        table.write(p, format="parquet")
        check = Table.read(p, format="parquet")
        assert check.colnames == table.colnames
        assert np.array_equal(check["objectId"], table["objectId"])
        for name in table.colnames:
            assert np.array_equal(np.ma.getmaskarray(check[name]),
                                  np.ma.getmaskarray(table[name])), name
        _write_manifest(staging / "download_manifest.sqlite", rows)
        document = {
            "status": "PASS", "scope": "catalog_and_manifest_snapshot_only",
            "where": provenance[0]["selection"]["where"],
            "objects": len(table), "sources": provenance,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "launcher_sha256": sha256_file(os.environ["LSST_DP2_RELEASE_SCRIPT"]),
        }
        Path(str(p) + ".selection.json").write_text(json.dumps(document, indent=2) + "\n")
        os.replace(staging, target)
    print(f"SNAPSHOT VERIFIED: {len(table):,} input candidates", flush=True)
    print("Only fully resolved patches will enter the output; downloads continue independently.", flush=True)
PY

  "$MMU_PYTHON" -u -m scripts.lsst_dp2.offline_cutouts prepare \
    --source-run "$LSST_DP2_RELEASE_SNAPSHOT" \
    --output-run "$LSST_DP2_RELEASE_RUN" --shards "$SHARDS"
  seq 0 "$((SHARDS - 1))" | xargs -P "$EXTRACT_JOBS" -I {} \
    "$MMU_PYTHON" -u -m scripts.lsst_dp2.offline_cutouts extract \
    --output-run "$LSST_DP2_RELEASE_RUN" --index {} --workers "$PATCH_WORKERS"
  "$MMU_PYTHON" -u -m scripts.lsst_dp2.offline_cutouts gather \
    --output-run "$LSST_DP2_RELEASE_RUN" --workers "$INGEST_WORKERS" \
    --pixel-threshold "$PIXEL_THRESHOLD"
fi

"$MMU_PYTHON" -u - <<'PY'
import json
import os
from pathlib import Path
from filelock import FileLock

run = Path(os.environ["LSST_DP2_RELEASE_RUN"]).resolve(strict=True)
report_path = run / "validation_report.json"
report = json.loads(report_path.read_text())
assert report["status"] == "PASS"
assert report["hats_rows"] == report["catalog_rows"] == report["unique_object_ids"]
if os.environ["LSST_DP2_AUTO_PUBLISH"] != "1":
    print(f"BUILD PASS: {report['hats_rows']:,} rows; {report['hats_path']}", flush=True)
else:
    live = Path(os.environ["LSST_DP2_PUBLISH_ROOT"]).absolute()
    archive = Path(os.environ["DP2_BASE"]) / (os.environ["LSST_DP2_RELEASE_NAME"] + "_previous_published")
    live.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(live.parent / ".lsst_dp2_publish.lock"), timeout=0):
        current = live / "validation_report.json"
        published = json.loads(current.read_text()) if current.exists() else None
        already_published = bool(published and
            published.get("plan_sha256") == report["plan_sha256"])
        inner = Path(report["hats_path"])
        if not already_published:
            collection = inner.parent
            assert collection.is_relative_to(run), "Unexpected source collection"
            assert (collection / "lsst_dp2/hats.properties").is_file()
            assert (collection / "lsst_dp2/dataset").is_dir()
            assert not archive.exists() and not archive.is_symlink(), "Archive already exists"
            assert not live.is_symlink(), "Publish root must be a collection directory"
            moved_old = False
            if live.exists():
                assert (live / "lsst_dp2/hats.properties").is_file(), "Unexpected current dataset"
                assert published and published.get("status") == "PASS", "Current report is missing or invalid"
                assert report["hats_rows"] >= published["hats_rows"], "Refusing to publish fewer rows"
                for key in ("hats_path", "hats_properties"):
                    Path(published[key]).relative_to(live)
                live.rename(archive)
                moved_old = True
            try:
                collection.rename(live)
            except BaseException:
                if moved_old:
                    archive.rename(live)
                raise
            if moved_old:
                for key in ("hats_path", "hats_properties"):
                    old = Path(published[key])
                    published[key] = str(archive / old.relative_to(live))
                (archive / "validation_report.json").write_text(json.dumps(published, indent=2) + "\n")
        new_inner = live / "lsst_dp2"
        assert (new_inner / "hats.properties").is_file()
        assert (new_inner / "dataset").is_dir()
        original = run / "validation_report.before_publication.json"
        if not original.exists():
            original.write_bytes(report_path.read_bytes())
        report["hats_path"] = str(new_inner)
        report["hats_properties"] = str(new_inner / "hats.properties")
        report["original_validation_report"] = str(original)
        report["previous_release_path"] = str(archive) if archive.exists() else None
        # Write the public report atomically after the directory has been moved.
        temporary = live / ".validation_report.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary, live / "validation_report.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"PUBLISHED PASS: {report['hats_rows']:,} rows", flush=True)
        print(f"HATS: {new_inner}", flush=True)
        print(f"Previous release: {report['previous_release_path']}", flush=True)
print(f"Selected global population coverage: {report['hats_rows'] / 10_358_863:.2%}", flush=True)
PY
