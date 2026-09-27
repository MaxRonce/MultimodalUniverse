"""Offline regression tests for extending a dense DP2 download campaign."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from astropy.table import Table

from scripts.lsst_dp2 import query_dense_patch_catalog as dense
from scripts.lsst_dp2.common import sha256_file


def source_run(tmp_path):
    run = tmp_path / "original"
    work = run / "catalog/.objects.dense_query"
    work.mkdir(parents=True)
    inventory = Table(
        {"tract": [1] * 5, "patch": [1, 2, 3, 4, 5], "n_objects": [4, 3, 2, 1, 1]}
    )
    inventory.write(work / "patch_inventory.parquet", format="parquet")
    dense.rank_inventory(inventory, 2).write(
        work / "selected_patches.parquet", format="parquet"
    )
    plan = {
        "where": dense.DEFAULT_WHERE,
        "tap_url": dense.DEFAULT_TAP_URL,
        "patch_limit": 2,
        "batch_patches": 200,
    }
    (work / "plan.json").write_text(json.dumps(plan))
    (run / "catalog/objects.parquet.selection.json").write_text(
        json.dumps(
            {
                **plan,
                "status": "PASS",
                "patches": 2,
                "objects": 7,
            }
        )
    )
    return run


def offline_tap(monkeypatch):
    calls = []
    monkeypatch.setenv("RSP_TOKEN", "test-token")
    monkeypatch.setattr(dense, "_authenticated_tap", lambda *args: object())
    monkeypatch.setattr(
        dense,
        "discover_columns",
        lambda service: {
            "objectId",
            "coord_ra",
            "coord_dec",
            "tract",
            "patch",
        },
    )

    def query(service, adql, maxrec, **kwargs):
        calls.append(adql)
        assert "GROUP BY" not in adql
        if "patch IN (3, 4)" in adql:
            ids, patches = [30, 31, 40], [3, 3, 4]
        elif "patch IN (5)" in adql:
            ids, patches = [50], [5]
        else:
            raise AssertionError(f"unexpected patch query: {adql}")
        return Table(
            {
                "objectId": ids,
                "tract": [1] * len(ids),
                "patch": patches,
                "coord_ra": [150.0] * len(ids),
                "coord_dec": [2.0] * len(ids),
            }
        )

    monkeypatch.setattr(dense, "run_async_query", query)
    return calls


def test_extension_queries_only_new_patches_and_resumes(tmp_path, monkeypatch):
    source = source_run(tmp_path)
    before = {p: sha256_file(p) for p in source.rglob("*") if p.is_file()}
    calls = offline_tap(monkeypatch)
    output = tmp_path / "extension/catalog/objects.parquet"
    args = ["--output", str(output), "--extend-from", str(source), "--patch-limit", "4"]
    assert dense.main(args) == 0
    assert len(calls) == 1
    table = pq.read_table(output)
    assert table["objectId"].to_pylist() == [30, 31, 40]
    assert table["dense_patch_rank"].to_pylist() == [3, 3, 4]
    report = json.loads(Path(str(output) + ".selection.json").read_text())
    assert (report["rank_start"], report["rank_stop"], report["patches"]) == (3, 4, 2)
    assert report["objects"] == 3
    assert dense.main(args) == 0
    assert len(calls) == 1
    assert all(sha256_file(path) == checksum for path, checksum in before.items())

    next_output = tmp_path / "next/catalog/objects.parquet"
    assert (
        dense.main(
            [
                "--output",
                str(next_output),
                "--extend-from",
                str(output.parent.parent),
                "--patch-limit",
                "5",
            ]
        )
        == 0
    )
    assert pq.read_table(next_output)["objectId"].to_pylist() == [50]


def test_extension_rejects_incompatible_selection_and_inventory(tmp_path):
    source = source_run(tmp_path)
    with pytest.raises(ValueError, match="source selection"):
        dense.load_extension_source(source, "i_cModelMag < 22", dense.DEFAULT_TAP_URL)
    selected_path = source / "catalog/.objects.dense_query/selected_patches.parquet"
    selected = Table.read(selected_path)
    selected["patch"][0] = 5
    selected.write(selected_path, format="parquet", overwrite=True)
    with pytest.raises(ValueError, match="ranked inventory"):
        dense.load_extension_source(source, dense.DEFAULT_WHERE, dense.DEFAULT_TAP_URL)


def test_extension_rejects_same_output_and_exhausted_inventory(tmp_path, monkeypatch):
    source = source_run(tmp_path)
    offline_tap(monkeypatch)
    with pytest.raises(SystemExit):
        dense.main(
            [
                "--output",
                str(source / "catalog/objects.parquet"),
                "--extend-from",
                str(source),
                "--patch-limit",
                "4",
            ]
        )
    with pytest.raises(SystemExit):
        dense.main(
            [
                "--output",
                str(tmp_path / "new/objects.parquet"),
                "--extend-from",
                str(source),
                "--patch-limit",
                "2",
            ]
        )


@pytest.mark.parametrize("fail_on", [0, 3])
def test_submit_chain_is_serial_and_keeps_receipt(tmp_path, fail_on):
    source = source_run(tmp_path)
    repo = Path(__file__).resolve().parents[1]
    commands = tmp_path / "bin"
    commands.mkdir()
    sbatch = commands / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        + """
import os
import sys
from pathlib import Path
assert "--dependency=singleton" in sys.argv
assert "--job-name=mmu-lsst-net" in sys.argv
assert "--time=20:00:00" in sys.argv
assert os.environ["LSST_DP2_DENSE_REQUESTS_PER_MINUTE"] == "59"
counter = Path(os.environ["FAKE_COUNTER"])
n = int(counter.read_text()) + 1 if counter.exists() else 1
counter.write_text(str(n))
if n == int(os.environ["FAKE_FAIL_ON"]):
    sys.exit(1)
print(100 + n)
"""
    )
    sbatch.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{commands}:{os.environ['PATH']}",
        "SCRATCH": str(tmp_path),
        "MMU_JZ_ROOT": str(tmp_path / "mmu"),
        "MMU_REPO": str(repo),
        "LSST_DP2_RUN_NAME": "new_run",
        "RSP_TOKEN": "secret-test-value",
        "FAKE_COUNTER": str(tmp_path / "counter"),
        "FAKE_FAIL_ON": str(fail_on),
    }
    # Simulate reconnecting from the previous run's exported paths.
    env["LSST_DP2_ROOT"] = str(source)
    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts/lsst_dp2/submit_dense_extension.sh"),
            "--source-run",
            str(source),
            "--jobs",
            "4",
            "--patch-limit",
            "5",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == (1 if fail_on else 0), result.stderr
    receipts = list((tmp_path / "mmu/runs/new_run/logs").glob("submitted-jobs.*.txt"))
    assert len(receipts) == 1
    assert receipts[0].read_text().splitlines() == (
        ["101", "102"] if fail_on else ["101", "102", "103", "104"]
    )
    assert (
        "secret-test-value"
        not in result.stdout + result.stderr + receipts[0].read_text()
    )
