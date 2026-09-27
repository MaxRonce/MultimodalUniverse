"""Test the Slurm handoff without contacting a scheduler or exposing tokens."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_offline_submission_dependencies_resources_and_credentials(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    scratch = tmp_path / "scratch"
    source = scratch / "mmu_lsst_dp2/runs/source"
    (source / "catalog").mkdir(parents=True)
    (source / "catalog/objects.parquet").touch()
    (source / "download_manifest.sqlite").touch()
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "calls.jsonl"
    sbatch = commands / "sbatch"
    sbatch.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"path = Path({str(log)!r})\n"
        "calls = path.read_text().splitlines() if path.exists() else []\n"
        "with path.open('a') as out:\n"
        "    out.write(json.dumps({'args': sys.argv[1:], 'token': os.environ.get('RSP_TOKEN'), "
        "'root': os.environ['LSST_DP2_ROOT']}) + '\\n')\n"
        "print(100 + len(calls))\n"
    )
    squeue = commands / "squeue"
    squeue.write_text("#!/bin/bash\nexit 0\n")
    for path in (squeue, sbatch):
        path.chmod(0o755)
    env = {
        **os.environ,
        "SCRATCH": str(scratch),
        "MMU_JZ_ROOT": str(scratch / "mmu_lsst_dp2"),
        "MMU_REPO": str(repo),
        "PATH": f"{commands}:{os.environ['PATH']}",
        "RSP_TOKEN": "test-placeholder-must-not-reach-sbatch",
        "LSST_DP2_ROOT": "wrong-download-run",
        "USER": "test",
    }
    result = subprocess.run(
        [
            "bash",
            str(repo / "scripts/lsst_dp2/submit_offline_cutouts.sh"),
            "--source-run",
            str(source),
            "--run-name",
            "new_mmu",
            "--concurrent",
            "32",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(calls) == 3
    assert all(call["token"] is None for call in calls)
    assert all(
        call["root"] == str(scratch / "mmu_lsst_dp2/runs/new_mmu") for call in calls
    )
    assert "--dependency=afterok:100" in calls[1]["args"]
    assert "--dependency=afterok:101" in calls[2]["args"]
    assert "--array=0-127%32" in calls[1]["args"]
    assert "--cpus-per-task=8" in calls[1]["args"]
    assert "--cpus-per-task=40" in calls[2]["args"]
    assert "Receipt:" in result.stdout
    receipts = list(
        (scratch / "mmu_lsst_dp2/runs/new_mmu/logs").glob("offline-jobs.*.txt")
    )
    assert receipts[0].read_text() == "prepare 100\nextract 101\ngather 102\n"


@pytest.mark.parametrize(
    ("settings", "expected", "exit_code"),
    [
        ({}, ("8", "512"), 0),
        (
            {"LSST_DP2_GATHER_WORKERS": "4", "LSST_DP2_GATHER_PIXEL_THRESHOLD": "1024"},
            ("4", "1024"),
            0,
        ),
        ({"LSST_DP2_GATHER_WORKERS": "0"}, None, 2),
        ({"LSST_DP2_GATHER_PIXEL_THRESHOLD": "invalid"}, None, 2),
    ],
)
def test_gather_worker_settings_without_reextracting(
    tmp_path, settings, expected, exit_code
):
    repo = Path(__file__).resolve().parents[1]
    output = tmp_path / "scratch/runs/output"
    recorder = tmp_path / "python-recorder"
    calls = tmp_path / "calls.json"
    recorder.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"Path({str(calls)!r}).write_text(json.dumps({{'args': sys.argv[1:], "
        "'token': os.environ.get('RSP_TOKEN'), 'root': os.environ['LSST_DP2_ROOT'], "
        "'tmp': os.environ['TMPDIR']}))\n"
    )
    recorder.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LSST_DP2_", "MMU_"))
    }
    env.update(
        {
            "SCRATCH": str(tmp_path / "scratch"),
            "MMU_JZ_ROOT": str(tmp_path / "scratch"),
            "MMU_REPO": str(repo),
            "MMU_PYTHON": str(recorder),
            "LSST_DP2_SOURCE_RUN": str(tmp_path / "source"),
            "LSST_DP2_OFFLINE_ROOT": str(output),
            "LSST_DP2_PIXEL_THRESHOLD": "256",  # Generic pilot default must not override gather.
            "RSP_TOKEN": "test-placeholder-must-not-reach-worker",
            "SLURM_JOB_ID": "1234",
            **settings,
        }
    )
    result = subprocess.run(
        ["bash", str(repo / "scripts/lsst_dp2/offline_cutouts.slurm"), "gather"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    if expected is None:
        assert "positive integers" in result.stderr
        assert not calls.exists()
        return
    call = json.loads(calls.read_text())
    assert call["args"] == [
        "-u",
        "-m",
        "scripts.lsst_dp2.offline_cutouts",
        "gather",
        "--output-run",
        str(output),
        "--workers",
        expected[0],
        "--pixel-threshold",
        expected[1],
    ]
    assert call["token"] is None and call["root"] == str(output)
    assert Path(call["tmp"]).is_relative_to(output)
    assert f"workers={expected[0]} pixel_threshold={expected[1]}" in result.stdout


def test_dense_healpix_cell_exceeds_256_but_fits_512():
    from hats.pixel_math.partition_stats import generate_alignment

    histogram = np.zeros(12, dtype=np.int64)
    histogram[0] = 280
    with pytest.raises(
        ValueError, match="single pixel row count 280 exceeds threshold 256"
    ):
        generate_alignment(histogram, highest_order=0, threshold=256)
    alignment = generate_alignment(histogram, highest_order=0, threshold=512)
    assert tuple(alignment[0]) == (0, 0, 280)
