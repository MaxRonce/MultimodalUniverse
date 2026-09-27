"""Test the Slurm handoff without contacting a scheduler or exposing tokens."""

import json
import os
import subprocess
import sys
from pathlib import Path


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
