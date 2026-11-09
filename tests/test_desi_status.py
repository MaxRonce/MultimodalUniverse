"""Monitor behavior without any cluster, scientific data, or mutation."""
import json
import os
from pathlib import Path
import subprocess

from scripts.desi import status_dr1_main as status


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_pending_job_is_found_without_slurm_logfile(tmp_path):
    log = tmp_path / "workflow/.snakemake/log/controller.log"
    log.parent.mkdir(parents=True)
    log.write_text("Job 3 has been submitted with SLURM jobid 7433308 (log: later).\n")
    prepare = tmp_path / "logs/prepare.log"
    prepare.parent.mkdir()
    prepare.write_text("Reading local zcatalog in place (no download, no copy): /shared/zall.fits\n")
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "7433308|RUNNING|6:08|worker5075\n", "")
    data = status.snapshot(tmp_path, query)
    result = status.render(data)
    assert "7433308" in result and "RUNNING" in result and "worker5075" in result
    assert "no download, no copy" in result
    assert len(calls) == 1 and calls[0][0] == "squeue"
    assert calls[0][calls[0].index("--jobs") + 1] == "7433308"


def test_completed_job_uses_accounting_without_shell_pipe(tmp_path):
    log = tmp_path / "workflow/.snakemake/slurm_logs/rule_prepare/7433308.log"
    log.parent.mkdir(parents=True)
    log.touch()
    calls = []
    def query(command, **kwargs):
        calls.append(command)
        text = "7433308|COMPLETED|00:07:00|0:0|\n" if command[0] == "sacct" else ""
        return subprocess.CompletedProcess(command, 0, text, "")
    result = status.render(status.snapshot(tmp_path, query))
    assert "COMPLETED" in result and "exit=0:0" in result
    assert [call[0] for call in calls] == ["squeue", "sacct"]
    assert not any("|" in arg or arg == "-1" for arg in calls[1])


def test_success_comes_from_validation_not_empty_queue(tmp_path):
    write_json(tmp_path / "plan.json", {"scope": "smoke", "groups": [{}, {}],
               "selected_rows": 7, "config": {"shards": 2}})
    for index, count in enumerate((3, 2)):
        write_json(tmp_path / f"reports/shard-{index:03d}.json", {"retained_rows": count})
    write_json(tmp_path / "validation.json", {"status": "PASS", "rows": 5,
               "collection": "/output/hats/desi_dr1_main", "source_warnings": ["Version unverified"]})
    rendered = status.render(status.snapshot(tmp_path))
    assert "SMOKE" in rendered and "2/2 morceaux" in rendered
    assert "PASS — 5 spectres" in rendered and "Version unverified" in rendered
    (tmp_path / "validation.json").unlink()
    assert "RÉSULTAT : PASS" not in status.render(status.snapshot(tmp_path))


def test_missing_slurm_commands_are_reported(tmp_path):
    log = tmp_path / "workflow/.snakemake/slurm_logs/rule_prepare/7433308.log"
    log.parent.mkdir(parents=True)
    log.touch()
    def absent(*args, **kwargs):
        raise FileNotFoundError("squeue")
    result = status.render(status.snapshot(tmp_path, absent))
    assert "module load slurm" in result and "ÉTAT NON DISPONIBLE" in result


def test_monitor_interrupt_does_not_cancel_jobs(tmp_path, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(status.time, "sleep", interrupted)
    assert status.main([str(tmp_path), "--watch"]) == 0


def test_launcher_status_needs_no_raw_data_or_sbatch(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = {**os.environ, "DESI_RUN_ROOT": str(tmp_path), "DESI_RAW_ROOT": str(tmp_path / "absent")}
    result = subprocess.run(["bash", str(repo / "scripts/desi/launch_dr1_main.sh"), "--status"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DESI DR1 Main" in result.stdout and "aucun" in result.stdout.lower()
    assert not (tmp_path / "workflow").exists()  # Read-only: nothing created.
