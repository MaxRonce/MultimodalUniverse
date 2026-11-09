"""Dedicated production DAG; never uses the shared MMU output defaults."""
from pathlib import Path
import shlex
import sys

REPO = str(Path(workflow.basedir).resolve().parents[1])
PYTHON = sys.executable
ROOT = str(Path(config["run_root"]).resolve())
RAW = config.get("raw_root", "/mnt/ceph/users/polymathic/external_data/astro/DESI_DR1")
ZCAT = config.get("zcatalog", str(Path(RAW) / "zall-pix-iron.fits"))
SOURCE_ARGS = ("--download-zcatalog-v1" if int(config.get("download_zcatalog_v1", 0))
               else "--zcatalog " + shlex.quote(ZCAT))
SHARDS = int(config.get("shards", 8))
WORKERS = int(config.get("workers", 16))
PARTITION = config.get("partition", "genx")
MAX_GROUPS = int(config.get("max_groups", 0))
LIMIT = f"--max-groups {MAX_GROUPS}" if MAX_GROUPS else ""
BUILDER = REPO + "/scripts/desi/dr1_main.py"

rule all:
    input:
        ROOT + "/validation.json"

rule prepare:
    input:
        BUILDER
    output:
        ROOT + "/plan.json"
    log:
        ROOT + "/logs/prepare.log"
    threads: 2
    resources:
        mem_mb=128000,
        runtime=180,
        slurm_partition=PARTITION,
    shell:
        "cd {REPO:q} && {PYTHON:q} -u -m scripts.desi.dr1_main prepare "
        "--run-root {ROOT:q} --raw-root {RAW:q} --shards {SHARDS} {SOURCE_ARGS} {LIMIT} > {log:q} 2>&1"

rule scatter:
    input:
        plan=ROOT + "/plan.json",
        builder=BUILDER,
    output:
        ROOT + "/reports/shard-{shard}.json"
    wildcard_constraints:
        shard=r"\d{3}"
    log:
        ROOT + "/logs/scatter-{shard}.log"
    threads: WORKERS
    resources:
        mem_mb=int(config.get("scatter_mem_mb", 128000)),
        runtime=720,
        slurm_partition=PARTITION,
    shell:
        "cd {REPO:q} && {PYTHON:q} -u -m scripts.desi.dr1_main scatter "
        "--run-root {ROOT:q} --raw-root {RAW:q} --shard {wildcards.shard} "
        "--workers {threads} > {log:q} 2>&1"

rule gather:
    input:
        reports=expand(ROOT + "/reports/shard-{shard}.json", shard=[f"{i:03d}" for i in range(SHARDS)]),
        builder=BUILDER,
    output:
        ROOT + "/validation.json"
    log:
        ROOT + "/logs/gather.log"
    threads: int(config.get("gather_workers", 8))
    resources:
        mem_mb=int(config.get("gather_mem_mb", 900000)),
        runtime=720,
        slurm_partition=PARTITION,
    shell:
        "cd {REPO:q} && {PYTHON:q} -u -m scripts.desi.dr1_main gather "
        "--run-root {ROOT:q} --raw-root {RAW:q} --workers {threads} > {log:q} 2>&1"
