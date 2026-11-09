#!/usr/bin/env bash
# Frontend launcher: all catalog I/O and computation run in Slurm jobs.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_ROOT=${DESI_RUN_ROOT:-/mnt/ceph/users/polymathic/mronceray/desi_dr1_main}
RAW_ROOT=${DESI_RAW_ROOT:-/mnt/ceph/users/polymathic/external_data/astro/DESI_DR1}
ZCATALOG=${DESI_ZCATALOG:-$RAW_ROOT/zall-pix-iron.fits}
DOWNLOAD_ZCATALOG_V1=0
STATUS=0
WATCH=0
SHARDS=${DESI_SHARDS:-8}
WORKERS=${DESI_WORKERS:-16}
JOBS=${DESI_CONCURRENT:-2}
GATHER_WORKERS=${DESI_GATHER_WORKERS:-8}
GATHER_MEM=${DESI_GATHER_MEM_MB:-900000}
MAX_GROUPS=0
DRY_RUN=()
while (($#)); do
  case "$1" in
    --smoke)
      RUN_ROOT="${DESI_RUN_ROOT:-/mnt/ceph/users/polymathic/mronceray/desi_dr1_main_smoke}"
      SHARDS=2; MAX_GROUPS=8; GATHER_WORKERS=2; GATHER_MEM=64000; shift ;;
    --dry-run) DRY_RUN=(--dry-run); shift ;;
    --status) STATUS=1; shift ;;
    --watch) STATUS=1; WATCH=1; shift ;;
    --download-zcatalog-v1) DOWNLOAD_ZCATALOG_V1=1; shift ;;
    --help|-h)
      echo "Usage: bash scripts/desi/launch_dr1_main.sh [--smoke] [--dry-run] [--status|--watch] [--download-zcatalog-v1]"
      echo "Default: read DESI_ZCATALOG or the shared zall-pix-iron.fits; no network/copy."
      echo "Monitoring only: --status (once) or --watch (every 10 seconds); no jobs submitted."
      echo "Environment: DESI_RUN_ROOT, DESI_RAW_ROOT, DESI_PARTITION (genx), DESI_CONCURRENT (2)."
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if ((STATUS)); then
  STATUS_ARGS=("$RUN_ROOT")
  if ((WATCH)); then STATUS_ARGS+=(--watch); fi
  exec "$REPO/.venv/bin/python" "$REPO/scripts/desi/status_dr1_main.py" "${STATUS_ARGS[@]}"
fi
for VALUE in "$SHARDS" "$WORKERS" "$JOBS" "$GATHER_WORKERS" "$GATHER_MEM"; do
  [[ "$VALUE" =~ ^[1-9][0-9]*$ ]] || { echo "Resource values must be positive integers" >&2; exit 2; }
done
PYTHON="$REPO/.venv/bin/python"
test -x "$PYTHON" || { echo "Install the MMU .venv first; see docs/BUILD_DESI_DR1.md" >&2; exit 2; }
command -v sbatch >/dev/null || { echo "Load Slurm first: module load slurm" >&2; exit 2; }
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Validate target paths and immutable run settings before creating anything.
cd "$REPO"
"$PYTHON" - "$RUN_ROOT" "$RAW_ROOT" "$SHARDS" "$MAX_GROUPS" "$ZCATALOG" "$DOWNLOAD_ZCATALOG_V1" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
from scripts.desi.dr1_main import validate_paths
run, raw = validate_paths(sys.argv[1], sys.argv[2])
download = bool(int(sys.argv[6]))
zcatalog = (run / "inputs/zall-pix-iron-v1.fits" if download else Path(sys.argv[5])).resolve()
if not raw.is_dir():
    raise SystemExit(f"Missing coadd mirror: {raw}")
if not download and not zcatalog.is_file():
    raise SystemExit(f"Local zcatalog missing: {zcatalog}. No download attempted.")
plan = run / "plan.json"
if plan.exists():
    config = json.loads(plan.read_text())["config"]
    builder = Path("scripts/desi/dr1_main.py")
    expected = {"raw_root": str(raw), "shards": int(sys.argv[3]),
                "max_groups": int(sys.argv[4]) or None,
                "builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
                "zcatalog_path": str(zcatalog), "catalog_source": "official_v1" if download else "local"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise SystemExit("Run settings/code changed; use DESI_RUN_ROOT pointing to a new run")
PY
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/workflow"
# An independent working directory avoids locking other MMU/LSST workflows.
EXTRA=()
if [[ -n "${DESI_ACCOUNT:-}" ]]; then
  EXTRA+=(--default-resources "slurm_account=$DESI_ACCOUNT")
fi
# Snakemake otherwise caps rule threads to --jobs on some versions.
CORES=$((JOBS * WORKERS))
if ((GATHER_WORKERS > CORES)); then CORES=$GATHER_WORKERS; fi
if ((CORES < 2)); then CORES=2; fi
if ((MAX_GROUPS)); then MODE_LABEL='SMOKE : 8 coadds maximum'; else MODE_LABEL='COMPLÈTE'; fi
echo "DESI DR1 Main — campagne $MODE_LABEL"
echo "Entrées : $RAW_ROOT (lecture seule)"
echo "Sortie  : $RUN_ROOT"
printf 'Suivi dans un autre terminal : %q %q %q --watch\n' "$PYTHON" "$REPO/scripts/desi/status_dr1_main.py" "$RUN_ROOT"
echo "Les messages suivants proviennent de Snakemake/Slurm ; les logs détaillés sont conservés."
exec "$PYTHON" -m snakemake --snakefile "$REPO/scripts/desi/dr1_main.smk" \
  --directory "$RUN_ROOT/workflow" --executor slurm --jobs "$JOBS" --cores "$CORES" \
  --rerun-incomplete --latency-wait 120 --printshellcmds \
  "${DRY_RUN[@]}" "${EXTRA[@]}" \
  --config "run_root=$RUN_ROOT" "raw_root=$RAW_ROOT" "shards=$SHARDS" \
  "workers=$WORKERS" "partition=${DESI_PARTITION:-genx}" "max_groups=$MAX_GROUPS" \
  "zcatalog=$ZCATALOG" "download_zcatalog_v1=$DOWNLOAD_ZCATALOG_V1" \
  "gather_workers=$GATHER_WORKERS" "gather_mem_mb=$GATHER_MEM"
