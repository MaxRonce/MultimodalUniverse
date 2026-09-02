#!/bin/bash
# Run the restartable LSST DP2 Snakemake workflow inside a prepost allocation.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=jeanzay_env.sh
source "$SCRIPT_DIR/jeanzay_env.sh"

if [[ ! -x "$MMU_PYTHON" ]]; then
  echo "ERROR: MMU virtual environment is missing: $MMU_PYTHON" >&2
  exit 2
fi
if [[ ! -f "$LSST_DP2_ROOT/catalog/objects.parquet" ]]; then
  echo "ERROR: query the catalog first: $LSST_DP2_ROOT/catalog/objects.parquet" >&2
  exit 2
fi

mkdir -p "$LSST_DP2_ROOT/logs"
cd "$MMU_REPO"

exec "${MMU_REPO}/.venv/bin/snakemake" \
  --snakefile scripts/lsst_dp2/Snakefile \
  --cores "${SLURM_CPUS_PER_TASK:-8}" \
  --printshellcmds \
  --rerun-incomplete \
  "$@"
