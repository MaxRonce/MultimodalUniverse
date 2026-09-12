#!/bin/bash
#SBATCH --job-name=mmu-lsst-net
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

set -euo pipefail

: "${SCRATCH:?SCRATCH must be exported by Jean-Zay}"
: "${RSP_TOKEN:?Read and export RSP_TOKEN before sbatch}"

export MMU_JZ_ROOT="${MMU_JZ_ROOT:-$SCRATCH/mmu_lsst_dp2}"
export MMU_REPO="${MMU_REPO:-$MMU_JZ_ROOT/MultimodalUniverse}"
export LSST_DP2_RUN_NAME="${LSST_DP2_RUN_NAME:-dense_i21_reff0p6_weekend_v1}"

cd "$MMU_REPO"
# shellcheck source=jeanzay_env.sh
source scripts/lsst_dp2/jeanzay_env.sh

if ! command -v uv >/dev/null 2>&1; then
  UV_BOOTSTRAP="$MMU_JZ_ROOT/venvs/uv"
  if [[ ! -x "$UV_BOOTSTRAP/bin/uv" ]]; then
    python3 -m venv "$UV_BOOTSTRAP"
    "$UV_BOOTSTRAP/bin/pip" install --upgrade pip uv
  fi
  export PATH="$UV_BOOTSTRAP/bin:$PATH"
fi

uv sync --frozen
"$MMU_PYTHON" -c "import astropy, pyarrow, pyvo, requests; print('installation OK')"

CATALOG="$LSST_DP2_ROOT/catalog/objects.parquet"
PATCH_LIMIT="${LSST_DP2_DENSE_PATCH_LIMIT:-6000}"
TASK_LIMIT="${LSST_DP2_DOWNLOAD_TASK_LIMIT:-18000}"
WHERE="${LSST_DP2_DENSE_WHERE:-i_cModelMag < 21 AND i_extendedness = 1 AND griz_model_extendedness >= 0.8 AND sersic_no_data_flag = 0 AND sersic_unknown_flag = 0 AND sersic_reff_major > 0.6}"

if [[ ! -f "$CATALOG" || ! -f "$CATALOG.selection.json" ]]; then
  "$MMU_PYTHON" -u -m scripts.lsst_dp2.query_dense_patch_catalog \
    --where "$WHERE" \
    --patch-limit "$PATCH_LIMIT" \
    --output "$CATALOG"
else
  echo "Reusing existing dense catalog: $CATALOG"
fi

"$MMU_PYTHON" - <<'PY'
import json
import os
import shutil
from pathlib import Path

root = Path(os.environ["LSST_DP2_ROOT"])
report = json.loads(Path(str(root / "catalog/objects.parquet") + ".selection.json").read_text())
print("Dense download plan:")
print(json.dumps(report, indent=2))
free_gib = shutil.disk_usage(root).free / 1024**3
required_gib = report["estimated_coadds_gib_at_31_mib_each"]
print(f"Filesystem free: {free_gib:.1f} GiB")
if free_gib < 1.25 * required_gib:
    raise RuntimeError(
        f"need at least 1.25 x {required_gib:.1f} GiB for the coadd mirror"
    )
PY

exec "$MMU_PYTHON" -u -m scripts.lsst_dp2.download_coadds \
  --catalog "$CATALOG" \
  --mirror-root "$LSST_DP2_ROOT/coadds" \
  --manifest "$LSST_DP2_ROOT/download_manifest.sqlite" \
  --workers "${LSST_DP2_DENSE_DOWNLOAD_WORKERS:-8}" \
  --requests-per-minute "${LSST_DP2_DENSE_REQUESTS_PER_MINUTE:-59}" \
  --task-limit "$TASK_LIMIT"
