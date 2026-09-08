#!/bin/bash
#SBATCH --job-name=mmu-lsst-50k
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
export LSST_DP2_RUN_NAME="${LSST_DP2_RUN_NAME:-multiregion_i22_50k_v1}"

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

uv sync --frozen --extra dev --extra viz
"$MMU_PYTHON" -c \
  "import mmu, astropy, pyarrow, pyvo, hats, lsdb; print('installation OK')"

CATALOG="$LSST_DP2_ROOT/catalog/objects.parquet"
if [[ ! -f "$CATALOG" ]]; then
  "$MMU_PYTHON" -u -m scripts.lsst_dp2.query_multiregion_catalog \
    --per-region 10000 \
    --seed 20260908 \
    --output "$CATALOG"
else
  echo "Reusing existing catalog: $CATALOG"
fi

"$MMU_PYTHON" - <<'PY'
import os
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

root = Path(os.environ["LSST_DP2_ROOT"])
table = pq.read_table(
    root / "catalog/objects.parquet",
    columns=["objectId", "tract", "patch", "dp2_region"],
)
ids = table["objectId"].to_pylist()
patches = set(zip(table["tract"].to_pylist(), table["patch"].to_pylist()))
regions = Counter(table["dp2_region"].to_pylist())
free_gib = shutil.disk_usage(root).free / 1024**3

assert len(ids) == len(set(ids)) == 50_000
assert len(regions) == 5 and set(regions.values()) == {10_000}
print("regions:", dict(sorted(regions.items())))
print("unique patches:", len(patches))
print("download tasks:", 6 * len(patches))
print("estimated coadds GiB:", round(6 * len(patches) * 31 / 1024, 1))
print("filesystem free GiB:", round(free_gib, 1))
PY

exec bash scripts/lsst_dp2/run_prepost.sh
