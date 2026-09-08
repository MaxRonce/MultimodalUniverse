#!/bin/bash
# Source this file before installing or running the LSST DP2 pipeline.

if [[ -z "${SCRATCH:-}" ]]; then
  echo "ERROR: SCRATCH is not set" >&2
  return 1 2>/dev/null || exit 1
fi

export MMU_JZ_ROOT="${MMU_JZ_ROOT:-$SCRATCH/mmu_lsst_dp2}"
export MMU_REPO="${MMU_REPO:-$MMU_JZ_ROOT/MultimodalUniverse}"
export LSST_DP2_RUN_NAME="${LSST_DP2_RUN_NAME:-pilot5000}"
export LSST_DP2_ROOT="${LSST_DP2_ROOT:-$MMU_JZ_ROOT/runs/$LSST_DP2_RUN_NAME}"
export MMU_HATS_ROOT="${MMU_HATS_ROOT:-$MMU_JZ_ROOT/hats}"
export LSST_DP2_HATS_OUTER="${LSST_DP2_HATS_OUTER:-$LSST_DP2_ROOT/hats/lsst_dp2}"
export MMU_PYTHON="${MMU_PYTHON:-$MMU_REPO/.venv/bin/python}"

export LSST_DP2_DOWNLOAD_WORKERS="${LSST_DP2_DOWNLOAD_WORKERS:-4}"
export LSST_DP2_REQUESTS_PER_MINUTE="${LSST_DP2_REQUESTS_PER_MINUTE:-50}"
export LSST_DP2_PATCH_WORKERS="${LSST_DP2_PATCH_WORKERS:-4}"
export LSST_DP2_INGEST_WORKERS="${LSST_DP2_INGEST_WORKERS:-2}"
export LSST_DP2_PIXEL_THRESHOLD="${LSST_DP2_PIXEL_THRESHOLD:-256}"

export XDG_CACHE_HOME="$MMU_JZ_ROOT/cache/xdg"
export UV_CACHE_DIR="$MMU_JZ_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$MMU_REPO/.venv"
export PIP_CACHE_DIR="$MMU_JZ_ROOT/cache/pip"
export HF_HOME="$MMU_JZ_ROOT/cache/huggingface"
export TORCH_HOME="$MMU_JZ_ROOT/cache/torch"
export MPLCONFIGDIR="$MMU_JZ_ROOT/cache/matplotlib"
export NUMBA_CACHE_DIR="$MMU_JZ_ROOT/cache/numba"
export IPYTHONDIR="$MMU_JZ_ROOT/cache/ipython"
export DASK_TEMPORARY_DIRECTORY="$MMU_JZ_ROOT/tmp/dask"
export JOBLIB_TEMP_FOLDER="$MMU_JZ_ROOT/tmp/joblib"
export JAX_COMPILATION_CACHE_DIR="$MMU_JZ_ROOT/cache/jax"
export PYTHONPYCACHEPREFIX="$MMU_JZ_ROOT/cache/pycache"
export TMPDIR="$MMU_JZ_ROOT/tmp/generic"

mkdir -p \
  "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" \
  "$TORCH_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$IPYTHONDIR" \
  "$DASK_TEMPORARY_DIRECTORY" "$JOBLIB_TEMP_FOLDER" \
  "$JAX_COMPILATION_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$TMPDIR" \
  "$LSST_DP2_ROOT/catalog" "$LSST_DP2_ROOT/logs" "$MMU_HATS_ROOT" \
  "$LSST_DP2_HATS_OUTER"
