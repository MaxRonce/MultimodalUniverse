#!/usr/bin/env bash
# A charger avec : source /chemin/vers/hsc_env.sh
# HF_HOME reste inchange pour reutiliser une connexion Hugging Face existante.
export HSC_BASE="${HSC_BASE:-/mnt/ceph/users/polymathic/mronceray}"
export HSC_DEST="${HSC_DEST:-$HSC_BASE/hsc_pdr3}"
export HSC_RAW="${HSC_RAW:-$HSC_BASE/hsc_pdr3_raw}"
export HF_DOWNLOAD_ENV="${HF_DOWNLOAD_ENV:-$HOME/.local/share/hf-download}"
export HF_DOWNLOAD_PYTHON="$HF_DOWNLOAD_ENV/bin/python"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HSC_RAW/.cache/huggingface/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HSC_RAW/.cache/huggingface/xet}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export UV_CACHE_DIR="$HSC_RAW/.cache/uv"
export TMPDIR="$HSC_RAW/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "$HSC_DEST" "$HSC_RAW/logs" "$TMPDIR" \
  "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_XET_CACHE"
