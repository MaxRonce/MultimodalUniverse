#!/usr/bin/env bash
# A charger sur le noeud choisi : source scripts/legacy_env.sh
# HF_HOME reste inchange pour reutiliser le login HF existant.
export LEGACY_BASE="${LEGACY_BASE:-/mnt/ceph/users/polymathic/mronceray}"
export LEGACY_DEST="${LEGACY_DEST:-$LEGACY_BASE/legacysurvey_dr10}"
export LEGACY_RAW="${LEGACY_RAW:-$LEGACY_BASE/legacysurvey_dr10_raw}"
export LEGACY_REPO="hugging-science/mmu_legacysurvey_dr10_south_21"
export HF_DOWNLOAD_ENV="${HF_DOWNLOAD_ENV:-$HOME/.local/share/hf-download}"
export HF_DOWNLOAD_PYTHON="$HF_DOWNLOAD_ENV/bin/python"
# Creer ces caches apres l'entree sur le noeud : /tmp est local et ephemere.
export LEGACY_LOCAL_TMP="$(mktemp -d /tmp/mmu-legacy.XXXXXX)"
export TMPDIR="$LEGACY_LOCAL_TMP/tmp"
export UV_CACHE_DIR="$LEGACY_LOCAL_TMP/uv"
export HF_HUB_CACHE="$LEGACY_LOCAL_TMP/hub"
export HF_XET_CACHE="$LEGACY_LOCAL_TMP/xet"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTHONUNBUFFERED=1
mkdir -p "$LEGACY_DEST" "$LEGACY_RAW/logs" "$TMPDIR" \
  "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$HF_XET_CACHE"
