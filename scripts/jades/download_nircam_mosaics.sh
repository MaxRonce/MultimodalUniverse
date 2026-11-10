#!/usr/bin/env bash
# Resumable JADES DR5 NIRCam download with live progress and per-band logs.
set -Eeuo pipefail

RAW="${RAW:-/mnt/ceph/users/polymathic/external_data/Astro/JADES_DR5}"
LOG="${LOG:-/mnt/ceph/users/polymathic/mronceray/jades_dr5_runs/logs}"
BASE_URL="${JADES_BASE_URL:-http://slate.ucsc.edu/~brant/jades-dr5}"
mkdir -p "$LOG"

SOUTH=(f070w f090w f115w f150w f162m f182m f200w f210m f250m f277w f300m f335m f356w f410m f430m f444w f460m f480m)
NORTH=(f070w f090w f115w f150w f162m f182m f200w f210m f277w f300m f335m f356w f410m f430m f444w f460m)
TOTAL=$(( ${#SOUTH[@]} + ${#NORTH[@]} ))
INDEX=0
DONE=0
ACTIVE=""

on_interrupt() {
  printf '\nInterrupted while downloading %s; rerun to resume. Completed: %d/%d.\n' "$ACTIVE" "$DONE" "$TOTAL" >&2
  exit 130
}
trap on_interrupt INT TERM

for FIELD in GOODS-S GOODS-N; do
  if [[ "$FIELD" == GOODS-S ]]; then
    BANDS=("${SOUTH[@]}")
  else
    BANDS=("${NORTH[@]}")
  fi
  mkdir -p "$RAW/mosaics/$FIELD"
  for BAND in "${BANDS[@]}"; do
    INDEX=$((INDEX + 1))
    ACTIVE="$FIELD $BAND"
    FILENAME="hlsp_jades_jwst_nircam_${FIELD,,}_${BAND}_v5.0_drz.fits"
    URL="$BASE_URL/$FIELD/hlsp/images/mosaics/$FILENAME"
    DEST="$RAW/mosaics/$FIELD/$FILENAME"
    LOG_FILE="$LOG/${FIELD}_${BAND}.log"

    # Never treat the presence of a partial FITS file as completion.
    SIZE=$(curl -fsSI --max-time 60 "$URL" | awk 'BEGIN {IGNORECASE=1} tolower($1)=="content-length:" {gsub("\r", "", $2); print $2}' | tail -n 1)
    if [[ ! "$SIZE" =~ ^[0-9]+$ ]]; then
      printf '[%d/%d] ERROR: missing Content-Length for %s\n' "$INDEX" "$TOTAL" "$ACTIVE" >&2
      exit 1
    fi

    LOCAL_SIZE=0
    [[ ! -f "$DEST" ]] || LOCAL_SIZE=$(stat -c %s "$DEST")
    if (( LOCAL_SIZE == SIZE )); then
      printf '[%d/%d] SKIP %s — complete (%s bytes)\n' "$INDEX" "$TOTAL" "$ACTIVE" "$SIZE"
      DONE=$((DONE + 1))
      continue
    fi
    if (( LOCAL_SIZE > SIZE )); then
      printf '[%d/%d] ERROR: local file larger than remote: %s\n' "$INDEX" "$TOTAL" "$DEST" >&2
      exit 1
    fi

    printf '\n[%d/%d] Downloading %s — resume %.1f%%\n' "$INDEX" "$TOTAL" "$ACTIVE" "$(awk -v n="$LOCAL_SIZE" -v d="$SIZE" 'BEGIN {printf "%.1f", 100*n/d}')"
    # Show wget's live bar on the terminal AND store logs. pipefail catches failures.
    if ! wget --continue --tries=20 --timeout=120 \
        --progress=bar:force:noscroll --output-document="$DEST" "$URL" 2>&1 | tee -a "$LOG_FILE"; then
      printf 'FAILED: %s — see %s\n' "$ACTIVE" "$LOG_FILE" >&2
      exit 1
    fi
    ACTUAL=$(stat -c %s "$DEST")
    if (( ACTUAL != SIZE )); then
      printf 'FAILED: %s size %s != expected %s\n' "$ACTIVE" "$ACTUAL" "$SIZE" >&2
      exit 1
    fi
    DONE=$((DONE + 1))
    printf '[%d/%d] DONE %s (%s bytes)\n' "$DONE" "$TOTAL" "$ACTIVE" "$ACTUAL"
  done
done

printf '\nALL %d NIRCAM MOSAICS DOWNLOADED (size-checked).\n' "$TOTAL"
