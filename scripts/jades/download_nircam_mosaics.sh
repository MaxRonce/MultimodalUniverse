#!/usr/bin/env bash
# JADES DR5: parallel, resumable downloads with live global progress.
set -Eeuo pipefail
RAW="${RAW:-/mnt/ceph/users/polymathic/external_data/Astro/JADES_DR5}"
LOG="${LOG:-/mnt/ceph/users/polymathic/mronceray/jades_dr5_runs/logs}"
BASE_URL="${JADES_BASE_URL:-http://slate.ucsc.edu/~brant/jades-dr5}"
JOBS=4
INTERVAL=10
usage() { echo "Usage: bash $0 [--jobs 1..8] [--interval seconds]"; }
while (( $# )); do
  case "$1" in
    --jobs) JOBS="${2:-}"; shift 2 ;;
    --interval) INTERVAL="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [[ ! "$JOBS" =~ ^[1-8]$ || ! "$INTERVAL" =~ ^[1-9][0-9]*$ ]]; then usage >&2; exit 2; fi
SOUTH=(f070w f090w f115w f150w f162m f182m f200w f210m f250m f277w f300m f335m f356w f410m f430m f444w f460m f480m)
NORTH=(f070w f090w f115w f150w f162m f182m f200w f210m f277w f300m f335m f356w f410m f430m f444w f460m)
declare -a NAMES URLS DESTS SIZES PIDS STATES
declare -A PID_TO_INDEX
mkdir -p "$LOG"
stop_children() {
  trap - INT TERM
  for pid in "${PIDS[@]}"; do [[ -z "${pid:-}" ]] || kill "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do [[ -z "${pid:-}" ]] || wait "$pid" 2>/dev/null || true; done
}
on_interrupt() {
  printf '\nInterrupted. Stopping active downloads; rerun to resume.\n' >&2
  stop_children
  exit 130
}
trap on_interrupt INT TERM

echo "Checking remote file sizes..."
for FIELD in GOODS-S GOODS-N; do
  if [[ "$FIELD" == GOODS-S ]]; then BANDS=("${SOUTH[@]}"); else BANDS=("${NORTH[@]}"); fi
  mkdir -p "$RAW/mosaics/$FIELD"
  for BAND in "${BANDS[@]}"; do
    FILENAME="hlsp_jades_jwst_nircam_${FIELD,,}_${BAND}_v5.0_drz.fits"
    URL="$BASE_URL/$FIELD/hlsp/images/mosaics/$FILENAME"
    DEST="$RAW/mosaics/$FIELD/$FILENAME"
    SIZE=$(curl -fsSI --max-time 60 "$URL" | awk 'tolower($1)=="content-length:" {gsub("\r", "", $2); print $2}' | tail -n 1)
    if [[ ! "$SIZE" =~ ^[1-9][0-9]*$ ]]; then echo "ERROR: missing Content-Length: $FIELD $BAND" >&2; exit 1; fi
    i=${#NAMES[@]}
    NAMES[i]="$FIELD $BAND"; URLS[i]="$URL"; DESTS[i]="$DEST"; SIZES[i]="$SIZE"
    PIDS[i]=""; STATES[i]="pending"
    local_size=0
    [[ ! -f "$DEST" ]] || local_size=$(stat -c %s "$DEST")
    if (( local_size > SIZE )); then echo "ERROR: local file larger than remote: $DEST" >&2; exit 1; fi
    if (( local_size == SIZE )); then STATES[i]="done"; fi
  done
done
TOTAL=${#NAMES[@]}
TOTAL_BYTES=0
BASELINE=0
for ((i=0; i<TOTAL; i++)); do
  TOTAL_BYTES=$((TOTAL_BYTES + SIZES[i]))
  [[ ! -f "${DESTS[i]}" ]] || BASELINE=$((BASELINE + $(stat -c %s "${DESTS[i]}")))
done
START=$(date +%s)

print_status() {
  local now elapsed current=0 done=0 active=0 rate eta bytes percent summary=""
  now=$(date +%s); elapsed=$((now - START))
  for ((i=0; i<TOTAL; i++)); do
    bytes=0
    [[ ! -f "${DESTS[i]}" ]] || bytes=$(stat -c %s "${DESTS[i]}")
    current=$((current + bytes))
    [[ "${STATES[i]}" != done ]] || done=$((done + 1))
    if [[ "${STATES[i]}" == running ]]; then
      active=$((active + 1)); percent=$((100 * bytes / SIZES[i]))
      summary+=" ${NAMES[i]}:${percent}%"
    fi
  done
  rate=0
  if (( elapsed > 0 && current >= BASELINE )); then rate=$(((current - BASELINE) / elapsed)); fi
  eta="unknown"
  if (( rate > 0 )); then
    remaining=$(((TOTAL_BYTES - current) / rate))
    eta="$((remaining / 3600))h $(((remaining % 3600) / 60))m"
  fi
  awk -v c="$current" -v t="$TOTAL_BYTES" -v r="$rate" -v d="$done" -v n="$TOTAL" -v a="$active" -v e="$eta" \
    'BEGIN {printf "[GLOBAL] %.1f%% | %.2f/%.2f TB | %d/%d complete | %d active | %.2f MB/s | ETA %s\n", 100*c/t, c/1e12, t/1e12, d, n, a, r/1e6, e}'
  [[ -z "$summary" ]] || printf '         %s\n' "$summary"
}

running=0; failed=0; next=0
printf 'Starting %d mosaics, %d parallel downloads.\n' "$TOTAL" "$JOBS"
print_status
while :; do
  while (( running < JOBS && next < TOTAL )); do
    i=$next; next=$((next + 1))
    if [[ "${STATES[i]}" == done ]]; then echo "SKIP ${NAMES[i]} (size complete)"; continue; fi
    field=${NAMES[i]%% *}; band=${NAMES[i]#* }
    log_file="$LOG/${field}_${band}.log"
    echo "START ${NAMES[i]} — $log_file"
    wget --continue --tries=20 --timeout=120 --progress=dot:giga \
      --output-document="${DESTS[i]}" "${URLS[i]}" >> "$log_file" 2>&1 &
    pid=$!; PIDS[i]="$pid"; PID_TO_INDEX[$pid]=$i; STATES[i]="running"
    running=$((running + 1))
  done
  if (( running == 0 )); then break; fi
  sleep "$INTERVAL" & wait "$!" || true
  for pid in "${!PID_TO_INDEX[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then continue; fi
    i=${PID_TO_INDEX[$pid]}
    if wait "$pid"; then status=0; else status=$?; fi
    unset 'PID_TO_INDEX[$pid]'
    PIDS[i]=""; running=$((running - 1))
    actual=0
    [[ ! -f "${DESTS[i]}" ]] || actual=$(stat -c %s "${DESTS[i]}")
    if (( status == 0 && actual == SIZES[i] )); then
      STATES[i]="done"; echo "DONE ${NAMES[i]} ($actual bytes)"
    else
      STATES[i]="failed"; failed=$((failed + 1))
      echo "FAILED ${NAMES[i]} (wget exit $status, $actual/${SIZES[i]} bytes)" >&2
    fi
  done
  print_status
done
print_status
if (( failed > 0 )); then echo "$failed downloads failed; rerun to resume." >&2; exit 1; fi
echo "ALL $TOTAL NIRCAM MOSAICS DOWNLOADED (size-checked)."
