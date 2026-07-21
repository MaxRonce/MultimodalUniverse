#!/bin/bash
#SBATCH --job-name=ztf_hats_finalize
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit this job from the MMU repository root}}"
PYTHON="${PYTHON:-/lustre/fswork/projects/rech/jrx/commun/aion/bin/python}"
INPUT_DIR="${INPUT_DIR:-/lustre/fsn1/projects/rech/jrx/urx63nr/ztf_dr23_lc_hats_full/ztf_dr23_lc-hats}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:?SCRATCH is not set}/TimeSeries/local/ztf}"
MANIFEST_DIR="${MANIFEST_DIR:-${SCRATCH}/TimeSeries/build/ztf_direct_manifests}"
SHARDS="${SHARDS:-512}"

[[ -x "${PYTHON}" ]] || { echo "Missing MMU Python environment: ${PYTHON}" >&2; exit 2; }
for shard in $(seq 0 $((SHARDS - 1))); do
  printf -v manifest "%s/manifest-direct-shard-%03d.json" "${MANIFEST_DIR}" "${shard}"
  [[ -s "${manifest}" ]] || { echo "Missing shard manifest: ${manifest}" >&2; exit 3; }
done
cd "${REPO_DIR}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MMU="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MMU:-0.0.0+hats}"

"${PYTHON}" -u -m scripts.ztf.build_parent_sample_hats \
  --input-dir "${INPUT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --scratch-dir "${MANIFEST_DIR}" \
  --finalize-direct-hats \
  --num-shards "${SHARDS}" \
  --shard-idx 0

CATALOG="${OUTPUT_ROOT}/ztf/ztf"
[[ -s "${CATALOG}/hats.properties" ]] || { echo "Missing HATS properties" >&2; exit 4; }
[[ -s "${CATALOG}/partition_info.csv" ]] || { echo "Missing partition info" >&2; exit 4; }
[[ -s "${CATALOG}/aion_quality_policy.json" ]] || { echo "Missing AION policy" >&2; exit 4; }
echo "Finalized ZTF HATS catalog: ${CATALOG}"
