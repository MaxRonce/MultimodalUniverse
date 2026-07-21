#!/bin/bash
#SBATCH --job-name=ztf_mmu_hats
#SBATCH --account=jrx@cpu
#SBATCH --partition=cpu_p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --array=0-511%32
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit this job from the MMU repository root}}"
PYTHON="${PYTHON:-/lustre/fswork/projects/rech/jrx/commun/aion/bin/python}"
INPUT_DIR="${INPUT_DIR:-/lustre/fsn1/projects/rech/jrx/urx63nr/ztf_dr23_lc_hats_full/ztf_dr23_lc-hats}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH:?SCRATCH is not set}/TimeSeries/local/ztf}"
MANIFEST_DIR="${MANIFEST_DIR:-${SCRATCH}/TimeSeries/build/ztf_direct_manifests}"
SHARDS="${SHARDS:-512}"

[[ -x "${PYTHON}" ]] || { echo "Missing MMU Python environment: ${PYTHON}" >&2; exit 2; }
[[ -d "${INPUT_DIR}/dataset" ]] || { echo "Missing input HATS dataset: ${INPUT_DIR}/dataset" >&2; exit 2; }
[[ "${SLURM_ARRAY_TASK_COUNT}" -eq "${SHARDS}" ]] || {
  echo "Slurm array has ${SLURM_ARRAY_TASK_COUNT} tasks but SHARDS=${SHARDS}" >&2
  exit 2
}
mkdir -p "${OUTPUT_ROOT}" "${MANIFEST_DIR}"
cd "${REPO_DIR}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MMU="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MMU:-0.0.0+hats}"

"${PYTHON}" -u -m scripts.ztf.build_parent_sample_hats \
  --input-dir "${INPUT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --scratch-dir "${MANIFEST_DIR}" \
  --direct-hats \
  --num-shards "${SHARDS}" \
  --shard-idx "${SLURM_ARRAY_TASK_ID}" \
  --num-processes "${SLURM_CPUS_PER_TASK}" \
  --arrow-threads 1 \
  --batch-size 10000 \
  --aion-relaxed-threshold 30 \
  --resume \
  --skip-ingest
