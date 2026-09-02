#!/bin/bash
#SBATCH --job-name=mmu-lsst-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00

set -euo pipefail

: "${SCRATCH:?SCRATCH must be set}"
ROOT_BASE="${MMU_JZ_ROOT:-$SCRATCH/mmu_lsst_dp2}"
REPO="${MMU_REPO:-$ROOT_BASE/MultimodalUniverse}"

cd "$REPO"
exec bash scripts/lsst_dp2/run_prepost.sh
