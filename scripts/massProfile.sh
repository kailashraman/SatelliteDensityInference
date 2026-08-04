#!/bin/bash
# Per-halo mass/velocity profiles: one array task covers CHUNK_SIZE=50000-halo
# groups (python/massProfile.py's CHUNK_SIZE) in batches of up to 32,
# dispatched to GNU parallel with --jobs=JOBS_PER_NODE. As currently sized
# (--cpus-per-task=1 on a shared node), JOBS_PER_NODE evaluates to 1, so the
# 32 groups per task run SERIALLY, not 32-to-a-node in parallel -- see the
# note below --mem for what it would take to actually pack them.
#
#   sbatch scripts/massProfile.sh                        # Diemer
#   sbatch --export=ALL,VERSION=Zhao scripts/massProfile.sh
#
# Sizing --array for a given catalog: n_groups = ceil(n_halos / CHUNK_SIZE),
# array = 0..ceil(n_groups/32)-1. Compute n_halos for the target version's h5
# (config.h5_path) before submitting and edit --array below accordingly --
# this script does not size itself, since a wrong array bound would either
# skip trailing halos (too small) or waste tasks on empty groups (too large).
#
# *** This stage is NOT finished when the array completes. ***
# massProfile.py writes per-group fragments:
#     results/paper_massProfile/<version>/massProfile/massProfile-<g>.npz
# but every downstream consumer (vcirc_scatter_300pc.py) reads the
# concatenated
#     results/paper_massProfile/<version>/massProfile.npz
# produced by a separate pass:
#     scripts/concat_massProfile.sh <version>
# concat_massProfile refuses to write unless the fragments are contiguous,
# share one r_grid, and tile the catalog exactly, so a partially-failed array
# is caught there rather than silently producing a short, misaligned array.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/massProfile_out/ must already exist. It is
# tracked with a .gitkeep for that reason. Submit from the repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=massProfile
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --time=24:00:00
# Diemer: 2432159 halos / CHUNK_SIZE 50000 = 49 groups needed; 32 groups per
# task -> --array=0-1, which submits groups 0-63 and reproduces the 64-fragment
# upstream tree (groups 49-63 come out empty and concat tolerates them). The
# value shipped upstream was 0-3, which submits 128 groups and writes 79 empty
# fragments. Recompute per catalog: m2e12 and mass_floor_7 both need 0-2.
#SBATCH --array=0-1
#SBATCH --cpus-per-task=1
# JOBS_PER_NODE below is SLURM_CPUS_ON_NODE / SLURM_CPUS_PER_TASK, which is 32
# only if this allocation actually owns the node. With --cpus-per-task=1 on a
# shared node it evaluates to 1 and the 32 groups run SERIALLY inside the 24 h
# wall clock -- which still completes, but is not what the header describes.
# --mem is per task, sized for one Python process: ~97 MB of catalog arrays,
# ~121 MB of output, 50000 Green instances and the Green+19 interpolation
# grids. Raise it in step with JOBS_PER_NODE if this is ever made to pack.
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/massProfile_out/slurm-%A_%a.out
#SBATCH --exclude=n0394.lr6,n0395.lr6,n0396.lr6,n0318.lr6,n0350.lr6,n0048.lr4,n0319.lr6,n0320.lr6,n0321.lr6,n0056.lr6,n0352.lr6,n0359.lr6

# SLURM copies this script to a node-local spool directory (/var/spool/slurmd)
# before executing it, so ${BASH_SOURCE[0]} does NOT point into the repository
# under sbatch -- it resolves to the spool copy and every repo-relative path
# below silently becomes wrong. Prefer SLURM_SUBMIT_DIR (these headers already
# require submission from the repository root); fall back to the script's own
# location for direct shell invocation. Validate either way and fail loudly
# rather than resolving to the wrong tree.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ ! -f "${REPO}/python/config.py" ]]; then
  echo "ERROR: REPO=${REPO} is not the SatelliteDensityInference root" >&2
  echo "       (no python/config.py). Submit from the repository root." >&2
  exit 2
fi
mkdir -p "${REPO}/scripts/massProfile_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc
module load parallel

set -euo pipefail
cd "${REPO}/python"

VERSION="${VERSION:-Diemer}"

# Export jobs per node
export JOBS_PER_NODE=$(( SLURM_CPUS_ON_NODE / SLURM_CPUS_PER_TASK ))

# Calculate group indices for this array task
idx0=$(( SLURM_ARRAY_TASK_ID * 32 ))
idx1=$(( SLURM_ARRAY_TASK_ID * 32 + 31 ))

echo "Start group: ${idx0}"
echo "Stop group: ${idx1}"

/usr/bin/time --verbose bash -c \
  "seq ${idx0} ${idx1} | parallel --jobs ${JOBS_PER_NODE} python -u massProfile.py ${VERSION} {}"
