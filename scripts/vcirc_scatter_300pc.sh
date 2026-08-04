#!/bin/bash
# Dex scatter on the weighted-median V_circ at r=300pc: one array task per
# (dwarf, SHMR) pair. Array size is 43 dwarfs x len(SHMR_NAMES) SHMRs
# (len(SHMR_NAMES) == 10 today) = 430 tasks (0-429), via
# dwarf_idx = task_id // len(SHMR_NAMES), shmr_idx = task_id % len(SHMR_NAMES)
# in python/vcirc_scatter_300pc.py.
#
# The dwarf count is data-derived (Jdata.dwarf_names, built from the pinned
# LVDB catalog) and the SHMR count from vcirc_scatter_params.SHMR_NAMES, so
# this --array bound must track len(dwarf_names)*len(SHMR_NAMES) by hand if
# either count ever changes. vcirc_scatter_300pc.py parses this --array line
# itself (LAUNCHER_N_TASKS) and asserts it equals
# len(dwarf_names)*len(SHMR_NAMES) at task 0, so a stale bound here fails
# loudly instead of silently dropping the tail dwarfs.
#
#   sbatch scripts/vcirc_scatter_300pc.sh                        # Diemer
#   sbatch --export=ALL,VERSION=Zhao scripts/vcirc_scatter_300pc.sh
#
# Requires results/paper_massProfile/<version>/massProfile.npz to already
# exist -- i.e. scripts/massProfile.sh followed by scripts/concat_massProfile.sh
# must have completed for this version first.
#
# *** This stage is NOT finished when the array completes. ***
# vcirc_scatter_300pc.py writes one CSV row per (dwarf, SHMR):
#     results/vcirc_scatter_300pc/<version>/<SHMR>/per_dwarf/row_<dwarf_idx>.csv
# Concatenate the rows into one CSV per SHMR with:
#     scripts/concat_vcirc_scatter_300pc.sh <version>
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/vcirc_scatter_300pc_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=vcirc_scatter_300pc
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
# Loads massProfile.npz's vcircProfile (n_halos x 101 float64 = 2.0 GB at
# Diemer) alongside a full ResampleMstar. Exceeds the 4590M default.
#SBATCH --mem=8G
# Upstream shipped 00:30:00, which is marginal rather than generous: in a full
# 430-task run the slowest COMPLETED task took 29:39 and 7 tasks hit the limit
# exactly (00:30:14) on loaded nodes. A timed-out task leaves the PREVIOUS
# run's CSV in place, so the tree still looks complete -- the survivors are
# only distinguishable by their missing provenance sidecar. Sized well clear of
# the observed maximum rather than just above it.
#SBATCH --time=02:00:00
#SBATCH --array=0-429
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/vcirc_scatter_300pc_out/slurm-%A_%a.out
#SBATCH --exclude=n0394.lr6,n0395.lr6,n0396.lr6,n0318.lr6,n0350.lr6,n0048.lr4,n0319.lr6,n0320.lr6,n0321.lr6,n0056.lr6

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
mkdir -p "${REPO}/scripts/vcirc_scatter_300pc_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/vcirc_scatter_300pc.py" "${SLURM_ARRAY_TASK_ID}" "${VERSION:-Diemer}"
