#!/bin/bash
# V_circ-r_max and rho150-Mpeak contours (both galactocentric-weighted and
# Jeans-prior-ladder): one array task per dwarf.
#
#   sbatch scripts/save_contours.sh                                    # Diemer, z0
#   sbatch --export=ALL,VERSION=Zhao,REDSHIFT=infall scripts/save_contours.sh
#
# Requires scripts/compute_weights.sh to have already completed for this
# version (reads data/additional/weights_gc/<version>/<dwarf>.npz) and
# scripts/compute_mhalf.sh's pass 2 for this version's rho150 file
# (data/additional/<version>_rho150.npz):
#
#   sbatch --array=0-0 --export=ALL,VERSION=Diemer,MHALF_GLOBALS=1 scripts/compute_mhalf.sh
#
# Like python/save_contours.py itself, VERSION here is used directly, not
# resolved through a sim_version alias -- only versions that are direct
# H5_REGISTRY keys (Diemer, Zhao, mass_floor_7, cut_m9, cut_m8p5, ...) work.
#
# Array size is 43 dwarfs (0-42), from the LVDB catalog -- see the same count
# named in scripts/Jdwarf.sh's header (the actual assertion lives in
# Jdwarf.py, not that comment) and asserted in tests/test_concat_Js.py.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/save_contours_out/ must already exist. It
# is tracked with a .gitkeep for that reason. Submit from the repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=save_contours
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# 2:00:00 is upstream's launcher value, not a walltime measured for this
# repo's task -- a single task does many FFTKDE evaluations at 1024^2 over up
# to ~2.4M halos, plus seven external Jeans reads. Restored after being cut
# to 00:30:00 with nothing behind the cut; a wall-clock kill still lands
# preferentially in the last write of a task, leaving a truncated tree, even
# though each write is now stamped atomically (see save_contours.py).
#SBATCH --time=2:00:00
#SBATCH --array=0-42
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/save_contours_out/slurm-%A_%a.out
#SBATCH --exclude=n0000.lr7,n0001.lr7,n0011.lr7,n0117.lr7,n0150.lr7,n0151.lr7,n0153.lr7

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
mkdir -p "${REPO}/scripts/save_contours_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/save_contours.py" "${SLURM_ARRAY_TASK_ID}" "${VERSION:-Diemer}" "${REDSHIFT:-z0}"
