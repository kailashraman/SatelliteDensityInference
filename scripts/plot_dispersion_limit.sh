#!/bin/bash
# sigma_los-vs-r_half detectability-contour figure (dispersion_limit_contours.pdf).
#
#   sbatch scripts/plot_dispersion_limit.sh
#
# Requires: the Diemer SatGen h5 (config.h5_path) and
# results/paper_massProfile/Diemer/massProfile.npz (scripts/massProfile.sh +
# scripts/concat_massProfile.sh). Version is fixed to 'Diemer' in
# python/plot_dispersion_limit.py, matching the notebook cell it reproduces.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/plot_dispersion_limit_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=plot_dispersion_limit
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/plot_dispersion_limit_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/plot_dispersion_limit_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/plot_dispersion_limit.py"
