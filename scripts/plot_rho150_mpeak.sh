#!/bin/bash
# rho150-vs-Mpeak inference figures for Draco, Segue 1, and Hydrus I
# (drafts_temp/temp_part1.tex fig:rho150_vs_Mpeak).
#
#   sbatch scripts/plot_rho150_mpeak.sh
#   sbatch --export=ALL,VERSION=Diemer scripts/plot_rho150_mpeak.sh
#   sbatch scripts/plot_rho150_mpeak.sh --supplementary                 # all 39 dwarfs, plots/supplementary/rho150_mpeak/
#
# Requires scripts/save_contours.sh to have already completed for this
# version (reads results/paper_contours/rho150_mpeak/<version>/<dwarf>_{
# unweighted,mhalf,F18,K24}.npz) and scripts/compute_mhalf.sh's pass 2 for
# this version's rho150 globals (data/additional/<version>_rho150.npz).
#
# Extra arguments beyond $VERSION (e.g. --supplementary) are forwarded to
# python/plot_rho150_mpeak.py verbatim.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/plot_rho150_mpeak_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=plot_rho150_mpeak
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# One shared 100x100 grid of NFW solutions (~10,000 brentq root-finds, see
# python/plot_rho150_mpeak.py's runtime note) plus three cheap contour plots.
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/plot_rho150_mpeak_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/plot_rho150_mpeak_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/plot_rho150_mpeak.py" --version "${VERSION:-mass_floor_7}" "$@"
