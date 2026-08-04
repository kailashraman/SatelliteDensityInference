#!/bin/bash
# Part I's main dwarf multipanel figure (sigma_LOS/M_peak/rho150 across the
# 39-dwarf canonical ordering): reads
# results/paper_quantiles/galactocentric/<version>/<dwarf>.npz and writes
# plots/panels/multipanel_<version>_paper1.pdf.
#
#   sbatch scripts/plot_multipanel_paper1.sh                          # Diemer
#   sbatch --export=ALL,VERSION=Zhao scripts/plot_multipanel_paper1.sh
#
# Requires scripts/compute_quantiles.sh to have already completed for this
# version (produces results/paper_quantiles/galactocentric/<version>/) --
# all 43 dwarves in Jdata.dwarf_names are read, not just the 39 plotted.
#
# Single task: this script renders one figure from already-computed
# quantiles, it does not recompute them, so there is no per-dwarf array to
# parallelise over.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/plot_multipanel_paper1_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=plot_multipanel_paper1
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/plot_multipanel_paper1_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/plot_multipanel_paper1_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/plot_multipanel_paper1.py" --version "${VERSION:-Diemer}"
