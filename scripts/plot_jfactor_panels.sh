#!/bin/bash
# J-factor comparison panels (panel_Jfactor_theta95, multipanel_jeans_priors,
# panel_Jfactor_literature): single task, all three by default.
#
#   sbatch scripts/plot_jfactor_panels.sh
#   sbatch --export=ALL,FIGURE=panel_Jfactor_literature scripts/plot_jfactor_panels.sh
#
# Requires results/paper_quantiles/galactocentric/Diemer/<dwarf>.npz for all
# 43 dwarfs (scripts/compute_quantiles.sh) and, for every dwarf/prior pair
# each figure plots, DwarfJeansAnalysis's
# <DJA_RESULTS_DIR>/<lvdb_key>/<prior>/derived.npz -- see
# python/plot_jfactor_panels.py's FIGURE_PRIORS and module docstring.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/plot_jfactor_panels_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=plot_jfactor_panels
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/plot_jfactor_panels_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/plot_jfactor_panels_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/plot_jfactor_panels.py" "${FIGURE:-all}"
