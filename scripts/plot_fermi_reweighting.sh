#!/bin/bash
# Fermi recasting figures (prior ladder, SHMR slice, prior envelope, MC band):
# reads results/fermi_reweighting_update/ (Circiello+26 SEDs, cited throughout
# the drafts) and writes plots/fermi/*.pdf.
#
#   sbatch scripts/plot_fermi_reweighting.sh                          # all four figures
#   sbatch --export=ALL,FIGURE=mc scripts/plot_fermi_reweighting.sh   # one figure
#
# --fermi-version update is passed explicitly below -- python/plot_fermi_reweighting.py's
# own default stays 'legacy' (matching fermi_reweighting.py's CLI default
# convention), so this launcher, not a buried default, is what makes 'update'
# the tree actually read.
#
# Requires scripts/fermi_reweighting_update.sh to have already completed for
# every non-skipped dwarf (see python/plot_fermi_reweighting.py's SKIP_DWARFS)
# and every weight index a requested figure reads.
#
# Single task: this script renders figures from already-computed TS grids, it
# does not recompute them, so there is no per-dwarf array to parallelise over.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/plot_fermi_reweighting_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=plot_fermi_reweighting
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/plot_fermi_reweighting_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/plot_fermi_reweighting_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/plot_fermi_reweighting.py" "${FIGURE:-all}" --fermi-version update
