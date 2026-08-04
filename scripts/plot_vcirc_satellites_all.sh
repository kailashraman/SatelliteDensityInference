#!/bin/bash
# V_circ(r) satellite comparison figures (satellites_vcirc_all_fattahi18,
# satellites_vcirc_all_kim24, satellites_vcirc_intro): single task, all three
# by default.
#
#   sbatch scripts/plot_vcirc_satellites_all.sh                                # all three, Diemer
#   sbatch --export=ALL,FIGURE=intro scripts/plot_vcirc_satellites_all.sh      # one figure
#   sbatch --export=ALL,VERSION=Zhao scripts/plot_vcirc_satellites_all.sh
#
# Requires, for the version being rendered: the SatGen h5 (config.h5_path),
# results/paper_massProfile/<version>/massProfile.npz (scripts/massProfile.sh
# + concat_massProfile.sh), results/paper_quantiles/galactocentric/<version>/
# (fattahi18/kim24 only), and results/vcirc_scatter_300pc/<version>/{Fattahi18,
# Kim24}/scatter_300pc.txt (scripts/vcirc_scatter_300pc.sh +
# concat_vcirc_scatter_300pc.sh; fattahi18/kim24 only).
#
# Like python/plot_vcirc_satellites_all.py itself, VERSION here is used
# directly, not resolved through a sim_version alias.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/plot_vcirc_satellites_all_out/ must
# already exist. It is tracked with a .gitkeep for that reason. Submit from
# the repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=plot_vcirc_satellites_all
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/plot_vcirc_satellites_all_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/plot_vcirc_satellites_all_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/plot_vcirc_satellites_all.py" \
  "${FIGURE:-all}" "${VERSION:-Diemer}"
