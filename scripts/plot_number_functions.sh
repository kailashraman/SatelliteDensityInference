#!/bin/bash
# "Number function" figures (sims_mpeak, sims_rho150, rho150, Jfactors_all,
# distance_statistics, distance_statistics_sims): single task, all six by
# default.
#
#   sbatch scripts/plot_number_functions.sh                                     # all six, Diemer, seed 0
#   sbatch --export=ALL,FIGURE=jfactors_all scripts/plot_number_functions.sh    # one figure
#   sbatch --export=ALL,VERSION=Zhao scripts/plot_number_functions.sh
#
# Requires, for the version being rendered: the SatGen h5 (config.h5_path),
# its rho150 file (data/additional/<version>_rho150.npz, from
# scripts/compute_mhalf.sh's pass 2), and scripts/compute_weights.sh's output
# (data/additional/weights_gc/<version>/<dwarf>.npz) for inference_rho150.
# jfactors_all additionally requires results/paper_Js/halo_position/<version>/
# (scripts/Jhalopos.sh + concat_Js.sh) and results/paper_Js/<version>/<dwarf>/
# (scripts/Jdwarf.sh + concat_Js.sh), and refuses any version but Diemer --
# see python/plot_number_functions.py's build_jfactors_all docstring.
#
# Like python/plot_number_functions.py itself, VERSION here is used directly,
# not resolved through a sim_version alias.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/plot_number_functions_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=plot_number_functions
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/plot_number_functions_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/plot_number_functions_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/plot_number_functions.py" \
  "${FIGURE:-all}" "${VERSION:-Diemer}" "${SEED:-0}"
