#!/bin/bash
# jeans_corner.pdf (Ursa Major II, loguniform prior) -- fetched from
# DwarfJeansAnalysis's own scripts/plot_posteriors.py, not reimplemented
# here (see python/fetch_jeans_corner.py's module docstring). Single task,
# no array.
#
#   sbatch scripts/fetch_jeans_corner.sh
#   sbatch --export=ALL,GALAXY_KEY=ursa_major_2,PRIOR=loguniform scripts/fetch_jeans_corner.sh
#
# Requires a completed DJA production run at
# $SDI_DJA_RESULTS_DIR/<GALAXY_KEY>/<PRIOR>/posterior_samples.npz (default
# config.DJA_RESULTS_DIR if that env var is unset).
#
# Deviates from this repo's other launchers in one respect: it activates the
# `DwarfJeans` conda environment, not `J_calc`. plot_posteriors.py imports
# the editable-installed `dwarfjeans` package, which only exists in
# `DwarfJeans`; this script's own python/fetch_jeans_corner.py needs nothing
# beyond numpy + stdlib, both present there too, so one environment covers
# the whole task.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/fetch_jeans_corner_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=fetch_jeans_corner
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/fetch_jeans_corner_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/fetch_jeans_corner_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate DwarfJeans

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/fetch_jeans_corner.py" \
    --galaxy-key "${GALAXY_KEY:-ursa_major_2}" --prior "${PRIOR:-loguniform}"
