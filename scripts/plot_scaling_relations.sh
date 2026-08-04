#!/bin/bash
# Draft scaling-relation figures (tidal_stripping, L_vs_Mpeak, L_vs_rho150,
# lvdb_kdsa_rho150): single task, all four by default.
#
#   sbatch scripts/plot_scaling_relations.sh                                  # all four
#   sbatch --export=ALL,FIGURE=tidal_stripping scripts/plot_scaling_relations.sh
#
# Requires, for every figure: scripts/compute_quantiles.sh's output
# (results/paper_quantiles/galactocentric/<version>/<dwarf>.npz) for the
# versions each figure reads -- Diemer (all four figures), mass_floor_7
# (L_vs_Mpeak/L_vs_rho150's high-res band) and lvdb (lvdb_kdsa_rho150's LVDB
# series). tidal_stripping additionally reads the splashback SatGen h5
# directly (data/additional/m12res8_10k_Diemer+scatter_sim_all.h5, no
# separate build step) and L_vs_Mpeak/L_vs_rho150 read
# data/additional/{Diemer,mass_floor_7}_{Mvir200,rho150}.npz
# (scripts/compute_mhalf.sh's pass 2).
#
# L_vs_Mpeak.pdf and L_vs_rho150.pdf draw their SatGen bands from unseeded
# scatter (halo_weights.logMstar_Fattahi18/logMstar_Kim24) -- neither figure
# is bit-reproducible run-to-run; see python/plot_scaling_relations.py's
# docstring.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/plot_scaling_relations_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=plot_scaling_relations
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/plot_scaling_relations_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/plot_scaling_relations_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/plot_scaling_relations.py" "${FIGURE:-all}"
