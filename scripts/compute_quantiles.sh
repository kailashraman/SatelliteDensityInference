#!/bin/bash
# 16/50/84th percentile observable quantiles per dwarf.
#
#   sbatch scripts/compute_quantiles.sh                        # Diemer
#   sbatch --export=ALL,VERSION=Zhao scripts/compute_quantiles.sh
#
# Requires, for this version, scripts/compute_mhalf.sh (both passes),
# scripts/compute_weights.sh, and scripts/Jdwarf.sh + scripts/concat_Js.sh (for
# the J-factor quantiles; a dwarf missing halo_Js.npz still gets the
# non-J quantiles, with a printed warning -- see compute_quantiles.py's
# `has_J` guard).
#
# VERSION=Symphony/MWest is unexercised in this repository: their weights and
# quantiles were built on a superseded, contaminated Symphony catalog and have
# been removed. The Symphony/MWest branch here is retained as a faithful port,
# but its logMhalf read has no supplying step. These two catalogs reach the
# paper figures only via get_h5 in plot_number_functions's
# build_distance_statistics_sims and via the data/symphony/ npz tree.
#
# Array size is 43 dwarfs (0-42), from the LVDB catalog -- see the same count
# asserted in scripts/Jdwarf.sh and tests/test_concat_Js.py.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/compute_quantiles_out/ must already exist.
# It is tracked with a .gitkeep for that reason. Submit from the repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=compute_quantiles
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --array=0-42
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/compute_quantiles_out/slurm-%A_%a.out
#SBATCH --exclude=n0000.lr7,n0001.lr7,n0011.lr7,n0117.lr7,n0150.lr7,n0151.lr7,n0153.lr7

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO}/scripts/compute_quantiles_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/compute_quantiles.py" "${SLURM_ARRAY_TASK_ID}" "${VERSION:-Diemer}"
