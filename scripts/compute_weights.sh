#!/bin/bash
# SHMR / kinematic reweighting of SatGen halos: one array task per dwarf.
#
#   sbatch scripts/compute_weights.sh                        # Diemer
#   sbatch --export=ALL,VERSION=Zhao scripts/compute_weights.sh
#
# Requires BOTH passes of scripts/compute_mhalf.sh to have already been run
# for this version's underlying catalog: compute_weights.py reads
# data/additional/<sim_version>_Mvir200.npz (pass 2, --globals) AND
# data/additional/mhalf/<sim_version>/<dwarf>.npz (pass 1, per-dwarf) via
# halo_weights.py's from_logMstar/from_logMhalf.
#
# Full regeneration covers all nine versions this repository carries:
#   Diemer, Diemer_0p12, Zhao, lmc, lmc_50, lvdb, m2e12, mass_floor_7,
#   mhalf_scatter (see docs/migration-notes.md). Submit this script once per
#   version, after both compute_mhalf.sh passes have completed for that
#   version's underlying sim_version (lmc/lmc_50/lvdb/mhalf_scatter all share
#   Diemer's).
#
# Regenerating changes mstar_weights and joint_weights relative to the
# migrated weights_gc/ products, for two independent reasons: (1) the
# from_logMstar fix that forwards logMhalf_scatter to obs.Dwarf instead of
# silently dropping it, and (2) this script now seeds the global NumPy RNG
# (see compute_weights.py's module docstring), so even a rerun with fix (1)
# alone draws a fresh, seeded SHMR-scatter realization rather than
# reproducing the migrated one. mhalf_weights is unaffected by either change:
# from_logMhalf already forwarded logMhalf_scatter, and it does not draw from
# the RNG at all -- it reproduces bit-exact.
#
# Downstream of mstar_weights/joint_weights (see docs/migration-notes.md,
# "SHMR weights are one unseeded realization"): panel 2 of
# multipanel_systematics.pdf (Fattahi18, Moster18, Kim+24, Danieli+23) and the
# rho150.pdf inference curves, plus every paper_quantiles/galactocentric
# product and everything built from it (both parts of the paper series share
# these upstream intermediates -- check both drafts, not just one, before
# treating a regeneration as scoped to a single figure).
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
# this script body runs, so scripts/compute_weights_out/ must already exist.
# It is tracked with a .gitkeep for that reason. Submit from the repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=compute_weights
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --array=0-42
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/compute_weights_out/slurm-%A_%a.out
#SBATCH --exclude=n0000.lr7,n0001.lr7,n0011.lr7,n0117.lr7,n0150.lr7,n0151.lr7,n0153.lr7

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO}/scripts/compute_weights_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/compute_weights.py" "${SLURM_ARRAY_TASK_ID}" "${VERSION:-Diemer}"
