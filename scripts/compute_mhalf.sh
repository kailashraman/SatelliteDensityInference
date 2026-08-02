#!/bin/bash
# Mass enclosed within dwarfs' half-light radii, plus four catalog-global
# halo properties (rho30, M30, rho150, Mvir200).
#
# *** Two passes, and they are NOT interchangeable. ***
#
# Pass 1 -- array, one task per dwarf, writes ONLY the per-dwarf product:
#     data/additional/mhalf/<version>/<dwarf>.npz
#
#   sbatch scripts/compute_mhalf.sh                        # Diemer
#   sbatch --export=ALL,VERSION=Zhao scripts/compute_mhalf.sh
#
# Pass 2 -- a SINGLE task, writes ONLY the four catalog-global products:
#     data/additional/<version>_{rho30,M30,rho150,Mvir200}.npz
#
#   sbatch --array=0-0 --export=ALL,VERSION=Diemer,MHALF_GLOBALS=1 scripts/compute_mhalf.sh
#
# Run pass 2 for a version BEFORE running compute_weights.sh for it --
# compute_weights.py reads <version>_Mvir200.npz. Pass 1 and pass 2 do not
# depend on each other and may run in either order relative to one another,
# but both must complete before compute_weights.sh.
#
# VERSION must be a real sim_version (Diemer, Zhao, m2e12, mass_floor_7,
# Diemer_0p12, ...), never a reweighting-only alias (lmc, lmc_50, lvdb,
# mcdaniel, mhalf_scatter). Both passes read the h5 via get_h5(), which does
# not resolve aliases -- `--globals lmc` raises KeyError rather than silently
# writing to mhalf/lmc/. Alias versions share their underlying sim_version's
# mhalf/ and globals output; run this script for that sim_version instead.
#
# Array size (pass 1) is 43 dwarfs (0-42), from the LVDB catalog -- see the
# same count asserted in scripts/Jdwarf.sh and tests/test_concat_Js.py.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/compute_mhalf_out/ must already exist. It
# is tracked with a .gitkeep for that reason. Submit from the repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=compute_mhalf
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --array=0-42
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/compute_mhalf_out/slurm-%A_%a.out
#SBATCH --exclude=n0000.lr7,n0001.lr7,n0011.lr7,n0117.lr7,n0150.lr7,n0151.lr7,n0153.lr7

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO}/scripts/compute_mhalf_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

if [ "${MHALF_GLOBALS:-0}" = "1" ]; then
    # Pass 2 must be a single task -- see the header. `--array=0-42` above is
    # unconditional, so a globals submission that forgets `--array=0-0` would
    # otherwise reinstate the race this split removed: 43 tasks all
    # overwriting the same four files.
    if [ "${SLURM_ARRAY_TASK_ID}" != "0" ]; then
        echo "compute_mhalf.sh: MHALF_GLOBALS=1 requires a single task; resubmit with --array=0-0 (got SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID})" >&2
        exit 1
    fi
    exec python -u "${REPO}/python/compute_mhalf.py" --globals "${VERSION:-Diemer}"
else
    exec python -u "${REPO}/python/compute_mhalf.py" "${SLURM_ARRAY_TASK_ID}" "${VERSION:-Diemer}"
fi
