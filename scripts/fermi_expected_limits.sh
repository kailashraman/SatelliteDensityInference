#!/bin/bash
# Fermi expected-limit (blank-field null distribution) TS grids: one array
# task per dwarf, looping over the 1008 blank-field SEDs internally.
#
#   sbatch scripts/fermi_expected_limits.sh                            # all variants
#   sbatch --export=ALL,VARIANTS=mhalf,Jeans scripts/fermi_expected_limits.sh
#
# No legacy/update choice here: unlike fermi_reweighting.py, this script never
# reads the dwarf's own SED, only data/fermi_legacy/blank_fields/, which has
# no _update counterpart (see python/fermi_expected_limits.py's docstring).
# results/fermi_expected_limits/ therefore does not carry a version suffix.
#
# Requires the paper_quantiles/galactocentric/Diemer tree (for mhalf/F18) and,
# for Jeans/satgen_shmr_fattahi18, DwarfJeansAnalysis posteriors under
# $SDI_DJA_RESULTS_DIR -- fermi_expected_limits.py skips (does not fail) the
# satgen_shmr_fattahi18 variant when a dwarf has no such posterior.
#
# This is the expensive stage: one convert_sed() Minuit-free profile fit per
# (blank field, variant), 1008 fields per dwarf per variant. Time below is a
# starting point, not a measured bound -- watch the first few tasks and raise
# --time if they are still running near the limit.
#
# Array size is 43 dwarfs (0-42), from the LVDB catalog -- see the same count
# asserted in scripts/Jdwarf.sh and tests/test_concat_Js.py.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/fermi_expected_limits_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=fermi_expected_limits
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --array=0-42
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/fermi_expected_limits_out/slurm-%A_%a.out
#SBATCH --exclude=n0000.lr7,n0001.lr7,n0011.lr7,n0117.lr7,n0150.lr7,n0151.lr7,n0153.lr7

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
mkdir -p "${REPO}/scripts/fermi_expected_limits_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

ARGS=("${SLURM_ARRAY_TASK_ID}")
if [[ -n "${VARIANTS:-}" ]]; then
  ARGS+=(--variants "${VARIANTS}")
fi

exec python -u "${REPO}/python/fermi_expected_limits.py" "${ARGS[@]}"
