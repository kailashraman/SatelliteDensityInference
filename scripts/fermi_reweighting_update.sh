#!/bin/bash
# Fermi TS grids for the 'update' Fermi-LAT SED tree (Circiello:2026inp, cited
# throughout the drafts): one array task per dwarf.
#
#   sbatch scripts/fermi_reweighting_update.sh                    # all variants
#   sbatch --export=ALL,VARIANTS=mhalf,F18,K24 scripts/fermi_reweighting_update.sh
#
# --legacy-version update is passed explicitly below -- python/fermi_reweighting.py's
# own default stays 'legacy' (see that module's docstring); this launcher, not
# a buried default, is what makes 'update' the variant actually run.
#
# Requires the paper_quantiles/galactocentric/{Diemer,mhalf_scatter} trees
# (scripts/compute_quantiles.sh) and, for the Jeans_satgen* variants,
# DwarfJeansAnalysis posteriors under $SDI_DJA_RESULTS_DIR to already exist for
# the dwarfs/priors being requested -- fermi_reweighting.py skips (does not
# fail) a Jeans_satgen* variant with no posterior for a given dwarf.
#
# mcdaniel is NOT buildable here: paper_quantiles/galactocentric/mcdaniel was
# deliberately not migrated (its weights are 17 GB and excluded; no in-repo
# producer -- see docs/migration-notes.md). The default variant set (no
# VARIANTS override) excludes mhalf_mcdaniel for this reason; requesting it
# explicitly via VARIANTS fails with a targeted error.
#
# Array size is 43 dwarfs (0-42), from the LVDB catalog -- see the same count
# asserted in scripts/Jdwarf.sh and tests/test_concat_Js.py. Dwarfs with no
# Fermi source name (obs.fermi_names_update[idx] is None) fail loudly with a
# ValueError rather than silently skipping -- see the fermi_names_update table
# in Jdata.py for which those are. One more hazard for anyone invoking
# fermi_reweighting.py directly without --legacy-version update (this
# launcher always passes it): obs.fermi_names[20] is 'Leo_6', not None, but
# no Leo_6_sed.fits exists under data/fermi_legacy/dSphs/SEDs/ either -- task
# 20 dies with FileNotFoundError from fits.open() rather than the documented
# ValueError.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/fermi_reweighting_update_out/ must already
# exist. It is tracked with a .gitkeep for that reason. Submit from the
# repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=fermi_reweighting_update
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-42
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/fermi_reweighting_update_out/slurm-%A_%a.out
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
mkdir -p "${REPO}/scripts/fermi_reweighting_update_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

ARGS=("${SLURM_ARRAY_TASK_ID}" --legacy-version update)
if [[ -n "${VARIANTS:-}" ]]; then
  ARGS+=(--variants "${VARIANTS}")
fi

exec python -u "${REPO}/python/fermi_reweighting.py" "${ARGS[@]}"
