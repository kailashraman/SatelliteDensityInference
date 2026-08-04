#!/bin/bash
# The two LaTeX row-tables drafts_temp/*.tex \input{}: dwarf-observables.tex
# (tab:meta, Part I) and dwarf_J_table.tex (tab:J_meta, Part II).
#
#   sbatch scripts/make_tables.sh
#
# Requires, for the Diemer version python/make_tables.py hardcodes (matching
# the source notebook cells' own hardcoded 'Diemer'): the 39-dwarf
# results/paper_quantiles/galactocentric/Diemer/<dwarf>.npz tree
# (scripts/compute_quantiles.sh) and DwarfJeansAnalysis's per-dwarf
# <lvdb_key>/{loguniform,satgen_shmr_fattahi18}/derived.npz under
# config.DJA_RESULTS_DIR.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/make_tables_out/ must already exist. It
# is tracked with a .gitkeep for that reason. Submit from the repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=make_tables
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/make_tables_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/make_tables_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/make_tables.py"
