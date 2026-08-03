#!/bin/bash
# J-factors of SatGen halos at their own positions -- the SatGen curve of
# Jfactors_all.pdf.
#
#   sbatch scripts/Jhalopos.sh                        # Diemer
#   sbatch --export=ALL,VERSION=Zhao scripts/Jhalopos.sh
#
# Array size is ceil(n_halos / CHUNK) with CHUNK=5000 in python/Jhalopos.py.
# For Diemer (2432159 halos) that is 487; the published run used 0-511, so the
# tail tasks wrote empty fragments. 0-511 is kept for parity.
#
# Writes results/paper_Js/halo_position/<version>/halo_Js/halo_Js-<g>.npz, then
#   scripts/concat_Js.sh <version>   is NOT used here --
# this tree is outside the per-dwarf layout concat_Js walks. Merge with:
#   python python/concat_Js.py halo_position <version>
#
# NOTE: this stage is not bit-reproducible. Its integral is unseeded vegas
# Monte Carlo, so a rerun differs within MC error. See the module docstring.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=Jhalopos
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-511
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/Jhalopos_out/slurm-%A_%a.out
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
mkdir -p "${REPO}/scripts/Jhalopos_out"

source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/Jhalopos.py" "${VERSION:-Diemer}" "${SLURM_ARRAY_TASK_ID}"
