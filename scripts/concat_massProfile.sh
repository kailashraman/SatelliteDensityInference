#!/bin/bash
# Merge the per-group massProfile fragments into one massProfile.npz.
#
#   sbatch scripts/concat_massProfile.sh Diemer
#
# Mandatory after scripts/massProfile.sh: the array writes fragments, and
# vcirc_scatter_300pc.py reads the concatenated massProfile.npz. Exits
# non-zero if any group is missing, r_grid disagrees between fragments, or the
# concatenated row count does not match the catalog, so a partially-failed
# array does not silently yield a short, misaligned array.

#SBATCH --job-name=concat_massProfile
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr6
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# The three output arrays are preallocated at full catalog length and held
# simultaneously: 3 x n_halos x 101 x 8 bytes = 5.9 GB at Diemer's 2.4M halos,
# plus one fragment. Measured MaxRSS is 16.0 GB, not 5.9 -- provenance.savez
# compresses, and np.savez_compressed buffers on top of the live arrays. Sized
# from the measurement, not the arithmetic. lr7's 4590M default OOM-kills this
# with no traceback.
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/concat_massProfile_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/concat_massProfile_out"

# Strict mode after the environment setup: ~/.bashrc and conda activate are not
# written to survive `set -u`/`set -e`.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch scripts/concat_massProfile.sh <version>" >&2
  exit 2
fi

exec python -u "${REPO}/python/concat_massProfile.py" "$@"
