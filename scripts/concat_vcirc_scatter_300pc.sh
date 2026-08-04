#!/bin/bash
# Merge the per-dwarf vcirc-scatter rows into one scatter_300pc.txt per SHMR.
#
#   sbatch scripts/concat_vcirc_scatter_300pc.sh Diemer
#
# Mandatory after scripts/vcirc_scatter_300pc.sh: the array writes per-dwarf
# rows, and this pass merges them into the one file per SHMR that a plot
# would read. Exits non-zero if an SHMR directory is missing or unexpected,
# a row count disagrees with len(Jdata.dwarf_names), or a row's provenance
# sidecar is missing or disagrees with the row it sits next to -- see
# python/concat_vcirc_scatter_300pc.py for the full validation and the
# two-phase validate-then-write guarantee.

#SBATCH --job-name=concat_vcirc_scatter_300pc
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/concat_vcirc_scatter_300pc_out/slurm-%j.out

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
mkdir -p "${REPO}/scripts/concat_vcirc_scatter_300pc_out"

# Strict mode after the environment setup: ~/.bashrc and conda activate are not
# written to survive `set -u`/`set -e`.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch scripts/concat_vcirc_scatter_300pc.sh <version>" >&2
  exit 2
fi

exec python -u "${REPO}/python/concat_vcirc_scatter_300pc.py" "$@"
