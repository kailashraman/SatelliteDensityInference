#!/bin/bash
# Merge the per-group J-factor fragments into one halo_Js.npz per dwarf.
#
#   scripts/concat_Js.sh Diemer
#   scripts/concat_Js.sh Diemer "Antlia II"
#
# Mandatory after scripts/Jdwarf.sh: the array writes fragments, and every
# downstream consumer reads the concatenated file. Exits non-zero if any dwarf
# is missing a group, so a partially-failed array does not silently yield a
# short, misaligned halo array.
#
# Cheap enough for a login node (a few minutes for 43 dwarfs).

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

# Strict mode after the environment setup: ~/.bashrc and conda activate are not
# written to survive `set -u`/`set -e`.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail

exec python -u "${REPO}/python/concat_Js.py" "$@"
