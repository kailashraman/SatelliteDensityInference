#!/bin/bash
# Build a SatGen halo catalog under data/additional/.
#
#   scripts/make_h5.sh build Diemer
#   scripts/make_h5.sh build splashback --unmasked
#   scripts/make_h5.sh sims Symphony
#
# Run on a login node for `sims` (seconds). `build` reads a ~2-4 GB compressed
# dataset and holds the selected satellites in memory -- use an interactive
# node with ~32 GB for the SatGen versions rather than a login node.
#
# NOTE: `build <version>` alone does not reproduce a published catalog for the
# versions that carry a logMhalf group. That group is added by
#   scripts/make_h5.sh augment-mhalf <version>
# which depends on data/additional/mhalf/<version>/ from compute_mhalf.py. The
# full order is: build -> compute_mhalf -> augment-mhalf.
#
# This script does not submit to SLURM. Submitting jobs is the user's call.

set -euo pipefail

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

source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

exec python -u "${REPO}/python/make_h5.py" "$@"
