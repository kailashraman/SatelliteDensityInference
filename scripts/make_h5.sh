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

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

exec python -u "${REPO}/python/make_h5.py" "$@"
