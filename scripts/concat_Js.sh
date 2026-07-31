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

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Strict mode after the environment setup: ~/.bashrc and conda activate are not
# written to survive `set -u`/`set -e`.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail

exec python -u "${REPO}/python/concat_Js.py" "$@"
