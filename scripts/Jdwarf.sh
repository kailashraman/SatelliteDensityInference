#!/bin/bash
# J-factor calculation: one array task per (dwarf, halo group).
#
#   sbatch scripts/Jdwarf.sh                        # Diemer
#   sbatch --export=ALL,VERSION=Zhao scripts/Jdwarf.sh
#
# Array size is 43 dwarfs x GROUPS(=20) - 1 = 0-859. GROUPS is defined in
# python/Jdwarf.py and the dwarf count comes from the LVDB catalog at import,
# so the two are coupled only by this comment -- Jdwarf.py asserts the task id
# is in range and fails loudly rather than skipping a dwarf.
#
# *** This stage is NOT finished when the array completes. ***
# Jdwarf.py writes per-group fragments:
#     results/paper_Js/<version>/<dwarf>/halo_Js/halo_Js-<g>.npz
# but every downstream consumer reads the concatenated
#     results/paper_Js/<version>/<dwarf>/halo_Js.npz
# produced by a separate pass:
#     scripts/concat_Js.sh <version>
# concat_Js refuses to write unless the fragments tile the catalog exactly, so
# a partially-failed array is caught there rather than silently producing a
# short, misaligned halo array.
#
# Jdwarf.py raises FileExistsError on an existing fragment, so resubmitting the
# whole array only fills in the gaps.
#
# --output below is resolved by SLURM against the submission directory BEFORE
# this script body runs, so scripts/Jdwarf_out/ must already exist. It is
# tracked with a .gitkeep for that reason. Submit from the repository root.
#
# This script does not submit itself. Submitting jobs is the user's call.

#SBATCH --job-name=Jdwarf
#SBATCH --account=pc_heptheory
#SBATCH --partition=lr7
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --array=0-859
#SBATCH --mail-type=NONE
#SBATCH --output=scripts/Jdwarf_out/slurm-%A_%a.out
#SBATCH --exclude=n0000.lr7,n0001.lr7,n0011.lr7,n0117.lr7,n0150.lr7,n0151.lr7,n0153.lr7

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${REPO}/scripts/Jdwarf_out"

# ~/.bashrc and conda activate are not written to survive `set -u`/`set -e`;
# enabling strict mode before them aborts every task before Python starts.
source ~/.bashrc
conda deactivate 2>/dev/null || true
conda activate J_calc

set -euo pipefail
cd "${REPO}"

exec python -u "${REPO}/python/Jdwarf.py" "${VERSION:-Diemer}" "${SLURM_ARRAY_TASK_ID}"
