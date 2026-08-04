"""Fetch jeans_corner.pdf (Ursa Major II) from DwarfJeansAnalysis.

This is deliberately NOT a port: `jeans_corner.pdf` is produced by
DwarfJeansAnalysis's own `scripts/plot_posteriors.py`, reading a completed
production posterior under `config.DJA_RESULTS_DIR`. Per CLAUDE.md's
"reproducibility contract", that pipeline is a tracked, reproducible repo we
invoke, not one we reimplement here. This script:

  1. invokes DJA's plot_posteriors.py for the requested galaxy/prior, using
     an explicit --run-dir built from config.DJA_RESULTS_DIR (so a test's
     $SDI_DJA_RESULTS_DIR override is honored -- DJA's own default resolves
     relative to its own repo root and would not see that override);
  2. copies the jeans_corner.pdf it wrote into plots/jeans/;
  3. writes a provenance sidecar recording which DJA snapshot (posterior
     chain + derived export, DJA git commit) the copy came from.

DJA's plot_posteriors.py hardcodes its own PLOTS_DIR (REPO/'plots', no CLI
override) and always writes all five diagnostic PDFs there, not just
jeans_corner.pdf -- there is no way to invoke it without that side effect.
See the module docstring in plot_posteriors.py for the other four
(jd_mhalf.pdf, m_J_corner.pdf, rs_J_corner.pdf, sigma_los_walker.pdf); this
script only harvests the one basename the draft references
(\\includegraphics{SatGen_take2/jeans_corner.pdf} in temp_part2.tex).

Requires the `DwarfJeans` conda environment (dwarfjeans editable-installed),
not this repository's usual `J_calc` -- see scripts/fetch_jeans_corner.sh.

Usage:
    python fetch_jeans_corner.py [--galaxy-key ursa_major_2] [--prior loguniform]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import config
import provenance

# The one draft figure this step produces: the Jeans posterior corner plot
# for Ursa Major II under the loguniform prior (established, per the task
# brief -- the *satgen* prior, not loguniform, is the one DJA's own docs
# describe as lognormal; there is no 'lognormal' entry in config.DJA_PRIORS).
DEFAULT_GALAXY_KEY = 'ursa_major_2'
DEFAULT_PRIOR = config.DJA_PRIOR  # 'loguniform'

DJA_REPO_ROOT = config.DJA_RESULTS_DIR.parent.parent
DJA_PLOT_SCRIPT = DJA_REPO_ROOT / 'scripts' / 'plot_posteriors.py'

DEST_DIR = config.PLOTS_DIR / 'jeans'
DEST_BASENAME = 'jeans_corner.pdf'


def _dja_git_state():
    """Commit and dirty flag for the DJA repo, or (None, None) if unreadable.

    Mirrors provenance._git_state, but that helper is scoped to
    config.REPO_ROOT (this repository); DJA is a separate checkout.

    `--untracked-files=no`: plain `git status --porcelain` counts untracked
    scratch files (e.g. staged-but-not-committed run outputs) as "dirty",
    which made `dja_git_dirty` true on effectively every invocation and
    therefore meaningless. Restricting to tracked-file changes means the flag
    actually answers "does the checked-out code differ from HEAD".
    """
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(DJA_REPO_ROOT), 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        commit = None
    try:
        status = subprocess.check_output(
            ['git', '-C', str(DJA_REPO_ROOT), 'status', '--porcelain',
             '--untracked-files=no'],
            stderr=subprocess.DEVNULL, text=True)
        dirty = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        dirty = None
    return commit, dirty


def _dja_cli_prior_args(prior):
    """(--prior value, --shmr value or None) for DJA's plot_posteriors.py.

    DJA's own --prior choices are ('uniform', 'loguniform', 'jeffreys',
    'satgen', 'satgen_box', 'satgen_shmr') -- the four satgen_shmr_<SHMR>
    members of config.DJA_PRIORS are not valid --prior values there; that
    family is selected via '--prior satgen_shmr --shmr <shmr>' instead.
    """
    prefix = 'satgen_shmr_'
    if prior.startswith(prefix):
        return 'satgen_shmr', prior[len(prefix):]
    return prior, None


def _dja_output_leaf(run_dir, requested_prior):
    """The plots/<galaxy_key>/<leaf> DJA's plot_posteriors.py actually wrote
    to, derived the same way that script derives it: from `run_dir`'s own
    audit.json (prior_name, and shmr for the satgen_shmr family), NOT from
    the --prior flag this script was invoked with. See that script's main()
    -- a --run-dir pointing at a run whose recorded prior differs from
    --prior would otherwise route around this and land in the wrong leaf
    silently. Raises if the derived leaf disagrees with `requested_prior`
    (the directory this script read `run_dir` from in the first place).
    """
    audit = json.loads((run_dir / 'audit.json').read_text())
    effective_prior = audit.get('prior_name', requested_prior)
    effective_shmr = audit.get('shmr')
    leaf = effective_prior
    if effective_prior == 'satgen_shmr' and effective_shmr:
        leaf = f'{effective_prior}_{effective_shmr}'
    if leaf != requested_prior:
        raise ValueError(
            f'{run_dir}/audit.json records prior {leaf!r}, but this run_dir '
            f'was read from the {requested_prior!r} directory -- refusing '
            'to harvest a PDF that may belong to a different prior')
    return leaf


def fetch(galaxy_key=DEFAULT_GALAXY_KEY, prior=DEFAULT_PRIOR):
    run_dir = config.DJA_RESULTS_DIR / galaxy_key / prior
    posterior_npz = run_dir / 'posterior_samples.npz'
    derived_npz = run_dir / 'derived.npz'
    if not posterior_npz.is_file():
        raise FileNotFoundError(
            f'no posterior_samples.npz at {run_dir} -- has the DJA production '
            f'run for {galaxy_key} ({prior}) completed?')

    # plot_posteriors.py derives its output leaf from audit.json's
    # prior_name/shmr, not from the --prior/--shmr flags it is invoked with
    # -- see that script's main(). Re-derive it the same way, from run_dir's
    # own audit.json, BEFORE invoking: raises immediately if it disagrees
    # with `prior` (the directory this script read run_dir from), rather
    # than running DJA's plotting pipeline first and only then discovering
    # `prior` cannot be trusted to name where the output landed.
    leaf = _dja_output_leaf(run_dir, prior)

    dja_prior, dja_shmr = _dja_cli_prior_args(prior)
    cmd = [sys.executable, str(DJA_PLOT_SCRIPT),
           '--lvdb-key', galaxy_key, '--prior', dja_prior,
           '--run-dir', str(run_dir)]
    if dja_shmr is not None:
        cmd += ['--shmr', dja_shmr]
    print('invoking:', ' '.join(cmd))
    subprocess.run(cmd, cwd=DJA_REPO_ROOT, check=True)

    dja_out_dir = DJA_REPO_ROOT / 'plots' / galaxy_key / leaf
    src_pdf = dja_out_dir / DEST_BASENAME
    if not src_pdf.is_file():
        raise FileNotFoundError(
            f'plot_posteriors.py did not write {src_pdf} -- see its stdout above')

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest_pdf = DEST_DIR / DEST_BASENAME
    shutil.copy2(src_pdf, dest_pdf)

    dja_commit, dja_dirty = _dja_git_state()
    # dja_files: both the chain plot_posteriors.py actually reads
    # (posterior_samples.npz, via postprocess.resolve_run_meta) and the
    # regenerable derived.npz side-export, so the sidecar records the full
    # snapshot this run_dir represents, not only the one file this
    # particular script happened to touch.
    sidecar = provenance.figure_manifest(
        dest_pdf, 'python/fetch_jeans_corner.py', inputs=(),
        dja_prior=prior, dja_files=(posterior_npz, derived_npz),
        galaxy_key=galaxy_key,
        dja_repo_root=str(DJA_REPO_ROOT),
        dja_git_commit=dja_commit, dja_git_dirty=dja_dirty,
        dja_source_pdf=str(src_pdf),
        dja_invocation=cmd)
    print(f'wrote {dest_pdf}')
    print(f'wrote {sidecar}')
    return dest_pdf


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--galaxy-key', default=DEFAULT_GALAXY_KEY)
    p.add_argument('--prior', default=DEFAULT_PRIOR, choices=config.DJA_PRIORS)
    args = p.parse_args()
    fetch(galaxy_key=args.galaxy_key, prior=args.prior)


if __name__ == '__main__':
    main()
