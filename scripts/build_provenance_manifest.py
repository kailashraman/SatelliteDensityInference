#!/usr/bin/env python
"""Assembles docs/provenance-manifest.json from the `.prov.json` sidecars
already written next to every product under plots/ -- the single place to
answer "which SatGen variant and which DwarfJeansAnalysis snapshot went into
this figure" without opening 34 (now 167) files by hand.

This is a report, not just a collector. It walks every `*.pdf` under plots/
and every `*.tex` under plots/tables/ (matching tests/test_figure_names.py's
globs, `.ipynb_checkpoints` excluded the same way) and, for each:

  - requires a `<file>.prov.json` sidecar to exist. A figure with no sidecar
    means "we don't know what produced it" -- exactly the silent-mixing
    failure mode CLAUDE.md calls the worst one here, so this is fatal: the
    script prints every offender and exits non-zero rather than writing a
    manifest with holes in it.
  - flags any INPUT recorded with null provenance -- but only for inputs
    that are DERIVED products this repository is expected to have stamped.
  - flags any figure whose own `satgen_version` disagrees with its siblings
    in the same family (family = the immediate subdirectory of plots/, e.g.
    `rmax_vmax`, `panels`, `tables` -- this mirrors the "one subdirectory per
    figure family" convention in the CLAUDE.md architecture section).

Distinguishing "not applicable" from "unknown" turned out to need TWO rules,
not the one originally anticipated -- the second was found empirically by
running this script, not designed in advance:

  1. DwarfJeansAnalysis posterior/derived.npz files are recorded under a
     record's `dja_snapshot.files`, not its top-level `inputs` -- by
     `provenance._dja_snapshot`'s construction, those fingerprint dicts never
     carry a `provenance` key at all (DJA is an external, independently-
     tracked repo; this codebase does not and should not stamp its outputs).
     That absence is EXPECTED and reported separately
     (`inputs_not_applicable`), never folded into
     `inputs_with_unknown_provenance`.
  2. A first version of this script (no DATA_DIR/RESULTS_DIR distinction)
     flagged 677 "unknown provenance" inputs across 8 figures, and inspecting
     them showed two very different populations bundled together: ~630 were
     genuinely unstamped `results/fermi_reweighting_update/<dwarf>/
     {mhalf,Jeans_<prior>}.npz` intermediates (a real gap -- see the printed
     report), but the rest were `data/literature/*.csv`, `data/lvdb/
     dwarf_all.csv`, and `data/symphony/{SymphonyMilkyWay,MWest}` -- raw,
     committed source data under config.DATA_DIR (see CLAUDE.md, "data/
     holds observational and literature inputs"), which provenance.py never
     stamps by design (its docstring: "every product THIS REPOSITORY WRITES
     carries a stamp" -- these are read, not written). Flagging committed
     source data as a provenance gap is noise that would bury the real one.
     So: an input under DATA_DIR but NOT under ADDITIONAL_DIR (the
     derived-product subtree living under data/, e.g. the SatGen h5 copies
     and per-dwarf weights) is "not applicable"; a null-provenance input
     anywhere else (results/, additional/) is "unknown" -- a real gap.

  Likewise, a record's own `satgen_version` being None is legitimate for a
  pure-Jeans or pure-observational product (e.g. a `Jeans_<prior>` contour,
  jeans_corner.pdf, or the Fermi-reweighting figures, which read no h5
  directly) -- those are excluded from the version-agreement check entirely
  (reported under `no_satgen_version`, not `version_disagreements`), rather
  than being treated as a stray None fighting for a majority within their
  family.

Usage:
    conda activate J_calc
    python scripts/build_provenance_manifest.py [--out PATH]

Exits non-zero (after printing the report) if any figure/table under plots/
lacks a sidecar. Does not exit non-zero for null-provenance inputs or
version disagreements -- those are surfaced in the printed report and in the
manifest's `report` block for a human to act on, since (unlike a missing
sidecar) they do not by themselves mean the artifact's provenance is
unrecoverable.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'python'))

import config
import provenance


def _glob_excluding_checkpoints(root, pattern):
    return sorted(p for p in root.glob(pattern) if '.ipynb_checkpoints' not in p.parts)


def _family(path):
    """A product's own containing directory under plots/, e.g.
    plots/rmax_vmax/foo.pdf -> 'rmax_vmax'.

    NOT just the top-level subdirectory of plots/: plots/supplementary/ (the
    full-39-dwarf sweeps excluded from the orphan check in
    tests/test_figure_names.py) nests TWO unrelated figure families under one
    directory, supplementary/rho150_mpeak/ and supplementary/rmax_vmax/, each
    legitimately built from a different version (mass_floor_7 vs Diemer) --
    grouping by the top-level component alone flagged that as a spurious
    "version disagreement" the first time this script ran. Using the file's
    own parent directory keeps a flat family (e.g. rmax_vmax/) as one group
    while splitting the nested supplementary/ case into two.
    """
    return str(path.parent.relative_to(config.PLOTS_DIR))


def _is_raw_source_data(path_str):
    """True for committed source data under data/ (excluding data/additional/,
    the derived-product subtree that happens to live under data/) -- never
    stamped by provenance.py by design. See module docstring, rule 2."""
    try:
        path = Path(path_str).resolve()
    except (TypeError, ValueError):
        return False
    return (config.DATA_DIR in path.parents
            and config.ADDITIONAL_DIR not in path.parents
            and path != config.ADDITIONAL_DIR)


def build_manifest():
    targets = (_glob_excluding_checkpoints(config.PLOTS_DIR, '**/*.pdf')
               + _glob_excluding_checkpoints(config.PLOTS_DIR / 'tables', '*.tex'))

    figures = {}
    missing_sidecar = []
    inputs_with_unknown_provenance = {}
    inputs_not_applicable = {}
    no_satgen_version = []
    version_by_family = defaultdict(lambda: defaultdict(list))

    for path in targets:
        rel = str(path.relative_to(config.PLOTS_DIR))
        sidecar = path.with_name(path.name + provenance.SIDECAR_SUFFIX)
        if not sidecar.exists():
            missing_sidecar.append(rel)
            continue

        with open(sidecar) as handle:
            record = json.load(handle)

        inputs = record.get('inputs') or []
        null_inputs = [inp.get('path') for inp in inputs if inp.get('provenance') is None]
        unknown_inputs = [p for p in null_inputs if not _is_raw_source_data(p)]
        na_inputs = [p for p in null_inputs if _is_raw_source_data(p)]
        if unknown_inputs:
            inputs_with_unknown_provenance[rel] = unknown_inputs
        if na_inputs:
            inputs_not_applicable[rel] = na_inputs

        dja_snapshot = record.get('dja_snapshot') or {}
        dja_files = dja_snapshot.get('files') or []
        if dja_files:
            # By construction (_dja_snapshot), these never carry a
            # 'provenance' key -- their absence of provenance is "not
            # applicable", not "unknown".
            inputs_not_applicable.setdefault(rel, []).extend(
                f.get('path') for f in dja_files)

        satgen_version = record.get('satgen_version')
        family = _family(path)
        if satgen_version is None:
            no_satgen_version.append(rel)
        else:
            version_by_family[family][satgen_version].append(rel)

        figures[rel] = {
            'script': record.get('script'),
            'satgen_version': satgen_version,
            'sim_version': record.get('sim_version'),
            'h5_file': record.get('h5_file'),
            'git_commit': record.get('git_commit'),
            'git_dirty': record.get('git_dirty'),
            'written_utc': record.get('written_utc'),
            'lvdb_version': record.get('lvdb_version'),
            'dja_prior': dja_snapshot.get('prior'),
            'dja_dir': dja_snapshot.get('dir'),
            'n_inputs': len(inputs),
            'n_dja_files': len(dja_files),
        }

    version_disagreements = {
        family: dict(versions)
        for family, versions in version_by_family.items()
        if len(versions) > 1
    }

    manifest = {
        'plots_dir': str(config.PLOTS_DIR),
        'n_products': len(targets),
        'n_with_sidecar': len(figures),
        'figures': figures,
        'report': {
            'missing_sidecar': missing_sidecar,
            'inputs_with_unknown_provenance': inputs_with_unknown_provenance,
            'inputs_not_applicable': inputs_not_applicable,
            'no_satgen_version': no_satgen_version,
            'version_disagreements': version_disagreements,
        },
    }
    return manifest


def _print_report(manifest):
    report = manifest['report']
    print(f"{manifest['n_with_sidecar']}/{manifest['n_products']} products under "
          f"{manifest['plots_dir']} have a provenance sidecar")

    if report['missing_sidecar']:
        print(f"\nMISSING SIDECAR ({len(report['missing_sidecar'])}):")
        for name in report['missing_sidecar']:
            print(f'  {name}')

    if report['inputs_with_unknown_provenance']:
        n = sum(len(v) for v in report['inputs_with_unknown_provenance'].values())
        print(f"\nINPUTS WITH UNKNOWN PROVENANCE -- a real gap, these are "
              f"derived products expected to carry a stamp ({n} across "
              f"{len(report['inputs_with_unknown_provenance'])} products):")
        for name, paths in report['inputs_with_unknown_provenance'].items():
            print(f'  {name}: {len(paths)} file(s), e.g. {paths[0]}')

    if report['inputs_not_applicable']:
        n = sum(len(v) for v in report['inputs_not_applicable'].values())
        print(f"\ninputs with no provenance by design -- committed source "
              f"data under data/ or external DJA files -- not a gap "
              f"({n} across {len(report['inputs_not_applicable'])} products, "
              f"not detailed here; see the manifest's report block)")

    if report['no_satgen_version']:
        print(f"\nNO SATGEN VERSION recorded (expected for pure-Jeans products, "
              f"{len(report['no_satgen_version'])} total):")
        for name in report['no_satgen_version']:
            print(f'  {name}')

    if report['version_disagreements']:
        print(f"\nSATGEN VERSION DISAGREEMENTS WITHIN A FAMILY "
              f"({len(report['version_disagreements'])} families):")
        for family, versions in report['version_disagreements'].items():
            print(f'  {family}:')
            for version, names in versions.items():
                print(f'    {version}: {len(names)} file(s), e.g. {names[0]}')

    if not (report['missing_sidecar'] or report['inputs_with_unknown_provenance']
            or report['version_disagreements']):
        print('\nNo missing sidecars, no unknown-provenance inputs, no '
              'version disagreements within a family.')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(config.DOCS_DIR / 'provenance-manifest.json'),
                        help='Output path (default: docs/provenance-manifest.json)')
    args = parser.parse_args()

    manifest = build_manifest()
    _print_report(manifest)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"\nwrote: {out_path}")

    if manifest['report']['missing_sidecar']:
        sys.exit(1)


if __name__ == '__main__':
    main()
