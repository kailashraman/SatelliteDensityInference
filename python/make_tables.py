"""Generates the two LaTeX row-tables drafts_temp/*.tex `\\input{}`s:

    dwarf-observables -> plots/tables/dwarf-observables.tex   (tab:meta, temp_part1.tex)
    dwarf_J_table     -> plots/tables/dwarf_J_table.tex       (tab:J_meta, temp_part2.tex)

Migrated from ../SatGen_Dwarf/jupyter/PaperPlots.ipynb cells 105
(dwarf-observables) and 106 (dwarf_J_table). Each of those cells is a
COMPLETE, self-contained row emitter -- the PDG formatter definitions
(`pdg_decimals`/`fmt_pdg`/`to_log10_J`/`fmt_pdg_logJ`, already ported to
pdg_format.py) plus its own `# Main loop` that builds and prints every row --
not merely a formatter definition, which is what was originally assumed.
Cell 107 is a THIRD variant: fixed-decimal (`.1f`/`.0f`/`.2f`), not
PDG-rounded, and its column set matches neither table's caption (tab:meta
lists rmax/vmax/Mstar/Mhalf with no J column; tab:J_meta lists J-factors, not
rmax/vmax). It is not migrated here.

Column mapping, confirmed by matching each cell's cached notebook output
against the corresponding table's caption text in drafts_temp:
  dwarf-observables (cell 105, 10 cols): d, r_1/2, sigma_los, logL_V,
    logM_1/2, logM_star, r_max(M_1/2-weighted), v_max(M_1/2-weighted),
    r_max(M_star-weighted, Fattahi18 SHMR), v_max(M_star-weighted, Fattahi18).
  dwarf_J_table (cell 106, 9 cols): d, r_1/2, sigma_los, logL_V, logM_1/2,
    logM_star, logJ(M_1/2-weighted), logJ(Jeans, standard/loguniform prior),
    logJ(Jeans, SatGen-informed/satgen_shmr_fattahi18 prior) -- prior naming
    follows the prior ladder in temp_part2.tex ("standard prior" ==
    loguniform, "SatGen-informed prior" == the SHMR-weighted log-normal fit,
    Fattahi18 in the main text).

Row ordering deliberately does NOT use dwarf_categories.idcs. `idcs` is
documented there as the canonical x-axis ordering for PANEL/MULTIPANEL
FIGURES (problem-ultrafaints -> ultrafaints -> classicals, each block sorted
by ascending rhalf); these are TABLES, a different artifact with a different
reading pattern -- a reader looks a dwarf up by name, so alphabetical-within-
category is the natural order, and, more importantly, it is what cells
105/106 themselves emit and therefore what produced the published tables.
Reproducing the artifact takes precedence over harmonizing with the figures'
convention: the order here is classicals -> ultrafaints -> problem-
ultrafaints, each block sorted ALPHABETICALLY by dwarf name, built from
dwarf_categories.classical_idcs/ultrafaint_idcs/problem_idcs (the same
39-dwarf partition the figures use) rather than from `idcs` itself. Do not
"fix" this to match the figure ordering.

No embedded/rendered .tex exists anywhere to diff against -- drafts_temp only
`\\input{}`s these two basenames, and the files themselves are absent from
both this repo and SatGen_Dwarf (confirmed by search). The only available
ground truth is each cell's CACHED notebook output. Two distinct kinds of
divergence from that cached output are expected, not both a bug:
  - mhalf-derived and observational columns (d, r_1/2, sigma_los, logL_V,
    logM_1/2, and the M_1/2-weighted rmax/vmax/J) are deterministic given the
    pinned LVDB/KDSA snapshot and this repo's compute_quantiles.py output,
    and should match the cached output to the last printed digit.
  - SHMR/M_star-derived columns (rmax/vmax M_star-weighted, and the Fattahi
    Jeans-prior J) are expected to differ for two independent reasons: (1)
    mstar_weights is an unseeded draw (see docs/migration-notes.md), so this
    repo's realization is not the one that produced the cached output, and
    (2) these quantities sit on the corrected infall->z=0 evolved_Mstar
    tidal-ratio mapping, which the cached output predates (see
    provenance.py's CONVENTION_BEARING_SCRIPTS docstring) -- so on top of
    realization noise there is a real, expected systematic shift.

Usage:
    conda activate J_calc
    python make_tables.py
"""

import argparse

import astropy.units as u
import numpy as np

import config
import dwarf_categories as dc
import provenance
import Jdata as obs
from pdg_format import fmt_pdg, fmt_pdg_logJ

VERSION = 'Diemer'

# Jeans priors dwarf_J_table plots: the "standard prior" (loguniform) and the
# "SatGen-informed prior" (SHMR-weighted, Fattahi18 in the main text), per the
# prior ladder in temp_part2.tex.
JEANS_PRIORS = {'standard': 'loguniform', 'satgen_informed': 'satgen_shmr_fattahi18'}


def _table_order():
    """39-dwarf index array: classicals -> ultrafaints -> problem-ultrafaints,
    each block sorted alphabetically by name -- cell 105/106's `index_list`,
    NOT dwarf_categories.idcs (see module docstring)."""
    return np.concatenate((
        dc.classical_idcs[np.argsort(dc.dwarf_names[dc.classical_idcs])],
        dc.ultrafaint_idcs[np.argsort(dc.dwarf_names[dc.ultrafaint_idcs])],
        dc.problem_idcs[np.argsort(dc.dwarf_names[dc.problem_idcs])],
    ))


def _quantiles_path(dwarf):
    return config.PAPER_QUANTILES_DIR / 'galactocentric' / VERSION / f'{dwarf}.npz'


def build_dwarf_observables():
    """Row strings and the quantile npz files read, for dwarf-observables.tex."""
    index_list = _table_order()
    dwarf_list = dc.dwarf_names[index_list]
    abbr_list = obs.abbreviations[index_list]
    kdsa_list = obs.kd_names

    rows = []
    files = []
    for idx, dwarf in enumerate(dwarf_list):
        dwarf_obj = obs.Dwarf(dwarf)
        d = dwarf_obj.distance.to(u.kpc).value
        d_err = dwarf_obj.distance_err.to(u.kpc).value
        rhalf = dwarf_obj.rhalf.value
        rhalf_err = dwarf_obj.rhalf_err.value
        dispersion = dwarf_obj.dispersion.value
        dispersion_err = dwarf_obj.dispersion_err.value
        logLv = dwarf_obj.logLv
        logLv_err = dwarf_obj.logLv_err
        logMhalf = dwarf_obj.logMhalf
        logMhalf_err = dwarf_obj.logMhalf_err
        logMstar = dwarf_obj.logMstar
        logMstar_err = dwarf_obj.logMstar_err
        footnote = r'$^a$' if dwarf in kdsa_list else r'$^b$'

        path = _quantiles_path(dwarf)
        files.append(path)
        quantiles_file = np.load(path)
        rmax_quantiles = quantiles_file['rmax_quantiles']
        vmax_quantiles = quantiles_file['vmax_quantiles']

        # M_1/2-weighted (weight index 0).
        rmax_med = rmax_quantiles[1, 0]
        rmax_err = (rmax_quantiles[1, 0] - rmax_quantiles[0, 0],
                    rmax_quantiles[2, 0] - rmax_quantiles[1, 0])
        vmax_med = vmax_quantiles[1, 0]
        vmax_err = (vmax_quantiles[1, 0] - vmax_quantiles[0, 0],
                    vmax_quantiles[2, 0] - vmax_quantiles[1, 0])

        # M_star-weighted, Fattahi18 SHMR (weight index 4 -- see
        # compute_quantiles.py's SHMR_names / weights_list construction).
        rmax_mstar_med = rmax_quantiles[1, 4]
        rmax_mstar_err = (rmax_quantiles[1, 4] - rmax_quantiles[0, 4],
                          rmax_quantiles[2, 4] - rmax_quantiles[1, 4])
        vmax_mstar_med = vmax_quantiles[1, 4]
        vmax_mstar_err = (vmax_quantiles[1, 4] - vmax_quantiles[0, 4],
                          vmax_quantiles[2, 4] - vmax_quantiles[1, 4])

        abbr = abbr_list[idx]
        row = (
            f"{abbr}{footnote} & "
            f"{fmt_pdg(d, d_err[0], d_err[1])} & "
            f"{fmt_pdg(rhalf, rhalf_err[0], rhalf_err[1])} & "
            f"{fmt_pdg(dispersion, dispersion_err[0], dispersion_err[1])} & "
            f"{fmt_pdg(logLv, logLv_err[0], logLv_err[1])} & "
            f"{fmt_pdg(logMhalf, logMhalf_err[0], logMhalf_err[1])} & "
            f"{fmt_pdg(logMstar, logMstar_err[0], logMstar_err[1])} & "
            f"{fmt_pdg(rmax_med, rmax_err[0], rmax_err[1])} & "
            f"{fmt_pdg(vmax_med, vmax_err[0], vmax_err[1])} & "
            f"{fmt_pdg(rmax_mstar_med, rmax_mstar_err[0], rmax_mstar_err[1])} & "
            f"{fmt_pdg(vmax_mstar_med, vmax_mstar_err[0], vmax_mstar_err[1])} \\\\"
        )
        rows.append(row)
    return rows, files


def build_dwarf_j_table():
    """Row strings, quantile npz files, and DJA derived.npz files read, for
    dwarf_J_table.tex."""
    index_list = _table_order()
    dwarf_list = dc.dwarf_names[index_list]
    abbr_list = obs.abbreviations[index_list]
    kdsa_list = obs.kd_names

    rows = []
    files = []
    dja_files = []
    for idx, dwarf in enumerate(dwarf_list):
        dwarf_obj = obs.Dwarf(dwarf)
        d = dwarf_obj.distance.to(u.kpc).value
        d_err = dwarf_obj.distance_err.to(u.kpc).value
        rhalf = dwarf_obj.rhalf.value
        rhalf_err = dwarf_obj.rhalf_err.value
        dispersion = dwarf_obj.dispersion.value
        dispersion_err = dwarf_obj.dispersion_err.value
        logLv = dwarf_obj.logLv
        logLv_err = dwarf_obj.logLv_err
        logMhalf = dwarf_obj.logMhalf
        logMhalf_err = dwarf_obj.logMhalf_err
        logMstar = dwarf_obj.logMstar
        logMstar_err = dwarf_obj.logMstar_err
        footnote = r'$^a$' if dwarf in kdsa_list else r'$^b$'

        path = _quantiles_path(dwarf)
        files.append(path)
        quantiles_file = np.load(path)
        J_quantiles = quantiles_file['J_quantiles']
        J_mhalf_16, J_mhalf_50, J_mhalf_84 = (
            J_quantiles[0, 0], J_quantiles[1, 0], J_quantiles[2, 0])

        dwarf_obj.get_Jeans_results(dja_prior=JEANS_PRIORS['standard'])
        if dwarf_obj.jeans_dja_path:
            dja_files.append(dwarf_obj.jeans_dja_path)
        J_loguni_16, J_loguni_50, J_loguni_84 = dwarf_obj.Jeans_J_quantiles

        dwarf_obj.get_Jeans_results(dja_prior=JEANS_PRIORS['satgen_informed'])
        if dwarf_obj.jeans_dja_path:
            dja_files.append(dwarf_obj.jeans_dja_path)
        J_fatt_16, J_fatt_50, J_fatt_84 = dwarf_obj.Jeans_J_quantiles

        abbr = abbr_list[idx]
        row = (
            f"{abbr}{footnote} & "
            f"{fmt_pdg(d, d_err[0], d_err[1])} & "
            f"{fmt_pdg(rhalf, rhalf_err[0], rhalf_err[1])} & "
            f"{fmt_pdg(dispersion, dispersion_err[0], dispersion_err[1])} & "
            f"{fmt_pdg(logLv, logLv_err[0], logLv_err[1])} & "
            f"{fmt_pdg(logMhalf, logMhalf_err[0], logMhalf_err[1])} & "
            f"{fmt_pdg(logMstar, logMstar_err[0], logMstar_err[1])} & "
            f"{fmt_pdg_logJ(J_mhalf_16, J_mhalf_50, J_mhalf_84)} & "
            f"{fmt_pdg_logJ(J_loguni_16, J_loguni_50, J_loguni_84)} & "
            f"{fmt_pdg_logJ(J_fatt_16, J_fatt_50, J_fatt_84)} \\\\"
        )
        rows.append(row)

    # Deduplicate dja_files, preserving order (each dwarf/prior pair reads a
    # distinct derived.npz, but repeated calls across dwarfs never touch the
    # same file, so this only guards against a future change, matching the
    # dedup already done in plot_jfactor_panels.py's load_jeans).
    seen = set()
    dja_files_unique = []
    for f in dja_files:
        if f not in seen:
            seen.add(f)
            dja_files_unique.append(f)
    return rows, files, dja_files_unique


def _write_table(name, rows):
    path = config.PLOTS_DIR / 'tables' / f'{name}.tex'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(rows) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    obs_rows, obs_files = build_dwarf_observables()
    obs_path = _write_table('dwarf-observables', obs_rows)
    provenance.figure_manifest(obs_path, 'python/make_tables.py', obs_files,
                                version=VERSION, table_key='dwarf-observables')
    print('saved:', obs_path)

    j_rows, j_files, dja_files = build_dwarf_j_table()
    j_path = _write_table('dwarf_J_table', j_rows)
    provenance.figure_manifest(j_path, 'python/make_tables.py', j_files,
                                version=VERSION, table_key='dwarf_J_table',
                                dja_prior=list(JEANS_PRIORS.values()),
                                dja_files=dja_files)
    print('saved:', j_path)


if __name__ == '__main__':
    main()
