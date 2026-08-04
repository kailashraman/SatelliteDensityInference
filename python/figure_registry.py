"""The figure-naming contract: which module produces each paper figure.

`FIGURES` and `TABLES` are derived by parsing drafts_temp/*.tex for
`\\includegraphics{...}` and `\\input{...}`, in the order they first appear
across the two drafts (temp_part1.tex then temp_part2.tex). The
`SatGen_take2/` prefix on `\\includegraphics` paths is the paper build tree,
not a directory in this repository, so only the basename is kept; dwarf
names contain spaces and are kept verbatim, not sanitised.

`REGISTRY` and `TABLE_REGISTRY` map each basename to the python/ module that
produces it. All 34 figures and both tables are now mapped.

These mappings are load-bearing, not documentation: tests/test_figure_names.py
keys its xfail on whether an entry is None, so registering a producer converts
that entry into a hard assertion that the artifact exists exactly once under
plots/. Never register a basename before its producer actually writes it.
"""

import re

import config

_INCLUDEGRAPHICS = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}')
_INPUT = re.compile(r'\\input\{([^}]*)\}')


def _parse_tex_names():
    """Basenames referenced by \\includegraphics and \\input, first-seen order.

    Does NOT strip LaTeX `%` comments before matching, so a commented-out
    `\\includegraphics{...}`/`\\input{...}` line would still register its
    basename as required. No line in either draft is commented out like that
    today (verified), so this is a documented limitation, not a live bug.
    """
    figures = []
    tables = []
    for tex_path in sorted(config.DRAFTS_DIR.glob('*.tex')):
        text = tex_path.read_text()
        for match in _INCLUDEGRAPHICS.finditer(text):
            name = match.group(1).rsplit('/', 1)[-1]
            if name not in figures:
                figures.append(name)
        for match in _INPUT.finditer(text):
            name = match.group(1).rsplit('/', 1)[-1]
            if name not in tables:
                tables.append(name)
    return figures, tables


# pdf basenames (with extension) and table basenames (without -- \input in
# these drafts never carries one), in first-seen order across the two drafts.
FIGURES, TABLES = _parse_tex_names()

# basename -> producing python/<module>.py name, or None until that step lands.
REGISTRY = {name: None for name in FIGURES}
TABLE_REGISTRY = {name: None for name in TABLES}

# Step 23: both LaTeX tables (python/make_tables.py). Their row order follows
# notebook cells 105/106 -- classicals, ultrafaints, problem-ultrafaints, each
# alphabetical -- NOT dwarf_categories.idcs, which is the figure x-axis order.
# The two artifacts order differently on purpose; see make_tables.py.
for _name in ('dwarf-observables', 'dwarf_J_table'):
    TABLE_REGISTRY[_name] = 'make_tables'

# Step 10: the "number function" figure family (python/plot_number_functions.py).
for _name in ('sims_mpeak.pdf', 'sims_rho150.pdf', 'rho150.pdf', 'Jfactors_all.pdf',
              'distance_statistics.pdf', 'distance_statistics_sims.pdf'):
    REGISTRY[_name] = 'plot_number_functions'

# Step 12: the Fermi recasting figures (python/plot_fermi_reweighting.py).
for _name in ('fermi_reweighting_all_priors.pdf', 'fermi_reweighting_SHMRs.pdf',
              'fermi_reweighting_prior_envelope.pdf', 'fermi_reweighting_MC.pdf'):
    REGISTRY[_name] = 'plot_fermi_reweighting'

# Step 13: the rmax/vmax figures (python/plot_rmax_vmax.py). Dwarf names keep
# their spaces -- these basenames are what the .tex references verbatim.
for _name in ('rmax_vmax_Ursa Major II_no_Jeans.pdf',
              'rmax_vmax_Segue 1_with_Jeans.pdf',
              'rmax_vmax_Antlia II_with_Jeans.pdf',
              'rmax_vmax_Crater II_with_Jeans.pdf'):
    REGISTRY[_name] = 'plot_rmax_vmax'

# Step 15: the systematics multipanel (python/plot_multipanel_systematics.py).
REGISTRY['multipanel_systematics.pdf'] = 'plot_multipanel_systematics'

# Step 18: the rho150-vs-Mpeak figures (python/plot_rho150_mpeak.py).
for _name in ('rho150_mpeak_Draco.pdf', 'rho150_mpeak_Segue 1.pdf',
              'rho150_mpeak_Hydrus I.pdf'):
    REGISTRY[_name] = 'plot_rho150_mpeak'

# Step 22: the appendix limit comparison (python/plot_fermi_limit_comparison.py).
REGISTRY['fermi_limit_comparison.pdf'] = 'plot_fermi_limit_comparison'

# Step 14: Part I's main multipanel (python/plot_multipanel_paper1.py). This
# figure's dwarf ordering is the one multipanel_systematics says it follows.
REGISTRY['multipanel_Diemer_paper1.pdf'] = 'plot_multipanel_paper1'

# Step 16: the Jeans J-factor panels (python/plot_jfactor_panels.py).
for _name in ('panel_Jfactor_theta95.pdf', 'multipanel_jeans_priors.pdf',
              'panel_Jfactor_literature.pdf'):
    REGISTRY[_name] = 'plot_jfactor_panels'

# Step 17: jeans_corner is NOT produced here. DwarfJeansAnalysis is a tracked,
# reproducible repository, so its own plot_posteriors.py renders the figure and
# python/fetch_jeans_corner.py copies it in with a provenance record of which
# snapshot was consumed. Reimplementing it here would duplicate a pipeline the
# reproducibility contract deliberately keeps external.
REGISTRY['jeans_corner.pdf'] = 'fetch_jeans_corner'

# Step 20: the detectability contours (python/plot_dispersion_limit.py).
REGISTRY['dispersion_limit_contours.pdf'] = 'plot_dispersion_limit'

# Step 21: the scaling-relation figures (python/plot_scaling_relations.py).
# Note L_vs_rho150 is reconstructed from cell 156's commented branch -- the
# notebook's saved state renders only the Mpeak variant. The published PDF and
# the draft caption both confirm the reconstruction, including that the
# diagonal Mpeak<1e8 hatch belongs to the Mpeak panel alone.
for _name in ('tidal_stripping.pdf', 'L_vs_Mpeak.pdf', 'L_vs_rho150.pdf',
              'lvdb_kdsa_rho150.pdf'):
    REGISTRY[_name] = 'plot_scaling_relations'

# Step 19: the power-law fits (python/plot_powerlaw.py). The only figures here
# produced by a sampler rather than a deterministic computation -- reproducing
# them needs the package versions pinned in docs/pymc-environment.md as well as
# upstream's random_seed=42, since PyTensor compiles the log-density and the
# compiled step function varies with pymc/pytensor/numpy/scipy.
for _name in ('powerlaw_posteriors.pdf', 'powerlaw_fit.pdf'):
    REGISTRY[_name] = 'plot_powerlaw'

# Step 11: the V_circ satellite figures (python/plot_vcirc_satellites_all.py).
# satellites_vcirc_intro is reproduced from the RENDERED figure, not from
# notebook cell 133 -- that cell's plotting is almost entirely commented out
# and would render nothing as it stands.
for _name in ('satellites_vcirc_all_fattahi18.pdf', 'satellites_vcirc_all_kim24.pdf',
              'satellites_vcirc_intro.pdf'):
    REGISTRY[_name] = 'plot_vcirc_satellites_all'
