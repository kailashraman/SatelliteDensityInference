# CLAUDE.md

Guidance for Claude Code when helping someone use this repository: redraw the
paper figures, or recompute the paper results from the deposited inputs.
`README.md` is the overview; `zenodo/README_DATA.md` describes the data record
and the full recompute recipe. Read those before giving instructions.

## What the repository is

Code for a two-part paper series (Raman, Folsom, Kaplinghat, Lisanti, Safdi)
on Milky Way dwarf spheroidal dark-matter halos. Part I infers ρ₁₅₀, peak
masses and related quantities from a SatGen subhalo population conditioned on
stellar mass or kinematics; Part II turns them into J-factors and recasts
Fermi-LAT ⟨σv⟩ limits.

```
python/   computation and figure code; every entry point takes CLI arguments
scripts/  SLURM launchers (scripts/<name>.sh wraps python/<name>.py) and the
          Zenodo fetch/pack/upload tools
data/     inputs; large files come from the Zenodo record, not git
results/  derived intermediates (from the Zenodo record or recomputed)
plots/    the published figures and tables, with .prov.json provenance sidecars
tests/    pytest suite
```

## Setting up

- Environment: `conda env create -f environment.yml && conda activate sdi`.
  LaTeX is not included; every figure needs a `latex` binary, found through
  `SDI_TEXLIVE_BIN` (the directory holding `latex`). The default in
  `python/config.py` is the authors' machine, so a figure fails loudly until
  this is set.
- Data: git holds only the code and small text inputs; the rest comes from
  the Zenodo record (next section).

## The Zenodo record

All large data lives in one Zenodo record (`zenodo/README_DATA.md` is its
README; DOI 10.5281/zenodo.24511712, record ID 24511712). Nothing
in the repository runs past a few unit tests without it. It has two tiers:

- **`figures`** (about 27 GB download): the intermediates every paper figure
  and table reads. With it, each figure redraws in minutes, no cluster needed.
  `jeans_corner.pdf` cannot be redrawn from it (it is drawn by
  DwarfJeansAnalysis; see `zenodo/README_DATA.md` for redrawing it from that
  repository's record). The supplementary rmax/vmax figures can be
  redrawn only after recomputing from `inputs`; the supplementary
  ρ₁₅₀–M_peak figures for all dwarfs except Draco, Hydrus I and Segue 1
  cannot be reproduced from the record. Use the published PDFs in `plots/`.
- **`inputs`** (about 22 GB download, 33 GB unpacked; `all` is about 40 GB
  download): the SatGen halo catalogs, the frozen published
  SHMR weight draw, the Fermi-LAT data and a DwarfJeansAnalysis results
  snapshot, from which the pipeline recomputes the paper results on a
  cluster.

Fetch from the repository root:

```bash
python scripts/zenodo_fetch.py --record 24511712 --tier figures   # or inputs, all
export SDI_DJA_RESULTS_DIR="$PWD/data/dja_results"
```

The fetch unpacks into the checkout (`--dest` to change), verifies every
archive against Zenodo's md5 and `MANIFEST.json`'s sha256 (`--files-from`
checks only the latter), and refuses to overwrite a differing local file
unless given `--force`. A network fetch needs `requests`; both need a Python with `tarfile.data_filter` (both satisfied by `environment.yml`),
and stages archives under `--dest` (or `--tmp DIR`), so allow about 10 GB
extra. `--files-from DIR` (DIR holding `MANIFEST.json` and the archives under
their record names) unpacks without the network.
`SDI_DJA_RESULTS_DIR` must be exported in every shell that runs the code;
without it the Jeans products are looked for in a sibling
`../DwarfJeansAnalysis` checkout.

`MANIFEST.json` lists every archive member with its sha256, and
`sdi_git_commit`, the public commit the record was packed from; checking it
out gives an exact match, though later commits normally change only
documentation. Provenance sidecars (`*.prov.json`) record the authors'
machine paths and `git_commit` values from unpublished history, which mostly will not resolve (none in the deposited
sidecars); `MANIFEST.json`'s `path_remap` table (data only, nothing applies
it) maps the paths to the deposited layout.
`scripts/zenodo_pack.py` and `scripts/zenodo_upload.py` are the authors'
packing and upload tools; a user does not need them.

## Running things

- **The `scripts/*.sh` launchers are specific to the authors' SLURM cluster**
  (account, partitions, a `J_calc` environment). Don't run them as they stand
  elsewhere, and don't submit cluster jobs without the user's say-so. Run
  `python/<name>.py` directly with the arguments the launcher's last line
  passes, including the environment variables it reads (`VERSION`,
  `WEIGHTS_SOURCE`, `VARIANTS`). For most array stages the task index is the dwarf index (0–42);
  `Jdwarf`, `Jhalopos`, `massProfile` and `vcirc_scatter_300pc` use other
  index spaces and argument orders, documented in each header. Leave the
  `--array` line of `vcirc_scatter_300pc.sh` unchanged: the Python reads it.
- **Figure drivers** (`python/plot_*.py`, `python/make_tables.py`) run in
  minutes from the `figures` tier and write into `plots/`, overwriting the
  published PDFs there. `git checkout plots/` restores them.
- **Pipeline stages** are array jobs over 10⁵–10⁶ halos per dwarf. Follow the
  stage order in `zenodo/README_DATA.md` ("Recomputing from `inputs`").
  Recompute in a separate clone fetched with `--tier inputs` only, not in a
  `figures` or `all` checkout: several stages overwrite the deposited
  products in `data/` and `results/` in place.

## Things that will bite

- **`<version>` is an argument that selects the SatGen catalog** (`Diemer`,
  `Zhao`, `m2e12`, `Diemer_0p12`, `mass_floor_7`, and reweighting variants such
  as `lmc`). Mixing versions corrupts results silently.
- **Dwarf names contain spaces** (`Ursa Major II`, `Segue 1`) and appear
  verbatim in filenames. Quote paths; don't sanitize names.
- **Two SHMR weight realizations.** `data/halo_weights_published/` is an
  unseeded draw that cannot be regenerated; the SHMR contours and Fermi
  products use it (`--weights-source published`). The quantiles, tables and
  most other figures use the seeded `data/halo_weights/`, which
  `compute_weights.py` regenerates exactly.
- **Not everything reproduces bit for bit**: the Fermi Monte Carlo bands, the
  halo-position J-factors and the scatter in two scaling-relation figures are
  statistical. `zenodo/README_DATA.md` ("Reproducibility notes") lists what
  does and doesn't, and which supplementary figures can be redrawn.
- **SatGen is the conda-forge `satgen` package**, not the fork that generated
  the trees. Regenerating the SatGen trees is outside this repository;
  `make_h5.py` builds catalogs from them (via `SDI_TREES_DIR`), except
  `mass_floor_7`, which cannot be rebuilt.

## Tests

`pytest` runs fast unit tests with no data. `pytest --rundata` checks stage
parity against the authors' full `data/` and `results/` trees (hundreds of
GB); the Zenodo tiers carry only a subset, so some of those tests fail until
the corresponding stages are recomputed. Don't read those failures as a bad
download. (`--runslow` selects no tests yet.) Say which tier was run.
