# SatelliteDensityInference

Reproduction code and figures for **"Semi-analytic Inference of Satellite
Densities in the Cold Dark Matter Model"** (Raman, Folsom, Kaplinghat, Lisanti
& Safdi), a two-part study of the dark-matter halos of Milky Way dwarf
spheroidals:

- **Part I — Comparison to Ultra-faint Dwarf Kinematics**
  ([arXiv:2607.27316](https://arxiv.org/abs/2607.27316)). Conditions a SatGen
  subhalo population on either stellar mass (via ten stellar-mass–halo-mass
  relations) or stellar kinematics, and infers ρ₁₅₀, peak masses, r_max/v_max
  and the V_circ–r₁ᐟ₂ relation for 39 dwarfs.
- **Part II — Implications for Dark Matter Indirect Detection Constraints**
  ([arXiv:2607.27326](https://arxiv.org/abs/2607.27326)).
  Turns those posteriors into J-factors, compares them against Jeans-analysis
  J-factors under a ladder of priors, and recasts Fermi-LAT ⟨σv⟩ limits.

Every figure and table in both papers is produced by a script in this
repository, except `jeans_corner.pdf`, which is drawn by DwarfJeansAnalysis
and committed here under `plots/jeans/`.

## Related repositories

| | |
|---|---|
| This repository | https://github.com/kailashraman/SatelliteDensityInference |
| SatGen fork used for our runs | https://github.com/folsomde/SatGen |
| SatGen upstream (Jiang et al.) | https://github.com/JiangFangzhou/SatGen |
| Jeans-analysis posteriors | https://github.com/kailashraman/DwarfJeansAnalysis |
| Local Volume Database (dwarf catalog) | https://github.com/apace7/local_volume_database |

The Jeans repository is a live dependency, not a historical reference: the
prior-ladder figures in Part II read its `results/` directly, and each figure
records which snapshot of it was consumed (`results/production/` in that
repository, or `data/dja_results/` from the Zenodo record). The SatGen fork is
what generated the merger trees and satellite catalogs the pipeline starts
from; the code here imports upstream SatGen from conda-forge.

## What is here

- **34 figures and 2 LaTeX tables**, matching the `\includegraphics` and
  `\input` names the papers expect, under `plots/`.
- **117 supplementary figures** under `plots/supplementary/` — the rmax/vmax
  and ρ₁₅₀–M_peak figures for all 39 dwarfs, not only the handful the papers
  show. Most of their inputs are not in the Zenodo record's `figures` tier.
  The rmax/vmax set can be redrawn after recomputing from the `inputs` tier.
  The ρ₁₅₀–M_peak figures for 36 of the 39 dwarfs read `mass_floor_7`
  contours built from the published weight draw, whose `mass_floor_7` part is
  deposited only for Draco, Hydrus I and Segue 1, so they cannot be redrawn from the record (see
  `zenodo/README_DATA.md`).
- **A provenance record beside the products.** Each figure, table and final
  intermediate carries a `.prov.json` sidecar naming the SatGen realization,
  the input files with their fingerprints, and the code that wrote it (the
  per-group J-factor fragments that are concatenated into the final products
  do not). The recorded git commits are from the authors' development history
  and mostly do not resolve in this repository. `docs/provenance-manifest.json`
  aggregates the figure sidecars; `scripts/build_provenance_manifest.py`
  rebuilds and audits it.
- **29 launchers under `scripts/`**: 14 pipeline stages and 15 figure and
  table drivers, plus the provenance and Zenodo utilities. The computation is
  in `python/`.

## Reproducing a figure

```bash
conda env create -f environment.yml && conda activate sdi
python scripts/zenodo_fetch.py --record 24511712 --tier figures   # see "Data"
export SDI_DJA_RESULTS_DIR="$PWD/data/dja_results"
export SDI_TEXLIVE_BIN=/path/to/texlive/bin  # directory holding `latex`
python python/plot_rmax_vmax.py "Segue 1"   # one figure
```

Figure drivers write into `plots/`, overwriting the published PDFs there;
`git checkout plots/` restores them.

Most launchers are `scripts/<name>.sh` wrapping `python/<name>.py`
(`fermi_reweighting_update.sh` wraps `fermi_reweighting.py`); a few drivers
fix the SatGen version internally. Figure drivers run in minutes on cached
intermediates; the pipeline stages that produce those intermediates are SLURM
array jobs over 10⁵–10⁶ halos per dwarf.

**The `scripts/*.sh` launchers are written for the authors' cluster and are
not portable as they stand.** Their `#SBATCH` headers name our account,
partitions, QOS and excluded nodes, and they replace the active conda
environment with the authors' own (`J_calc`, or `DwarfJeans` for
`fetch_jeans_corner.sh`) through `~/.bashrc`. Read them as the record of how
each stage was invoked. Elsewhere, run `python/<name>.py` with the arguments
the launcher's last line passes, or adapt the headers to your scheduler. For
most array stages the task index is the dwarf index (0–42); `Jdwarf`,
`Jhalopos`, `massProfile` and `vcirc_scatter_300pc` use other index spaces
and argument orders, documented in each header. Leave the `--array` line of
`vcirc_scatter_300pc.sh` unchanged: the Python reads it. Two machine-specific defaults in
`python/config.py` need overriding off this cluster: `SDI_TEXLIVE_BIN` (the
directory holding `latex`, which every figure needs) and `SDI_TREES_DIR` (the
SatGen trees, read only by `make_h5.py`). `environment.yml` pins the
Python packages the papers were produced with (linux-64); see
`docs/pymc-environment.md` for why the PyMC pins matter.

The default test run is fast and needs no data:

```bash
pytest                # unit tests only
pytest --rundata      # adds stage-parity checks against the cached results tree
```

(`--runslow` is reserved for a mock-pipeline calibration run; no test uses it
yet.)

`tests/test_figure_names.py` is the standing contract: it parses the paper
sources and asserts each figure and table exists exactly once under `plots/`,
with a registered producer (an unregistered one is reported as xfail), and
that no output outside `plots/supplementary/` exists which the papers do not
reference. The paper sources are not part of this
repository, so the module skips unless a local snapshot is present; point
`SDI_DRAFTS_DIR` at one to run it.

## Data

On the authors' machine `data/` holds about 267 GB (inputs plus the
halo-level products the pipeline writes there) and `results/` about 67 GB.
Neither is in git — the repository holds the recipe, not the bytes; the
Zenodo record below carries what reproduction needs. Small
provenance-critical text inputs (the pinned LVDB catalog, literature limit
curves, dwarf tables) *are* committed, because a silent catalog swap would
change every downstream number.

Two upstream dependencies sit outside the chain: the SatGen trees
(`TREES_DIR`) and the DwarfJeansAnalysis results (`DJA_RESULTS_DIR`), each a
single constant in `python/config.py` overridable by environment variable.
Two deposited inputs also cannot be rebuilt here: the `mass_floor_7` catalog
and the published SHMR weight draw.

### Zenodo deposition

The data needed to reproduce both papers is in a single Zenodo record,
[doi:10.5281/zenodo.24511712](https://doi.org/10.5281/zenodo.24511712)
(record 24511712). `zenodo/README_DATA.md` describes the record;
`zenodo/deposit_spec.json` lists its archives. It supports two tiers:

- **figures**: the intermediates every paper figure and table reads (except
  `jeans_corner.pdf`, see above), so each one redraws in minutes.
- **inputs**: the SatGen halo catalogs, the frozen published SHMR weight draw,
  the Fermi-LAT data and a DwarfJeansAnalysis results snapshot, from which the
  pipeline recomputes the paper results.

```bash
python scripts/zenodo_fetch.py --record 24511712 --tier figures   # or inputs, all
export SDI_DJA_RESULTS_DIR="$PWD/data/dja_results"
```

`scripts/zenodo_pack.py` builds the archives and `MANIFEST.json` (it needs
the paper sources via `SDI_DRAFTS_DIR`, and refuses to build if any recorded
figure input is missing from the deposit, checking identity by size);
`scripts/zenodo_upload.py` uploads them to a draft, and never publishes.

### Reproducibility caveats

- **The paper uses two SHMR weight realizations.** The SHMR contours and the
  Fermi recast were built from an unseeded draw migrated from the original
  analysis (`data/halo_weights_published/`), which cannot be regenerated. It
  is deposited as a frozen input for `Diemer` (all dwarfs) and `mass_floor_7`
  (Draco, Hydrus I, Segue 1). The contour stage reads it directly
  (`--weights-source published`, its default), and the Fermi stages read
  quantiles built from it (`compute_quantiles.sh` with
  `WEIGHTS_SOURCE=published`). Rerunning on it reproduced the stored Segue 1
  contours and Draco `mass_floor_7` contours bit for bit. Of the Fermi
  products the figures use, only `mhalf` was compared (Draco, Antlia II); the
  plotted SHMR variant `F18_MC` reproduces only statistically (see
  `zenodo/README_DATA.md`).
  The stellar-mass
  quantiles, tables and the figures built on them use the seeded weights
  `compute_weights.py` regenerates exactly. Quantities conditioned on the
  half-light mass draw no random numbers and are identical in both.
- **The Fermi Monte Carlo bands** were drawn with a per-process `hash()` seed
  and reproduce only statistically. The code now seeds them deterministically
  (`fermi_funcs.mc_seed`), so reruns from here on are reproducible.
- **The `mass_floor_7` prior contours** in the ρ₁₅₀–M_peak figures use a
  host selection: hosts with a satellite of M_peak > 10¹¹ M☉ within 100 kpc,
  minus each host's most massive satellite (`get_mask(lmc_selection=True)`).
- **Two more steps are statistical** besides the Fermi Monte Carlo bands:
  the scatter drawn in the
  L_vs_Mpeak and L_vs_rho150 figures (about 0.02 dex, unseeded), and the
  VEGAS integration in `Jhalopos`. Redraws of those two figures differ
  slightly from the published ones.
- **The power-law posteriors need pinned package versions** as well as their
  seed, because PyMC compiles the log-density; see `docs/pymc-environment.md`.

## Layout

```
python/    computation — importable modules and analysis entry points
scripts/   run harness — SLURM launchers and figure drivers
data/      input datasets (gitignored, except small text inputs)
results/   derived intermediates, one subdirectory per product (gitignored)
plots/     paper figures; plots/supplementary/ holds the all-39-dwarf sets
tests/     pytest suite (default and --rundata tiers)
docs/      provenance manifest, migration notes, review checklist, development guide
zenodo/    the data record's README, archive spec and metadata
```

`CLAUDE.md` is a guide for using the repository with Claude Code: setup, the
Zenodo record, how to run the figure drivers and pipeline stages, and the
pitfalls. The authors' development rules are in `docs/claude.development.md`.

## Third-party code and data

The Fermi-LAT analysis builds on the dwarf-spheroidal legacy analysis:

- **McDaniel et al. 2024**, Phys. Rev. D 109, 063024 (arXiv:2311.04982):
  the Fermi-LAT dwarf and blank-field SEDs and TS profiles under
  `data/fermi_legacy/`, and the analysis code that `python/fermi_funcs.py`
  is adapted from.
- **Circiello et al. 2026**, ApJL 1004, L33 (arXiv:2605.25586): the updated
  dwarf SEDs and TS profiles under `data/fermi_legacy_update/`, which the
  paper's Fermi figures use.
- **PPPC 4 DM ID** (Cirelli et al. 2011, JCAP 03, 051, arXiv:1012.4515;
  electroweak corrections from Ciafaloni et al. 2011, JCAP 03, 019,
  arXiv:1009.0224): the annihilation spectra under `data/fermi_legacy/PPPC4DM/`.

The McDaniel et al. and Circiello et al. releases are licensed CC BY 4.0.
Please cite these works alongside ours if you use the Fermi products.

The N-body comparison catalogs (`data/symphony/` in git, and
`SymphonyMilkyWay.h5` and `MWest.h5` in the Zenodo record) are reduced from
two simulation suites hosted at http://web.stanford.edu/group/gfc/gfcsims:

- **Symphony**: Nadler et al. 2023, ApJ 945, 159 (arXiv:2209.02675), with
  its Milky Way hosts from Mao, Williamson & Wechsler 2015 (arXiv:1503.02637).
- **Milky Way-est**: Buch et al. 2024, ApJ 971, 79 (arXiv:2404.08043).

These files are not covered by the repository's MIT licence. Please cite
these works if you use them, and include the acknowledgement each suite's
credits page requests (`zenodo/README_DATA.md` links both).

## Citation

If you use the data, cite the Zenodo record,
[doi:10.5281/zenodo.24511712](https://doi.org/10.5281/zenodo.24511712), as
well as the papers:

```bibtex
@article{Raman:2026mky,
    author = "Raman, Kailash and Folsom, Dylan and Kaplinghat, Manoj and Lisanti, Mariangela and Safdi, Benjamin R.",
    title = "{Semi-analytic Inference of Satellite Densities in the Cold Dark Matter Model Part I. Comparison to Ultra-faint Dwarf Kinematics}",
    eprint = "2607.27316",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.GA",
    month = "7",
    year = "2026"
}

@article{Raman:2026zbr,
    author = "Raman, Kailash and Folsom, Dylan and Kaplinghat, Manoj and Lisanti, Mariangela and Safdi, Benjamin R.",
    title = "{Semi-analytic Inference of Satellite Densities in the Cold Dark Matter Model Part II. Implications for Dark Matter Indirect Detection Constraints}",
    eprint = "2607.27326",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "7",
    year = "2026"
}
```

## Notes for contributors

`docs/claude.development.md` holds the working rules: what counts as
verified, when a change needs review, and the conventions that keep this code
aligned with the analysis it reproduces.

Three documents are worth reading before changing anything:

- **`docs/migration-notes.md`** — what a reader should know before trusting a
  number out of this pipeline: which stages are deterministic, which are not,
  and which products were not carried over.
- **`docs/review-checklist.md`** — recurring failure modes to check for when
  reviewing a change to the pipeline.
- **`docs/provenance.md`** — how the `.prov.json` sidecars and figure
  manifests are produced and what each field means.
