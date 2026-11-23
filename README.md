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
repository.

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
records which snapshot of it was consumed. The SatGen fork is what generated
the merger trees and satellite catalogs the pipeline starts from.

## What is here

- **34 figures and 2 LaTeX tables**, matching the `\includegraphics` and
  `\input` names the papers expect, under `plots/`.
- **117 supplementary figures** under `plots/supplementary/` — the rmax/vmax
  and ρ₁₅₀–M_peak figures for all 39 dwarfs, not only the handful the papers
  show.
- **A provenance record beside every product.** Each figure, table and derived
  intermediate carries a `.prov.json` sidecar naming the SatGen realization,
  the input files with their fingerprints, and the code that wrote it.
  `docs/provenance-manifest.json` aggregates all of it;
  `scripts/build_provenance_manifest.py` rebuilds and audits it.
- **Fifteen pipeline launchers and sixteen figure drivers** under `scripts/`,
  with the computation in `python/`.

## Reproducing a figure

```bash
conda env create -f environment.yml && conda activate sdi
export SDI_TEXLIVE_BIN=/path/to/texlive/bin  # directory holding `latex`
python python/plot_rmax_vmax.py "Segue 1"   # one figure
bash scripts/plot_rmax_vmax.sh --supplementary   # all 39 dwarfs
```

Every driver is `scripts/<name>.sh` wrapping `python/<name>.py`, and takes its
parameters as arguments rather than hardcoding them. Figure drivers run in
minutes on cached intermediates; the pipeline stages that produce those
intermediates are SLURM array jobs over 10⁵–10⁶ halos per dwarf.

**The `scripts/*.sh` launchers are written for the authors' cluster and are
not portable as they stand.** Their `#SBATCH` headers name our account,
partitions, QOS and excluded nodes, and they activate a conda environment
called `J_calc` through `~/.bashrc`. Read them as the record of how each stage
was invoked. Elsewhere, run `python/<name>.py` with the arguments the launcher
passes (for an array job, the task index is the dwarf index, 0–42), or adapt
the headers to your scheduler. Two machine-specific defaults in
`python/config.py` need overriding off this cluster: `SDI_TEXLIVE_BIN` (the
directory holding `latex`, which every figure needs) and `SDI_TREES_DIR` (the
SatGen trees, read only by `make_h5.py`). `environment.yml` pins the
Python packages the papers were produced with (linux-64); see
`docs/pymc-environment.md` for why the PyMC pins matter.

Tests come in three tiers, so the default run is fast and needs no data:

```bash
pytest                # unit tests only, seconds
pytest --rundata      # adds stage-parity checks against the cached results tree
pytest --runslow      # adds the mock-pipeline calibration run
```

`tests/test_figure_names.py` is the standing contract: it parses the paper
sources and asserts each figure and table exists exactly once under `plots/`,
with a registered producer, and that no output exists which the papers do not
reference. It passes in both directions. The paper sources are not part of this
repository, so the module skips unless a local snapshot is present; point
`SDI_DRAFTS_DIR` at one to run it.

## Data

The pipeline reads about 267 GB of inputs and produces about 67 GB of derived
intermediates. Neither is in git — the repository holds the recipe, not the
bytes. Small provenance-critical text inputs (the pinned dwarf catalog,
literature limit curves, SHMR tables) *are* committed, because a silent catalog
swap would change every downstream number.

The large inputs currently live on the group filesystem, with the SatGen trees
and satellite catalogs as the one upstream dependency the reproducibility chain
does not itself contain. Both external paths resolve from single constants in
`python/config.py` (`TREES_DIR`, `DJA_RESULTS_DIR`), each overridable by
environment variable.

### Zenodo deposition

The data needed to reproduce both papers is packaged for a single Zenodo
record (DOI to be added on publication). `zenodo/README_DATA.md` describes the
record; `zenodo/deposit_spec.json` lists its archives. It supports two tiers:

- **figures**: the intermediates every paper figure and table reads, so each
  one redraws in minutes.
- **inputs**: the SatGen halo catalogs, the frozen published SHMR weight draw,
  the Fermi-LAT data and a DwarfJeansAnalysis results snapshot, from which the
  pipeline recomputes the paper results.

```bash
python scripts/zenodo_fetch.py --record <RECORD_ID> --tier figures   # or inputs, all
export SDI_DJA_RESULTS_DIR="$PWD/data/dja_results"
```

`scripts/zenodo_pack.py` builds the archives and `MANIFEST.json` (it refuses
to build if any recorded figure input is missing from the deposit);
`scripts/zenodo_upload.py` uploads them to a draft, and never publishes.

### Reproducibility caveats

- **The paper uses two SHMR weight realizations.** The SHMR contours and the
  Fermi recast were built from an unseeded draw migrated from the original
  analysis (`data/halo_weights_published/`), which cannot be regenerated; it
  is deposited as a frozen input, and the contour, quantile and Fermi stages
  read it with `--weights-source published` (the default for the contour and
  Fermi stages), reproducing those products bit for bit. The stellar-mass
  quantiles, tables and the figures built on them use the seeded weights
  `compute_weights.py` regenerates exactly. Quantities conditioned on the
  half-light mass draw no random numbers and are identical in both.
- **The Fermi Monte Carlo bands** were drawn with a per-process `hash()` seed
  and reproduce only statistically. The code now seeds them deterministically
  (`fermi_funcs.mc_seed`), so reruns from here on are reproducible.
- **The `mass_floor_7` prior contours** in the ρ₁₅₀–M_peak figures use a
  host selection: hosts with a satellite of M_peak > 10¹¹ M☉ within 100 kpc,
  minus each host's most massive satellite (`get_mask(lmc_selection=True)`).
- **The power-law posteriors need pinned package versions** as well as their
  seed, because PyMC compiles the log-density; see `docs/pymc-environment.md`.

## Layout

```
python/    computation — importable modules and analysis entry points
scripts/   run harness — SLURM launchers and figure drivers
data/      input datasets (gitignored, except small text inputs)
results/   derived intermediates, one subdirectory per product (gitignored)
plots/     paper figures; plots/supplementary/ holds the all-39-dwarf sets
tests/     three opt-in tiers
docs/      provenance manifest, migration notes, review checklist
jupyter/   exploration only, not part of the reproducibility chain
```

## Citation

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

`CLAUDE.md` holds the working rules: what counts as verified, when a change
needs review, and the conventions that keep this code aligned with the analysis
it reproduces.

Three documents are worth reading before changing anything:

- **`docs/migration-notes.md`** — what a reader should know before trusting a
  number out of this pipeline: which stages are deterministic, which are not,
  and which products were not carried over.
- **`docs/review-checklist.md`** — recurring failure modes to check for when
  reviewing a change to the pipeline.
- **`docs/provenance.md`** — how the `.prov.json` sidecars and figure
  manifests are produced and what each field means.
