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
conda activate J_calc                       # SatGen is a pip dependency of this env
python python/plot_rmax_vmax.py "Segue 1"   # one figure
bash scripts/plot_rmax_vmax.sh --supplementary   # all 39 dwarfs
```

Every driver is `scripts/<name>.sh` wrapping `python/<name>.py`, and takes its
parameters as arguments rather than hardcoding them. Figure drivers run in
minutes on cached intermediates; the pipeline stages that produce those
intermediates are SLURM array jobs over 10⁵–10⁶ halos per dwarf.

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

### Planned: Zenodo deposition

The input datasets and derived intermediates will be deposited on Zenodo so the
chain is reproducible off this machine. The DOI will be added here once the
deposition is made.

### Reproducibility caveats

Three stages are not bit-reproducible, and it matters which numbers you take
from them:

- **The SHMR-conditioned weights are one realization.** `halo_weights.py`
  applies each relation's intrinsic scatter through the global NumPy RNG.
  `compute_weights.py` now seeds it deterministically per (version, dwarf), so
  a rerun reproduces the tree shipped here — but quantities conditioned on
  stellar mass still differ between *differently seeded* runs. Those
  conditioned on the half-light mass (`mhalf_weights`) do not draw from the RNG
  and are bit-reproducible either way.
- **The Fermi Monte Carlo ensemble cannot be regenerated exactly**, only
  reproduced statistically: it is seeded from `hash()` of a string, which
  Python randomizes per process unless `PYTHONHASHSEED` is pinned.
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
