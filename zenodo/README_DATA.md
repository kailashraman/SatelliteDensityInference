# Data for "Semi-analytic Inference of Satellite Densities in the Cold Dark Matter Model" (Parts I and II)

Raman, Folsom, Kaplinghat, Lisanti & Safdi.
Part I: arXiv:2607.27316. Part II: arXiv:2607.27326.
Code: https://github.com/kailashraman/SatelliteDensityInference

This record holds the data the SatelliteDensityInference repository needs to
reproduce both papers. Every archive stores paths relative to the repository
root and unpacks there. `MANIFEST.json` lists every member with its size and
sha256, the archives' checksums, and the git commits of this repository and of
DwarfJeansAnalysis at packing time.

## Two ways to use it

**`figures` — redraw every paper figure and table** from the intermediates
the figures were made from. Minutes per figure; needs the Python environment
and a LaTeX installation, not a cluster. It covers the figures and tables in
the papers, not the repository's `plots/supplementary/` figures for the other
dwarfs, whose inputs are not deposited.

**`inputs` — recompute the paper results** from the minimal inputs: halo
catalogs, the frozen SHMR weight draw, Fermi-LAT data and the Jeans-analysis
results. The pipeline stages are SLURM array jobs over 10⁵–10⁶ halos per
dwarf (hundreds of CPU-hours in total).

```bash
git clone https://github.com/kailashraman/SatelliteDensityInference
cd SatelliteDensityInference
python scripts/zenodo_fetch.py --record <RECORD_ID> --tier figures   # or inputs, or all
export SDI_DJA_RESULTS_DIR="$PWD/data/dja_results"                   # printed by the fetch
```

The fetch checks every archive against Zenodo's md5 and `MANIFEST.json`'s
sha256 before unpacking, and refuses to overwrite a differing local file
unless given `--force`. Network fetches need `requests` (in
`environment.yml`; `--files-from` does not), and the fetch needs a Python
release with `tarfile.data_filter` (3.12, or a 3.8–3.11 patch release
from mid-2023 onward).

## Running the code elsewhere

The `scripts/*.sh` launchers are the SLURM jobs the authors ran on their own
cluster. Their `#SBATCH` headers (account, partition, QOS, excluded nodes) and
their environment setup (`source ~/.bashrc; conda activate J_calc`) are
specific to that machine, so they will not run unmodified anywhere else. Each
launcher runs `python/<stage>.py` with the arguments on its last line; their
order differs between stages (e.g. `<version> <task>` for `Jdwarf`, `<index>
<version>` for `compute_weights`). Its header documents the array size and what
the task index means, which also differs by stage (a dwarf
index 0–42 for most; dwarf × halo group for `Jdwarf`, halo chunks for
`Jhalopos`, dwarf × SHMR for `vcirc_scatter_300pc`, batches of halo groups run
through GNU `parallel` for `massProfile`). Run the Python entry points
directly, or adapt the headers to your scheduler. One exception:
`vcirc_scatter_300pc.py` reads the `#SBATCH --array=0-N` line of
`scripts/vcirc_scatter_300pc.sh` and checks that it covers every (dwarf, SHMR)
pair, 430 tasks, even when run directly; leave that file in place and that
line unchanged.

Before drawing figures, set `SDI_TEXLIVE_BIN` to the directory containing
`latex`: the figures use LaTeX text, and the default points at a module path on
the authors' cluster. The repository's `environment.yml` pins the versions of
the Python packages the papers were produced with (`conda env create -f
environment.yml`); some of those packages came from the `defaults` channel
there, so build strings may differ. It does not include LaTeX or GNU
`parallel`.

## Archives

| Archive | Contents | Tier |
|---|---|---|
| `sdi_satgen_catalogs.tar` | `data/satgen_catalogs/`: per-version SatGen halo catalogs (`.h5`) and per-halo summary arrays | figures, inputs |
| `sdi_halo_weights_seeded.tar.gz` | `data/halo_weights/Diemer/`: seeded per-dwarf conditioning weights, read directly by four figures (`plot_number_functions.py`, `plot_powerlaw.py`) | figures |
| `sdi_halo_weights_published.tar.gz` | `data/halo_weights_published/`: the unseeded SHMR weight draw the published contours and Fermi results were built from (Diemer, all dwarfs; mass_floor_7 for Draco, Hydrus I and Segue 1) | inputs |
| `sdi_fermi_legacy.tar.gz` | `data/fermi_legacy/`: Fermi-LAT dwarf and blank-field SEDs and TS profiles, and the PPPC4DM spectra | inputs |
| `sdi_fermi_legacy_update.tar.gz` | `data/fermi_legacy_update/`: updated dwarf SEDs and TS profiles (Circiello et al. 2026) | figures, inputs |
| `sdi_results.tar` | `results/`: every intermediate a paper figure or table reads, with its provenance sidecar | figures |
| `dja_results_snapshot.tar.gz` | `data/dja_results/`: DwarfJeansAnalysis per-dwarf, per-prior Jeans results (`derived.npz`) and the one posterior chain a figure plots | figures, inputs |

Not included: the SatGen merger trees (the pipeline's upstream input; only
`make_h5.py` reads them), and the full seeded weight trees for other SatGen
versions, which `compute_weights.py` regenerates deterministically.

## Recomputing from `inputs`

Fetch `--tier inputs` into its own clone rather than on top of `figures`
(`--tier all` is for downloading everything, not for recomputing in one tree): the
contour, Fermi and weight stages write into the same `results/` and `data/`
paths the `figures` tier deposits, and overwrite them in place.

The stages must be run for these SatGen versions. Catalog versions, each with its own
halo catalog: `Diemer`, `Diemer_0p12`, `Zhao`, `m2e12`, `mass_floor_7`.
Reweighting variants of `Diemer` (no catalog of their own): `lmc`, `lmc_50`,
`lvdb`, `mhalf_scatter`.

Stage order. The launcher names are the authors' SLURM jobs; each header
documents the arguments of the `python/<stage>.py` it wraps (see "Running the
code elsewhere"):

1. `compute_mhalf.sh`, pass 1 (per-dwarf half-light masses), for each catalog
   version. Its pass 2, the per-version global arrays, is already in
   `sdi_satgen_catalogs.tar`.
2. `compute_weights.sh` (seeded conditioning weights) for all nine versions.
3. For `Diemer` only: `Jdwarf.sh` then `concat_Js.sh Diemer`; `Jhalopos.sh`
   then `python python/concat_Js.py halo_position Diemer`; `massProfile.sh`
   then `concat_massProfile.sh Diemer`; `vcirc_scatter_300pc.sh` then
   `concat_vcirc_scatter_300pc.sh Diemer`. `compute_quantiles` reads the
   `Jdwarf` J-factors when they exist and otherwise writes its output without
   J quantiles, with only a warning. That is expected for the non-Diemer
   versions (the figures use only their ρ₁₅₀ quantiles); the Fermi stage needs
   them for Diemer.
4. `compute_quantiles.sh` (seeded) for all nine versions, and once more for
   `VERSION=Diemer` with `WEIGHTS_SOURCE=published`.
5. `save_contours.sh` for `Diemer` (all dwarfs), and for `mass_floor_7` only
   for Draco, Hydrus I and Segue 1 (dwarf indices 11, 27 and 39, e.g.
   `--array=11,27,39`; only these have deposited `mass_floor_7` weights, and the
   other tasks fail). The supplementary ρ₁₅₀–M_peak figures for the other 36
   dwarfs read `mass_floor_7` contours that this tier cannot regenerate; they
   come from the `figures` tier only. Then `fermi_reweighting_update.sh`.
   Both read the frozen published weights by default. The paper's Fermi
   figures use nine variants:
   `python python/fermi_reweighting.py <dwarf_index> --legacy-version update --weights-source published --variants Jeans,Jeans_satgen,Jeans_satgen_box,Jeans_satgen_shmr_fattahi18,Jeans_satgen_shmr_kim24,Jeans_satgen_shmr_danieli23_const,Jeans_satgen_shmr_moster18,mhalf,F18_MC`
   (through the launcher, set `VARIANTS` as an environment prefix rather than
   with `sbatch --export`, which splits on commas). The six dwarfs with no
   Fermi data (indices 13, 20, 21, 23, 26, 41) fail by design. The launcher's
   default variant list is longer; no figure uses the extra variants, and two
   of them (`mhalf_scatter`, `mdyn_errani`) need published `mhalf_scatter`
   weights that are not deposited.
6. The figure and table drivers: `python/plot_*.py` and
   `python/make_tables.py` (launchers `scripts/plot_*.sh` and
   `make_tables.sh`).

## Reproducibility notes

- **Two SHMR weight draws.** The stellar-mass-conditioned quantiles, tables
  and the figures built on them use the seeded weights, which
  `compute_weights.py` regenerates exactly. The r_max–v_max and ρ₁₅₀–M_peak
  contours and the Fermi recast were built from an earlier unseeded draw that
  cannot be regenerated; it is deposited as a frozen input. Rerunning the
  current code on it reproduced the stored products bit for bit wherever we
  compared them: the Segue 1 contours, the Draco and Antlia II Fermi limits
  for the mhalf, F18 and K24 variants, and the Draco `mass_floor_7`
  contours. Other dwarfs were not individually compared, except the
  `mass_floor_7` prior contours below. Weights
  conditioned on the half-light mass involve no randomness and are identical
  in both.
- **The `mass_floor_7` prior contours** (`*_unweighted` under
  `results/paper_contours/rho150_mpeak/mass_floor_7/`; the gray contours in
  the ρ₁₅₀–M_peak figures) use a host selection: keep hosts with any
  satellite of M_peak > 10¹¹ M☉ within 100 kpc of the host centre, then
  remove each host's most massive satellite by M_peak, wherever it lies.
  `save_contours.py` applies it for `mass_floor_7` only, records it as
  `prior_lmc_selection` in the provenance. Rerunning the script reproduces
  Draco's deposited prior bit for bit, and a scan repeating the same contour
  calls matched all 43 dwarfs exactly.
- **The Fermi Monte Carlo bands** (`*_MC` products) were drawn with a
  per-process random seed. The code now seeds them deterministically, so a
  rerun reproduces the published bands statistically, not bit for bit. The
  deposited `results/` products are the published draws.
- **`L_vs_Mpeak.pdf` and `L_vs_rho150.pdf`** draw their Fattahi+18 and
  Kim+24 bands with unseeded random scatter (about 0.02 dex), so a redraw
  differs slightly from the published figure, as much as two redraws differ
  from each other.
- **The halo-position J-factors** (`Jhalopos`) use an unseeded VEGAS
  integration, so they reproduce only statistically.
- **`jeans_corner.pdf` cannot be redrawn from this record.** It is drawn by
  DwarfJeansAnalysis's own plotting script
  (https://github.com/kailashraman/DwarfJeansAnalysis), which also reads the
  run's `audit.json` and that repository's star catalogs, neither of them in
  this record, and runs in that repository's environment. This record
  carries the posterior chain it plots; the published PDF is in the
  repository under `plots/jeans/`.
- Provenance sidecars record the paths on the machine that produced them,
  and the `git_commit` of the checkout that wrote them. Nearly all of those
  commits belong to the authors' unpublished development history and are not
  in the public repository; `MANIFEST.json`'s `sdi_git_commit` is the public
  commit the record was packed from.
  `MANIFEST.json`'s `path_remap` table maps the recorded paths to the
  deposited layout. The fetch checks file checksums, but the pack's check
  that each deposited file is the one a figure recorded compares sizes only.
- Archive members carry no timestamps; the fetch stamps extracted files with
  the extraction time.

## How this record was checked

Before publication, from a fresh clone of the repository holding only this
record's archives (unpacked with `zenodo_fetch.py`) and an environment built
from `environment.yml`. The figure drivers ran under `strace`, which found no
read from the authors' own data, checkouts or environment:

- `figures`: every paper figure and table driver ran. 33 of the 35 outputs
  are pixel-identical to the published ones at 100 dpi; the other two are the
  scaling-relation figures noted above. `jeans_corner.pdf` was not redrawn
  (see above).
- `inputs`, for Segue 1 and `Diemer`: `compute_mhalf`, `compute_weights`
  and `compute_quantiles` reproduced the deposited half-light masses, seeded
  weights and quantile arrays bit for bit, and one of Segue 1's twenty
  `Jdwarf` groups (121,608 halos) reproduced the deposited J-factors bit for
  bit.
- The full unit-test suite passes in that environment.

## Licence

CC BY 4.0. The Fermi-LAT and PPPC4DM tables are redistributed from their
public releases; cite the original works as listed in the papers.
