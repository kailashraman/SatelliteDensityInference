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
and a LaTeX installation, not a cluster.

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
unless given `--force`. It needs only the Python standard library, from a
release with `tarfile.data_filter` (3.12, or a 3.8–3.11 patch release from
mid-2023 onward).

## Running the code elsewhere

The `scripts/*.sh` launchers are the SLURM jobs the authors ran on their own
cluster. Their `#SBATCH` headers (account, partition, QOS, excluded nodes) and
their environment setup (`source ~/.bashrc; conda activate J_calc`) are
specific to that machine, so they will not run unmodified anywhere else. Each
launcher is a thin wrapper: it runs `python/<stage>.py` with the arguments
shown on its last line, and for array jobs the task index is the dwarf index
(0–42). Run the Python entry points directly, or adapt the headers to your
scheduler.

Before drawing figures, set `SDI_TEXLIVE_BIN` to the directory containing
`latex`: the figures use LaTeX text, and the default points at a module path on
the authors' cluster. The repository's `environment.yml` pins the
Python packages the papers were produced with (`conda env create -f
environment.yml`); it does not include LaTeX.

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

Stage order. The launcher names below are the authors' SLURM jobs; each
header documents the arguments of the `python/<stage>.py` it wraps (see
"Running the code elsewhere"):

1. `compute_mhalf.sh`, pass 1 (per-dwarf half-light masses). Its pass 2, the
   per-version global arrays, is already in `sdi_satgen_catalogs.tar`.
2. `compute_weights.sh` (seeded conditioning weights; needs step 1)
3. `Jdwarf.sh` then `concat_Js.sh <version>`; `Jhalopos.sh` then
   `python python/concat_Js.py halo_position <version>`; `massProfile.sh`
   then `concat_massProfile.sh`; `vcirc_scatter_300pc.sh` then
   `concat_vcirc_scatter_300pc.sh <version>`
4. `compute_quantiles.sh` for each SatGen version (seeded), and once more for
   `VERSION=Diemer` with `WEIGHTS_SOURCE=published`
5. `save_contours.sh` and `fermi_reweighting_update.sh` (both read the frozen
   published weights by default). The paper's Fermi figures use nine
   variants:
   `python python/fermi_reweighting.py <dwarf_index> --legacy-version update --weights-source published --variants Jeans,Jeans_satgen,Jeans_satgen_box,Jeans_satgen_shmr_fattahi18,Jeans_satgen_shmr_kim24,Jeans_satgen_shmr_danieli23_const,Jeans_satgen_shmr_moster18,mhalf,F18_MC`
   (through the launcher, set `VARIANTS` as an environment prefix rather than
   with `sbatch --export`, which splits on commas). The default run also
   includes `mhalf_scatter` and `mdyn_errani`, which need
   published `mhalf_scatter` weights that are not deposited.
6. the figure and table drivers (`python/plot_*.py` and `python/make_tables.py`; their launchers are `scripts/plot_*.sh` and `make_tables.sh`)

## Reproducibility notes

- **Two SHMR weight draws.** The stellar-mass-conditioned quantiles, tables
  and the figures built on them use the seeded weights, which
  `compute_weights.py` regenerates exactly. The r_max–v_max and ρ₁₅₀–M_peak
  contours and the Fermi recast were built from an earlier unseeded draw that
  cannot be regenerated; it is deposited as a frozen input, and rerunning the
  current code on it reproduces those products bit for bit. Weights
  conditioned on the half-light mass involve no randomness and are identical
  in both.
- **The `mass_floor_7` prior contours** (`*_unweighted` under
  `results/paper_contours/rho150_mpeak/mass_floor_7/` and
  `results/paper_contours/galactocentric/z0/mass_floor_7/`; the gray contours
  in the ρ₁₅₀–M_peak figures) use a host selection: keep hosts with any
  satellite of M_peak > 10¹¹ M☉ within 100 kpc of the host centre, then
  remove each host's most massive satellite by M_peak, wherever it lies.
  `save_contours.py` applies it for `mass_floor_7` only, records it as
  `prior_lmc_selection` in the provenance, and reproduces the deposited
  contours bit for bit.
- **The Fermi Monte Carlo bands** (`*_MC` products) were drawn with a
  per-process random seed. The code now seeds them deterministically, so a
  rerun reproduces the published bands statistically, not bit for bit. The
  deposited `results/` products are the published draws.
- **The halo-position J-factors** (`Jhalopos`) use an unseeded VEGAS
  integration, so they reproduce only statistically.
- **`jeans_corner.pdf`** is drawn by DwarfJeansAnalysis's own plotting code,
  so redrawing it needs a checkout of
  https://github.com/kailashraman/DwarfJeansAnalysis; this record carries the
  chain it plots.
- Provenance sidecars record the paths on the machine that produced them.
  `MANIFEST.json`'s `path_remap` table maps the recorded paths to the
  deposited layout. The fetch checks file checksums, but the pack's check
  that each deposited file is the one a figure recorded compares sizes only.
- Archive members carry no timestamps; the fetch stamps extracted files with
  the extraction time.

## Licence

CC BY 4.0. The Fermi-LAT and PPPC4DM tables are redistributed from their
public releases; cite the original works as listed in the papers.
