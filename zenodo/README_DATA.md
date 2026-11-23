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
the figures were made from. Minutes per figure, no cluster needed.

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
unless given `--force`.

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

Stage order (each `scripts/<stage>.sh` launches `python/<stage>.py`; see its
header for arguments):

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
   variants; select them with
   `VARIANTS=Jeans,Jeans_satgen,Jeans_satgen_box,Jeans_satgen_shmr_fattahi18,Jeans_satgen_shmr_kim24,Jeans_satgen_shmr_danieli23_const,Jeans_satgen_shmr_moster18,mhalf,F18_MC sbatch scripts/fermi_reweighting_update.sh`
   (an environment prefix, not `--export`, which splits on commas). The
   default run also includes `mhalf_scatter` and `mdyn_errani`, which need
   published `mhalf_scatter` weights that are not deposited.
6. the figure and table drivers (`scripts/plot_*.sh`, `make_tables.sh`)

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
  `results/paper_contours/rho150_mpeak/mass_floor_7/`,
  the gray contours in the ρ₁₅₀–M_peak figures) were built from halos in
  hosts with an LMC-analog satellite, excluding that satellite. The current
  code does not apply this selection, so rerunning it changes those gray
  contours; the deposited products are the published ones.
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
