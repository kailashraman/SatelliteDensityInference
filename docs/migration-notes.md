# Migration notes

What a reader must know before trusting a number out of this pipeline: which
stages are deterministic, which are one realization, and which products cannot
be rebuilt here. Each entry is **fixed** (corrected, with the test that pins
it), **flagged** (reproduced faithfully, hazard recorded), or **not carried
over** (absent by decision, with the reason).

Throughout, **reference** means the catalogs, products and figures the
pre-migration source tree produced — the artifacts this port was checked
against. It does not refer to the papers.

---

## Fixed

### evolved_Mstar applies a per-halo tidal ratio, not a constant offset

`ResampleMstar.evolved_Mstar` maps a SHMR's infall-epoch stellar mass to z=0
before comparison with the observed `logMstar`. It previously added a scalar
`Mstar_ratio = 1.2` to `log10(Mstar)` — a constant **+1.2 dex**, a factor of ~16
upward. Three faults compounded: a linear ratio applied in log space (in log the
same ratio is 0.079 dex); a constant standing in for a value the data carries per
halo (the catalog holds both `stellar_mass` and `tpeak_stellar_mass`, tree-
building having already applied the Errani+18 tidal track, so the mapping is a
lookup); and the wrong sign, since the quantity is the fraction of stellar mass
remaining, bounded above by 1 — the real per-halo ratio runs −0.38 to +0.005 dex
on Diemer, and to −6.2 dex on `splashback`, which retains disrupted halos.

**The fix.** A per-halo `logMstar_ratio = log10(stellar_mass) −
log10(tpeak_stellar_mass)`, clipped at 0 (1–4% of halos carry a positive ratio of
at most +0.006 dex — tree-building noise). `evolved_Mstar` raises on a length
mismatch, since it runs before any mask and a shorter array would pair each halo
with another halo's stripping history, and on Symphony/MWest, which carry no
stellar-mass fields. `ResampleMstar.__init__` raises on a non-finite ratio: one
`-inf` from `log10(0)` reaching `_kde2D` poisons every weight for that dwarf.

**Why the convention is now stamped.** Products built under the two conventions
are indistinguishable by name, shape, dtype or version string, and eight of the
nine weights versions carried here were built without the offset while
`mass_floor_7` was built with it. Any comparison mixing them confounds the
convention change with the effect being measured: between conventions Fornax's
M₁ᐟ₂ column moves +0.002 dex while its Kim+24 column moves −0.825 dex, and
Fornax's median `logMpeak` of 10.2 cannot be touched by a 10⁷ mass floor.
Products therefore record `mstar_evolution`, and
`provenance.assert_single_version` raises when inputs disagree on it; absence is
its own class, meaning *predates this change*, never *agrees*.
`compute_quantiles.py` re-stamps it top-level because `_input_record` strips
embedded `inputs[]`.

One shipped figure is exactly such a comparison: `multipanel_systematics.pdf`'s
mass-floor panel plots `Diemer` against `mass_floor_7`. Because
`plot_multipanel_systematics.py` calls `provenance.assert_single_version` once
per version, and that check compares `mstar_evolution` only within a single
call's `paths`, a convention mismatch *between* the eight versions is invisible
to the guard. Check `mstar_evolution` across all eight before trusting that
panel. The trees on disk currently all stamp `per_halo_tidal`, so this is a
latent trap rather than a live discrepancy.

Two stages call `evolved_Mstar` directly rather than reading stored weights:
`vcirc_scatter_300pc.py` (flagged below) and `plot_scaling_relations.py`, whose
Fattahi+18 68% band in `logM_peak` sits 0.25–0.40 dex higher than under the old
convention — deterministically, where unseeded SHMR scatter over 2.4M halos
contributes about 0.02 dex.

Pinned by `tests/test_evolved_mstar.py`: seven of eight tests fail on the pre-fix
code, verified by sabotage. A `needs_data` tier asserts the bound on every real
catalog, and that the catalog's own infall mass round-trips to `stellar_mass`.

### Two differences between the catalog-builder notebook and what built the catalogs

Found by rebuilding the Diemer catalog and diffing all 24 arrays against the
reference file; the rebuild now matches exactly.

**1. `rvir = 259` — wrong by 476 satellites.** The host virial radius in the
Diemer dataset is **258.92401 kpc**, in the dataset's own `hosts` table. 259 keeps
2432635 satellites; the real value keeps **2432159**, the reference count.
`make_h5.host_virial_radius` derives it and asserts uniqueness. Not
Diemer-specific — `m2e12`'s host radius is **326.22382 kpc** — and the derived
recipe reproduces the reference count for all four rebuildable versions (Diemer
2432159, Diemer_0p12 2432772, Zhao 2410133, m2e12 4769107).

**2. The LMC-host search must run on the *unmasked* catalog.** A satellite
stripped below the 1% mass-loss floor, or now beyond the virial radius, still
means its host had an LMC. Searching the masked set finds 647 of Diemer's 773
hosts — a strict subset, missing 126. Had both differences stood, `mw_hosts_lmc`
would differ in 25923 rows.

**Two LMC definitions coexist, deliberately.** The catalog flag is on
**present-day** virial mass (`lmc_host_ids`, `virial_mass > 1e11`, 773 Diemer
hosts); `halo_weights.get_mask(lmc_selection=True)` *recomputes* the set inline
from **peak** mass (`logMpeak > 11`, 8829 hosts) and `gc_distance`, never reading
the stored column. The first reproduces the reference `mw_hosts_lmc`, the second
produced the `lmc`/`lmc_50` weights; they are not interchangeable.

Pinned by `tests/test_make_h5.py`, including a boundary test on the
259-vs-258.924 difference and an end-to-end test that fails if the LMC search is
moved after the mask.

### compute_mhalf.py array-task race on four global files

All 43 array tasks rewrote the same `<version>_{rho30,rho150,M30,Mvir200}.npz`,
so their contents depended on scheduling. Split into
`compute_mhalf.py <idx> <version>` (per-dwarf `logMhalf`/`logMdyn_errani`) and
`compute_mhalf.py --globals <version>`. The split is **value-preserving**, and
that is not an assumption: the global arrays are evaluated at fixed radii (0.03,
0.15 kpc) or are a mass-definition change of `Green_params`, so none reads the
dwarf. Verified bit-exact on a real-catalog slice by
`tests/test_compute_mhalf.py`. Every task computed the same numbers, so the race
was an integrity hazard — concurrent writers leaving a partial file — not a value
hazard.

### compute_quantiles.py referenced an undefined `sats` on the Symphony/MWest path

That branch read `sats['logMhalf'][dwarf][()]`, and `sats` is defined nowhere in
the module or in anything it star-imports — a `NameError` that made the branch
unrunnable. Ported as the open h5 handle's `logMhalf` group, matching
`halo_weights.from_logMhalf`. This corrects the branch's *shape*, not its
runnability: the rebuilt `data/additional/{SymphonyMilkyWay,MWest}.h5` carry no
`logMhalf` group (see below), so it now raises `KeyError` instead. Any reference
Symphony/MWest quantiles predate this line as written.

### from_logMstar accepted logMhalf_scatter and silently discarded it

`halo_weights.from_logMstar` takes `logMhalf_scatter` (default `0.0`) and built
`obs.Dwarf(...)` without forwarding it, so `obs.Dwarf.__init__`'s own default of
**0.1** applied instead; that term enters `Mhalf_err` in quadrature and reaches
the weight through `logMhalf_pdf`. So `mhalf_weights` used 0.0 while all ten
`joint_weights` columns carried an extra 0.1 dex, and the `mhalf_scatter`
variant — whose purpose is +0.1 dex on log M₁ᐟ₂ — moved only `mhalf_weights`/
`mdyn_errani_weights` while its `joint_weights` matched Diemer's. Fixed by
forwarding the argument.

### Ursa Major III orphan products removed; its catalog code is live and kept

`weights_gc/Diemer/Ursa Major III.npz` and `mhalf/Diemer/Ursa Major III.npz` were
orphans: the name is commented out of `dwarf_names`, so no task in `--array=0-42`
maps to it. Removed, with the dead commented UMa III lines in `Jdata.py`.
**Kept, and must stay:** the `gc_ambiguous` table load, the `uma3_row` /
`uma3_info` construction, and the `if name == 'Ursa Major III':` branch in
`Dwarf.__init__` — `obs.Dwarf('Ursa Major III')` places the UMa3/U1 upper-limit
point in `dispersion_limit_contours.pdf`. `tests/test_jdata_module.py` asserts it
constructs with finite `rhalf`/`rhalf_err`.

### Paths, imports and the network read

Four defects of one class — code resolving locations outside the repository or
outside `config`:

- **`Jdata.py` fetched the LVDB catalog over HTTPS at import**, making every
  script network-dependent and the dwarf sample silently mutable if the release
  were re-tagged. `dwarf_all.csv` and `gc_ambiguous.csv` are vendored under
  `data/lvdb/` with recorded sha256s, verified against the pre-vendoring module:
  eight name/count arrays identical element-for-element in order, and all 43
  dwarfs × six scalar fields identical — 258 comparisons, zero differences.
- **`util.py` appended an external tree to `sys.path`** and imported
  `Jdata as obs` from it, a name it never referenced. Both lines removed, which
  also breaks a needless `Jdata` ↔ `util` import cycle.
- **`fermi_funcs.py`'s `dSphs_csv_path` pointed one directory too high**
  (`'../fermi_legacy/'`, where every sibling constant carried the `data/`
  segment). No code path read it, so it would have fired the first time a new
  variant reached that branch, looking like the new caller's fault. All seven
  constants derive from `config.DATA_DIR`; pinned by `tests/test_fermi_paths.py`.
- **cwd-dependent imports.** Every script assumed `cwd == python/`, and seven
  files carried a dead `sys.path.append` to a retired tree. All paths derive from
  `config`, which resolves from the module file. Relatedly, `get_h5` had no
  else-branch and raised `UnboundLocalError` on an unknown version;
  `config.h5_path` raises `KeyError` naming the valid versions.

### splashback is registered normally — it is live

`splashback` is consumed: `version = 'splashback'` supplies `disrupted_fraction` for
`tidal_stripping.pdf`. Its catalog is the `_all` variant that **retains**
disrupted halos, and that figure's lower panel is `f_disrupted` — identically ~0
in `Diemer`, where those halos are already masked out. The obvious repair,
pointing `splashback` at `Diemer`, would render normally with a wrong lower panel
and biased upper-panel quantiles.

### Provenance stamps: one claim per product, verified only where diffed

An early bulk stamping of the copied `paper_Js` tree applied one note to every
product — *"reproduced bit-exactly by Jdwarf.py + concat_Js.py"* — when a single
product had been diffed, and two could not have been produced by those scripts at
all: `paper_Js/halo_position/halo_Js.npz` is written by `Jhalopos.py` in
5000-halo chunks (512 fragments, not 20), with values differing from
`Diemer/<dwarf>/` (median 2.4960e17 vs 2.5123e17); and
`paper_Js/Diemer_backup/<dwarf>/halo_Js.npz` stores only `green_Js` and
`theta95`, where the current `Jdwarf.py` also writes `full_Js`. A sidecar reads
as evidence, so a false one terminates the audit that would have caught it.

`scripts/stamp_migrated.py` records the real producer per product, marks
`verified` only for what was diffed, ties that flag to the product's bytes via
`verified_fingerprint`, and never downgrades an existing `verified: true` — it
performs no diff itself, so a lost flag cannot be reconstructed by rerunning.
Pinned by `tests/test_stamp_migrated.py`. `scripts/check_provenance.py` counts the
~5700 per-task fragments separately rather than demanding stamps: fragments are
inputs to an aggregation step whose stamp records each by path.

### Jhalopos.py is migrated, with two hazards in its docstring

It produces `paper_Js/halo_position/`, the SatGen curve of `Jfactors_all.pdf`. It
is not a variant of `Jdwarf.py`: it evaluates each halo at *its own* heliocentric
distance rather than at a dwarf's, using the small-angle radius
`sqrt((L-d)^2 + (L*theta)^2)` and unseeded **vegas** Monte Carlo, and writes only
`green_Js`. `concat_Js.py halo_position <version>` reproduces the reference
2432159-halo array bit-exactly from its 512 fragments.

* **Not bit-reproducible.** vegas is stochastic and unseeded, so a rerun differs
  within Monte Carlo error — the only migrated product that cannot be verified by
  an exact diff.
* **The random rotation is dead code.** The module draws a random z-rotation and
  builds `gc_rotated`, then takes its distance from the *unrotated* `gc`, so the
  result does not depend on the draw. Ported unchanged; removing it would be a
  behaviour change to a reference product.

**Layout changed deliberately.** The two sources disagree: the script writes
`paper_Js/<version>/halo_position/`, while the on-disk artifact and the plot
script use `paper_Js/halo_position/` with no version, where a second version's
run would silently overwrite the first. Neither is safe — the first puts a
non-dwarf directory beside the per-dwarf ones, where `concat_Js` would treat it
as a dwarf. This repository uses `paper_Js/halo_position/<version>/`.

### SymphonyMilkyWay_old.h5 mixed two simulation suites

`SymphonyMilkyWay_old.h5` carries **63 hosts**: the 45 real Symphony hosts plus 18
MWest-only hosts (`Halo327`/`Halo349` exist in both suites, and the old file holds
the genuine Symphony versions of those two). `MWest_old.h5` has the right 20 hosts
but only 3123 of 7113 halos, reproducing exactly a `mvir > 1.2e8` cut. The suite
was selected by toggling a commented `base_dir` and a commented `np.savez`
destination, with the mass cut a third commented line: a run written into the
Symphony output directory with the MWest settings active was swept up by the h5
builder, which globs whatever is in the directory.

Values never changed — for every shared halo all eight fields are bit-identical,
and the old catalogs are strict subsets of the new. The rebuilds (45 and 20 hosts,
the true suite sizes) are what `config.H5_REGISTRY` names and rebuild from the
vendored `data/symphony/` tree via `make_h5.py sims`; every figure that reads them
was produced afterwards. The rebuild dropped the `logMhalf` group, which the
`from_logMhalf` path reads for these two suites only; no figure carried here needs
it, so `augment-mhalf` is deliberately not run and the Symphony/MWest weights and
quantiles were removed rather than regenerated.

**Live hazard, not history:** `make_h5.py sims` still trusts the contents of a
directory. Anything stray in `data/symphony/<suite>/` becomes part of the catalog.

### A figure's series set is defined by the rendered figure, not by comment state

A commented-out `plot`/`errorbar`/`savefig` in a source module is not evidence
that a series is absent from the rendered figure, and uncommenting it is not
evidence it was present. Symphony/MWest were recorded as reaching no figure
because their `errorbar` calls in the `multipanel_systematics` cell are commented
out; they reach `sims_rho150.pdf`, `sims_mpeak.pdf` and
`distance_statistics_sims.pdf` by another data path. And
`multipanel_systematics.pdf`, planned as ten versions in one panel, renders as
**six panels** over eight `weights_gc` versions — c-M relation (`Diemer`, `Diemer_0p12`, `Zhao`), SHMRs (weight *columns*
of `Diemer`, not versions), mass estimators (`Diemer`, `mhalf_scatter`), host mass
(`Diemer`, `m2e12`), LMC analogue (`Diemer`, `lmc_50`, `lmc`), mass floor
(`Diemer`, `mass_floor_7`).

`pdftotext` returns nothing here (`text.usetex=True` renders text as paths), so
absence of a label in extracted text proves nothing either; rasterize with pymupdf
at ~110 dpi and read the legend. Three series sets settled that way, now fixed in
the ported drivers:

* `fermi_reweighting_prior_envelope.pdf`'s band is
  `band_indices = [0, 6, 7, 8, 9, 10, 11]`; an adjacent comment lists
  `0, 1, 6, … 11`, and index 1 is `mhalf`, which is not a Jeans analysis.
* `Jfactors_all.pdf` plots five curves: SatGen, mhalf, Fattahi+18, Jeans with the
  standard prior, and Jeans with the SatGen-informed prior.
* `fermi_limit_comparison.pdf` has **no** Galactic Center Excess band — five
  legend entries plus a thin thermal-relic line — although its producing cell
  carries an uncommented
  `files = [Abazajian2015.csv, Daylan2014.csv, Calore2014.csv]` reaching nothing
  rendered. `sigma_thermal.csv` is used, for the relic line; literature inputs
  live under `data/literature/`.

An array diff against an intermediate `.py` driver is not evidence that a figure
is right: those drivers are themselves transcriptions of the notebook cells, and a
diff cannot catch a defect shared by both sides.

---

## Flagged — reproduced faithfully, hazard recorded

### SHMR weights are one unseeded realization, not a function of the catalog

`halo_weights.py` applies each SHMR's intrinsic scatter through the global NumPy
RNG with no seed in the module — nine live draw sites (`stats.norm.rvs` in
`logMstar_Kim24`; `np.random.normal` in `logMstar_B13`, `_RP17`, `_Moster13`,
`_Moster18`, `_Fattahi18`, `_Behroozi19`, `_Munshi21`, `_Danieli23` at both call
sites). A tenth, `np.random.rand`, is gated on `floor_width is not None`,
hardcoded `None` with no setter — dead on the reachable path.

Confirmed by rerunning `compute_weights.py 0 Diemer` against real data:
`mhalf_weights` reproduces bit-exact; `mstar_weights` and `joint_weights` do
**not**. Verification must therefore be stated per array — bit-exact for
`mhalf_weights`, distributional at a quoted tolerance for the rest.

**The `weights_gc/` tree carried here is a local regeneration, not the reference
bytes.** Every sidecar under it records `migrated_from: null` and
`mstar_evolution: per_halo_tidal`, so the tree is reproducible by rerunning
`compute_weights.py` at the same (version, dwarf) — but it is *not* the
realization behind the reference SHMR numbers, and that realization is not
preserved anywhere in this repository. Any value drawn from `mstar_weights` or
`joint_weights` is this repository's realization.
A parity claim covering the whole file is an overclaim. The contour stage inherits
the same split, measured on one task (`Segue 1`, `Diemer`, `z0`): `_unweighted`
and `_mhalf` exact on all eight arrays; `_F18`, `_K24`, `_joint_F18` exact in
`X_hist`/`Y_hist`/`X_kde`/`Y_kde` while `hist_hist`/`hist_kde` differ 3–11% and
contour levels 3e-3 to 1.2e-2.

**Seeded from here on.** `compute_weights.py` seeds the global RNG once, after the
version dispatch and before any draw, from `zlib.crc32(f'{version}:{dsph_idx}')` —
deterministic and distinct per (version, dwarf). `hash()` is deliberately not
used: Python randomizes string hashing per process unless `PYTHONHASHSEED` is
pinned. `vcirc_scatter_300pc.py` seeds the same way per (version, dwarf, SHMR),
replacing a module-level `SEED = 0` that fed only a *local* `default_rng`. No seed
recovers a realization drawn before it existed: regenerating moves `mstar_weights`
and `joint_weights` relative to the copied products, while `mhalf_weights` is
deterministic either way. Reach: panel 2 of `multipanel_systematics.pdf` and the
`rho150.pdf` inference curves.

### The vcirc_scatter_300pc tree was regenerated, and its numbers moved

`vcirc_scatter_300pc.py` calls `ResampleMstar.evolved_Mstar` directly rather than
reading stored weights, so it takes the infall → z=0 stellar-mass correction at
full strength; the copied tree predates it. Regenerating all 430 (dwarf, SHMR)
products under the per-halo Errani ratio moves `V_median` by a pooled median of
**+16.5%** (max +55.6%), upward for **96.3%** of pairs, and `sigma_dex` by a
pooled median of 0.0061 dex (max 0.125).

The sign is the expected one: removing the constant +1.2 dex lowers the
*predicted* stellar mass of every halo, so matching a dwarf's fixed *observed*
`logMstar` selects more massive halos, which have larger V_circ at 300 pc. The
observation side is untouched — `logMstar`, `logMstar_err_lo` and
`logMstar_err_hi` are bit-identical across all 430 rows, confirming `Jdata.py`'s
Woo+2008 `ML_ratio = 1.2` (a genuine linear M/L) was not caught up in the change.
Part of the `sigma_dex` movement is not physics: this stage also gained RNG
seeding, which alone changes `sigma_dex` by ~13% between runs.

Consumers (`satellites_vcirc_all_*`, `powerlaw_*`, `dispersion_limit_contours`)
build from the regenerated tree, so they are verified against the same code run on
this repository's regenerated inputs rather than against the older rendered PDFs.
Any `sigma_dex` or `V_median` value carried over from the older tree will not
reproduce against the regenerated one, and the gap is far wider than the ~13%
run-to-run spread seeding alone accounts for. Use the regenerated tree.

### vcirc_median_uncertainty replaces the observed M* errors with a constant

The Monte Carlo *draw* of the central value uses the observed asymmetric errors
(`_sample_twosided(rng, dwarf.logMstar, lowerr, higherr, ...)`); the conditioning
kernel that then weights the halos does not — the mock dwarf's `logMstar_err` is
overwritten with `NARROW_SIGMA = 0.16` dex, symmetric and identical for every
dwarf. Measured, not assumed: 0.16 dex is **narrower** than the observed mean
`logMstar_err` for all 43 dwarfs (0.196 min / 0.28 median / 0.616 max), so the
kernel tightens the conditioning for every dwarf. The parameter name
(`narrow_sigma`) suggests this is deliberate. The direction of the effect on the
reported `sigma_dex` has **not** been measured; do not assume it. Reproduced
without change; affects `satellites_vcirc_all_fattahi18.pdf`,
`satellites_vcirc_all_kim24.pdf` and the per-dwarf scatter values.

### The two sims panels mask Symphony/MWest differently

In `plot_number_functions.py`, `build_sims` (`sims_rho150.pdf`, `sims_mpeak.pdf`)
masks the Symphony/MWest reductions on `(logmvir - logmpeak) > -2`, `d < 259`,
`mpeak > 1.2e8`, while `sim_dist_quantiles` (`distance_statistics_sims.pdf`)
applies only `d < 259` and `mpeak > 1.2e8` — **no mass-loss cut** — against a
SatGen side whose catalog is already survival-masked. One panel therefore compares
survival-masked SatGen against unmasked Symphony, and the two sims figures treat
the same suites differently. A property of the reference figures, not a porting
artifact; reproduced as-is.

### util.quantile's len(c) - 2 clip is a real off-by-one, dormant at production scale

`util.quantile` clips its `searchsorted` index to `len(c) - 2`, so it can never
return the largest element: with unit weights on five values it returns the
*second* largest for `q = 1.0`, and the clip truncates every quantile above the
second-to-last cumulative weight.

| N | 0.16 / 0.5 / 0.84 / 0.95 affected? |
|---|---|
| 5 | yes, grossly |
| 100 | yes, adjacent-value off-by-one |
| 10,000 | yes, ~3e-5 relative |
| 2,432,159 (the Diemer catalog) | **no** |

At the real catalog size the clip engages only for `q > 0.999999178`, and every
quantile quoted from this pipeline is 0.16/0.5/0.84 — so no number moves. Left
as-is and pinned by a regression test. **The hazard to watch is small N:** masking
can cut a dwarf's surviving halo count by orders of magnitude, and at a few
hundred halos the off-by-one becomes visible. Any future stage that quantiles a
heavily masked subset, a per-dwarf tail, or a bootstrap resample should be checked
against this entry first.

### The Fermi MC ensemble is seeded with hash() and is not reproducible

`fermi_reweighting.py` seeds its 1000-realization Monte Carlo with
`np.random.default_rng(abs(hash(name)) % (2**32))`. Python randomizes string
hashing per process unless `PYTHONHASHSEED` is set, and no launcher sets it, so
**every run draws a different ensemble**: `hash('Draco')` returns 480799978 /
878290448 / 2581502141 in three successive processes. The three artifacts on disk
(legacy `F18_MC`, update `F18_MC`, legacy `F18_MC_test`) are three distinct
realizations. The module docstring records this as the seeding convention to
preserve, while `compute_weights.py` documents rejecting exactly this construction
in favour of `zlib.crc32`. The 68%/95% containment bands of
`fermi_reweighting_MC.pdf` are this ensemble, so a rerun gives a statistically
equivalent but numerically different band. Two layers sit underneath: the F18
quantiles the MC samples from derive from `mstar_weights`, themselves an unseeded
realization, and the draw is a split-normal fit to the 16/50/84 quantiles rather
than a full J-factor PDF. Left as-is deliberately: changing the seed changes an
archived band, and the docstring must change in the same edit.

### build_jfactors_all's version guard is right; its stated reason is not

`plot_number_functions.py`'s `build_jfactors_all` refuses any version but
`Diemer`, justified by a comment claiming the other per-version subdirectories of
`results/paper_Js/` are empty. They are not — each holds 43 populated
`halo_Js.npz` files whose lengths match the corresponding catalog exactly (Diemer
2432159, Diemer_0p12 2432772, Zhao 2410133, mass_floor_7 4142592), all finite,
with medians 3.7–5.6e17 for the Diemer family and 8.6e16 for `mass_floor_7`. The
**guard is correct for a different reason**: the figure also reads
`paper_Js/halo_position/` and the per-dwarf inference curves, which exist only for
Diemer. The same function carries a **`Tucana V` column-fill hazard** — the Jeans
curve fills one fewer dwarf column than the other three curves. Both ported
verbatim.

### ps18_names and n_stars carry one element more than there are dwarfs

Both have 44 entries against 43 in `dwarf_names`, `abbreviations`, `fermi_names`
and `fermi_names_update`. The extra sits past the end: all 43 valid indices align
correctly (verified at `LMC`, `Fornax`, `Ursa Major II`, `Segue 1`, `Leo V`). Not
a live off-by-one — but appending a dwarf would make it one, so the lengths are
pinned in `tests/test_jdata_module.py`.

### The DwarfJeansAnalysis snapshot a figure consumes can move under its audit

`jeans_corner.pdf` and the prior-ladder figures read `config.DJA_RESULTS_DIR`
directly, and that tree is a live dependency updated in place. Inside
`production/ursa_major_2/loguniform/`, `derived.npz` and `posterior_samples.npz`
are dated later than the `audit.json` and `run.log` beside them: the posteriors
were regenerated after the audit was written, so the audit does not describe the
bytes the figure consumes. Provenance records `dja_snapshot` by fingerprinting
the consumed files individually rather than by directory mtime, precisely
because an in-place re-run leaves the directory looking unchanged. Re-check the
audit's currency before quoting anything from that chain.

### The stored fermi_reweighting tree is superseded, not a parity baseline

A single-task check (`Antlia II`, `--variants mhalf`) shows `TS_array` differing
from the copied product by 1.7138e-04 relative, with `mass_vec`/`sigmav_vec`
exact. The port itself is exact: the code this was ported from, run today against
the same inputs, reproduces this repository's output bit-for-bit, and both differ
from the stored tree by the same 1.7138e-04.

The cause is the input, not code drift. The `mhalf` path is numerically unchanged
over the period — a surviving `.pyc` disassembles against the current source with
zero numeric-constant differences — while
`results/paper_quantiles/galactocentric/Diemer/<dwarf>.npz` was overwritten by a
later `compute_quantiles` array run following a `Jdwarf` rebuild, after the stored
Fermi product was written. Reconstructing the earlier values for Antlia II gives
`logJ_50` −3.85e-5 dex and `sigmaJ` +3.2e-5 dex relative to today; feeding those
two scalars into the current code reproduces the stored `TS_array` to 6e-13
median, and a `sigmaJ` shift is not degenerate with a flux or J rescale, so the
fit is specific. The 95% upper limit moves by ≤8.3e-5 dex (0.019%).

Treat the stored tree as *superseded but numerically equivalent*: re-run it for
provenance hygiene, and point a future parity gate at a fresh run rather than
these bytes. The same caution applies to `fermi_expected_limits/`. Ruled out with
evidence: the Fermi input data, the low-side interpolation clamp (reverting is
bit-identical), `Jdata.py`'s `vmax_min` cut (it touches only `Jeans*` variants),
the weights, and the regenerated `paper_quantiles` — re-running against the
byte-identical `galactocentric_published` tree gave the *identical* 1.7138e-04.

Separately unresolved: `paper_quantiles/galactocentric` regenerated here differs
from `galactocentric_published` by up to 4.5e-2 relative (`cV_quantiles` 4.5e-2,
`rho150_quantiles` 2.1e-2, `J_quantiles` 9.4e-3), most of which the
`evolved_Mstar` change explains — but `Mhalf_quantiles` also differs, by 4.1e-3,
while `mhalf_weights` is bit-exact between the trees. Identical weights giving
different quantiles implies the *sample* entering the quantile changed, not the
weighting. Worth settling before a new figure consumes `Mhalf_quantiles`.

---

## Not carried over

### Diemer_backup is stale and was not copied

`paper_Js/Diemer_backup/<dwarf>/` supplied the per-dwarf inference curves of
`Jfactors_all.pdf`. Compared across all 43 dwarfs it is **bit-identical to
`paper_Js/Diemer/`** on both arrays it holds (`green_Js`, `theta95`); it simply
predates `full_Js`. Not copied (3.2 GB saved), and `plot_number_functions.py`
reads `paper_Js/Diemer/<dwarf>/halo_Js.npz` instead — a change that provably
cannot move a number. Guarded by
`tests/test_concat_Js.py::test_diemer_backup_is_not_reintroduced`.

### Products removed as stale or orphaned rather than migrated

Four product trees were deleted after being copied:

| removed | reason |
|---|---|
| `weights_gc/{Symphony,MWest}` | keyed to the contaminated 63-host catalog; not regenerable; no figure consumes them |
| `paper_quantiles/{Symphony,MWest}` | descend from those weights |
| `paper_quantiles/mcdaniel` | its 17 GB of weights were deliberately excluded, leaving no in-repo producer |
| `paper_quantiles/Geha` | same, and it reaches no figure (below) |

The `mcdaniel`/`Geha` pair was an inconsistency in the copy itself: the quantiles
tree was taken wholesale while those versions' weights were excluded by scope,
leaving derived products nothing here could rebuild. `fermi_reweighting.py`
therefore excludes `mhalf_mcdaniel` from its default variant set and fails with a
targeted message if it is requested explicitly, as
`scripts/fermi_reweighting_update.sh` records. `weights_gc/` and
`paper_quantiles/galactocentric/` now hold the same nine versions: `Diemer`,
`Diemer_0p12`, `Zhao`, `lmc`, `lmc_50`, `lvdb`, `m2e12`, `mass_floor_7`,
`mhalf_scatter`. `scripts/stamp_migrated.py` treats `Symphony`, `MWest`,
`mcdaniel` and `Geha` as simply absent from the local tree, not as an error.

### Geha reaches no figure

A reverse trace over every notebook cell mentioning `geha`, checked for whether a
`savefig` depends on it, finds no figure using it: the cells building `geha_*`
arrays contain no `savefig`, the cell rendering `tidal_stripping.pdf` references
none of them, and the one cell writing `L_vs_rho_geha.pdf` is commented out and
that basename is not a paper figure. So `weights_gc/Geha/` (9.6 GB) and
`paper_quantiles/galactocentric/Geha/` are not copied, and `config.H5_REGISTRY`
maps `Geha` to `None`. Two things would block a rebuild, both recorded in
`config.py` and `docs/provenance.md`: the `Geha -> Diemer` `sim_version` branch is
commented out in both `compute_weights.py` and `compute_quantiles.py`, so
`version='Geha'` falls through to a `sim_version` with no branch; and the same
disabled block set `dwarf_names = obs.geha_names`, which no longer exists in
`Jdata.py`. The artifacts hold 25 dwarfs, not 43, and being named
`Geha/<dwarf>.npz` the list is recoverable from the filenames — `kd_names` with
`Leo V` and `Leo T` replaced by `Leo VI` and `Pisces II`.

### mass_floor_7 cannot be rebuilt in this repository

It is built by re-filtering `26-02-Diemer_mass_floor_7.h5` (**80 GB**, not carried
over) rather than from a tree dataset; the filtered 920 MB result is copied in and
stamped `migrated_from`. Recorded in `make_h5.NOT_REBUILDABLE` so
`build mass_floor_7` fails with the reason instead of a missing-file error, and
asserted by `tests/test_make_h5.py` so a registered catalog can never be silently
unaccounted for. Two consequences: the `mwID < 2000` cut belongs to the re-filter
path, not the general build path, and is exposed as `--max-mwid` rather than
applied unconditionally; and `mass_floor_7`'s `mw_hosts_lmc` follows the
**post-mask** convention (130 LMC hosts over `mwID < 2000`), unlike the five
rebuildable catalogs, which use the pre-mask convention. Do not assume all six
catalogs share one flag definition. Same class:
`halo_weights.get_mask(use_z50=...)` / `use_z90` read `h5['z50']` / `h5['z90']`,
which none of the eight catalogs carried here has — those writes are commented out
in the builder, and no driver passes the keyword.

### paper_contours/rho150_mpeak/Diemer is required, not spare

The 2.1 GB `paper_contours/rho150_mpeak/Diemer/` tree looks prunable next to the
per-dwarf contour products, but it is what `rho150_mpeak` reads. Do not prune it
on size grounds.

### Launchers must resolve the repository root from SLURM_SUBMIT_DIR

`scripts/<name>.sh` derives `REPO` from `SLURM_SUBMIT_DIR`. Invoked by hand from
inside a session the scheduler did not spawn, that variable is unset or points
elsewhere, and `REPO` resolves into an unrelated tree — the script then runs, but
against the wrong paths. Submit these from the repository root, or set
`SLURM_SUBMIT_DIR` explicitly when running one by hand.

### Everything else not carried over

| Item | Reason |
|---|---|
| `26-02-Diemer_mass_floor_7.h5` (80 GB) | unfiltered source of the `_updated` catalog; see above |
| `fermi_legacy` `.zip`/`.tar` archives (~21 GB) | redundant with the extracted directories; no path constant references them |
| TeX binary paths | hardcoded to two different texlive module directories; now `config.TEXLIVE_BIN`, overridable |
| `plot_vcirc_green_vs_nfw_segue1.py` | diagnostic, not a paper figure; also wrote a PNG into the repo root |
| notebook cells 53/54, 56/60 | duplicates, de-duplicated on port |
| notebook cells 29/30 | both write `panel_Jeans.pdf`, not a paper figure |
| `toyAnalysis*`, `jeans_pace/`, `jeans/`, `Jcluster`, `JofE`, `make_contours`, `sim_load`, `Janalysis` | out of scope; several target retired trees |
| `data/kinematics/` (6.3 MB) | consumed only by `jeans/` and `jeans_pace/`, both out of scope; the Jeans figures read the DwarfJeansAnalysis `results/` tree instead |
| `ProcessTrees.py`, `compute_zform.py`, `compute_rmaxvmax.py` and their products (~3.2 GB) | forward trace: their only consumers are three notebook cells whose every downstream `savefig` is commented out, producing `zforms.pdf` and `stats_with_jeans/rho150_zform50_5.pdf`, neither a paper figure |
| `F18_MC_test.npz` / `K24_MC_test.npz` (38 dwarfs) | output of an early array run from a since-overwritten revision, named for no variant in current code, and consumed by nothing: the only mention is a commented-out line in the MC-list cell, they exist only in the legacy tree while the figure reads the `update` tree, and their 60-point `sigmav` grid (`logspace(-30,-23,60)`) would fail the current code's 70-point assertion. Left on disk, unused |
