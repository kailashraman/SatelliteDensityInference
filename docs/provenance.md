# Provenance

Every derived product in `results/`, every `.h5` in `data/additional/`, and every
figure in `plots/` carries a record of what produced it. This exists because the
worst failure mode in this pipeline is invisible: a figure combining `Diemer`
quantiles with `Zhao` weights looks entirely normal, and nothing in the arrays
says otherwise.

## Three layers

| Layer | Where | Why this one |
|---|---|---|
| In-band | `_provenance` key inside the `.npz` | travels with the file; survives copying |
| Sidecar | `<file>.prov.json` beside it | greppable without loading a multi-GB array; the only option for `.h5` and for copied-in files we must not rewrite |
| Figure manifest | `plots/<family>/<figure>.pdf.prov.json` | embeds the full stamp of every input, so a figure alone answers "which snapshot?" |

The in-band key is a 0-d string array holding JSON. It survives `np.savez` /
`np.load` and is ignored by any consumer that indexes by name, so it can be
added without touching readers.

## Fields

```
script            repo-relative producing script, e.g. python/compute_quantiles.py
argv              its arguments
git_commit        this repo's HEAD when written
git_dirty         whether the working tree was dirty (a dirty stamp is not reproducible)
written_utc       ISO 8601, UTC
satgen_version    the version argument as given, e.g. 'lmc'
sim_version       the underlying halo catalog, e.g. 'Diemer'  <- what mixing is checked on
h5_file           catalog filename
h5_fingerprint    {path, size, mtime} of that catalog
lvdb_version      pinned observational catalog release
dja_snapshot      {dir, prior, files[]} -- each DJA file actually read is
                  fingerprinted individually; a directory mtime would not
                  change when a chain is overwritten in place, which is how a
                  DwarfJeansAnalysis re-run updates it
inputs[]          {path, size, mtime, provenance} per input, one hop deep
migrated_from     set when copied in from the SatGen_Dwarf source tree rather
                  than computed here; see "Absolute paths are retained
                  deliberately" below
stochastic_arrays which array names in this product are RNG draws, e.g.
                  ('mstar_weights', 'joint_weights')
stochastic_caveat prose explaining what those draws depend on (the RNG seed)
                  and which arrays in the same product are unaffected
mstar_evolution   the infall -> z=0 stellar-mass convention evolved_Mstar used,
                  e.g. 'per_halo_tidal' (the Errani+18 tidal-track ratio read
                  per halo from stellar_mass/tpeak_stellar_mass) or
                  'not_applicable' (Symphony/MWest, which never reach the SHMR
                  path). Absence means "unknown, predates the fix" -- products
                  written before evolved_Mstar stopped adding a constant
                  +1.2 dex are unstamped, and that is NOT equivalent to
                  agreement with a stamped product. `assert_single_version`
                  raises if inputs disagree on this field, same as it does for
                  `sim_version`. compute_quantiles.py re-stamps it as a
                  top-level field on its own record (read off its weights
                  input) rather than leaving it to travel only inside
                  `inputs[]`: `_input_record` strips the `inputs` key when
                  embedding an upstream record one hop deep, so a quantiles
                  product's own `mstar_evolution` would otherwise be buried
                  inside `inputs[weights].provenance.mstar_evolution` and
                  vanish the next time that quantiles product is itself
                  embedded as an input -- e.g. in a figure manifest -- because
                  that embedding strips quantiles' `inputs[]` wholesale.
```

`inputs` embeds each input's stamp but not *its* inputs. Part II's aggregation
points combine ~39 dwarfs across several priors, and every leaf shares the same
h5 ancestor; recursing fully would copy that identical subtree once per leaf.
One hop still names each input's producing script and version, and records its
fan-in as `n_inputs`.

`satgen_version` and `sim_version` differ for reweighting-only variants: `lmc`,
`lmc_50`, `lvdb`, `mcdaniel`, and `mhalf_scatter` all reweight the `Diemer`
catalog. Mixing those with `Diemer` is legitimate; mixing `Diemer` with `Zhao`
is not. `assert_single_version` compares on `sim_version` for that reason.

`Geha` is **not** in that list, despite a `Geha -> Diemer` branch existing
upstream — it is commented out in both `compute_weights.py` and
`compute_quantiles.py`. See `docs/migration-notes.md`.

## Usage

Writing a product:

```python
import provenance

record = provenance.stamp('python/compute_quantiles.py',
                          version=version, argv=sys.argv[1:],
                          inputs=[js_path, weights_path])
provenance.savez(out_path, record, J_quantiles=..., rho150_quantiles=...)
```

Reading inputs, with the guard:

```python
provenance.assert_single_version([js_path, weights_path], expected=version)
```

## Absolute paths are retained deliberately

`migrated_from`, `h5_fingerprint.path`, `dja_snapshot.dir` and the `inputs[]`
paths are recorded as the literal absolute paths on the machine the stage ran
on, and are not rewritten to a portable stem. This is a deliberate retention,
not an oversight, and it is the reason ~1150 sidecars name a home directory and
the `SatGen_Dwarf` source tree:

- `scripts/stamp_migrated.py` re-derives these strings from `UPSTREAM` and
  refuses (`_target_missing`) to write a stamp whose target does not exist.
  Rewriting the committed strings would make them disagree with what a rerun of
  the tracked stamper produces — a sidecar that reads as evidence while being
  unverifiable by the repository's own tool, which is the failure this layer
  exists to prevent.
- `scripts/check_provenance.py` and `docs/provenance-manifest.json` are keyed on
  these records, so a textual scrub would need the manifest regenerated in the
  same change and would still leave the two out of step with the stamper.
- A provenance record's value is in being literal. "These bytes were copied from
  this exact path on this date" is the claim; a redacted path claims less.

The consequence to know: a path in a sidecar does not resolve off the machine
the migration ran on, and is a record of where bytes came from rather than an
instruction for finding them.

Stamping something we cannot rewrite (an `.h5`, or an intermediate copied in
from `SatGen_Dwarf`):

```python
record = provenance.stamp('scripts/migrate_data.sh', version='Diemer',
                          migrated_from=old_path)
provenance.stamp_existing(new_path, record)
```

A figure:

```python
provenance.figure_manifest(fig_path, 'python/plot_number_functions.py',
                           inputs, version=version)
```

## Checking

```
python scripts/check_provenance.py            # all of results/
python scripts/check_provenance.py results/paper_quantiles
```

Exits non-zero if any product is unstamped. An unstamped file is one whose
version is *unknown*, which is treated as an error rather than assumed
consistent.
