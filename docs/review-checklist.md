# Review checklist

Recurring bug classes in this repository. Reviewers consult this before a pass;
**only the main session writes to it**. When a reviewer catches a bug whose
class is not listed, it surfaces the proposed class in its findings — a one-line
class name, where it appears, what goes wrong when it slips through, and what to
flag going forward — and the main session makes the edit. Serializing writes
here keeps entries general enough to catch near-misses, not just the exact bug
that triggered them.

Entries describe the class, not the instance. Do not put the triggering
`file:line` in the "what to flag" body.

---

## Silent SatGen version mixing

**Where:** anywhere two derived products are loaded together — quantiles with
weights, contours with an h5, a plot script reading several `results/` trees.

**What goes wrong:** `<version>` selects the SatGen realization and cosmology.
Combining products built from different realizations produces a figure that
looks entirely normal; nothing in the arrays records which run they came from.
The error propagates into a published number with no symptom.

**What to flag:** a load of two or more derived products with no
`provenance.assert_single_version` guard; a version string threaded through some
paths but hardcoded in others; a reweighting variant (`lmc`, `lmc_50`, `lvdb`,
`mcdaniel`, `mhalf_scatter`) compared against its underlying catalog without
going through `config.sim_version`; a product written without a stamp. Note
`Geha` is *not* such a variant — see the next two entries.

## Path constants no code path exercises

**Where:** module-level path constants in library modules, especially ones that
are only read by a rarely-used branch.

**What goes wrong:** a wrong constant sits undetected for as long as nothing
reaches it, then fails — or worse, silently reads a stale sibling directory —
the first time a new figure or variant touches that branch. Because it never
broke before, the failure looks like it was introduced by the new caller.

**What to flag:** a path constant with no test asserting the target exists; a
constant whose relative depth differs from its siblings in the same block; a
directory constant assembled by string concatenation where its neighbours use a
path join.

## Array-task races on shared output files

**Where:** SLURM array entry points that write both a per-task file and an
aggregate file.

**What goes wrong:** every task in the array rewrites the same aggregate, so its
contents depend on scheduling. The file is present and well-formed afterwards,
so nothing looks wrong; it simply holds whichever task finished last, possibly
interleaved.

**What to flag:** a write to a path that does not contain the task index; an
aggregate assembled inside the per-task code path rather than in a separate
single-task pass; a `savez` of a global array inside a loop over one dwarf.

## Disabled upstream branch ported as live behavior

**Where:** any table, registry, or conditional in migrated code that encodes a
rule read out of the source tree — version aliases, name maps, prior ladders,
default selections.

**What goes wrong:** the branch being copied is commented out, dead, or
overridden later in the same file, so the migrated code reproduces behavior the
source no longer has. Ported code, its docstring, and its tests then all agree
with each other and look internally consistent; the divergence is visible only
by diffing against the *currently reachable* lines upstream. Worse, the
artifacts that branch once produced usually still exist on disk, so the mapping
appears corroborated by the data.

**What to flag:** any migrated table whose upstream counterpart has commented-out
code adjacent to it; a claim that code "mirrors" a source file, without a quote
of the active lines; an entry justified by the existence of an output directory
rather than by reachable code; a test that asserts the same assumption the code
makes, and so cannot fail independently of it. Check reachability, not presence:
that the branch sits inside a conditional actually evaluated, and is not
superseded by a later unconditional assignment.

## Live upstream branch dropped as unused

**Where:** registries, version maps, and "not migrated" tables built while
porting — especially entries justified by a prose claim about what consumes
them. The mirror image of the entry above.

**What goes wrong:** a branch that is reachable and figure-producing upstream is
recorded as dead, usually because the search covered `.py` drivers but not the
notebook where most figures are actually rendered. The registry, its comment,
and its tests then agree with each other. When someone later restores the
entry, the obvious substitute is a sibling catalog with different content,
producing a plausible figure with a silently wrong panel.

**What to flag:** a registry value of `None`, or a not-migrated note, whose
justification is a claim about consumers rather than a search result across
`.py`, `.ipynb`, **and** the drafts' `\includegraphics` basenames; a dropped
constant whose upstream branch was uncommented. Require the reverse trace:
figure basename in the `.tex` → the `savefig` cell → the variables that cell
reads → the cell that computed them.

## Comment state read as the figure specification

**Where:** any claim about what a migrated figure contains — which series it
plots, how many panels it has, which versions or weight indices it consumes —
derived by reading the source rather than the published PDF.

**What goes wrong:** the upstream sources were left in whatever comment state
the last exploratory run ended in, so commented-out draw calls are post-run
residue, not evidence of absence — and uncommented ones are not evidence of
presence. Scope and copy-list decisions built on that reading are wrong in
both directions: a live series gets dropped as dead, or a dead one is
reproduced. Nothing downstream contradicts it, because the code, its port, and
its tests all inherit the same misreading.

**What to flag:** any statement of the form "this series is commented out, so
it reaches no figure", or a panel/series count asserted without the rendered
figure; a copy-list or exclusion justified by comment state; a port whose
series set was never compared against the published artifact. Note that
`pdftotext` returns nothing on these figures — `text.usetex=True` renders text
as paths — so "the PDF has no such label" is not evidence either. Rasterize
and look at it.

## Artifacts outliving the code that made them

**Where:** migrated intermediates whose producing branch has since changed shape
— different dwarf list, different column order, different units.

**What goes wrong:** the arrays load fine and have plausible values, but are
keyed to an index the current code no longer constructs. Indexing them with
today's list silently mis-assigns every row.

**What to flag:** an intermediate whose element count differs from the current
catalog length; a directory whose file count does not match the number of
dwarfs; any consumer that indexes a migrated product positionally without
asserting its length; a version present in `results/` but absent from, or
disabled in, the producing script.

## Regenerated record overwriting the one field that was not re-derivable

**Where:** bulk provenance sweeps that rebuild every field of a record from
current state — producer, source path, fingerprint, commit, timestamp.

**What goes wrong:** almost every field *should* be regenerated, which makes
the sweep look safe. But a record can also carry the result of a human or
external action — that someone diffed the bytes against upstream and they
matched. That fact is not re-derivable from the file or the repo, so
regenerating it silently downgrades it to "not verified", destroying the only
evidence anyone ever checked. Guards written to protect locally-produced
products do not catch it: a migrated product that was *also* verified has a
legitimate migration origin and often no in-band record, so it falls between
them.

**What to flag:** any bulk stamper that writes a verification field
unconditionally; a verification flag with no tie to the content fingerprint it
attests to, which can outlive the bytes it was granted for; a claim that
nothing was lost, based on a scan run *after* the sweep — the post-run state
cannot distinguish "never set" from "just cleared". Establish the prior state
from an independent record, or say the question is open.

## Two-layer provenance where only one layer is rewritten

**Where:** products carrying both an in-band stamp and a sidecar, touched by a
tool that writes only one of them — bulk stampers, fix-up sweeps, re-stamps
after a partial regeneration.

**What goes wrong:** the reader prefers one layer, so a stale or false record
in that layer wins over a truthful one beside it, and nothing ever compares the
two. The failure is invisible precisely because the product looks
doubly-documented. It compounds downstream: every consumer that embeds an
input's provenance copies the wrong layer forward.

**What to flag:** a sidecar-only writer applied to a file that could have been
written by the both-layers writer; a re-stamp path that can run over locally
produced products; a record asserting a migration origin for a file whose
in-band stamp names a local script. Require the writer to update both layers or
refuse when they would disagree.

## Race fixed in the code but re-openable from the submit line

**Where:** launchers for a stage split into a per-task mode and a single-task
aggregate mode, where the split exists to remove an array-wide write race.

**What goes wrong:** the Python is correct and the migration note records the
race as fixed, but the `#SBATCH --array` directive still requests the full
array and the "run this one once" constraint lives only in a comment. One
submission with the mode flag and the default array re-creates the original
race — now with an N× runtime penalty — and the note says it cannot happen.

**What to flag:** a launcher mode switch whose task-count requirement is
documented rather than asserted in the script body; an aggregate/`--globals`
mode reachable from a multi-task submission; any "race fixed" claim whose
enforcement is not visible in the code path that actually runs.

## Provenance caveat naming arrays the product does not contain

**Where:** stochastic, verification, or staleness annotations propagated from a
stage to a derived tree whose arrays have different names or a different axis
layout.

**What goes wrong:** the caveat is doubly wrong — false about the arrays it
names, which are absent, and silent about the ones actually affected, which are
present under other names or as columns of a stacked array. A reader checking
whether a quantity is reproducible finds it unlisted and concludes it is safe.

**What to flag:** an annotation list not cross-checked against the product's
own `.files`; a caveat copied to a derived product without translating to that
product's array names, columns, or axes; per-file marking where the property
varies per array or per column.

## Bulk stamper minting provenance it never validated

**Where:** tools that sweep a migrated tree and write provenance records
derived from the directory layout — version from a path component, source path
by string-joining onto an upstream root.

**What goes wrong:** the stamp is generated, not observed. It asserts a
producer that may be unable to produce the file, an upstream source that may
not exist, and a version taken from a directory name nobody checked. Because
the record looks identical to a genuine one, it actively corroborates a wrong
story for any later reader. A second sweep is worse: it overwrites real records
from a local run — `argv`, `inputs[]`, `git_commit` — with migration records,
destroying the audit trail with no diagnostic.

**What to flag:** a stamper that writes without checking the `migrated_from`
target exists; a per-dwarf sweep that stamps a basename absent from the current
name list; an unconditional overwrite of an existing sidecar with no
producer check or explicit re-stamp flag; a producer string attached to a
product the named script cannot reproduce.

## Guard whose equivalence class is coarser than what it protects

**Where:** consistency checks that canonicalize before comparing — resolving an
alias, a variant, or a version family to one key, then asserting on the key.

**What goes wrong:** the guard is documented as preventing mixing, and does
prevent it *between* families while permitting it freely *within* one. Products
keyed on disk by the finer name pass the check against any sibling, so swapping
one variant's inputs for another's is invisible: lengths match, stamps match,
the output is written with labels describing a selection it was not computed
under.

**What to flag:** a comparison against a resolved/canonical version where the
product is stored under the raw variant name; a guard whose alias table maps
several live variants onto one catalog; any check that passes when two inputs
differ in a field the output's meaning depends on. Test it by placing one
variant's file in another's slot and confirming it raises.

## Parity test that re-implements the pipeline instead of invoking it

**Where:** `needs_data` tests written to certify a migrated stage against its
published artifact.

**What goes wrong:** the test recomputes the quantity with its own transcribed
copy of the loop, then compares to the stored array. It validates the artifact
against the test's physics, not the script's, so it stays green through any
unit, radius, or constant slip introduced in the code it claims to cover — the
kpc/pc and per-halo-factor errors it exists to catch.

**What to flag:** a parity test that does not call the production entry point;
assertions against locally recomputed values where the script's own output was
available; a fixture that leaves the entry point free to read the real catalog
instead of the fixture. Confirm by sabotage: break a constant in the script and
require the test to fail.

## Stochastic stage treated as a deterministic function of its inputs

**Where:** any migrated stage that applies intrinsic scatter, resampling, or a
Monte-Carlo integral — SHMR draws, mass-half scatter, vegas integration — where
the RNG is the module-level global and nothing seeds it.

**What goes wrong:** the stage looks like a pure function of the catalog, so
its output is described, stamped, and verified as reproducible. It is not: the
stored arrays are one realization. A rerun yields statistically equivalent but
numerically different values, so a bit-exact parity check "fails" and invites a
hunt for a porting bug that does not exist — or, worse, passes on the one
deterministic array in the file and is reported as covering all of them. The
published numbers then exist only in the copied bytes, and nothing says so.

**What to flag:** `np.random.*` or `scipy.stats.*.rvs` with no `default_rng`,
no seed argument, and no seed set by any caller; a parity claim covering a file
whose arrays are not all deterministic; a provenance record that implies
regenerability for such a product; a verification described as "bit-exact"
without naming which arrays were compared. Determine per array, not per file:
one file routinely holds both kinds. Never resolve this by adding a seed to
match — a seed cannot recover a realization drawn before it existed, and
changes every downstream number the next time the stage runs.

## Catalog rebuilt underneath its derived products

**Where:** version registries pointing at an h5 that has been regenerated since
the `weights_gc/`, `mhalf/`, or `paper_quantiles/` products keyed to it were
written — especially where a superseded `*_old` or `*_backup` file still sits
beside it in the source tree.

**What goes wrong:** the registry, the code, and the provenance stamp all name
the new catalog while the arrays on disk came from the old one. Halo counts
differ, so a rerun either crashes far from the cause or — if the counts happen
to match — misaligns every halo index in silence. A group the rebuild dropped
turns a live branch into a `KeyError` that reads like a porting mistake, which
sends review at the port instead of at the catalog.

**What to flag:** a derived product whose leading-axis length differs from the
registered catalog's halo count; a registry entry whose file has an `_old` or
`_backup` sibling upstream; a guard that length-checks one input pair while a
sibling pair loaded in the same block goes unchecked; an h5 whose key set is
missing a group a consumer indexes unconditionally. Check lengths against the
catalog, not against each other.

## Regression test restating an invariant the new code enforces structurally

**Where:** tests written alongside a fix that removed the failure mode by
construction — a split that stops a mode from reading an input, a guard that
makes a state unreachable.

**What goes wrong:** the test exercises the new code path, where the input the
invariant is about is no longer present, so it passes trivially and would pass
against almost any implementation. The claim it appears to certify is usually
about the *source* being ported, which only a source-vs-port parity check can
carry. The suite gains a green test and no coverage.

**What to flag:** a regression test that does not fail against the pre-fix
behaviour; a test whose failure would require nondeterminism rather than
incorrectness; an equivalence claim about migrated code with no `needs_data`
parity check against the published artifact.

## Hardcoded constant standing in for a value the data carries

**Where:** selection cuts, radii, mass thresholds, and grid bounds in migrated
code — especially round numbers, and especially where a commented-out line
nearby derives the same quantity.

**What goes wrong:** the literal is close enough that nothing looks broken. It
shifts a boundary by a fraction of a percent, so counts, masks, and downstream
flags differ by a small number of rows that no plot resolves. Because the
rounded value looks deliberate, review reads it as a choice rather than a
regression.

**What to flag:** a suspiciously round constant (259, 100, 1e11) where the
input file has a field of the same name; any literal that replaced a
commented-out derivation; a cut applied to a filtered set where the quantity it
tests is a property of the unfiltered one. Verify by rebuilding and diffing row
counts against the published artifact — not by reading the code, which will
look reasonable either way.

## Provenance stamped before the artifact is final

**Where:** builders that write a product and then modify it — graft, append,
augment in place — or that stamp inside the same function that writes.

**What goes wrong:** the fingerprint records an intermediate state. The
swap-detection guard then reports a swap that never happened, permanently, and
the stale record propagates into every downstream product's `inputs[]`. Nobody
notices because nothing compares the stamp against what is on disk.

**What to flag:** a `stamp_existing` call that is not the last statement
touching the file; any documented post-processing ("grafted", "appended",
"augmented") with no re-stamp afterwards; the absence of a test comparing the
recorded fingerprint to `os.stat`.

## Destructive rebuild of a partially-unreproducible artifact

**Where:** `--force` or `mode='w'` in a migrated builder whose output carries
groups produced by a stage that has not been ported yet.

**What goes wrong:** the rebuild is verified by a diff and looks perfect, then
a later re-run silently drops the un-ported content. The loss is recoverable
only from the very source tree the migration is supposed to retire.

**What to flag:** `File(path, 'w')` where the existing target has top-level
members the writer does not produce; a `--force` that never reads back what it
is about to overwrite; a build documented as depending on a script that does
not exist in this repository yet.

## Stored selection that no consumer reads, shadowed by a recomputation

**Where:** boolean flag columns and derived masks carried inside catalogs,
where the consumer recomputes the same concept inline.

**What goes wrong:** the stored column and the live recomputation drift apart —
different mass variable, different radius — and a migration audits the stored
column as though it were load-bearing, spending effort there while the
definition that actually sets the figure goes unreviewed. It also hides the
reverse: the stored column being wrong is harmless right up until someone uses
it.

**What to flag:** an attribute assigned in `__init__` with no other reference
in the module; a concept computed in both producer and consumer; any severity
claim of the form "this flag drives figure X" unaccompanied by a grep of the
attribute name across `.py`, live `.ipynb` cells, and the drafts.

## Dead clause in a migrated cut

**Where:** selection expressions carried from notebooks, especially with `:=`,
chained comparisons, or `&`/`|` precedence.

**What goes wrong:** a clause that reads as an active cut evaluates to a
constant — a boolean array compared against a number, or precedence swallowing
a term. Code that "removes" it is then right for the wrong reason, and the
migration note records a false causal story that misleads the next reader.

**What to flag:** a walrus inside a comparison; a comparison whose left operand
is a boolean array; any claimed behavioural difference from a source expression
whose actual value was never evaluated.

## cwd-dependent relative reads

**Where:** any module that opens `'../data/...'` at import or call time.

**What goes wrong:** the module works only when invoked from one directory. A
test, a notebook, or a driver script run from elsewhere reads a different path
or fails to import, and the failure is attributed to the caller.

**What to flag:** a relative path outside the module's own directory; a
`sys.path.append` with `..` in it; any path not derived from `config`.

## Repository root resolved from the running script's own location

**Where:** launcher scripts under `scripts/` that derive a project root from
`${BASH_SOURCE[0]}`, `$0`, or `__file__` for later `cd`, `python <root>/...`,
or relative-path construction.

**What goes wrong:** a scheduler stages the submitted script to a node-local
spool path before executing it, so self-referential resolution returns the
staging directory instead of the source tree, and every path built from it is
wrong. The failure is invisible outside the scheduler: run the same script
from a shell and `BASH_SOURCE` is correct, so both a manual smoke test and a
diff-only review pass while every array task dies in seconds.

**What to flag:** a root-detection line derived solely from the invoking
script's own path, with no scheduler-provided submission-directory variable as
an alternative; any such resolution with no validation that the resolved root
contains a sentinel file; a resolution snippet duplicated across launchers with
no test that executes it from *outside* the tree. Verifying a launcher by
running it in place does not exercise this — the check must simulate the spool
copy, or run under the real scheduler.

The fix has a mirror failure worth flagging in the same review. A
scheduler-provided submission directory is exported into *every* shell inside
an allocation, not only into batch jobs — an interactive session spawned by the
scheduler (a notebook server, an `salloc` shell, an on-demand portal job)
inherits it pointing at whatever directory launched that session. So a launcher
that prefers the scheduler variable unconditionally resolves to an unrelated
tree when run by hand from inside such a session, and the script-location
fallback it was given never executes. Preferring one source over the other is
correct only when the resolved root is then validated against a sentinel file;
without that check, both orderings are silently wrong in one of the two
environments. Flag any preference chain between environment- and
self-derived roots that ends without a validation step, and treat "it works
when I run it manually" as evidence about one environment only.

## Test asserting local presence where the contract is fresh-clone presence

**Where:** tests over directories or files that a driver requires to pre-exist,
especially anything under a gitignored path (`scripts/*_out/`, `results/`,
`data/`).

**What goes wrong:** the test asserts `is_dir()` or `exists()`, which passes on
the machine where the artifact was created by hand and keeps passing forever,
while the actual requirement — that a clone reproduces it — is unmet. Paired
with a `.gitignore` rule that excludes the parent directory, a `.gitkeep` added
in good faith is silently never tracked, and prose asserting it *is* tracked
can survive in a script header uncontradicted. Git does not descend into an
excluded directory, so a negation for a file inside one never matches; the rule
must exclude the directory's *contents* instead.

**What to flag:** any existence assertion about a path the repository is
supposed to supply, not one the test itself created — it should assert tracked
state (`git ls-files`), not filesystem presence; a `.gitkeep` whose parent
matches an ignore rule ending in `/`; any header or doc sentence claiming a
path is tracked, without a `git ls-files` check behind it.

## Semantic convention change invisible to the version guard

**Where:** any fix that changes what a stored array *means* without changing
its name, shape, dtype, or version string, in a tree regenerated dwarf-by-dwarf
or version-by-version rather than atomically.

**What goes wrong:** old and new products coexist under one version. Every
existing guard compares versions, not conventions, so a partially regenerated
tree passes validation while an aggregate figure blends two definitions. The
products are byte-plausible under either reading, so nothing downstream can
detect it and no reviewer looking at one file can either.

**What to flag:** a convention recorded as a provenance field with no assertion
that reads it — writing a field does not prevent mixing, only checking it does;
a regeneration that cannot complete atomically across a tree, with no interim
"unusable" marker; any claim that a new field "prevents" recurrence without a
guard that raises on disagreement. Absence of the field must be a third class,
distinct from agreement, or every legacy product silently certifies itself.

## Provenance field dropped by depth-limited input embedding

**Where:** fields added to one stage's stamp that must be read at the figure or
aggregate level, in a scheme that embeds each input's record only one hop deep.

**What goes wrong:** the field is genuinely written and genuinely unreadable at
the point of use, because the intermediate stage's own `inputs[]` is stripped
when embedded. An auditor reads "unknown" for a product that is fine — or worse
assumes the field would have shown up if it mattered, and treats its absence as
evidence of agreement.

**What to flag:** a new stamp field whose consumer is more than one stage
downstream; any field added to a leaf producer without a matching top-level
re-stamp in each intermediate that carries it forward. Verify by writing the
whole record chain and reading the terminal manifest — inspecting the leaf's
own sidecar proves nothing about what survives the hop.

## Linear ratio applied as a dex offset

**Where:** any correction expressed as a ratio ("×1.2", "20% more", "the
remaining fraction") that is applied to a quantity already held in log10 --
stellar masses, luminosities, densities, J-factors.

**What goes wrong:** the ratio is added instead of its logarithm, so a factor
of 1.2 becomes 1.2 dex, a factor of 16. Nothing raises: the units are
dimensionless on both sides, the array shape and dtype are unchanged, and the
result stays finite and plausibly ordered. It survives review because the
constant matches the documented physical value exactly -- the number is right
and only the space is wrong. Downstream it moves a whole distribution relative
to a fixed observational target, which reads as a physical shift rather than a
bug, and can be mistaken for the very effect a figure claims to measure.

**What to flag:** a bare numeric constant added to a variable whose name begins
`log`; a comment describing a multiplicative relation next to an additive
operation; the same physical ratio appearing in two places with `np.log10()`
in one and not the other -- the correct idiom being present elsewhere in the
repo is evidence of a slip, not of intent. Check magnitude against the
quantity's real dynamic range: a correction larger than the spread of the
thing it corrects is almost never right. For a bounded quantity (a remaining
fraction, an efficiency), assert the bound rather than trusting it.

## Unseeded global RNG inside a stage that advertises a seed

**Where:** a stage that constructs a local `np.random.default_rng(SEED)` and
exposes a `SEED` constant, while a library function it calls draws from the
legacy global RNG (`np.random.normal`, `np.random.rand`, `scipy.stats.*.rvs`).

**What goes wrong:** the visible seed covers one stream and not the other, so
the stage looks deterministic and is not. The symptom is asymmetric and easy to
misread: central values move only slightly between runs while derived spreads
(a scatter, an error bar, an effective sample size) move by tens of percent,
because the spread is a difference of two independently noisy quantiles. A
parity check against a stored reference then fails by a fraction of a percent
with no code difference at all, and the natural conclusion -- that the port is
wrong -- is the wrong one. Two seeded generators in one file do not compose:
seeding a local Generator never constrains the global one.

**What to flag:** a `SEED` constant or `default_rng(...)` call in a stage that
also calls into a module drawing from the global RNG -- grep the callee, not
just the caller; a parity gate over a stage whose stochasticity has not been
established by running it twice; any product whose provenance names a seed
without naming which streams that seed covers. Establish the run-to-run spread
before attributing a small difference to a code change.

A seed *expression* can also be the defect, not just its absence. Deriving a
seed from `hash()` of a string looks deterministic and is not: Python
randomises string hashing per process unless `PYTHONHASHSEED` is set, so the
stage draws a fresh realization on every run while appearing seeded to every
reader. The giveaway is that nothing in the launcher sets that variable. The
same applies to any seed built from `id()`, an address, a dict iteration order,
or a wall-clock value. Flag a seed derived from anything that is not a pure
function of the run's declared inputs, and treat a docstring describing such an
expression as "the seeding convention" as a reason to check it rather than to
trust it -- a repository can contain both this mistake and its own written
refutation in two files that were never reconciled.

## Partial array rerun leaves stale survivors that look like a complete tree

**Where:** any stage whose SLURM array writes one file per task into a tree that
already holds a previous run's output -- a regeneration after a convention or
bug fix, a rerun after a wall-clock or memory limit was raised, a resubmission
of a subset of task IDs.

**What goes wrong:** a task that times out, is cancelled, or dies before its
write leaves the PREVIOUS run's file in place. The directory still holds the
full expected set of filenames with plausible values in every one, so a file
count, a row count, a glob, and a concatenation step all pass. The tree is
silently a mixture of two conventions, and the mixed rows are invisible in the
data itself: nothing about a stale row's numbers distinguishes it from a fresh
one, and a downstream concat merges them without complaint. This is the
version-mixing failure mode arriving through the scheduler rather than through
code, so a version guard that compares input provenance cannot see it -- the
inputs were identical for both runs.

**What to flag:** any rerun over an existing output tree that is verified by
counting files rather than by inspecting each product's provenance; a
completion check that reads the scheduler's exit states without cross-checking
them against what is on disk; a concat or aggregation step downstream of an
array with no per-fragment convention check. Require that the stage stamp the
convention it wrote under, and that the sweep after a rerun classify every
product by that stamp.

A presence-only sidecar check is not enough, and is the easy thing to settle
for. A missing sidecar identifies a survivor of a run that predates sidecar
writing — nothing more. Once the stage stamps its output, every subsequent
run's leftovers carry a complete, well-formed sidecar of their own and pass a
presence check unchanged, so the detector silently stops working at exactly the
point the tree starts looking trustworthy. Validate sidecar *content* against
the row it sits beside: the convention field, the version, and the identity the
product claims (which object, which variant) matched against the path it
occupies. Flag any freshness check whose predicate is `-f`, `exists()`, or a
file count. Treat a wall-clock limit inside the observed run-time distribution as a
latent instance of this class: the slowest successful task finishing seconds
under the limit means the next run will produce stale survivors, not failures.

## Guard disabled by a sibling producer landed in the same change

**Where:** a validating consumer that keys on an optional provenance field,
landed alongside a bulk stamper, a migration copier, or any second producer
that can fill the same tree without writing that field.

**What goes wrong:** the guard degrades to a weaker fallback on exactly the
tree it was written to protect, and the degradation announces itself only in a
job log that nobody reads after the run. The product then carries a stamp
implying it passed a check that never executed, and the review that approved
the change saw the guard and the stamper as separate halves and neither in
light of the other. The property being checked may still hold — but by luck,
not by verification, and nothing downstream can tell the two apart.

**What to flag:** a consumer using `record.get(X)` guarded by
`if X is None: <weaker path>`, where `X` is written by only some of the
producers that can populate that tree — enumerate every writer and check each
emits it; any degradation path that does not record its own occurrence in the
output's provenance; an integrity property that is derivable from the data
itself (row counts, index ranges, array shapes, filenames) but implemented
against an optional stamp field instead. Prefer the data-derived check as the
hard gate and keep the stamp field as a cross-check. Also flag a validated
quantity that is then not *used*: a check that confirms fragments tile a range
while placement is computed independently leaves transposed or misordered
inputs passing every assertion.

## Regeneration impact scoped to figures while the papers quote the numbers in prose

**Where:** migration notes, commit messages, or change reports for a corrected
pipeline stage, especially one whose consuming plot scripts are not yet built.

**What goes wrong:** the fix legitimately shifts a distribution, the affected
figures are correctly enumerated, and — because those figures do not exist yet
— the change is recorded as having no downstream consequence. Meanwhile the
papers quote specific values of that same quantity in running text. The prose
claim silently becomes false, and since no figure regenerates, nothing ever
surfaces the contradiction. The reproducibility contract is broken in the one
place that has no rebuild step to catch it.

**What to flag:** an impact statement whose scope is a list of figure
basenames; a "no consumers exist yet, nothing to regenerate" justification; any
regenerated per-dwarf or per-object table whose quantity is named with a value
somewhere in the drafts. Require a grep of the drafts for the *quantity and its
units*, not for the figure filename, and a before/after comparison restricted
to the specific objects the text names. Check the claim against the sample the
text is about — a statement about the analysis sample is not tested by
evaluating it over the full catalog, and a claim can be already-false upstream,
which changes whether the fix broke it or merely worsened it.

## Aggregation loop that skips an empty input group and still exits zero

**Where:** concat and merge drivers that iterate over discovered subdirectories
under `nullglob`, with a per-group completeness check inside the loop.

**What goes wrong:** a group whose tasks all died leaves an empty directory
behind — entry points routinely `mkdir` their output directory before doing any
work, so absence of data does not mean absence of a directory. The loop logs a
skip, continues, and the driver exits 0. The per-group row-count check never
runs for that group, because it lives inside the branch the skip bypassed: the
completeness check only ever validates the groups that happen to have data. Any
previous run's aggregate for the skipped group stays on disk and is read as
current by every downstream consumer.

**What to flag:** a `continue` on an empty glob inside an aggregation loop; a
success criterion of `found > 0` rather than `found == expected`, where the
expected count is resolved from the same data-derived source the per-item check
uses; an aggregate output left in place when its inputs are absent — deleting
or refusing is safer than leaving a stale file that looks fresh; a per-group
completeness check with no check that the group *set* is complete.

## Placement invariant numerically identical to the offset it replaces

**Where:** concat and merge steps hardened by replacing a running write offset
with bounds derived from each input's index.

**What goes wrong:** for correctly-sized inputs the two schemes compute the same
destinations, so the reordering-detection the change is described as adding does
not exist. The genuine benefit is narrower — a per-group size check — but the
comment, the docstring, and the commit message all claim the broader property.
The claim survives review because the accompanying test perturbs the input's
*metadata* rather than its *data*, so it exercises a cross-check that most real
inputs do not carry, and passes against the unhardened implementation too.

**What to flag:** a hardening claim about ordering or identity with no
data-derived identity carried inside each input — an index column, a first/last
key, a checksum — as opposed to a value recomputed from the input's position; a
fixture that never swaps two same-size inputs; any placement property asserted
in prose but demonstrated only against a mutated stamp. Ask what the check
would compare the input against if the metadata were absent, since for migrated
or bulk-stamped trees it usually is. And beware the repair that backfills the
missing metadata *from the index*: a field derived from position cannot
cross-check a bound derived from the same position, so backfilling flips the
verified flag to true while verifying nothing.

## Range assert one-sided against a hardcoded fan-out

**Where:** array-task entry points asserting `task_id < len(items) * k`, where
the launcher's `--array` upper bound is a literal.

**What goes wrong:** the assert fires loudly when the item list shrinks, and
never fires when it grows — but growth is the silent case it was written to
prevent. With more items than the array covers, every dispatched task id is
still in range and the tail items are simply never computed; the output tree
looks complete because nothing ever names the missing entries. The adjacent
comment typically claims the direction the assert does not cover.

**What to flag:** an in-range assert with no matching *equality* assert against
the task count the launcher declares; a launcher whose `--array` bound is a
literal while the entry point derives its fan-out from data; a comment claiming
an assert prevents "silently never reaching the last item" when only the
opposite bound is checked.

## Freshness check keyed on slot-invariant fields

**Where:** completeness or staleness sweeps over per-task output trees that
validate a sidecar's *content* against the product it sits beside — the
version, the object identity, a convention constant, the directory name.

**What goes wrong:** every one of those fields is a pure function of the slot,
so two runs of the same code under the same convention write byte-identical
values. A survivor of any rerun on the near side of the transition that
motivated the check passes it cleanly. The check therefore detects exactly one
historical event — the migration or fix that was on everyone's mind when it was
written — and stops working the moment the whole tree is regenerated. Because
it is real, present, and passing, it reads as continuous protection, and the
incident that prompted it is precisely the one that will no longer be
detectable next time.

**What to flag:** a freshness predicate built only from fields that cannot vary
between two runs of the same code; a stale-survivor guard justified by an
incident whose distinguishing signal was a *missing* or *outdated* field, once
no product on disk still lacks it; any per-run identity requirement with no
run-scoped token — a scheduler job id, a batch UUID, a submission timestamp —
recorded per product. Ask what the check would compare if every product had
been written by the same commit under the same convention; if the answer is
"nothing", it is a placement check, not a freshness check. Note also that
requiring a *single* run token across a tree is usually wrong: resubmitting
stragglers is legitimate and produces a genuinely multi-token tree. Record the
set and surface it; do not reject it.

## Cross-file constant asserted against its own copy

**Where:** entry points guarding a data-derived quantity against a scheduler- or
driver-supplied fan-out, where the guard's right-hand side is a literal
transcribed from the other file rather than read out of it.

**What goes wrong:** the assert is written as the fix for a silent-truncation
bug and cannot detect that bug. It compares the data against a duplicate, so it
fires only when the data and *both* copies disagree. The case that matters —
the authoritative file, the one the scheduler actually parses, going stale while
the transcribed copy is updated — passes every check, and the tail items are
simply never dispatched. The comment next to the copy usually asserts the
opposite, and the failure surfaces stages later as a count mismatch far from its
cause.

**What to flag:** a constant whose comment says "must be kept in step with
<other file>"; an equality assert whose operand is a literal that also appears
in a `#SBATCH` directive, a config file, or a launcher; any "fails loudly
instead of silently dropping" claim where the enforcing code never opens the
file it names. Require the value be parsed from the authoritative file at run
time, and require a test that mutates that file and observes the failure.

## Aggregate written by a shell driver escapes the provenance contract

**Where:** concat or merge steps implemented as a bash loop redirecting into a
text product, alongside sibling stages implemented in Python through the
provenance module.

**What goes wrong:** the per-task inputs are fully stamped and validated, and
the artifact every consumer actually opens carries nothing — no version, no
input fingerprints, no producing script. The tree looks thoroughly documented
because its leaves are, so the gap is invisible in exactly the place a reader
checks; every downstream manifest embeds a null and the audit chain terminates
silently at the last Python stage. Path constants in the shell driver drift
from the config module for the same underlying reason: there is no import to
keep them honest, so a `results/` path rebuilt by string concatenation merely
happens to agree with the constant it duplicates. The same driver also tends to
interleave validation with writing, because a two-pass structure is awkward in
bash — leaving a failed run with some outputs fresh and some stale.

**What to flag:** a product under `results/` with no sidecar beside it; a driver
resolving a `results/` path by string concatenation where a config constant
exists; any shell aggregation whose Python sibling stamps its output; a
validation performed inside the same loop that writes. Prefer moving the
aggregation into `python/` over bolting a stamping subprocess onto the shell.

## Optional grammar accepted as non-semantic in a parsed cross-file constant

**Where:** entry points that parse a scheduler directive, config key, or CLI
bound out of another file in order to assert a data-derived count against it.

**What goes wrong:** the parser tolerates an optional syntax element that
actually changes meaning — a step, a stride, a mask, a units suffix — and
discards it before computing the value it asserts on. The guard then passes on
every dispatched unit while the dispatched *set* differs from the one the count
implies, so the truncation the guard exists to prevent happens underneath a
green check. The comment beside the regex typically names the element and calls
it routine or cosmetic, which is true for one such element and false for its
neighbour.

**What to flag:** a non-capturing optional group in a directive parser whose
alternative is never validated; a comment describing a syntax element as
"non-semantic" without stating the values for which that holds; any parsed
bound converted straight into a count without checking the elements that would
make the count wrong. Enumerate the target grammar and, for each optional
element, ask whether it changes the *set* or only the *rate* — a throttle
changes the rate, a step changes the set.

## Coarse guard pre-empting the specific guard it is credited alongside

**Where:** ordered validation chains where a count or size check precedes an
identity, contiguity, or content check over the same collection.

**What goes wrong:** the coarse check fires first, and its message names only
one failure direction — "incomplete", "truncated", "missing" — so a surplus or
a substitution is reported as a shortfall. The specific check is unreachable
for most of the cases its docstring claims to cover, and its accompanying test
can only construct the residual case where both conditions coincide. The test
is therefore green, the prose is still wrong, and an operator debugging a real
tree is sent looking for the opposite problem.

**What to flag:** a size-mismatch refusal whose message asserts a direction the
comparison cannot distinguish; a claim that a later check "also catches X"
where X perturbs the quantity an earlier check tests; a validation order in
which the strictly-subsuming check runs second. Verify by constructing the
exact tree the docstring describes and reading the message that actually comes
out — not by reading the checks in order and believing their comments.

## Two-phase write whose commit phase has no failure path

**Where:** validate-then-commit drivers where phase 1 has a rich refusal
reporter and phase 2 stages into temporaries before replacing.

**What goes wrong:** the staging loop is wrapped in cleanup and the commit loop
is not. An exception during commit leaves a partially updated tree, orphaned
staging files, and nothing but a traceback — while the well-tested "these were
NOT refreshed" reporter is reachable only from the phase that modifies nothing.
Because each committed product carries a fresh self-consistent stamp and each
skipped one carries a valid previous-run stamp, no per-file inspection
distinguishes a half-committed tree from a complete one.

**What to flag:** a commit loop outside the `try`/`except` protecting the
staging loop; `except Exception` where preemption and interrupt are the stated
threat model, since that catches neither `KeyboardInterrupt` nor `SystemExit`
and a Ctrl-C then leaks every temporary; a refusal or stale-state reporter
invoked from one phase only; per-product stamps in a multi-product commit with
no shared run token linking them. Require that a half-committed set be
detectable from the products alone, and that the commit phase contain nothing
but rename-class operations — hoist record construction, fingerprinting, and
subprocess calls into staging.

## Sidecar written after its data, in a reader that prefers the sidecar

**Where:** any writer pairing a data file with a preferred-layer provenance
record — and especially the shared library writer that sibling stages call,
when one or two stages have been hardened by hand.

**What goes wrong:** a kill, an OOM, or a full disk between the data write and
the record write leaves fresh or truncated data beside the *previous* run's
fully valid record. The reader prefers that layer, so the new product
self-certifies under the old stamp, and any correct in-band record written
alongside the data is shadowed. Both artifacts are well-formed, so nothing
looks wrong at any later inspection. It is doubly easy to miss when the
ordering has just been fixed at a call site: the reviewed diff displays the
correct idiom while the unreviewed shared writer keeps the wrong one.

**What to flag:** a stamp or sidecar write that follows the data write with no
prior unlink of the stale record; a fix for this ordering applied at a call
site rather than in the writer every sibling stage shares — enumerate the
writer's callers, not the diff's files; a large or slow product write with no
temp-plus-rename. Prefer unlink, then write, then stamp, so an interruption
yields unstamped data that a consumer's missing-record check rejects outright.

## Atomic-write hardening that silently changes file permissions

**Where:** any conversion of a direct `open(path, 'w')` into the
write-to-temp-then-`os.replace` idiom, in writers whose products live on a
shared group tree.

**What goes wrong:** `tempfile.mkstemp` creates its file 0600 by design, and
`os.replace` preserves the *source* file's mode rather than the destination's.
So a change made purely for crash-safety also revokes group and world access to
every product written through it. Nothing fails, no test notices, and the
author's own reads keep working — the breakage appears only for a collaborator,
on a different day, as a permission error on a file that visibly exists. Older
products written before the change keep the umask-derived mode, so a directory
listing shows a mix that looks arbitrary.

**What to flag:** `mkstemp` followed by `os.replace` with no explicit `chmod`
between them; any atomicity or durability refactor whose diff touches no
permission logic, in a repository whose outputs are read by more than one user;
a hardcoded `0o644` where the umask should be respected instead. Check the mode
of a product the new code actually wrote against one the old code wrote — the
two sit side by side in the same directory, so the comparison is a single
`stat`. The same applies to directories created by a hardened writer.

## Atomic-write temp file whose name matches the product glob beside it

**Where:** any writer hardened with `tempfile.mkstemp` + `os.replace` inside an
output tree that a concat, inventory sweep, or catalog-building step later
discovers by pattern — especially per-task shard directories.

**What goes wrong:** two correct requirements collide. The temp must live in the
destination directory so the replace stays intra-filesystem, and it usually
carries the product's own stem and extension so the writer's extension handling
behaves. Together those make it match the downstream glob. Meanwhile the
cleanup handler cannot run for `SIGKILL` or a default `SIGTERM` — which is
precisely how a scheduler cancels a job, how a wall-clock limit expires, and how
the OOM-killer acts — so the orphan is exactly what the threat model guarantees
will be left behind, at full product size, once per killed run, accumulating
forever because nothing sweeps them.

Downstream it goes one of two ways, and the quiet one is worse. Either an index
parser built on fixed-width slicing crashes with a message about the wrong
thing, blocking a stage until someone hand-deletes a file; or a classifier — a
shard detector, a "skip non-products" branch — absorbs it silently and inflates
a count in the very sweep whose job was to inventory the tree.

**What to flag:** a `mkstemp` whose `prefix`/`suffix` reconstruct the product's
naming convention in the product's own directory; any atomicity or durability
hardening landed without a corresponding change to the discovery glob or a
temp-sweep step; an index or identity parsed out of a discovered filename by
fixed-width slicing rather than a strict pattern match; a cleanup `except`
presented as covering preemption when the stated kill path delivers an
uncatchable signal. Check the temp name against every glob that reads that
directory, and require the inventory sweep to *report* orphans rather than
classify them away. A leading dot is usually enough, since `glob` skips
dotfiles by default.

## Verification run overwrites the reference it is verifying against

**Where:** single-task parity checks during a migration, where the ported stage
is executed with its real output paths pointing at the tree of migrated
products being used as ground truth.

**What goes wrong:** the stage writes where it normally writes, so the baseline
is replaced by the very output under test. Any later re-check then compares the
new code against itself and passes trivially. The damage is quiet: the tree
still holds a full set of well-formed products with plausible values, and a
file count, a size sweep, and a provenance sweep all pass. It is easy to
under-estimate the blast radius, because one task usually writes *several*
products — one per weighting variant or per output tree — while the person
running it backs up only the one or two they intended to inspect.

**What to flag:** a parity run invoked with production output paths rather than
a scratch directory; a backup taken of "the file I plan to diff" rather than of
everything the task can write — enumerate the stage's writes first, do not
infer them from the diff you have in mind; a verification claim made after a
run that could have replaced its own baseline. Prefer redirecting the stage's
output root for the run, or copying the whole affected subtree first. Note the
baseline may be recoverable from the source tree the stage was ported from,
which makes this recoverable rather than terminal — but only while that tree
still exists, and a migration exists precisely to stop depending on it.

## Cached intermediate silently rebuilt underneath a downstream product

**Where:** any multi-stage tree where a stage's output is cached in `results/`
and a later stage reads it, and where stages are re-run independently rather
than as one pipeline.

**What goes wrong:** a stored product stops reproducing even though its own code
never changed, because an upstream cached file was overwritten in place between
the two runs. Every instinct points at the wrong place: the code is diffed, the
libraries are checked, the environment is questioned, and all of them come back
clean — because the defect is that the *input* is no longer the input. Nothing
records the substitution, since the upstream file was replaced at the same path
with no version stamp and no archived predecessor, and the older copy is
usually unrecoverable. The effect can be per-object rather than global, so a
spot-check on one object may reproduce fine while another is badly off, which
reads as noise rather than as a systematic.

**What to flag:** an input file whose mtime postdates the output it supposedly
fed — this single comparison is the cheapest detector and should be routine
before diagnosing any parity failure; a `results/` product overwritten in place
by a re-run rather than written to a new versioned path; a stage whose stored
output carries no record of the input snapshot it consumed, only the input's
path. Require derived products to record an input fingerprint, not just a
filename, so a later reader can tell whether the file at that path is still the
one that was read. When a parity check fails, compare mtimes across the whole
input chain before assuming code or library drift, and check more than one
object before concluding the difference is uniform.

## Selection literal taken from an adjacent comment that disagrees with the code

**Where:** migrating index sets, cuts, sample lists, or thresholds out of
notebooks and scratch scripts into drivers.

**What goes wrong:** a descriptive comment sitting directly above a live literal
has drifted from it, and the port transcribes the human-readable line. What a
published curve aggregates changes silently. This is distinct from reading
commented-*out* code as the specification: here the code is live and correct,
and the *prose* beside it is stale — so the usual instinct to distrust
commented-out blocks offers no protection.

**What to flag:** any migrated list, set, range, or threshold whose value is
also spelled out in an adjacent comment, docstring, or caption. Re-derive the
literal from the executed line, and where a published artifact exists, confirm
against it rather than against the prose. Treat a comment that enumerates the
same values as the code beneath it as a reason to diff the two, not as
corroboration.

## Parity gate anchored on an intermediate port rather than the published artifact

**Where:** multi-hop migrations — notebook to an intermediate driver to the
repository module — verified by array equality against the immediately
preceding hop.

**What goes wrong:** a defect introduced at the first hop is reproduced
bit-exactly at the second and reported as passing parity. Every compared array
agrees, the verification reads as unusually strong precisely because the
agreement is exact, and the figure still disagrees with the published one. The
intermediate looks like an authoritative source because it is code rather than
a notebook, but it is itself a migration and carries whatever its author
mistranscribed.

**What to flag:** any parity claim naming an intermediate `.py` as the
reference; a verification that reports agreement in aggregate across a family
of figures rather than per figure, since one outlier among ten exact matches is
the signal and a mean hides it. Require at least one check against the terminal
source of truth — the cell carrying the live `savefig`, the rasterized
published figure, or the caption's enumeration of curves — for any quantity
that selects, filters, or aggregates.

## Published figure reproduced with a neighbouring figure's series list

**Where:** modules rendering several figures through a shared series/render
helper, especially where consecutive figures share a weight-column or prior
vocabulary.

**What goes wrong:** the shared helper makes it cheap to carry the previous
figure's series list forward, so one figure inherits its neighbour's columns or
priors. A curve the paper describes disappears and one it never mentions
appears, while axes, limits, colour cycle, annotation and legend layout all
still match. The resulting pixel difference lands in the same range as
realization noise from unseeded weights and gets attributed to that. A
docstring claiming the series definitions are character-identical to the source
then reads as corroboration, because it was written from the same paste.

**What to flag:** two figures in one module whose series lists differ by
exactly one entry; a series list whose labels were not transcribed one-for-one
against the caption's enumeration of curves; a per-figure pixel difference an
order of magnitude above its siblings, which is a defect signature rather than
noise. The caption's curve list, not the source cell, is the acceptance
criterion.

## Impact claim asserting a deviation the artifact does not show

**Where:** scope-decision notes attached to a migrated cut, floor, or limit that
the source left commented out.

**What goes wrong:** the note claims the choice "intentionally changes the
published curves", inferred from comment state and never measured. In fact the
change is what *reproduces* the published artifact, and the comment state was
post-run residue. The note then reads as an unresolved discrepancy with the
paper, and the natural remedy — reverting it — silently corrupts the figure.
This is worse than a wrong content claim, because it is framed as awaiting
someone's approval and so survives review as a deferred decision rather than
being caught as a false statement.

**What to flag:** an intentional-deviation note whose effect is described
qualitatively rather than as a measured number over a stated range; a scope
decision justified against comment state with no diff against the rendered
artifact; a regenerated figure that matches the published one pixel-for-pixel
while its own docstring claims it should not. Treat "identical to published" as
refuting a deviation claim outright.

## Sibling-repo output path derived from the caller's flag, not the callee's routing key

**Where:** drivers that shell out to an external repository's script and then
harvest a file from its output tree.

**What goes wrong:** the callee routes its output on its own authority — a
manifest, an audit record, resolved metadata — while the caller rebuilds the
path by joining the arguments it passed. When the two disagree the caller reads
a pre-existing file from the wrong leaf, passes its existence check, and stamps
provenance describing the run it *requested* rather than the bytes it
*copied*. The hazard is acute where the external output tree is pre-populated
from earlier runs, which is normal for a results directory.

**What to flag:** any harvest path built by string-joining the caller's own
arguments; an existence check treated as confirmation that the file came from
this invocation. Derive the path from the callee's routing key, or assert the
two agree before invoking. Check whether the target directory already contains
plausible files from previous runs — if it does, existence proves nothing.

## Redaction that scrubs the prose while the machine-readable layer still carries the literal

**Where:** any pass preparing a repository, dataset or artifact set for an
outside audience, especially where the sensitive string appears both in
narrative documentation and in bulk generated metadata — provenance sidecars,
aggregate manifests, cached logs, path constants, test fixtures.

**What goes wrong:** the hand-authored surface is edited and reads clean, so
the pass is declared done, while thousands of generated files, one environment
default, or a single module constant still carry the literal. This is worse
than no redaction: the euphemism in the prose signals that the information was
considered and protected, so nobody greps for it, and the inconsistency between
the two layers advertises that something was hidden. The same shape appears
whenever a claim is scoped to what a human wrote and silently not to what a
generator emitted.

**What to flag:** a diff touching only `.md` files and docstrings; a rewrite
substituting a generic phrase for a specific name. Require the grep to run over
the whole tracked tree rather than the diff, and require counts — one surviving
hit and a thousand are the same failure. Require the pass to declare, per
literal, whether it is being removed or deliberately retained, with the
retention decision recorded in a tracked document. Reject a change that removes
a literal from prose while leaving it in tracked machine-readable output with
no such declaration.
