# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working with code in this repository.

## Behavioral Guidelines

### Think before coding
- State assumptions explicitly; ask if uncertain.
- If multiple interpretations exist, present them — don't pick silently.
- Push back when a simpler approach exists.
- Surface semantic fallbacks and scope decisions before shipping. Mapping prior A → prior B in a code path with no analog, leaving a hardcoded list untouched on grounds it "looks scoped to something else", deciding which files an enum sweep covers — these are choices, not implementation details. Name them in the response or the plan; don't make them silently.
### Simplicity first
- No features, abstractions, or error handling beyond what was asked.
- If 200 lines could be 50, rewrite it.
### Surgical changes
- Match existing style. Don't refactor or "improve" adjacent code.
- Remove imports/variables your changes made unused; leave pre-existing dead code alone (mention it, don't delete it).
- Every changed line should trace to the user's request.
- Extend existing scripts; don't clone them. If a pipeline script already produces the needed output, run it — a broader run beats a parallel copy that duplicates logic and drifts. To narrow what it does (e.g. recompute only one output type), add a flag to that script; don't copy its logic into a new file.
### Goal-driven execution
- Transform tasks into verifiable goals with success criteria.
- For multi-step work, state a brief plan with verification checks.
- Loop until verified — don't declare success without checking.
- When a change touches code called >10⁵ times per run (likelihoods, `prior_transform`, integrands, inner integrators), the plan must include a runtime/complexity bullet — table-lookups vs. scipy.stats overhead, vectorisation, allocation in the inner loop. Don't add it only when prompted.
- After fixing a bug, the fix lands with a regression test that fails on the pre-fix code and passes after. "Manually reran and the symptom is gone" is not verification — the test suite did not catch the bug, so a new test is needed to keep it caught.

### Shared compute
Never submit jobs to SLURM or any cluster scheduler. Prepare the script, surface its parameters, and stop. The user runs it.

## Working style and routing

Default to the main session for small edits, tight back-and-forth, and final judgment. Delegate to subagents when output would be verbose, work is parallelizable, or a task is self-contained.

Subagents:

- **worker** (Sonnet): implements specified changes after the approach is decided.
- **reviewer** (Sonnet): default review pass after non-trivial worker output.
- **deep-reviewer** (Opus): numerical methods, the inference pipeline, calibration claims, `.tex`-vs-code drift, or anything flagged critical (see "Adversarial review" for the routing rule).
- **analyst** (Opus): fresh-context reasoning when the main thread is noisy or the question is orthogonal. Read-only.
- **test-runner** (Haiku): runs tests, returns only failures.
- **researcher** (Haiku): external docs, library APIs, web lookup.
- **Explore** (built-in, Haiku): read-only codebase lookup. Used automatically.

Rules:

1. After non-trivial worker output, invoke reviewer (or deep-reviewer for high-stakes code) before declaring done.
2. After worker completes, invoke test-runner unless the change is purely cosmetic.
3. Fan out parallel subagents for independent investigations across multiple files.
4. Never run tests inline — delegate to test-runner so logs stay out of context.
5. Resolve ambiguity, wide design space, or subagent questions in the main session before (re-)dispatching. Worker does not make architectural decisions.

## Adversarial review

After any non-trivial code change or numerical result the user is likely to act on, run an **adversarial code review**. Assume there is a bug; hunt for sign errors, off-by-ones, unit/coordinate slips, boundary handling, silent fallbacks, and biased defaults. Report findings before declaring final.

Routing: **reviewer** for routine passes; **deep-reviewer** when the change touches the analysis pipeline, numerical methods, calibration, or anything where a subtle bug propagates into results. Concretely: deep-reviewer for anything that changes a number, curve, or contour appearing in a paper figure, anything on the reweighting / quantile / J-factor / Fermi-limit path, and any edit that reconciles a figure against the `.tex` claims. Reviewer for plotting cosmetics, data-ingest adapters, and provenance bookkeeping. **analyst** is read-only fresh-context reasoning — use it for orthogonal investigations or to classify failure modes, not as a substitute for reviewer. Do all of the above *unprompted* whenever the cost of being wrong exceeds a few minutes of agent time. Skip only for trivial edits (typo, rename, doc).

Reviewers consult `docs/review-checklist.md` for recurring bug classes in this repo. **Only the main session writes to this file**. When a reviewer (or any subagent) catches a bug whose class isn't already listed, it must surface the proposed class in its findings — a one-line class name, where the bug typically appears, what goes wrong when it slips through, and what reviewers should flag going forward — and leave the file edit to the main session. Serializing writes through the main session keeps framing consistent and entries general enough to catch near-misses, not just the exact bug that triggered them.

Entries describe the class, not the instance: name the failure mode, its consequences, the symptoms, and the review-time signals. Do not put the triggering file:line in the "what to flag" body.

When you delegate a verification or review task, never assert the expected current state in the brief ("X is staged but not yet run," "Y still exists"): state drifts between when you frame the task and when the agent runs it — especially for user-gated actions and long background jobs — so a stale premise reads as authoritative and produces a false PASS. Hand the agent the artifacts and have it derive current state itself (`ls`/`git log`/dry-run), and ask it to flag any contradiction with your assumptions rather than confirm them.

### When the gate fires

The gate fires **before each commit** that touches non-trivial code — three commits = three reviewer dispatches. The diff under review is the diff being committed.

### What does NOT substitute for a reviewer pass

- **Parity gates / regression numbers / unit tests passing.** These verify what they were aimed at; a reviewer finds what you didn't think to check. Parity can't catch bugs shared by both paths or in code outside the comparison surface.
- **"Framework-only, no calibration."** Import order, registry dispatch, docstring claims, and silent fallbacks are reviewable even when no numerics change.
- **"The figure looks the same as the published one."** Visual agreement hides sign and unit errors that cancel at plot resolution. Compare the underlying arrays.
- **The user already pointed out a bug.** That raises the prior of more bugs; the fix itself needs scrutiny for edge cases and adjacent interactions.
- **"Small extension of already-reviewed code."** Non-trivial new code gets its own pass.

### Gold-standard rule

**Mock data is the gold standard for testing analysis pipelines.** Unit tests check the math; mocks check the *whole pipeline* end-to-end. For any calibration claim ("recovers X to within Y"), generate synthetic input at known truth, run the pipeline, and report bias and dispersion across multiple realizations. Single-step checks and single runs on real data hide systematics.

Delegate mock runs: **worker** to set up and execute, **test-runner** for the runs, **deep-reviewer** to audit the bias/dispersion claim before it is reported to the user or lands in a figure.

## Version control

- Commit logically separable changes as distinct commits; don't bundle unrelated work.
- Never force-push to shared branches. Never commit secrets or large data files — they live locally under `data/` and `results/`, gitignored. See "Datasets are self-contained".
- Commit messages explain *why*, not just *what*.
- Never `git push`. Stage and commit locally, then surface that the branch is ready to push — the user runs `git push`.
- **Before each commit on non-cosmetic code** (this is a gate, not a suggestion):
  1. reviewer / deep-reviewer dispatched on this diff (route by the rule under "Adversarial review");
  2. test-runner green on the affected tests at the tier the change touches (`--rundata` for anything on the pipeline), plus the intermediate → figure step rerun and its output arrays diffed against the previous version;
  3. any figure the diff changes has been regenerated, not left stale.

## Project purpose

Plots and reproduction code for a two-part paper series (Raman, Folsom, Kaplinghat, Lisanti, Safdi) on Milky Way dwarf spheroidal DM halos. **Part I** conditions a SatGen subhalo population on stellar mass (SHMR) or stellar kinematics to infer ρ₁₅₀, peak masses, r_max/v_max, and the V_circ–r_{1/2} relation. **Part II** turns those posteriors into J-factors for 39 dwarfs, compares Jeans-analysis J-factors under a prior ladder, and recasts Fermi-LAT ⟨σv⟩ limits.

**`drafts_temp/` is read-only and is not part of this repository** — a local snapshot of the paper sources, which are versioned elsewhere. It is the *specification*: it defines what each figure must show and which numbers are claimed. Never edit it, including to match code or fix typos. When code and prose disagree, surface it and stop — fixing the code or refreshing the snapshot is the user's call. Cite by `\label` or section title, never line number.

## The reproducibility contract

**Everything in the papers reproduces from this repo alone, with one exception: generation of the SatGen trees and satellite catalogs**, which live under `config.TREES_DIR` (`TREES_*`, `SAT_*`, one per cosmology/resolution variant; overridable via `SDI_TREES_DIR`) and are fixed upstream input. Everything downstream — J-factors, reweighting, quantiles, contours, Jeans posteriors, Fermi recasting, figures — belongs here.

- **Full reproduction is infeasible as a test** (SLURM array over ~10⁵–10⁶ halos per dwarf). Verify at the level one task or a cached intermediate supports, and say which level you reached.
- **Cache intermediates, not just figures.** Every figure regenerates from an `.npz`/`.h5` via a script that lives here, and each intermediate records its SatGen variant, catalog snapshot, and producing script — silent version mixing is invisible in the figures and is the worst failure mode here.
- Migrating a figure from `SatGen_Dwarf` means porting the *code path*, not the PDF.
- The Jeans-side figures depend on `../DwarfJeansAnalysis`, which is itself a tracked, reproducible repo — so the chain stays intact without duplicating its pipeline here. Record which of its `results/` a figure consumed.

## External sources (read freely; edit only when asked)

- `../SatGen_Dwarf` — the pre-migration source tree this code was ported from. Not public, not tracked, and not part of the reproducibility chain: it is a source to **migrate from**, never to read at runtime.
- `../DwarfJeansAnalysis` — tracked repo, and a **live dependency**: the Jeans-analysis figures in Part II (`jeans_corner.pdf`, `multipanel_jeans_priors.pdf`, the prior ladder, the Jeans J-factor comparisons) consume its `results/` directly. Read those outputs; don't vendor its `src/`. See its `ARCHITECTURE.md`.
- `config.TREES_DIR` — SatGen trees and catalogs.

The two paths a script may legitimately point outside the repo are `config.TREES_DIR` and `config.DJA_RESULTS_DIR`; each comes from a single configurable constant, never hardcoded at the call site, and a figure built from either records which snapshot it used.

## Repository architecture

This section is the **where things go and why** — the equivalent of an `ARCHITECTURE.md`, kept inline while the repo is small. Naming mirrors `SatGen_Dwarf` so ported code moves with minimal edits; most directories are scaffolding to fill as code is ported.

```
python/       computation source code — importable modules + analysis entry points
scripts/      run harness — SLURM launchers and plot-production drivers
data/         all input datasets, self-contained (see below)
results/      derived intermediates (.npz/.h5), one subdirectory per product
plots/        final paper figures, one subdirectory per figure family
jupyter/      exploration only
tests/        pytest suite — three opt-in tiers (see below)
docs/         review-checklist.md, provenance.md, and any repo-internal notes
drafts_temp/  read-only .tex snapshot — the specification
```

The three that carry real rules:

- **`python/` is computation, not execution.** Modules import cleanly and are callable without a scheduler; each analysis entry point takes its parameters as CLI args. No SLURM directives and no hardcoded task counts. Figure-making code lives here too (`python/plot_*.py`, following `SatGen_Dwarf`) — "not execution" bars the scheduler and the job parameters, not plotting.
- **`scripts/` is how things run.** SLURM array launchers *and* the drivers that turn `results/` into `plots/`. Convention: `scripts/<name>.sh` launches `python/<name>.py` with stdout in `scripts/<name>_out/`. Every paper figure has a driver here — a figure with no runnable producer does not satisfy the reproducibility contract. Prepare these; never submit them.
- **`plots/` mirrors the drafts.** Basenames match what the `.tex` expects.

Update this section in the same change that adds a top-level directory, moves a responsibility between `python/` and `scripts/`, or alters a repo-wide convention (units, output paths, dataset layout). Routine edits inside an existing module do not require an update — if you find yourself adding function-level detail here, it belongs in a docstring. Promote this section to a standalone `ARCHITECTURE.md` once the layout outgrows a screen.

### Datasets are self-contained

**Every dataset the pipeline reads lives under `data/` in this repository**, except the two declared upstream dependencies: `config.TREES_DIR` and `config.DJA_RESULTS_DIR`. No runtime reads from `../SatGen_Dwarf/data/`, from `$HOME`, or from any other absolute path outside the repo. Migrating an analysis means migrating the data it consumes, not pointing at where it currently sits.

The split is **local, not committed**: data lives in the working tree so the repo is self-contained on this machine; git tracks the code that reads and produces it. Large files under `data/` and `results/` are gitignored — self-containment on GitHub means the recipe is in-repo, not the bytes. Never resolve a size problem by moving a file outside the repo.

`data/` holds observational and literature inputs — dwarf catalogs, stellar kinematics, published limit curves, SHMR tables — one subdirectory per source. Small text inputs are committed; they are the provenance-critical layer, where a silent catalog swap changes every downstream number.

`results/` holds derived intermediates, gitignored. Each product records the SatGen variant, input catalog snapshot, and producing script, and rebuilds from `data/` plus the trees via a script in `scripts/`.

## Things that will bite you

- **SatGen is a pip dependency of the `J_calc` conda env**, not vendored. `Green` / `Dekel` / `NFW` imports fail outside it.
- **`<version>` is an argument, not a tag** — it selects the SatGen realization/cosmology (`Diemer`, `Zhao`, `Correa`, …). Mixing versions corrupts results silently.
- **Dwarf names contain spaces** (`Ursa Major II`, `Segue 1`) and appear verbatim in filenames the `.tex` references. Quote paths; don't sanitize names. The `SatGen_take2/` prefix in `\includegraphics` is the paper build tree, not a directory here — match the basenames.
- **Part I and Part II share upstream intermediates.** A reweighting or catalog change moves figures in both; check both drafts before calling a change scoped.
- **`tests/` has three tiers, and the default run covers none of the pipeline.** `pytest` alone runs unit tests only — seconds, no data, no network. `pytest --rundata` adds stage-parity checks against the migrated `results/` tree; `pytest --runslow` adds the mock-pipeline calibration run. A green default run says nothing about the pipeline; say which tier you reached.
- **"Verified" still means arrays, not PDFs.** For any figure change, diff the produced arrays against the previous version — visual agreement hides sign and unit errors that cancel at plot resolution.
