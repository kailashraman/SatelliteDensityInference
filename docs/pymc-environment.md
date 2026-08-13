# PyMC sampling environment

`python/plot_powerlaw.py` is the only figure-producing script in this
repository that runs a PyMC NUTS sampler (the V_circ-r_1/2 power-law fits
behind `powerlaw_posteriors.pdf` and `powerlaw_fit.pdf`). Its `pm.sample()`
calls pass `random_seed=42` (the `SatGen_Dwarf` source's own literal, not added during
migration), but a fixed seed alone does not make NUTS output reproducible
across environments: PyMC compiles the model's log-density and gradient into
a PyTensor graph, and the compiled step function -- and therefore the exact
sequence of floating-point operations the sampler executes for a given
seed -- depends on the installed pymc/pytensor/numpy/scipy versions. A
different version of any of the four can change the trace even with
`random_seed=42` held fixed.

The versions below are what this migration ran under (`J_calc` conda
environment, read with `python -c "import pymc; print(pymc.__version__)"`
etc., not guessed or copied from a requirements file):

| package  | version |
|----------|---------|
| pymc     | 5.25.1  |
| arviz    | 0.21.0  |
| pytensor | 2.31.7  |
| numpy    | 1.26.4  |
| scipy    | 1.14.1  |

A future rerun of `python/plot_powerlaw.py` intended to reproduce the
published `powerlaw_posteriors.pdf` / `powerlaw_fit.pdf` must match these
versions, not just the seed. Each figure's `provenance.figure_manifest`
sidecar records `pymc_random_seed`, `pymc_sampler_kwargs` (draws/tune/chains/
target_accept), and `package_versions` (read the same way, at run time) for
exactly this reason -- so a rerun's sidecar can be diffed against this table
before its output is compared to the published figure.

This file documents an environment snapshot, not a pinned lockfile: nothing
in the repository enforces these versions at import time (unlike, say,
`plot_style.py`'s TEXLIVE_BIN check), since a broken PyMC import would block
every other figure step in this environment file's own history. Verifying a
future rerun's numbers means diffing its `package_versions` sidecar field
against this table by hand.
