"""Regression test for the satellites_vcirc_intro.pdf containment band.

Bug this pins: `build_intro` drew TWO stacked `fill_between` calls in the
SatGen median's color -- the 68% band (quantiles_arr indices 2/4) plus a
second, uncontained band (indices 1/5) -- where the published
satellites_vcirc_intro.pdf has only one (temp_part1.tex's fig:vcirc caption:
"the median (black line) and 68% containment band (gray)"). Verified by flat-
fill RGB decomposition of the rendered PDF: a single gray level, consistent
with one alpha-0.2 fill over white, not two overlapping fills.

Fails against the pre-fix source, which left a second `fill_between` call in
`build_intro` and drew two PolyCollection artists in the SatGen median's
color instead of one. The commented-out state of the source notebook cell is
not the oracle here (per CLAUDE.md/memory, comment state is post-run
residue) -- this asserts on the RENDERED artist count instead.

Requires the J_calc environment: plot_vcirc_satellites_all imports
halo_weights/Jdata/colossus. See conftest.py's `needs_satgen` marker.
"""
import numpy as np
import matplotlib.colors as mcolors
import astropy.units as u
import pytest

pvsa = pytest.importorskip('plot_vcirc_satellites_all', reason='needs the J_calc environment')

pytestmark = pytest.mark.needs_satgen


class _FakeDwarf:
    """Stands in for Jdata.Dwarf -- only the attributes build_intro reads."""

    def __init__(self, name):
        self.rhalf = 0.1 * u.kpc
        self.rhalf_err = [0.01, 0.01] * u.kpc
        self.Vcirc = 10.0 * u.km / u.s
        self.Vcirc_err = [1.0, 1.0] * u.km / u.s


def _render_intro(tmp_path, monkeypatch):
    """Run build_intro on small fake inputs and return its Axes."""
    # Small fake r_grid/vcirc_quantiles -- _load()'s real ~1.4 GB inputs
    # (h5, massProfile.npz, scatter_300pc.txt) are not needed to exercise the
    # plotting logic, and importing this module no longer loads them eagerly
    # (see W4 in the review this test accompanies).
    r_grid = np.linspace(0.02, 10, 20)
    quantiles = np.tile(np.linspace(1, 10, 20)[:, None], (1, 7))
    # raising=False: these are set by _load() at runtime (module scope no
    # longer defines them at import time, see W4) -- not present yet here.
    monkeypatch.setattr(pvsa, 'r_grid', r_grid, raising=False)
    monkeypatch.setattr(pvsa, 'vcirc_quantiles', quantiles, raising=False)
    monkeypatch.setattr(pvsa, 'h5_path', tmp_path / 'fake.h5', raising=False)
    monkeypatch.setattr(pvsa, 'massprofile_path', tmp_path / 'fake.npz', raising=False)
    monkeypatch.setattr(pvsa, 'version', 'Diemer', raising=False)
    # One dwarf, not in large_dwarf, so build_intro's errorbar series and
    # legend (which needs a real eb1 handle) still exercise normally.
    monkeypatch.setattr(pvsa.obs, 'dwarf_names', np.array(['Fake Dwarf']))
    monkeypatch.setattr(pvsa.obs, 'Dwarf', _FakeDwarf)
    monkeypatch.setattr(pvsa, 'out_path', lambda which: tmp_path / f'{which}.pdf')
    monkeypatch.setattr(pvsa.provenance, 'figure_manifest', lambda *a, **k: None)

    captured = {}
    orig_subplots = pvsa.plt.subplots

    def capture_subplots(*args, **kwargs):
        fig, ax = orig_subplots(*args, **kwargs)
        captured['ax'] = ax
        return fig, ax

    monkeypatch.setattr(pvsa.plt, 'subplots', capture_subplots)

    pvsa.build_intro()
    return captured['ax']


def test_build_intro_draws_exactly_one_satgen_band(tmp_path, monkeypatch):
    ax = _render_intro(tmp_path, monkeypatch)
    satgen_rgb = mcolors.to_rgb(pvsa.colors[0])
    satgen_bands = [
        c for c in ax.collections
        if type(c).__name__ == 'PolyCollection'
        and tuple(np.round(c.get_facecolor()[0][:3], 6)) == tuple(np.round(satgen_rgb, 6))
    ]
    assert len(satgen_bands) == 1, (
        f'build_intro drew {len(satgen_bands)} fill_between band(s) in the '
        "SatGen median's color; the published satellites_vcirc_intro.pdf has "
        'exactly one (the 68% containment band, quantiles_arr indices 2/4).')


def test_intro_nfw_curve_uses_diemer19(tmp_path, monkeypatch):
    """The reference NFW curve must use the Diemer+19 c(M) relation.

    Bug this pins: build_intro drew the 10^8.5 Msun, z=2 NFW reference with
    ishiyama21 concentrations (faithfully migrated from the rendered upstream
    PDF), while temp_part1.tex's fig:vcirc caption states Diemer+19. Fails
    on the pre-fix source: its green median line matches ishiyama21, not
    diemer19 (c = 7.69 vs 7.97; V_circ offset up to ~2.4%, far above rtol).
    Also pins the +/-0.16 dex band so it cannot keep a different c(M).
    """
    from colossus.halo import concentration, profile_nfw

    ax = _render_intro(tmp_path, monkeypatch)
    green = mcolors.to_rgb(pvsa.cm_color)
    lines = [ln for ln in ax.get_lines()
             if tuple(np.round(mcolors.to_rgb(ln.get_color()), 6)) == tuple(np.round(green, 6))]
    assert len(lines) == 1, f'expected one NFW median line, found {len(lines)}'
    r_kpc, v_drawn = (np.asarray(a) for a in lines[0].get_data())

    h = pvsa.obs.col_co.h
    M, z = 10 ** 8.5 * h, 2

    def vcirc(model, nsig=0):
        c = concentration.concentration(M, 'vir', z, model=model) * 10 ** (nsig * 0.16)
        return profile_nfw.NFWProfile(M=M, c=c, z=z, mdef='vir').circularVelocity(r_kpc * h)

    np.testing.assert_allclose(v_drawn, vcirc('diemer19'), rtol=1e-10)
    assert not np.allclose(v_drawn, vcirc('ishiyama21'), rtol=1e-3)

    bands = [c for c in ax.collections
             if type(c).__name__ == 'PolyCollection'
             and tuple(np.round(c.get_facecolor()[0][:3], 6)) == tuple(np.round(green, 6))]
    assert len(bands) == 1, f'expected one NFW band, found {len(bands)}'
    verts = bands[0].get_paths()[0].vertices
    lo, hi = vcirc('diemer19', -1), vcirc('diemer19', 1)
    # every r on the grid must have a band vertex on each envelope
    on_lo = np.array([np.any(np.isclose(verts[:, 1][verts[:, 0] == r], v, rtol=1e-10)) for r, v in zip(r_kpc, lo)])
    on_hi = np.array([np.any(np.isclose(verts[:, 1][verts[:, 0] == r], v, rtol=1e-10)) for r, v in zip(r_kpc, hi)])
    assert on_lo.all() and on_hi.all(), 'NFW band edges are not diemer19 * 10^(+/-0.16 dex)'
