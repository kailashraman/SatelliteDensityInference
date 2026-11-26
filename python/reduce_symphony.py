"""Reduce the Symphony / Milky Way-est symlib halo catalogs to per-host .npz.

The output, data/symphony/<suite>/<host>.npz, is the input to
`make_h5.py sims`. Upstream this lived in an untracked notebook
(../Symphony/Symphony.ipynb); it is ported here unchanged in substance.

Subcommands
-----------
    download <Symphony|MWest>   fetch the z=0 halo catalogs with symlib
                                -> data/symphony_raw/<suite>/<host>/halos/
    reduce <Symphony|MWest>     halo catalogs -> data/symphony/<suite>/<host>.npz

`download` reads the Symphony server credentials from SYMLIB_USER and
SYMLIB_PASSWORD (request them via the Symphony data-access page); they are never
stored in the repository.

Per host, the reduction keeps subhalos that symlib flags `ok` at the last
snapshot, excluding the host itself (index 0), and records

    rmax, vmax, mvir, x            z=0 values (kpc, km/s, Msun, kpc)
    mpeak                          from the merger history
    rs, rhos                       NFW parameters matched to (rmax, vmax):
                                   rs = rmax / 2.1626, rhos = 1.7213 (vmax/rmax)^2 / G
    rho150                         that NFW profile's density at 150 pc
    cV                             2 (vmax / (H0 rmax))^2

using the suite's cosmology from symlib.simulation_parameters (which symlib
keys on the suite directory, not the host).
"""

import argparse
import os
import re
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import quote

import numpy as np
import symlib
from astropy import constants as const
from astropy import units as u
from colossus.cosmology import cosmology as co
from colossus.halo import profile_nfw

import config
import provenance

# CLI name (matching make_h5.py sims) -> symlib suite name.
SUITES = {
    'Symphony': 'SymphonyMilkyWay',
    'MWest': 'MWest',
}


def cmd_download(args):
    """Download every host's halo catalogs for one suite."""
    user = os.environ.get('SYMLIB_USER')
    password = os.environ.get('SYMLIB_PASSWORD')
    if not user or not password:
        raise SystemExit('set SYMLIB_USER and SYMLIB_PASSWORD to download')
    config.SYMPHONY_RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        symlib.download_files(user, password, SUITES[args.suite], None,
                              str(config.SYMPHONY_RAW_DIR), target='halos')
    except Exception as exc:
        # The original message (and its chained traceback) can carry the
        # credentials, e.g. in the request URL; scrub them and drop the chain.
        # One pass, longest first, so a secret that is a prefix of the other
        # is not left half-redacted; include the percent-encoded forms, which
        # is how an HTTP library may re-render them in a URL.
        secrets = sorted({user, password, quote(user, safe=''),
                          quote(password, safe='')}, key=len, reverse=True)
        message = re.sub('|'.join(map(re.escape, secrets)), '***', str(exc))
        raise RuntimeError(f'symlib download of {SUITES[args.suite]} failed '
                           f'({type(exc).__name__}): {message}') from None


def reduce_host(host_dir, name):
    """The notebook's per-host reduction; returns the arrays to save."""
    params = symlib.simulation_parameters(str(host_dir))
    co.addCosmology(name, flat=params['flat'], H0=params['H0'],
                    Om0=params['Om0'], Ob0=params['Ob0'],
                    sigma8=params['sigma8'], ns=params['ns'])
    col_co = co.setCosmology(name)

    halos, histories = symlib.read_subhalos(str(host_dir))
    halo_selection = halos['ok'][:, -1].copy()
    halo_selection[0] = False
    rmax = halos['rvmax'][:, -1][halo_selection] * u.kpc
    vmax = halos['vmax'][:, -1][halo_selection] * u.km / u.s
    mvir = halos['mvir'][:, -1][halo_selection]
    mpeak = histories['mpeak'][halo_selection]
    x = halos['x'][:, -1][halo_selection]
    rs = 0.46241029979236 * rmax.value
    rhos = 1.7212585601570 * (vmax / rmax)**2 / const.G
    rhos = rhos.to(u.Msun / u.kpc**3).value

    rho150 = np.zeros_like(rs)
    cV = np.zeros_like(rs)
    for i in range(len(rmax)):
        # colossus works in kpc/h and Msun h^2 / kpc^3.
        profile = profile_nfw.NFWProfile(rhos=rhos[i] / col_co.h**2,
                                         rs=rs[i] * col_co.h)
        rho150[i] = profile.density(0.15 * col_co.h) * col_co.h**2
        cV[i] = 2 * (vmax[i] / (rmax[i] * (col_co.H0 * u.km / u.s / u.Mpc))
                     ).to(u.dimensionless_unscaled) ** 2

    return dict(rho150=rho150, cV=cV, mvir=mvir, x=x, mpeak=mpeak,
                vmax=vmax.value, rmax=rmax.value, rs=rs, rhos=rhos)


def cmd_reduce(args):
    """Write one .npz per host of a suite."""
    suite = SUITES[args.suite]
    raw_dir = config.SYMPHONY_RAW_DIR / suite
    out_dir = (args.out_dir if args.out_dir is not None
               else config.SYMPHONY_DIR / suite)
    hosts = sorted(p for p in raw_dir.iterdir()
                   if p.is_dir() and not p.name.startswith('.')) \
        if raw_dir.is_dir() else []
    if not hosts:
        raise SystemExit(f'no host directories under {raw_dir}; '
                         f'run `download {args.suite}` first')

    # Host set by name: a partial download or the wrong catalog is refused
    # rather than written as a short set.
    expected = set(symlib.util.DEFAULT_HALO_NAMES[suite])
    found = {p.name for p in hosts}
    if found != expected:
        raise SystemExit(f'{raw_dir} host directories differ from symlib\'s '
                         f'{suite} set: missing {sorted(expected - found)}, '
                         f'unexpected {sorted(found - expected)}; '
                         'incomplete download?')
    out_paths = [out_dir / f'{host_dir.name}.npz' for host_dir in hosts]
    # make_h5.py sims globs out_dir, so a stray .npz would be read as a host.
    stray = sorted(p.name for p in out_dir.glob('*.npz')
                   if p not in out_paths) if out_dir.is_dir() else []
    if stray:
        raise SystemExit(f'{out_dir} holds .npz files that are not outputs of '
                         f'this reduction ({stray}); make_h5.py sims would '
                         'read them as hosts. Remove them first')
    # Check every output before writing any, so a refusal never leaves a
    # part-written out_dir.
    existing = [p for p in out_paths if p.exists()]
    if existing and not args.force:
        raise SystemExit(f'{existing[0]} exists ({len(existing)} of '
                         f'{len(out_paths)}); pass --force to overwrite')

    # Phase 1: reduce every host and build its stamp in memory (~1 MB in
    # all), so a failure on host k leaves no partial output, and git_dirty is
    # read before any tracked file is rewritten.
    versions = {name: metadata.version(name)
                for name in ('symlib', 'colossus', 'astropy')}
    results = []
    for host_dir, out_path in zip(hosts, out_paths):
        arrays = reduce_host(host_dir, host_dir.name)
        inputs = sorted(p for p in (host_dir / 'halos').iterdir() if p.is_file())
        # version=None: a stamped version resolves to the downstream catalog
        # (MWest.h5), which is built from this output.
        record = provenance.stamp('python/reduce_symphony.py',
                                  version=None, argv=args.argv,
                                  inputs=inputs, suite=args.suite,
                                  symlib_version=versions['symlib'],
                                  colossus_version=versions['colossus'],
                                  astropy_version=versions['astropy'],
                                  n_halos=int(len(arrays['mvir'])))
        results.append((out_path, record, arrays))

    # Phase 2: write.
    for out_path, record, arrays in results:
        provenance.savez(out_path, record, **arrays)
        print(f'wrote {out_path} ({len(arrays["mvir"])} subhalos)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)

    download = sub.add_parser('download', help='fetch halo catalogs with symlib')
    download.add_argument('suite', choices=sorted(SUITES))
    download.set_defaults(func=cmd_download)

    reduce = sub.add_parser('reduce', help='halo catalogs -> per-host .npz')
    reduce.add_argument('suite', choices=sorted(SUITES))
    reduce.add_argument('--out-dir', type=Path,
                        default=None,
                        help='write here instead of data/symphony/<suite>/ '
                             '(e.g. for a parity check)')
    reduce.add_argument('--force', action='store_true')
    reduce.set_defaults(func=cmd_reduce)

    args = parser.parse_args(argv)
    args.argv = sys.argv[1:] if argv is None else list(argv)
    args.func(args)


if __name__ == '__main__':
    main()
