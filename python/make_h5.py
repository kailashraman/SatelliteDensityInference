"""Build the SatGen halo catalogs (data/additional/*.h5) from the raw datasets.

This is the root of the pipeline DAG: every downstream stage reads one of these
files via config.h5_path(). Upstream the recipe lived only in
jupyter/make_h5.ipynb and could not be run outside Jupyter, which left the
repository unable to rebuild its own inputs.

Subcommands
-----------
    build <version>          dfolsom dataset -> data/additional/<file>.h5
    sims <Symphony|MWest>    per-host .npz tree -> a much smaller h5
    augment-mhalf <version>  add the logMhalf/<dwarf> group in place

`build` applies the survival mask unless the version is in UNMASKED_VERSIONS.
That is a property of the version rather than a flag: `splashback` shares
Diemer's source and differs only by being unmasked, so a flag would allow a
masked catalog to be written under the canonical unmasked filename.

Two things in the notebook differ from what actually built the catalogs, and
are handled here rather than carried over (see docs/migration-notes.md):

* `rvir` was hardcoded to 259. The actual host virial radius in the Diemer
  dataset is 258.924 kpc, and using 259 admits 476 extra satellites -- the
  difference between reproducing the published Diemer catalog exactly
  (2432159) and not (2432635). rvir is read from the dataset's own `hosts`
  table, as the notebook's own commented-out code did.
* The LMC-host search runs before masking; the notebook searched the masked
  arrays and so missed 126 of Diemer's 773 hosts. Its trailing `dist < 259`
  clause is inert -- the walrus binds `dist` to a boolean -- so it is not a
  difference.

`mwID < 2000` appears only in the cell that rebuilt `mass_floor_7` from an
existing h5. It is not part of the dfolsom build path, and is exposed as
--max-mwid rather than applied unconditionally.
"""

import argparse
import sys

import h5py
import numpy as np

import config
import provenance

# Satellite fields copied straight through from the dataset to the h5, in the
# order the notebook wrote them.
SAT_FIELDS = [
    'mwID', 'virial_mass', 'stellar_mass', 'position', 'order', 'groupID',
    'r_half', 'r_max', 'v_max', 'peak_mass', 'tpeak_stellar_mass',
    'tpeak_position', 'tpeak_order', 'tpeak_groupID', 'tpeak_r_half',
    'tpeak_r_max', 'tpeak_v_max', 'tpeak', 'tpeak_conc', 'Green_params',
    'tperi', 'rperi', 'surviving',
]

# version -> dfolsom dataset filename. Taken from
# Green_m12res8_datasets/README.md where listed, and confirmed for every row by
# rebuilding and matching the published satellite and LMC-host counts exactly
# (the README does not mention 25-10-2e12_Diemer.npz).
# 'Correa' is deliberately absent: it exists upstream but no draft figure uses
# it, and it is not in config.H5_REGISTRY.
BUILD_SOURCES = {
    'Diemer': 'Green_m12res8_datasets/25-06-Diemer+scatter-10k.npz',
    'splashback': 'Green_m12res8_datasets/25-06-Diemer+scatter-10k.npz',
    'Diemer_0p12': 'Green_m12res8_datasets/25-10-0p12dex_Diemer.npz',
    'Zhao': 'Green_m12res8_datasets/25-10-z30zhao.npz',
    'm2e12': 'Green_m12res8_datasets/25-10-2e12_Diemer.npz',
}

# Versions written without the survival mask. A property of the version, not a
# flag: `splashback` shares Diemer's source and differs only by being unmasked,
# so a flag would let `build splashback` write a masked catalog under the
# canonical unmasked filename, and vice versa.
UNMASKED_VERSIONS = {'splashback'}

# Registered catalogs that cannot be rebuilt here, with the reason. Kept
# explicit so the gap is visible rather than looking like an oversight.
NOT_REBUILDABLE = {
    'mass_floor_7': (
        'built by re-filtering 26-02-Diemer_mass_floor_7.h5 (80 GB), which was '
        'not migrated; the notebook cell that produced it read that h5 rather '
        'than a dfolsom dataset. The filtered 920 MB result is copied in. See '
        'docs/migration-notes.md'),
}

# Per-host .npz trees for the external N-body comparison suites.
SIM_SOURCES = {
    'Symphony': 'SymphonyMilkyWay',
    'MWest': 'MWest',
}

SIM_FIELDS = ['rho150', 'cV', 'mvir', 'mpeak', 'vmax', 'rmax', 'rs', 'rhos']

MASS_LOSS_FLOOR = 0.01


def survival_mask(sats, rvir, max_mwid=None):
    """Satellites that retain >1% of peak mass and lie inside the host.

    `rvir` must come from the dataset's own hosts table; see the module
    docstring on the 259-vs-258.924 discrepancy.
    """
    mass_loss = sats['virial_mass'] / sats['peak_mass']
    radius = np.linalg.norm(sats['position'], axis=1)
    mask = (mass_loss > MASS_LOSS_FLOOR) & ~(radius > rvir)
    if max_mwid is not None:
        mask &= sats['mwID'] < max_mwid
    return mask


def host_virial_radius(hosts):
    """The single host virial radius shared by every MW in a dataset."""
    values = np.unique(hosts['virial_radius'])
    if len(values) != 1:
        raise ValueError(
            f'expected one host virial radius, found {len(values)}: {values[:5]}')
    return float(values[0])


def lmc_host_ids(sats):
    """MW IDs whose satellite set contains an LMC analogue.

    An LMC analogue is a satellite above 1e11 Msun that is either currently
    within 100 kpc or has a pericentre inside 100 kpc.

    The mass cut is on **present-day virial mass** (773 Diemer hosts), which is
    what reproduces the reference catalogs. Note that
    halo_weights.get_mask(lmc_selection=True) deliberately uses a *different*,
    peak-mass definition (logMpeak > 11, 8829 hosts) and never reads the column
    written here; the two are not interchangeable. See docs/migration-notes.md.

    The one load-bearing difference from the notebook is that this runs over
    the **unmasked** satellite set. A satellite stripped below the 1%
    mass-loss floor, or now beyond the virial radius, still means its host had
    an LMC; the notebook searched the already-masked arrays and so found 647 of
    the 773 Diemer hosts, a strict subset missing 126.

    The notebook's trailing `& (dist < 259)` is **not** a difference: its
    walrus binds `dist` to the boolean `norm(position) < 100`, so the clause
    compares a boolean array to 259 and is True everywhere. It is inert there,
    and omitting it here changes nothing.
    """
    dist = np.linalg.norm(sats['position'], axis=1)
    is_lmc = ((dist < 100) | (sats['rperi'] < 100)) & (sats['virial_mass'] > 1e11)
    return np.unique(sats['mwID'][is_lmc])


def cmd_build(args):
    version = args.version
    if version in NOT_REBUILDABLE:
        raise SystemExit(f'{version!r} cannot be rebuilt: {NOT_REBUILDABLE[version]}')
    if version not in BUILD_SOURCES:
        raise SystemExit(
            f'no dfolsom source registered for {version!r}; '
            f'known: {sorted(BUILD_SOURCES)}')

    source = config.TREES_DIR / BUILD_SOURCES[version]
    out_path = config.h5_path(version)
    unmasked = version in UNMASKED_VERSIONS

    # Groups the existing catalog carries that this builder does not produce --
    # for the SatGen versions that means logMhalf/, which comes from
    # compute_mhalf (plan step 4). Overwriting would destroy the only copy in
    # this repository, and the diff against the source would still look clean.
    carried_over = []
    if out_path.exists():
        if not args.force:
            raise SystemExit(f'{out_path} exists; pass --force to overwrite')
        produced = set(SAT_FIELDS) | {'mw_hosts_lmc'}
        with h5py.File(out_path, 'r') as existing:
            carried_over = [k for k in existing if k not in produced]
        if carried_over and not args.drop_extra:
            print(f'preserving {carried_over} from the existing catalog '
                  f'(pass --drop-extra to discard)')

    print(f'reading {source}')
    with np.load(source) as data:
        hosts = data['hosts']
        sats = data['sats']
        rvir = host_virial_radius(hosts)
        print(f'host virial radius = {rvir:.5f} kpc')

        # Computed before masking, deliberately -- see lmc_host_ids.
        lmc_ids = lmc_host_ids(sats)
        print(f'{len(lmc_ids)} hosts have an LMC analogue')

        if unmasked:
            mask = np.ones(len(sats), dtype=bool)
            print(f'{version} is unmasked: keeping all {len(sats)} satellites')
        else:
            mask = survival_mask(sats, rvir, max_mwid=args.max_mwid)
            print(f'mask keeps {mask.sum()} of {len(sats)} satellites')

        selected = sats[mask]

    # Read what we are preserving before truncating the file.
    preserved = {}
    if carried_over and not args.drop_extra:
        with h5py.File(out_path, 'r') as existing:
            for key in carried_over:
                preserved[key] = {name: existing[key][name][()]
                                  for name in existing[key]} \
                    if isinstance(existing[key], h5py.Group) else existing[key][()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'writing {out_path}')
    with h5py.File(out_path, 'w') as handle:
        for field in SAT_FIELDS:
            handle.create_dataset(field, data=selected[field])
        handle.create_dataset('mw_hosts_lmc',
                              data=np.isin(selected['mwID'], lmc_ids))
        for key, value in preserved.items():
            if isinstance(value, dict):
                group = handle.create_group(key)
                for name, array in value.items():
                    group.create_dataset(name, data=array)
            else:
                handle.create_dataset(key, data=value)

    # Stamped last, after every write, so the fingerprint describes the file as
    # it now stands rather than an intermediate state.
    record = provenance.stamp('python/make_h5.py', version=version,
                              argv=sys.argv[1:], inputs=[source],
                              n_satellites=int(mask.sum()),
                              n_source=int(len(mask)),
                              host_virial_radius=rvir,
                              n_lmc_hosts=int(len(lmc_ids)),
                              masked=not unmasked,
                              carried_over=sorted(preserved))
    provenance.stamp_existing(out_path, record)
    print(f'wrote {out_path} ({mask.sum()} satellites)')


def cmd_sims(args):
    """Concatenate a suite's per-host .npz files into one h5.

    Unlike `build` these are external N-body catalogs, already reduced to
    per-halo scalars; there is no mask.
    """
    name = args.suite
    source_dir = config.SYMPHONY_DIR / SIM_SOURCES[name]
    out_path = config.h5_path(name)
    if out_path.exists() and not args.force:
        raise SystemExit(f'{out_path} exists; pass --force to overwrite')

    files = sorted(source_dir.glob('*.npz'))
    if not files:
        raise SystemExit(f'no .npz files under {source_dir}')
    print(f'{len(files)} hosts under {source_dir}')

    columns = {field: [] for field in SIM_FIELDS}
    host_id, positions, host_names = [], [], []
    for index, path in enumerate(files):
        host_names.append(path.stem)
        with np.load(path) as data:
            for field in SIM_FIELDS:
                columns[field].append(data[field])
            host_id.append(np.full(len(data['mvir']), index, dtype=np.int32))
            positions.append(data['x'])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as handle:
        for field in SIM_FIELDS:
            handle.create_dataset(field, data=np.concatenate(columns[field]))
        handle.create_dataset('host_id', data=np.concatenate(host_id))
        handle.create_dataset('x', data=np.concatenate(positions))
        handle.create_dataset('host_names',
                              data=np.array(host_names, dtype=object),
                              dtype=h5py.special_dtype(vlen=str))

    total = sum(len(c) for c in columns['mvir'])
    record = provenance.stamp('python/make_h5.py', version=name,
                              argv=sys.argv[1:], inputs=files,
                              n_halos=int(total), n_hosts=len(files))
    provenance.stamp_existing(out_path, record)
    print(f'wrote {out_path} ({total} halos from {len(files)} hosts)')


def cmd_augment_mhalf(args):
    """Add the logMhalf/<dwarf> group to an existing catalog.

    Symphony and MWest only. halo_weights.from_logMhalf reads this group for
    those two catalogs; for the SatGen versions it reads
    data/additional/mhalf/<version>/ instead, produced by compute_mhalf (plan
    step 4). This computes logMhalf from the NFW parameters the N-body
    reductions carry, which the SatGen catalogs do not have.

    A Symphony/MWest catalog rebuilt without this step raises KeyError in
    from_logMhalf rather than falling back -- which is exactly how the current
    upstream files came to be unusable for regeneration.
    """
    import astropy.units as u
    from colossus.halo import profile_nfw

    import Jdata as obs

    out_path = config.h5_path(args.version)
    with h5py.File(out_path, 'a') as handle:
        group = handle['logMhalf'] if 'logMhalf' in handle \
            else handle.create_group('logMhalf')
        # float64: the stored columns are float32, and letting that propagate
        # through enclosedMass shifts the result by ~5e-6 dex against upstream.
        rs = np.asarray(handle['rs'][:], dtype=float)
        rhos = np.asarray(handle['rhos'][:], dtype=float)

        for dwarf in obs.dwarf_names:
            if dwarf in group and not args.force:
                continue
            rhalf = obs.Dwarf(dwarf).rhalf.to(u.kpc).value
            profile = profile_nfw.NFWProfile(rhos=rhos, rs=rs)
            logMhalf = np.log10(profile.enclosedMass(rhalf))
            if dwarf in group:
                del group[dwarf]
            group.create_dataset(dwarf, data=logMhalf)
            print(f'  {dwarf}')

    print(f'augmented {out_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)

    build = sub.add_parser('build', help='build a SatGen catalog')
    build.add_argument('version', choices=sorted(BUILD_SOURCES))
    build.add_argument('--max-mwid', type=int, default=None,
                       help='additionally require mwID < MAX_MWID (not part of '
                            'any published catalog; see NOT_REBUILDABLE)')
    build.add_argument('--force', action='store_true',
                       help='overwrite, preserving groups this builder does '
                            'not produce')
    build.add_argument('--drop-extra', action='store_true',
                       help='with --force, also discard those groups')
    build.set_defaults(func=cmd_build)

    sims = sub.add_parser('sims', help='build a Symphony/MWest catalog')
    sims.add_argument('suite', choices=sorted(SIM_SOURCES))
    sims.add_argument('--force', action='store_true')
    sims.set_defaults(func=cmd_sims)

    mhalf = sub.add_parser('augment-mhalf',
                           help='add the logMhalf group (Symphony/MWest only)')
    mhalf.add_argument('version', choices=sorted(SIM_SOURCES))
    mhalf.add_argument('--force', action='store_true',
                       help='recompute dwarfs already present')
    mhalf.set_defaults(func=cmd_augment_mhalf)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
