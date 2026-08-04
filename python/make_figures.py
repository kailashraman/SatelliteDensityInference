"""Report the figure-naming contract's coverage: which basenames in
figure_registry.REGISTRY / TABLE_REGISTRY have a producer registered, and
which do not yet.

No plot scripts exist yet (steps 10-22 add them and register themselves in
figure_registry.py), so today every entry is None and this prints the full
pending list. It does not call any producer -- the calling convention for a
registered producer is a decision for the step that adds the first one, not
this one.
"""

import argparse

import figure_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    for label, registry in [('figure', figure_registry.REGISTRY),
                             ('table', figure_registry.TABLE_REGISTRY)]:
        done = [name for name, producer in registry.items() if producer is not None]
        pending = [name for name, producer in registry.items() if producer is None]
        print(f'{len(done)}/{len(registry)} {label}s have a registered producer')
        for name in pending:
            print(f'  pending: {name}')


if __name__ == '__main__':
    main()
