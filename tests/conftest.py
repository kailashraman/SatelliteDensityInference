"""Shared pytest configuration.

Three tiers, so the default run is fast and network-free:

    pytest                  unit tests only -- seconds, no data needed
    pytest -m needs_data    also runs stage parity checks against results/
    pytest -m slow          also runs the mock-pipeline calibration run

`needs_data` and `slow` are deselected by default and opt-in via --rundata /
--runslow, because they depend on the ~250 GB of migrated intermediates.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'python'))


def pytest_addoption(parser):
    parser.addoption('--rundata', action='store_true', default=False,
                     help='run tests that need the migrated results/ tree')
    parser.addoption('--runslow', action='store_true', default=False,
                     help='run slow calibration tests')


def pytest_configure(config):
    config.addinivalue_line('markers', 'needs_data: requires migrated results/')
    config.addinivalue_line('markers', 'needs_satgen: requires the J_calc conda env')
    config.addinivalue_line('markers', 'slow: long-running calibration test')


def pytest_collection_modifyitems(config, items):
    skip_data = pytest.mark.skip(reason='needs --rundata')
    skip_slow = pytest.mark.skip(reason='needs --runslow')
    for item in items:
        if 'needs_data' in item.keywords and not config.getoption('--rundata'):
            item.add_marker(skip_data)
        if 'slow' in item.keywords and not config.getoption('--runslow'):
            item.add_marker(skip_slow)
