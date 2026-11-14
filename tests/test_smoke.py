"""Step 1: prove the package is installed and importable from the venv."""

import importlib

import pytest

import sardrone


def test_package_has_version():
    assert sardrone.__version__ == "0.1.0"


@pytest.mark.parametrize("name", ["sardrone.mcap", "sardrone.sim", "sardrone.fo"])
def test_subpackages_import(name):
    assert importlib.import_module(name) is not None
