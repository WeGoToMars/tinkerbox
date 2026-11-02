"""Compiled bundles for model tests: an environment override, else this Mac's installed catalog bundle."""

import os
from pathlib import Path

import pytest

from tinkerbox_ane import models


@pytest.fixture(scope='session')
def bundle():
    def find(model, variable=None):
        path = Path(os.environ[variable]) if variable and os.environ.get(variable) else models.cached(model)
        if not (path / 'model.json').is_file():
            pytest.skip(f'requires a compiled {model} bundle (install it, or set {variable or "its asset variable"})')
        return path
    return find
