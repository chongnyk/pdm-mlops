# Its presence puts the project folder on sys.path for pytest.
from pathlib import Path

import pytest


def pytest_configure(config):
    if not (Path(__file__).parent / "model.joblib").exists():
        raise pytest.UsageError("model.joblib not found. Run train.py first: the API tests need the model.")
