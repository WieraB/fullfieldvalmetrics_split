"""Shared fixtures for the FullSurrogate FMU verification suite.

The suite has two halves:

* **data / description tests** -- pure Python + pandas, no Wine (fast).
* **FMU tests** -- evaluate the surrogate through the Wine pipe and compare
  against the vendor's reference FEA results in ``train-valid-data/``.

Heavy batches (209 training + 24 validation points) are computed once in
session-scoped fixtures and reused by several test modules, so the whole
suite costs only a handful of Wine processes (~40 s total).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                    # surrogate_test_case_1/
DATA_DIR = ROOT / "train-valid-data"
FMU_PATH = ROOT / "FullSurrogate.fmu"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from surrogate_fmu import Surrogate  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "wine: exercises the Wine pipe directly (auto-skipped if the FMU "
        "ever ships a native Linux binary and the backend switches)",
    )


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def train_doe() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "TrainingDOE.csv")


@pytest.fixture(scope="session")
def train_res() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "TrainingResults.csv")


@pytest.fixture(scope="session")
def valid_doe() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ValidationDOE.csv")


@pytest.fixture(scope="session")
def valid_res() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "ValidationResults.csv")


# --------------------------------------------------------------------------
# FMU
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fmu_path() -> Path:
    return FMU_PATH


@pytest.fixture(scope="session")
def surrogate() -> Surrogate:
    return Surrogate(FMU_PATH)


@pytest.fixture(scope="session")
def wine_surrogate(surrogate):
    """The shared Surrogate, but only while the backend is actually Wine.

    Wine-specific tests skip (not fail) if SmartUQ ever re-exports the FMU
    with ``binaries/x86_64-linux`` and the wrapper auto-switches to fmpy.
    """
    if surrogate.backend != "wine":
        pytest.skip(f"backend is {surrogate.backend!r}, Wine pipe not in use")
    return surrogate


@pytest.fixture(scope="session")
def input_names(surrogate):
    return surrogate.input_names


@pytest.fixture(scope="session")
def output_names(surrogate):
    return surrogate.output_names


# --------------------------------------------------------------------------
# pre-computed FMU sweeps (the expensive part, shared across modules)
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fmu_train(surrogate, train_doe, output_names) -> pd.DataFrame:
    """FMU evaluated at all 209 training design points (one Wine session)."""
    df = surrogate.evaluate_batch(train_doe.to_dict("records"))
    assert list(df.columns[-len(output_names):]) == output_names
    return df


@pytest.fixture(scope="session")
def fmu_valid(surrogate, valid_doe, output_names) -> pd.DataFrame:
    """FMU evaluated at all 24 validation design points (one Wine session)."""
    df = surrogate.evaluate_batch(valid_doe.to_dict("records"))
    assert list(df.columns[-len(output_names):]) == output_names
    return df


# --------------------------------------------------------------------------
# helper points
# --------------------------------------------------------------------------
@pytest.fixture(scope="session")
def midpoint(train_doe) -> dict:
    """Centre of the training design box (definitely in-domain)."""
    return {
        c: float((train_doe[c].min() + train_doe[c].max()) / 2)
        for c in train_doe.columns
    }
