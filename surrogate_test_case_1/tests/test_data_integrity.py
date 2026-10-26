"""Integrity of the loaded training/validation CSVs.

Pure pandas -- runs before any Wine work, so a swapped or corrupted data
file fails in well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from metrics import (
    DESIGN_SPACE,
    N_INPUTS,
    N_OUTPUTS,
    N_TRAIN,
    N_VALID,
)


def test_shapes(train_doe, train_res, valid_doe, valid_res):
    assert train_doe.shape == (N_TRAIN, N_INPUTS)
    assert train_res.shape == (N_TRAIN, N_OUTPUTS)
    assert valid_doe.shape == (N_VALID, N_INPUTS)
    assert valid_res.shape == (N_VALID, N_OUTPUTS)


def test_all_values_finite_and_numeric(train_doe, train_res, valid_doe, valid_res):
    for name, df in [("TrainingDOE", train_doe), ("TrainingResults", train_res),
                     ("ValidationDOE", valid_doe), ("ValidationResults", valid_res)]:
        assert not df.isna().any().any(), f"{name} contains NaN"
        numeric = df.select_dtypes(include=[np.number])
        assert numeric.shape == df.shape, f"{name} has non-numeric columns"
        assert np.isfinite(numeric.to_numpy()).all(), f"{name} has inf values"


def test_doe_and_results_headers_are_aligned(train_doe, train_res,
                                             valid_doe, valid_res):
    """DOE and Results files must describe the same rows and the same
    variables (18 inputs / 40 outputs, identical names across splits)."""
    assert list(train_doe.columns) == list(valid_doe.columns)
    assert list(train_res.columns) == list(valid_res.columns)
    assert len(train_doe.columns) == N_INPUTS
    assert len(train_res.columns) == N_OUTPUTS


def test_no_duplicate_design_points(train_doe, valid_doe):
    assert not train_doe.duplicated().any(), "duplicate training points"
    assert not valid_doe.duplicated().any(), "duplicate validation points"


def test_validation_points_inside_training_box(train_doe, valid_doe):
    """Validation must interpolate the design space, not extrapolate."""
    lo, hi = train_doe.min(), train_doe.max()
    outside = (valid_doe < lo) | (valid_doe > hi)
    offenders = [c for c in valid_doe.columns if outside[c].any()]
    assert not offenders, f"validation outside training box for {offenders}"


def test_design_space_matches_frozen_expectations(train_doe):
    """Guard against reading the wrong/scaled CSV: every input must live in
    the design box observed when the suite was written (+/-1 % of span)."""
    for col, (lo, hi) in DESIGN_SPACE.items():
        assert col in train_doe.columns, f"missing input column {col!r}"
        pad = 0.01 * (hi - lo)
        assert train_doe[col].min() >= lo - pad, (
            f"{col}: min {train_doe[col].min():.6g} below expected {lo:.6g}")
        assert train_doe[col].max() <= hi + pad, (
            f"{col}: max {train_doe[col].max():.6g} above expected {hi:.6g}")


def test_temperatures_physically_plausible_in_reference(train_res, valid_res):
    """Coarse sanity of the FEA truth the surrogate is judged against."""
    tc_cols = [c for c in train_res.columns if c.startswith("Temperature (degC), TC")]
    assert len(tc_cols) == 10
    for name, df in [("training", train_res), ("validation", valid_res)]:
        tc = df[tc_cols].to_numpy()
        assert tc.min() > 0, f"{name}: non-positive TC temperature"
        assert tc.max() < 1200, f"{name}: TC temperature above 1200 degC"
        smax = df["Sample Maximum Temperature (degC)"].to_numpy()
        assert smax.min() > 0, f"{name}: non-positive sample max temperature"
        assert smax.max() < 1200, f"{name}: sample max above 1200 degC"
