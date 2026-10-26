"""Core scientific verification: FMU output vs the vendor's reference FEA.

Two very different expectations, because the two datasets play different
roles:

* **Training** (209 points) -- these are the surrogate's *own* Kriging
  training points, and ordinary Kriging interpolates them exactly.  The FMU
  must reproduce ``TrainingResults.csv`` to CSV round-trip precision
  (observed max |err| = 5.0e-8).  This is the strongest end-to-end proof
  that the Wine pipe feeds the right inputs into the right model: any
  input-mapping, scaling or plumbing bug would move temperatures by
  whole degrees.
* **Validation** (24 points) -- independent FEA runs the surrogate never
  saw.  Small residuals here are the surrogate's *modelling* error and are
  expected (observed: max 5.73 degC on Sample Maximum Temperature,
  R2 >= 0.964).  These tests bound that error; a failure means the FMU's
  accuracy degraded (or the data was swapped), not that the pipe is broken.
"""

from __future__ import annotations

import numpy as np

from metrics import (
    TRAIN_ATOL,
    TRAIN_NRMSE_MAX,
    VALID_MAX_ABS,
    VALID_MEAN_ABS,
    VALID_NRMSE_MAX,
    VALID_R2_MIN,
    describe_worst,
    nrmse_by_range,
    r2_scores,
)


def test_training_points_reproduced_exactly(fmu_train, train_res, output_names):
    ref = train_res.to_numpy(dtype=float)
    got = fmu_train[output_names].to_numpy(dtype=float)
    err = np.abs(got - ref)
    assert err.max() <= TRAIN_ATOL, (
        describe_worst(err, output_names, "training |err|")
        + f"; tolerance {TRAIN_ATOL:g} (observed 5.0e-8 when suite was "
          "written -- Kriging must interpolate its training points)"
    )


def test_training_normalized_rmse(fmu_train, train_res, output_names):
    ref = train_res.to_numpy(dtype=float)
    got = fmu_train[output_names].to_numpy(dtype=float)
    n = nrmse_by_range(got, ref)
    worst = int(np.argmax(n))
    assert n[worst] <= TRAIN_NRMSE_MAX, (
        f"worst normalised RMSE {n[worst]:.3e} on {output_names[worst]!r} "
        f"(limit {TRAIN_NRMSE_MAX:g}, observed 2.2e-4)"
    )


def test_validation_r2_floor(fmu_valid, valid_res, output_names):
    ref = valid_res.to_numpy(dtype=float)
    got = fmu_valid[output_names].to_numpy(dtype=float)
    r2 = r2_scores(got, ref)
    worst = int(np.argmin(r2))
    assert r2[worst] >= VALID_R2_MIN, (
        f"worst R2 {r2[worst]:.4f} on {output_names[worst]!r} "
        f"(floor {VALID_R2_MIN}, observed 0.9645) -- surrogate accuracy "
        "degraded or FMU/data mismatch"
    )


def test_validation_absolute_error_bounds(fmu_valid, valid_res, output_names):
    ref = valid_res.to_numpy(dtype=float)
    got = fmu_valid[output_names].to_numpy(dtype=float)
    err = np.abs(got - ref)

    assert err.max() <= VALID_MAX_ABS, (
        describe_worst(err, output_names, "validation |err|")
        + f"; limit {VALID_MAX_ABS:g} (observed 5.73 degC)"
    )
    assert err.mean() <= VALID_MEAN_ABS, (
        f"mean |err| {err.mean():.4f} over all 24x40 values; limit "
        f"{VALID_MEAN_ABS:g} (observed 0.059)"
    )


def test_validation_normalized_rmse(fmu_valid, valid_res, output_names):
    ref = valid_res.to_numpy(dtype=float)
    got = fmu_valid[output_names].to_numpy(dtype=float)
    n = nrmse_by_range(got, ref)
    worst = int(np.argmax(n))
    assert n[worst] <= VALID_NRMSE_MAX, (
        f"worst normalised RMSE {n[worst]:.4f} on {output_names[worst]!r} "
        f"(limit {VALID_NRMSE_MAX}, observed 0.0405)"
    )
