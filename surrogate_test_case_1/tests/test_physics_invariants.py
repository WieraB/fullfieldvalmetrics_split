"""Physical invariants and response directions.

Pointwise tolerance tests can pass while systematically biased; these tests
check *structure* instead:

* invariants that hold in the FEA truth (sample max above every TC, sample
  average below the sample max) must also hold in every FMU prediction;
* the FMU's output envelope must sit close to the reference envelope;
* for every strong input/output influence measured in the FEA data, the FMU
  must respond in the same direction (Spearman sign agreement) -- this is
  what "the surrogate model behaves like the physics" means.
"""

from __future__ import annotations

import numpy as np
import pytest

from metrics import (
    SENSITIVITY_RHO_CUTOFF,
    SENSITIVITY_SIGN_AGREEMENT,
)

SAMPLE_MAX = "Sample Maximum Temperature (degC)"
SAMPLE_AVG = "Sample Average Temperature (degC)"
TC_PREFIX = "Temperature (degC), TC"

# How far the FMU envelope may poke outside the reference envelope:
# 20 % of each output's reference span (observed worst: 0.8 %), plus a
# small absolute floor for near-constant outputs.
ENVELOPE_SPAN_FRACTION = 0.20
ENVELOPE_ABS_FLOOR = 1e-6


def _tc_cols(df):
    cols = [c for c in df.columns if c.startswith(TC_PREFIX)]
    assert len(cols) == 10, f"expected 10 TC columns, found {len(cols)}"
    return cols


def _assert_invariants(df, label):
    tc = df[_tc_cols(df)].to_numpy(dtype=float)
    smax = df[SAMPLE_MAX].to_numpy(dtype=float)
    savg = df[SAMPLE_AVG].to_numpy(dtype=float)

    bad_max = np.where(smax < tc.max(axis=1))[0]
    assert bad_max.size == 0, (
        f"{label}: sample max temperature below a TC in rows "
        f"{bad_max[:5].tolist()}"
    )
    bad_avg = np.where(savg > smax)[0]
    assert bad_avg.size == 0, (
        f"{label}: sample average above sample max in rows "
        f"{bad_avg[:5].tolist()}"
    )


def test_invariants_hold_in_reference_data(train_res, valid_res):
    _assert_invariants(train_res, "TrainingResults")
    _assert_invariants(valid_res, "ValidationResults")


def test_invariants_hold_in_fmu_output(fmu_train, fmu_valid):
    _assert_invariants(fmu_train, "FMU@training")
    _assert_invariants(fmu_valid, "FMU@validation")


def test_fmu_envelope_close_to_reference(fmu_train, train_res, fmu_valid,
                                         valid_res, output_names):
    """The FMU must cover roughly the same range as the FEA data it
    approximates -- catches scale errors, swapped outputs, sign flips."""
    problems = []
    for label, fmu, ref_df in [
        ("training", fmu_train, train_res),
        ("validation", fmu_valid, valid_res),
    ]:
        ref = ref_df[output_names].to_numpy(dtype=float)
        got = fmu[output_names].to_numpy(dtype=float)
        span = ref.max(axis=0) - ref.min(axis=0)
        floor = ENVELOPE_ABS_FLOOR * np.maximum(np.abs(ref).max(axis=0), 1.0)
        pad = ENVELOPE_SPAN_FRACTION * span + floor

        lo_bad = np.where(got.min(axis=0) < ref.min(axis=0) - pad)[0]
        hi_bad = np.where(got.max(axis=0) > ref.max(axis=0) + pad)[0]
        for col in set(lo_bad.tolist()) | set(hi_bad.tolist()):
            problems.append(
                f"{label}/{output_names[col]!r}: FMU range "
                f"[{got[:, col].min():.6g}, {got[:, col].max():.6g}] vs "
                f"ref [{ref[:, col].min():.6g}, {ref[:, col].max():.6g}]"
            )
    assert not problems, "FMU envelope deviates from reference:\n" + "\n".join(problems)


def test_sensitivity_signs_agree_with_fea(train_doe, valid_doe, train_res,
                                          valid_res, fmu_train, fmu_valid,
                                          output_names):
    """For every input/output pair with a strong influence in the FEA data
    (|Spearman rho| > 0.3), the FMU must push in the same direction.

    This is direction-only, so it is scale-free and catches inverted or
    mis-wired inputs even when pointwise tolerances are relaxed."""
    pytest.importorskip("scipy")
    from scipy.stats import spearmanr

    doe = np.vstack([
        train_doe.to_numpy(dtype=float),
        valid_doe.to_numpy(dtype=float),
    ])
    ref = np.vstack([
        train_res[output_names].to_numpy(dtype=float),
        valid_res[output_names].to_numpy(dtype=float),
    ])
    got = np.vstack([
        fmu_train[output_names].to_numpy(dtype=float),
        fmu_valid[output_names].to_numpy(dtype=float),
    ])

    in_names = list(train_doe.columns)
    strong = agree = 0
    disagreements = []
    for i, iname in enumerate(in_names):
        for j, oname in enumerate(output_names):
            rho_ref, _ = spearmanr(doe[:, i], ref[:, j])
            rho_fmu, _ = spearmanr(doe[:, i], got[:, j])
            if not np.isfinite(rho_ref) or not np.isfinite(rho_fmu):
                continue
            if abs(rho_ref) > SENSITIVITY_RHO_CUTOFF:
                strong += 1
                if np.sign(rho_ref) == np.sign(rho_fmu):
                    agree += 1
                else:
                    disagreements.append(
                        f"{iname} -> {oname}: rho_ref={rho_ref:+.2f} "
                        f"rho_fmu={rho_fmu:+.2f}"
                    )

    assert strong >= 50, (
        f"only {strong} strong pairs found -- the reference data no longer "
        "shows structure (wrong files loaded?)"
    )
    ratio = agree / strong
    assert ratio >= SENSITIVITY_SIGN_AGREEMENT, (
        f"sensitivity sign agreement {agree}/{strong} = {ratio:.2%} "
        f"(floor {SENSITIVITY_SIGN_AGREEMENT:.0%}); disagreements:\n"
        + "\n".join(disagreements[:10])
    )
