"""Does the Wine pipe evaluate the FMU *correctly*?

These tests pin down the plumbing itself -- determinism, batching,
instance reuse, error handling -- independently of any reference data, so a
plumbing failure is distinguishable from a surrogate-accuracy failure.
All tests here are marked ``wine`` and skip if the backend ever switches to
a native Linux binary.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from surrogate_fmu import Surrogate

pytestmark = pytest.mark.wine


def test_wine_toolchain_available(wine_surrogate):
    wine = wine_surrogate.check_wine()
    assert wine and (Path(wine).exists() or shutil.which(wine)), (
        f"wine executable {wine!r} not resolvable")
    assert Path(wine_surrogate.wine_prefix).is_dir(), (
        "Wine prefix missing -- see README_surrogate.md for setup")
    assert wine_surrogate.wine_prefix  # configured


def test_driver_reports_model_over_wine(wine_surrogate, input_names,
                                        output_names):
    """``fmi3_wine.py --list`` inside Windows Python: proves the Windows
    side launches, opens the FMU and parses its XML."""
    meta = json.loads(wine_surrogate._run_driver(["--list"]))
    assert meta["fmi_version"] == "3.0"
    assert [v["name"] for v in meta["inputs"]] == input_names
    assert [v["name"] for v in meta["outputs"]] == output_names


def test_deterministic_within_one_process(wine_surrogate, midpoint):
    """Same input twice in one batch (one FMU instance) -> identical."""
    df = wine_surrogate.evaluate_batch([midpoint, midpoint])
    a = df.iloc[0].to_numpy(dtype=float)
    b = df.iloc[1].to_numpy(dtype=float)
    assert np.array_equal(a, b), (
        f"non-deterministic within a session: max diff {np.abs(a - b).max()}")


def test_deterministic_across_processes(wine_surrogate, midpoint):
    """Two independent Wine processes / fresh instances -> bit-identical."""
    a = wine_surrogate.evaluate(midpoint)
    b = wine_surrogate.evaluate(midpoint)
    outs = list(a)
    diffs = {k: abs(a[k] - b[k]) for k in outs}
    worst = max(diffs, key=diffs.get)
    assert diffs[worst] == 0.0, f"cross-run drift at {worst!r}: {diffs[worst]}"


def test_batch_equals_single_evaluations(wine_surrogate, train_doe):
    """A batch (one instance, inputs re-set after init) must equal
    isolated single evaluations -- validates the fast batch path."""
    pts = [train_doe.iloc[i].to_dict() for i in (0, 70, 140, 208)]
    batch = wine_surrogate.evaluate_batch(pts)
    for i, p in enumerate(pts):
        single = wine_surrogate.evaluate(p)
        for col in batch.columns:
            assert batch.iloc[i][col] == single[col], (
                f"batch != single at point {i}, column {col!r}: "
                f"{batch.iloc[i][col]} vs {single[col]}")


def test_batch_is_order_independent(wine_surrogate, train_doe):
    """No state may leak from one design point to the next."""
    idx = [3, 77, 140, 12, 205]
    perm = [0, 4, 1, 3, 2]
    pts = [train_doe.iloc[i].to_dict() for i in idx]

    ordered = wine_surrogate.evaluate_batch(pts).to_numpy(dtype=float)
    shuffled = wine_surrogate.evaluate_batch([pts[j] for j in perm])
    restored = shuffled.to_numpy(dtype=float)[np.argsort(perm)]

    assert np.array_equal(ordered, restored), (
        "results depend on evaluation order -- inputs are not fully "
        "re-applied between points")


def test_fresh_instance_equivalence(wine_surrogate, fmu_path, train_doe,
                                    output_names):
    """Values produced by a reused instance (inputs set after init mode)
    must equal a brand-new process with inputs set *inside* init mode."""
    idx = [5, 60, 130, 200]
    pts = [train_doe.iloc[i].to_dict() for i in idx]

    reused = wine_surrogate.evaluate_batch(pts)[output_names].to_numpy(float)

    fresh = Surrogate(fmu_path)          # new wrapper -> new Wine process
    assert fresh.backend == "wine"
    cold = fresh.evaluate_batch(pts)[output_names].to_numpy(float)

    assert np.array_equal(reused, cold), (
        f"reused vs fresh instance differ: max diff "
        f"{np.abs(reused - cold).max():.3e}")


def test_unknown_input_rejected_before_running(wine_surrogate):
    with pytest.raises(KeyError) as excinfo:
        wine_surrogate.evaluate({"nonsense_input": 1.0})
    assert "nonsense_input" in str(excinfo.value)


def test_result_frame_has_all_inputs_and_outputs(wine_surrogate, midpoint,
                                                 input_names, output_names):
    df = wine_surrogate.evaluate_batch([midpoint])
    assert len(df) == 1
    assert set(df.columns) == set(input_names) | set(output_names)
    assert list(df.columns[-len(output_names):]) == output_names
    # inputs echo back exactly what we asked for
    for k, v in midpoint.items():
        assert df.iloc[0][k] == v


def test_outputs_finite_across_domain(wine_surrogate, midpoint, train_doe):
    """Nominal point, design-box centre and a real design point must all
    produce finite numbers (Kriging extrapolation is allowed to be wrong,
    not to emit NaN/inf)."""
    points = [
        {},                                        # wrapper nominal defaults
        midpoint,
        train_doe.iloc[0].to_dict(),
    ]
    df = wine_surrogate.evaluate_batch(points)
    numeric = df.to_numpy(dtype=float)
    assert np.isfinite(numeric).all(), (
        "non-finite outputs: "
        f"{df.columns[~np.isfinite(numeric).all(axis=0)].tolist()}")


def test_raw_start_values_are_evaluable(wine_surrogate):
    """nominal=False passes the FMU's own degenerate start values (all 0)
    through untouched; must still return finite outputs."""
    out = wine_surrogate.evaluate({}, nominal=False)
    bad = {k: v for k, v in out.items() if not np.isfinite(float(v))}
    assert not bad, f"non-finite outputs at start values: {bad}"


def test_repr_summarises_model(surrogate, input_names, output_names):
    r = repr(surrogate)
    assert "18 inputs" in r and "40 outputs" in r
