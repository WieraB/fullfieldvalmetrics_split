#!/usr/bin/env python3
"""Per-output accuracy report: FullSurrogate FMU vs vendor reference FEA.

Runs the FMU at every training (209) and validation (24) design point
through the Wine pipe, then prints one table row per output with:

    MAE, RMSE, MaxErr   absolute residuals (FMU - FEA), same unit as output
    R2                  coefficient of determination vs the FEA truth
    NRMSE               RMSE normalised by the output's reference range

Training rows should be at CSV precision (the Kriging model interpolates
its own training points) -- a non-trivial value there means a pipe/data
bug, not surrogate error.  Validation rows are the surrogate's genuine
prediction error and are what you report in an accuracy study.

Usage (from anywhere):

    python3 tests/accuracy_report.py                 # print tables
    python3 tests/accuracy_report.py --csv acc.csv   # also write CSV
    python3 tests/accuracy_report.py --skip-fmu      # metrics from a
                                                     # previous run's CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                              # surrogate_test_case_1/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from metrics import nrmse_by_range, r2_scores  # noqa: E402
from surrogate_fmu import Surrogate  # noqa: E402

DATA = ROOT / "train-valid-data"
FMU = ROOT / "FullSurrogate.fmu"


def evaluate(fmu_path: Path, doe: pd.DataFrame) -> pd.DataFrame:
    s = Surrogate(fmu_path)
    print(f"  backend={s.backend}, evaluating {len(doe)} points ...",
          end="", flush=True)
    df = s.evaluate_batch(doe.to_dict("records"))
    print(" done")
    return df


def metrics_table(fmu: np.ndarray, ref: np.ndarray, names) -> pd.DataFrame:
    err = fmu - ref
    return pd.DataFrame(
        {
            "MAE": np.abs(err).mean(axis=0),
            "RMSE": np.sqrt((err ** 2).mean(axis=0)),
            "MaxErr": np.abs(err).max(axis=0),
            "R2": r2_scores(fmu, ref),
            "NRMSE": nrmse_by_range(fmu, ref),
        },
        index=pd.Index(names, name="output"),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fmu", type=Path, default=FMU)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--csv", type=Path, help="write full report to CSV")
    args = ap.parse_args(argv)

    train_doe = pd.read_csv(args.data / "TrainingDOE.csv")
    train_res = pd.read_csv(args.data / "TrainingResults.csv")
    valid_doe = pd.read_csv(args.data / "ValidationDOE.csv")
    valid_res = pd.read_csv(args.data / "ValidationResults.csv")

    print(f"FMU: {args.fmu}")
    fmu_train = evaluate(args.fmu, train_doe)
    fmu_valid = evaluate(args.fmu, valid_doe)

    names = list(train_res.columns)
    got_tr = fmu_train[names].to_numpy(float)
    got_va = fmu_valid[names].to_numpy(float)

    tbl_tr = metrics_table(got_tr, train_res.to_numpy(float), names)
    tbl_va = metrics_table(got_va, valid_res.to_numpy(float), names)
    report = pd.concat({"training": tbl_tr, "validation": tbl_va}, axis=1)

    fmt = "%.4g"
    print("\n=== per-output accuracy (FMU - FEA) ===")
    print(report.to_string(float_format=lambda v: fmt % v))

    print("\n=== summary ===")
    for label, tbl, fmu, ref in [
        ("training", tbl_tr, got_tr, train_res.to_numpy(float)),
        ("validation", tbl_va, got_va, valid_res.to_numpy(float)),
    ]:
        err = np.abs(fmu - ref)
        print(
            f"  {label:<11} n={len(ref):3d}  max|err|={err.max():.4g}  "
            f"mean|err|={err.mean():.4g}  min R2={tbl['R2'].min():.5f}  "
            f"max NRMSE={tbl['NRMSE'].max():.4g}"
        )

    if args.csv:
        report.to_csv(args.csv)
        print(f"\nreport written to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
