"""Check the metric implementation against the paper's published numbers.

Rebuilds the bound rows from the paper's own FEA samples
(``STC_ProbSim_FieldsFull_25X_v4/SamplingResultsOnlyPointSensors.csv``),
recomputes the eight-row MAVM table with ``vm.mavm`` and compares it with
``images_pointsensors_pulse25X_v4/pointsensors_mavm.csv``. It then applies
:func:`mavm_nuisance_split.model_form_bounds` and compares the result with
the ``model_form_plus``/``model_form_minus`` rows of
``images_pointsensors_pulse25X_v4/``
``pointsensors_total_uncertainty_temperature.csv``.

Usage:
    ~/anaconda3/envs/pyvale/bin/python scripts/validate_against_paper.py

Exit code is 0 when both tables are reproduced to within 1e-9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mavm_nuisance_split as mns
import valmetrics as vm  # noqa: F401  (imported for the side effect of the
#                         module being the one the pipeline uses)

ROOT = Path(__file__).resolve().parent.parent
SIM_CSV = ROOT / "STC_ProbSim_FieldsFull_25X_v4" \
    / "SamplingResultsOnlyPointSensors.csv"
PAPER_DIR = ROOT / "images_pointsensors_pulse25X_v4"
PAPER_MAVM = PAPER_DIR / "pointsensors_mavm.csv"
PAPER_TOTAL = PAPER_DIR / "pointsensors_total_uncertainty_temperature.csv"

# the paper's sampling sizes (main_thermocouples_pulses25X.py)
EPIS_N = 250
ALEA_N = 400

# column of the point-sensor sampling file per sensor
SIM_KEYS = {"TC1": 0, "TC2": 1, "TC3": 2, "TC4": 3, "TC5": 4, "TC6": 5,
            "TC7": 6, "TC8": 7, "TC9": 8, "TC10": 9, "CV": 10}

TOL = 1e-9


def check_mavm_table(sim: np.ndarray, exp_bounds: dict,
                     paper_mavm: pd.DataFrame) -> float:
    """Recompute the paper's eight-row MAVM table and compare it.

    Parameters
    ----------
    sim : np.ndarray
        Point-sensor sampling results, shape=(n_rows, n_sensors).
    exp_bounds : dict[str, dict[str, numpy.ndarray]]
        Experimental bounds per sensor, as returned by
        :func:`mavm_nuisance_split.load_experiment`.
    paper_mavm : pandas.DataFrame
        The paper's ``pointsensors_mavm.csv``, indexed by bound combination.

    Returns
    -------
    float
        The largest absolute difference found between the recomputed and the
        published values.
    """
    worst = 0.0
    for kk, col in SIM_KEYS.items():
        if kk not in exp_bounds or kk not in paper_mavm.columns:
            continue
        exp = exp_bounds[kk]
        if np.any(np.isnan(exp["nom"])):
            continue
        bounds = mns.sim_bounds(sim[:, col].reshape(EPIS_N, ALEA_N))
        for ss in ("min", "max"):
            for es in ("min", "max"):
                res = vm.mavm(bounds[ss], exp[es])
                for key in ("d+", "d-"):
                    ref = paper_mavm.loc[f"sim-{ss}_exp-{es}-{key}", kk]
                    worst = max(worst, abs(res[key] - ref))
    return worst


def check_model_form(sim: np.ndarray, exp_bounds: dict,
                     paper_total: pd.DataFrame) -> float:
    """Compare the model-form selection with the paper's published rows.

    Parameters
    ----------
    sim : np.ndarray
        Point-sensor sampling results, shape=(n_rows, n_sensors).
    exp_bounds : dict[str, dict[str, numpy.ndarray]]
        Experimental bounds per sensor, as returned by
        :func:`mavm_nuisance_split.load_experiment`.
    paper_total : pandas.DataFrame
        The paper's ``pointsensors_total_uncertainty_temperature.csv``.

    Returns
    -------
    float
        The largest absolute difference found between the recomputed and the
        published model-form values.
    """
    worst = 0.0
    for kk, col in SIM_KEYS.items():
        if kk not in exp_bounds or kk not in paper_total.columns:
            continue
        exp = exp_bounds[kk]
        if np.any(np.isnan(exp["nom"])):
            continue
        bounds = mns.sim_bounds(sim[:, col].reshape(EPIS_N, ALEA_N))
        mf = mns.model_form_bounds(bounds, exp)
        worst = max(worst,
                    abs(mf["d+"] - paper_total.loc["model_form_plus", kk]))
        worst = max(worst,
                    abs(mf["d-"] - paper_total.loc["model_form_minus", kk]))
    return worst


def main() -> None:
    """Run both checks and report the deviations.

    Returns
    -------
    None
        The deviations are printed; a non-zero exit status signals a
        mismatch.
    """
    sim = pd.read_csv(SIM_CSV).to_numpy()
    exp_bounds, _ = mns.load_experiment()
    paper_mavm = pd.read_csv(PAPER_MAVM, index_col=0)
    paper_total = pd.read_csv(PAPER_TOTAL, index_col=0)

    d_table = check_mavm_table(sim, exp_bounds, paper_mavm)
    print(f"paper MAVM table (8 rows x sensors): max |difference| = "
          f"{d_table:.3e}")

    d_mf = check_model_form(sim, exp_bounds, paper_total)
    print(f"paper model_form_plus/minus:         max |difference| = "
          f"{d_mf:.3e}")

    if max(d_table, d_mf) > TOL:
        raise SystemExit(f"FAILED: deviation above {TOL:g}")
    print("OK: the metric implementation reproduces the published values.")


if __name__ == "__main__":
    main()
