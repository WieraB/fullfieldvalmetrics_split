"""MAVM with the epistemic set split between nuisance and transferrable uncertainties.

Built upon the Test Case 1 paper's validation process. Please refer to ``topyvale_tcs_pulse25X.py``
for the reference MAVM implementation, and to ``main_thermocouples_pulses25X.py``
for the model-form convention of step 4.


Split method example numbers:
    Keeps the E_n blocks separate.
    9 blocks (8 Sobol + the centre) for one sensor, each contributed n_et × n_a = 16 × 100 = 1600 samples. 
    Per block, the p-box bounds are the two rows (out of the 16 E_t rows) whose empirical CDF has the smallest 
    (min bound) and largest (max bound) quantile sum. 
    That's sim_bounds, and it's evaluated on a (16, 100) array.

Merged baseline example numbers:
    No E_n separation. 
    It stacks all nine blocks into one (144, 100) array and picks min/max bound rows over all 144 candidate 
    rows at once, i.e. one single epistemic box spanning the whole nuisance spread.
    E_n and E_t are grouped into one epistemic axis.


The pipeline is as follows:

1.  Every epistemic input is labelled as
    **transferrable** (E_t: carries over to the use context, e.g. material properties) /
    or **nuisance** (E_n: an artefact of the specific setup used to collect the experimental data, /
    e.g. the coil current).
2.  The surrogate is sampled on a tensor design of shape
    ``E_n x E_t x A``, meaning that for each nuisance realisation, a p-box over
    transferrable-epistemic x aleatory (common random numbers in A across
    E_n).
3.  MAVM computed like the reference pipeline (Eqs. 13-14): 
    min/max bound selection by quantile-sum, ``vm.mavm``
    over sim{min,max} x exp{min,max}) is evaluated **per E_n block**, then
    the final metric is ``max over E_n of d+ / d-``.
    d+ is the nuisance configuration that produces the largest positive deviation.
    d- is the nuisance configuration that produces the largest negative deviation (it can be a different block).
    The selected block indices are saved in ``mavm_summary.csv`` as split_d+_en and split_d-_en, respectively.
4.  The reported metric uses the paper's model-form convention
    (``model_form_bounds``): d+ against the sim-max bound row and
    d- against the sim-min bound row only, which is the quantity the paper
    publishes as ``model_form_plus``/``model_form_minus``.
    The reference convention (``mavm_over_bounds``, the max over all four
    bound combinations) is kept and can be selected with
    ``--convention reference``; the outputs always carry one convention
    only, never both.

Why: 
In the paper pipeline the nuisance and transferrable sets are pooled
into one epistemic axis, so the nuisance width (dominated by the +/-94.2 A
coil current) stretches the simulation p-box until it swallows the
experiment box and d+/d- collapse to zero. Here E_n never enters a CDF, as it only indexes the outer max.
So the metric can no longer be masked by rig-measurement uncertainty. E_n is not double-counted
(strict partition: E_n = index, E_t = sub-box, A = inside sub-box,
output-sensor errors = experiment box), and is not collapsed to a single
CDF (each block keeps its own p-box bounds).

The split is as follows:
* E_n = {I_coil_rms, F_coil, T_cool_sample}   (HIVE loading)
* E_t = {9 material scalings, HTC_sample}     (incl. Dittus-Boelter error)
* A   = {alpha_tb, beta_tb, dx, dy, dz}       (Table 1 aleatory-dominant)

Usage:
    pyvale conda env
    ~/anaconda3/envs/pyvale/bin/python scripts/mavm_nuisance_split.py

Outputs:
    ``scripts/mavm_nuisance_split/nen*_net*_na*/``:

* ``block_XX.csv``          -- one FMU batch per E_n block (inputs + outputs,
                               existing blocks are reused)
* ``design_meta.json``      -- seed, sizes, partition, containment check
* ``mavm_profile.csv``      -- d+/d- per sensor x E_n block (the profile)
                               in the selected convention (default: the
                               paper's model-form convention -- d+ vs the
                               sim-max bound row, d- vs the sim-min bound
                               row)
* ``mavm_summary.csv``      -- max over E_n + merged-box baseline per sensor
                               (like in the paper), same convention
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent          # repo root
SCRIPT_DIR = ROOT / "scripts"
SURR = ROOT / "surrogate_test_case_1"
FMU_PATH = SURR / "FullSurrogate.fmu"
EXP_DIR = ROOT / "STC_Exp_TCs_25X"
META_DIR = ROOT / "STC_ProbSim_MetaData_25X"
OUT_ROOT = SCRIPT_DIR / "mavm_nuisance_split"

for p in (str(SURR), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from surrogate_fmu import Surrogate  # noqa: E402
import valmetrics as vm  # noqa: E402  (needs the pyvale conda env)


# ---------------------------------------------------------------------------
# Uncertainty split (Table 1) formatted as (mean, uncertainty), dominant type only.
#
# E_n: epistemic, nuisance w.r.t. the context of use (HIVE loading).  
#      Table 1 gives F_coil in kHz; the FMU input is in Hz, hence 123500 +/- 500.
# E_t: epistemic, transferrable (materials; Dittus-Boelter correlation error on the coolant HTC).
# A:   aleatory-dominant inputs; normal distributions, sigma (Table 1),
#      truncated at +/-3 sigma (stays inside the training design box).
#
# * Cu_sigma_scaling is exposed by the FMU but Table 1 marks "Electrical
#   conductivity, copper" as a non-probabilistic input ("No") (refer to the repo
#   README's known-issues section). TODO: Clarify this?
# ---------------------------------------------------------------------------
E_N_BOX = {
    "I_coil_rms (A)": (1550.0, 94.2),         # Coil current [A]
    "F_coil (Hz)": (123500.0, 500.0),         # Coil frequency [kHz in Table 1]
    "T_cool_sample (degC)": (149.0, 2.2),     # Monoblock coolant temperature [degC]
}

E_T_BOX = {
    "HTC_sample": (43600.0, 4300.0),          # Monoblock coolant HTC [W m^-2 K^-1]
    "St_alpha_scaling": (1.0, 0.05),          # Thermal expansion, steel [%]
    "St_sigma_scaling": (1.0, 0.05),          # Electrical conductivity, steel [%]
    "St_k_scaling": (1.0, 0.05),              # Thermal conductivity, steel [%]
    "St_E_scaling": (1.0, 0.05),              # Young's modulus, steel [%]
    "St_nu_scaling": (1.0, 0.05),             # Poisson's ratio, steel [%]
    "St_cp_scaling": (1.0, 0.05),             # Specific heat capacity, steel [%]
    "Cu_sigma_scaling": (1.0, 0.05),          # Electrical conductivity, copper [%]*
    "Cu_k_scaling": (1.0, 0.05),              # Thermal conductivity, copper [%]
    "Cu_rho_scaling": (1.0, 0.05),            # Density, copper [%]
}

A_NORMAL = {
    "alpha_tb (deg)": (0.70, 0.47),           # Misalignment angle, alpha [deg]
    "beta_tb (deg)": (-0.32, 0.35),           # Misalignment angle, beta [deg]
    "dx (mm)": (-1.42, 0.26),                 # Misalignment, dx' [mm]
    "dy (mm)": (0.50, 0.30),                  # Misalignment, dy' [mm]
    "dz (mm)": (-0.23, 0.26),                 # Misalignment, dz' [mm]
}
# Truncation of the A-set normal draws in standard deviations.
# 3 sigma is the largest integer truncation that keeps every A
# sample inside the pinned training box.
# The binding inputs are dy and dz, the box limit is +-3.91σ, 
# and at 3σ the tightest is dz with 0.24 mm of margin.
# E_n and E_t need no truncation: they are Sobol samples over an explicitly bounded box.
A_TRUNC_SIGMA = 3.0

SENSORS = [f"TC{k}" for k in range(1, 11)] + ["CV"]
SIM_COL = {f"TC{k}": f"Temperature (degC), TC{k}" for k in range(1, 11)}
SIM_COL["CV"] = "V_coil_rms (V)"

# Number of surrogate points per Wine call.  The paper-scale run has
# 250 x 400 = 100 000 points per E_n block; one call with all of them
# would need several GB of RAM for the JSON payload.
EVAL_CHUNK = 5000


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _load_design_space():
    """Pinned training-box bounds from the FMU test code (metrics.py).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping of input name to (lower, upper) bound of the surrogate's
        training design box, as pinned in
        ``surrogate_test_case_1/tests/metrics.py``.
    """
    path = SURR / "tests" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_fmu_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DESIGN_SPACE

def _assert_inside_design_space(design):
    """Check no sample extrapolates beyond the training box.

    Parameters
    ----------
    design : dict
        Design as returned by :func:`build_design`, i.e. keys ``"E_n"``,
        ``"E_t"`` and ``"A"`` mapping to ``(names, points, is_centre)``
        tuples.

    Returns
    -------
    None
        Does nothing when every sample is contained.  Raises
        ``KeyError`` for an input missing from the pinned design space and
        ``SystemExit`` listing every input beyond the training box otherwise.
    """
    box = _load_design_space()
    violations = []
    for set_name, (names, pts, _) in design.items():
        for j, nm in enumerate(names):
            if nm not in box:
                raise KeyError(f"{nm} missing from pinned DESIGN_SPACE")
            b_lo, b_hi = box[nm]
            p_lo, p_hi = pts[:, j].min(), pts[:, j].max()
            tol = 1e-9 * (b_hi - b_lo)
            if p_lo < b_lo - tol or p_hi > b_hi + tol:
                violations.append(f"{set_name}/{nm}: [{p_lo}, {p_hi}] "
                                  f"outside [{b_lo}, {b_hi}]")
    if violations:
        raise SystemExit("Design leaves the training box:\n  "
                         + "\n  ".join(violations))

def build_design(seed: int, n_en: int, n_et: int, n_a: int):
    """Build the ``E_n x E_t x A`` tensor design.

    E_n is the best-estimate centre point first, then ``n_en``
    scrambled-Sobol points over the nuisance boxes (order stored in the
    metadata).  
    E_t is n_et scrambled-Sobol points over the
    transferrable boxes.  
    A is n_a independent and identically distributed normal draws clipped at
    +/-3 sigma, identical for every E_n block.

    Parameters
    ----------
    seed : int
        Base seed for the scrambled-Sobol draws; E_t uses `seed + 1`
        and the aleatory draws use `seed + 2`.
    n_en : int
        Number of Sobol points in the nuisance set E_n.  The returned design
        has ``n_en + 1`` E_n points because the best-estimate centre is
        included.
    n_et : int
        Number of Sobol points in the transferrable epistemic set E_t.
    n_a : int
        Number of aleatory draws per (E_n, E_t) point.

    Returns
    -------
    dict
        Keys ``"E_n"``, ``"E_t"`` and ``"A"``, each mapping to a tuple
        ``(names, points, is_centre)`` where 
        ``names`` is the list of input names, 
        ``points`` is an array of shape=(n_points, len(names)) and
        ``is_centre`` is a boolean array for E_n (``None`` for E_t and A).
        Checks if any sample leaves the training box before returning
        to avoid extrapolating the surrogate.
    """
    from scipy.stats import qmc

    # E_n: centre + Sobol
    en_names = list(E_N_BOX)
    centre = np.array([E_N_BOX[k][0] for k in en_names])
    lo = np.array([E_N_BOX[k][0] - E_N_BOX[k][1] for k in en_names])
    hi = np.array([E_N_BOX[k][0] + E_N_BOX[k][1] for k in en_names])
    sob = qmc.Sobol(d=len(en_names), scramble=True, seed=seed).random(n_en)
    en_pts = np.vstack([centre, lo + sob * (hi - lo)])
    en_is_centre = np.array([True] + [False] * n_en)

    # E_t: Sobol only (the centre is inside the sub-box by construction)
    et_names = list(E_T_BOX)
    lo_t = np.array([E_T_BOX[k][0] - E_T_BOX[k][1] for k in et_names])
    hi_t = np.array([E_T_BOX[k][0] + E_T_BOX[k][1] for k in et_names])
    sob_t = qmc.Sobol(d=len(et_names), scramble=True, seed=seed + 1).random(n_et)
    et_pts = lo_t + sob_t * (hi_t - lo_t)

    # A: truncated normals
    rng = np.random.default_rng(seed + 2)
    z = np.clip(rng.standard_normal((n_a, len(A_NORMAL))), -A_TRUNC_SIGMA,
                A_TRUNC_SIGMA)
    a_names = list(A_NORMAL)
    means = np.array([A_NORMAL[k][0] for k in a_names])
    sigmas = np.array([A_NORMAL[k][1] for k in a_names])
    a_pts = means + z * sigmas

    design = {
        "E_n": (en_names, en_pts, en_is_centre),
        "E_t": (et_names, et_pts, None),
        "A": (a_names, a_pts, None),
    }
    _assert_inside_design_space(design)
    return design

def build_rows(design, en_idx: int) -> list[dict]:
    """Build all ``E_t x A`` input dicts for one E_n block.

    Rows are emitted in row-major order, ``((et * n_a) + a)``, which are then 
    reshaped to ``(n_et, n_a)``.

    Parameters
    ----------
    design : dict
        Design as returned by :func:`build_design`, i.e. keys ``"E_n"``,
        ``"E_t"`` and ``"A"`` mapping to ``(names, points, is_centre)``
        tuples.
    en_idx : int
        Index of the E_n realisation to hold fixed, in ``0..n_en_total - 1``
        (index 0 is the best-estimate centre point).

    Returns
    -------
    list[dict]
        One input dict per surrogate evaluation, each with all 18 FMU input
        names set, length is ``n_et * n_a``.
    """
    (en_names, en_pts, _), (et_names, et_pts, _), (a_names, a_pts, _) = (
        design["E_n"], design["E_t"], design["A"])
    en = dict(zip(en_names, en_pts[en_idx]))
    rows = []
    for et in et_pts:
        etd = dict(zip(et_names, et))
        for a in a_pts:
            ad = dict(zip(a_names, a))
            rows.append({**en, **etd, **ad})
    all_names = set(E_N_BOX) | set(E_T_BOX) | set(A_NORMAL)
    missing = all_names - set(rows[0])
    if missing:
        raise RuntimeError(f"partition does not cover inputs: {sorted(missing)}")
    return rows


# ---------------------------------------------------------------------------
# Experiment functions (copied from the reference pipeline)
# ---------------------------------------------------------------------------
def load_experiment():
    """Load the experimental pulse data and its epistemic errors.

    Reads the steady-state fields of pulses 253/254/255 (TCs 1 and 3-10 from
    the DIC files, TC2 from HIVE, CV from PICO) and the sensor epistemic
    errors from ``STC2_1550A_SteadySummary.csv``, applying the reference
    pipeline's convention of moving the first error entry (coil voltage) to
    the end.

    Returns
    -------
    exp_bounds : dict[str, dict[str, np.ndarray]]
        Per sensor, keys ``"nom"``, ``"min"`` and ``"max"`` holding the
        nominal measurement time series and its nominal +/- epistemic-error
        bounds (dead sensors carry NaN).
    exp_err : dict[str, float]
        Per sensor, the scalar 1-sigma epistemic error applied above.
    """
    file_pre = ["Pulse253", "Pulse254", "Pulse255"]
    file_post = ["SteadyDICData", "SteadyHIVEData", "SteadyPICOData"]
    file_slices = [slice(2, 11), slice(2, 3), slice(1, 2)]
    data_keys = [["TC1", "TC3", "TC4", "TC5", "TC6", "TC7", "TC8", "TC9",
                  "TC10"],
                 ["TC2"],
                 ["CV"]]

    exp_data: dict[str, np.ndarray] = {}
    for pp in file_pre:
        for ii, ss in enumerate(file_post):
            arr = pd.read_csv(EXP_DIR / f"{pp}_{ss}.csv").to_numpy()
            arr = arr[:, file_slices[ii]]
            for jj, kk in enumerate(data_keys[ii]):
                exp_data[kk] = arr[:, jj]

    err_df = pd.read_csv(META_DIR / "STC2_1550A_SteadySummary.csv").to_numpy()
    err = np.array(err_df[-1, -11:], dtype=np.float64)
    # reference convention: move the first entry (coil voltage) to the end
    tmp = np.copy(err)
    err[-1] = tmp[0]
    err[0:-1] = tmp[1:]

    exp_bounds = {}
    for ii, kk in enumerate(exp_data):
        exp_bounds[kk] = {
            "nom": exp_data[kk],
            "min": exp_data[kk] - err[ii],
            "max": exp_data[kk] + err[ii],
        }
    return exp_bounds, dict(zip(exp_data, err))


# ---------------------------------------------------------------------------
# MAVM with the paper's bound-selection convention
# 1. sim_bounds: quantile-sum bounds
# 2. mavm_over_bounds: max over all four bound pairs
# 3. model_form_bounds: the paper's outer-bound-only rule
# ---------------------------------------------------------------------------
def sim_bounds(block: np.ndarray) -> dict[str, np.ndarray]:
    """Select the min/max bound rows of an ``(n_et, n_a)`` block.

    Same rule as ``topyvale_tcs_pulse25X.py``: the empirical CDF whose
    quantile sum is smallest/largest is taken as the bound of the p-box.

    Parameters
    ----------
    block : np.ndarray
        Simulated outputs for one E_n block, shape=(n_et, n_a), where each
        row is the aleatory sample vector at one transferrable-epistemic
        point.

    Returns
    -------
    dict[str, np.ndarray]
        Keys ``"min"`` and ``"max"`` mapping to the selected bound rows,
        each shape=(n_a,).
    """
    sums = [float(np.sum(stats.ecdf(block[i]).cdf.quantiles))
            for i in range(block.shape[0])]
    sums = np.asarray(sums)
    return {"min": block[int(np.argmin(sums))],
            "max": block[int(np.argmax(sums))]}


def mavm_over_bounds(sim: dict[str, np.ndarray],
                     exp: dict[str, np.ndarray]) -> dict:
    """Max d+/d- over sim{min,max} x exp{min,max}, reference convention.

    Exp bounds containing NaN (dead sensors) are skipped, as in the
    reference pipeline.

    Parameters
    ----------
    sim : dict[str, np.ndarray]
        Simulated p-box bounds for one sensor and one E_n block, keys
        ``"min"`` and ``"max"``, each shape=(n_a,), as returned by
        :func:`sim_bounds`.
    exp : dict[str, np.ndarray]
        Experimental bounds for the same sensor, keys ``"nom"``, ``"min"``
        and ``"max"``, as returned by :func:`load_experiment`.

    Returns
    -------
    dict
        Keys ``"d+"`` and ``"d-"`` (the maximised metric values),
        ``"d+_stat"`` and ``"d-_stat"`` (the winning ``sim-X_exp-Y``
        combination, or ``None`` if nothing was evaluated) and
        ``"n_combos"`` (number of bound pairs passed to ``vm.mavm``).
    """
    best = {"d+": 0.0, "d-": 0.0, "d+_stat": None, "d-_stat": None,
            "n_combos": 0}
    for es in ("min", "max"):
        e_arr = exp[es]
        if np.any(np.isnan(e_arr)):
            continue
        for ss in ("min", "max"):
            res = vm.mavm(sim[ss], e_arr)
            best["n_combos"] += 1
            key = f"sim-{ss}_exp-{es}"
            if res["d+"] > best["d+"]:
                best["d+"], best["d+_stat"] = res["d+"], key
            if res["d-"] > best["d-"]:
                best["d-"], best["d-_stat"] = res["d-"], key
    return best


def model_form_bounds(sim: dict[str, np.ndarray],
                      exp: dict[str, np.ndarray]) -> dict:
    """Model-form contribution using the paper's convention.

    ``main_thermocouples_pulses25X.py`` derives the model-form uncertainty
    from the outer simulation bound only: 
    ``d+`` is measured against the sim-max bound row and 
    ``d-`` against the sim-min bound row. 
    Of the two experimental bounds it keeps the one that makes
    the resulting total bound most extreme, 
    i.e. the largest ``sum(F_ + d+)`` for ``d+`` and the smallest ``sum(F_ - d-)`` for ``d-``.

    This differs from `mavm_over_bounds` function, which maximises over all
    four bound combinations and therefore also picks up the gap between the
    experiment and the inner simulation bound.

    Parameters
    ----------
    sim : dict[str, np.ndarray]
        Simulated p-box bounds for one sensor and one E_n block, keys
        ``"min"`` and ``"max"``, each shape=(n_a,), as returned by
        :func:`sim_bounds`.
    exp : dict[str, np.ndarray]
        Experimental bounds for the same sensor, keys ``"nom"``, ``"min"``
        and ``"max"``, as returned by :func:`load_experiment`.

    Returns
    -------
    dict
        Keys ``"d+"`` and ``"d-"`` (the model-form metric values),
        ``"d+_stat"`` and ``"d-_stat"`` (the winning ``sim-X_exp-Y``
        combination, or ``None`` if nothing was evaluated) and
        ``"n_combos"`` (number of bound pairs passed to ``vm.mavm``).
    """
    plus = None
    minus = None
    n_combos = 0
    for es in ("min", "max"):
        e_arr = exp[es]
        if np.any(np.isnan(e_arr)):
            continue
        res = vm.mavm(sim["max"], e_arr)
        n_combos += 1
        check = float(np.sum(res["F_"] + res["d+"]))
        if plus is None or check > plus[0]:
            plus = (check, float(res["d+"]), f"sim-max_exp-{es}")
        res = vm.mavm(sim["min"], e_arr)
        n_combos += 1
        check = float(np.sum(res["F_"] - res["d-"]))
        if minus is None or check < minus[0]:
            minus = (check, float(res["d-"]), f"sim-min_exp-{es}")
    return {
        "d+": 0.0 if plus is None else plus[1],
        "d-": 0.0 if minus is None else minus[1],
        "d+_stat": None if plus is None else plus[2],
        "d-_stat": None if minus is None else minus[2],
        "n_combos": n_combos,
    }


# The two bound-selection conventions, both kept:
#   "model_form" -- the paper's own rule (main_thermocouples_pulses25X.py):
#                   d+ against the sim-max bound row, d- against the
#                   sim-min bound row.  This is what the paper publishes and
#                   the default of this pipeline.
#   "reference"  -- the max over all four sim x exp bound combinations
#                   (topyvale_tcs_pulse25X.py, i.e. plain vm.mavm).
# The reported numbers use one convention only; the other one stays
# available through --convention.
CONVENTIONS = {"model_form": model_form_bounds,
               "reference": mavm_over_bounds}


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------
def parse_args():
    """Parse the command-line arguments.

    Returns
    -------
    argparse.Namespace
        Fields ``n_en``, ``n_et``, ``n_a``, ``seed``, ``convention`` and
        ``force``, with the defaults documented in ``--help``.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-en", type=int, default=8,
                    help="Sobol points in the nuisance set (default 8; the "
                         "best-estimate centre adds one block)")
    ap.add_argument("--n-et", type=int, default=16,
                    help="Sobol points in the transferrable set (default 16)")
    ap.add_argument("--n-a", type=int, default=100,
                    help="aleatory draws per (E_n, E_t) point (default 100)")
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--convention", choices=sorted(CONVENTIONS),
                    default="model_form",
                    help="bound-selection convention of the reported metric: "
                         "'model_form' (the paper's own rule: d+ against the "
                         "sim-max bound row, d- against the sim-min row) or "
                         "'reference' (max over all four sim x exp bound "
                         "combinations)")
    ap.add_argument("--force", action="store_true",
                    help="regenerate samples even if block files exist")
    return ap.parse_args()


def evaluate_in_chunks(surr, rows, chunk=EVAL_CHUNK):
    """Evaluate points through the surrogate in chunks and concatenate.

    The Wine bridge sends the whole batch as one JSON payload and reads the
    whole answer back, so a single call with 100 000 points would need
    several GB of RAM on both sides.  Chunking keeps the pipe small; the
    per-point cost is unchanged because the FMU instance is re-created per
    chunk.

    Parameters
    ----------
    surr : surrogate_fmu.Surrogate
        The surrogate to evaluate.
    rows : list[dict]
        Input dicts, one per point.
    chunk : int
        Number of points per surrogate call.

    Returns
    -------
    pandas.DataFrame
        The concatenated frames of all chunks, one row per point.
    """
    frames = []
    for start in range(0, len(rows), chunk):
        frames.append(surr.evaluate_batch(rows[start:start + chunk]))
    return pd.concat(frames, ignore_index=True)


def get_blocks(out_dir, design, n_en_tot, n_et, n_a, force):
    """Evaluate missing E_n blocks through the Wine pipe.

    Existing ``block_XX.csv`` files are reused unless ``force`` is set, so an
    interrupted run continues where it stopped.

    Parameters
    ----------
    out_dir : pathlib.Path
        Results directory holding (and receiving) the ``block_XX.csv`` files.
    design : dict
        Design as returned by :func:`build_design`.
    n_en_tot : int
        Total number of E_n blocks to evaluate, including the best-estimate
        centre point.
    n_et : int
        Number of transferrable-epistemic points per block, used to validate
        the row count of reused files.
    n_a : int
        Number of aleatory draws per (E_n, E_t) point, used to validate the
        row count of reused files.
    force : bool
        If True, re-evaluate every block even when a file already exists.

    Returns
    -------
    list[pandas.DataFrame]
        One frame per E_n block, in block order, each with
        ``n_et * n_a`` rows of FMU inputs and outputs.
    """
    blocks = []
    todo = []
    for en in range(n_en_tot):
        path = out_dir / f"block_{en:02d}.csv"
        if path.is_file() and not force:
            blocks.append((en, path))
        else:
            todo.append((en, path))

    if todo:
        print(f"Evaluating {len(todo)} block(s) of {n_et * n_a} points "
              f"each through the Wine pipe...")
        surr = Surrogate(FMU_PATH)
        for en, path in todo:
            rows = build_rows(design, en)
            t0 = time.perf_counter()
            df = evaluate_in_chunks(surr, rows)
            dt = time.perf_counter() - t0
            df.to_csv(path, index=False)
            blocks.append((en, path))
            blocks.sort()
            print(f"  block {en:02d}: {len(rows)} pts in {dt:6.1f} s "
                  f"({1e3 * dt / len(rows):5.2f} ms/pt) -> {path.name}")
    else:
        print(f"All {n_en_tot} block files present -- reusing samples.")

    frames = []
    for en, path in blocks:
        df = pd.read_csv(path)
        if len(df) != n_et * n_a:
            raise SystemExit(f"{path} has {len(df)} rows, expected "
                             f"{n_et * n_a} -- stale file? use --force.")
        frames.append(df)
    return frames


def main():
    """Run the partitioned MAVM end to end and write the result CSVs.

    Builds the design, evaluates the missing E_n blocks through the Wine
    pipe, computes the MAVM per E_n block and the merged paper-style
    baseline, then writes ``design_meta.json``, ``mavm_profile.csv`` and
    ``mavm_summary.csv`` into the results directory.

    The reported numbers use the bound-selection convention selected with
    ``--convention`` (default ``model_form``, the paper's own rule); the
    other convention stays available in
    :data:`CONVENTIONS` but is not written out.

    Returns
    -------
    None
        Results are written to disk and summarised on stdout.
    """
    args = parse_args()
    n_en_tot = args.n_en + 1  # + best-estimate centre block

    print(80 * "=")
    print("MAVM with E_n (nuisance) x E_t (transferrable) x A partition")
    print(80 * "=")
    print(f"E_n={n_en_tot} (centre + {args.n_en} Sobol)  "
          f"E_t={args.n_et} Sobol  A={args.n_a}  seed={args.seed}")
    print(f"  E_n = {list(E_N_BOX)}")
    print(f"  E_t = {list(E_T_BOX)}")
    print(f"  A   = {list(A_NORMAL)}  (normal, +/-{A_TRUNC_SIGMA:g} sigma)")

    design = build_design(args.seed, args.n_en, args.n_et, args.n_a)
    print("Design-space containment: OK (all sets inside the pinned "
          "training box)")

    out_dir = OUT_ROOT / (f"nen{n_en_tot}_net{args.n_et}_na{args.n_a}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "design_meta.json"
    meta = {
        "seed": args.seed,
        "n_en_sobol": args.n_en,
        "n_en_total": n_en_tot,
        "n_et": args.n_et,
        "n_a": args.n_a,
        "row_order": "((en * n_et) + et) * n_a + a",
        "E_N_BOX": E_N_BOX,
        "E_T_BOX": E_T_BOX,
        "A_NORMAL": {k: {"mean": v[0], "sigma": v[1]}
                     for k, v in A_NORMAL.items()},
        "A_TRUNC_SIGMA": A_TRUNC_SIGMA,
        "E_n_centre_first": True,
    }
    if meta_path.is_file() and not args.force:
        old = json.loads(meta_path.read_text())
        # compare through JSON so that tuples and lists match
        if old != json.loads(json.dumps(meta)):
            raise SystemExit(f"{meta_path} describes a different design -- "
                             f"use --force to regenerate.")
    else:
        meta_path.write_text(json.dumps(meta, indent=2))

    # -- stage 1: surrogate samples (resumable) ----------------------------
    frames = get_blocks(out_dir, design, n_en_tot, args.n_et, args.n_a,
                        args.force)
    missing_cols = [c for c in SIM_COL.values() if c not in frames[0].columns]
    if missing_cols:
        raise SystemExit(f"FMU output columns missing: {missing_cols}\n"
                         f"available: {list(frames[0].columns)}")

    # flat -> (en, et, a) per sensor
    data = {}
    for kk in SENSORS:
        flat = np.concatenate([df[SIM_COL[kk]].to_numpy() for df in frames])
        data[kk] = flat.reshape(n_en_tot, args.n_et, args.n_a)

    # -- stage 2: experiment ----------------------------------------------
    exp_bounds, exp_err = load_experiment()
    print(f"Experiment loaded: {len(exp_bounds)} sensors "
          f"(pulses 253/254/255 steady state)")

    # -- stage 3: MAVM per E_n block --------------------------------------
    convention = CONVENTIONS[args.convention]
    print(80 * "-")
    print("MAVM per E_n block (sub-box = E_t x A) + merged paper-style box")
    print(f"  convention: {args.convention}")
    profile_rows = []
    summary_rows = []
    for kk in SENSORS:
        if kk not in exp_bounds:
            continue
        exp = exp_bounds[kk]
        per_en = []
        for en in range(n_en_tot):
            bounds = sim_bounds(data[kk][en])
            res = convention(bounds, exp)
            if res["n_combos"] == 0:
                continue
            per_en.append((en, res))
            profile_rows.append({
                "sensor": kk,
                "en_idx": en,
                "en_is_centre": bool(en == 0),
                **{k: design["E_n"][1][en, j]
                   for j, k in enumerate(design["E_n"][0])},
                "d+": res["d+"],
                "d+_stat": res["d+_stat"],
                "d-": res["d-"],
                "d-_stat": res["d-_stat"],
            })

        if not per_en:
            print(f"  {kk}: excluded (experiment data contains NaN)")
            continue

        # split metric: max over E_n of d+/d-
        en_dplus, dplus = max(((en, r["d+"]) for en, r in per_en),
                              key=lambda t: t[1])
        en_dminus, dminus = max(((en, r["d-"]) for en, r in per_en),
                                key=lambda t: t[1])
        dplus_stat = next(r["d+_stat"] for en, r in per_en if en == en_dplus)
        dminus_stat = next(r["d-_stat"] for en, r in per_en
                           if en == en_dminus)

        # paper-style baseline: one merged epistemic box over ALL samples
        merged_block = data[kk].reshape(n_en_tot * args.n_et, args.n_a)
        merged_bounds = sim_bounds(merged_block)
        merged = convention(merged_bounds, exp)

        summary_rows.append({
            "sensor": kk,
            "convention": args.convention,
            "split_d+": dplus,
            "split_d+_en": en_dplus,
            "split_d+_stat": dplus_stat,
            "split_d-": dminus,
            "split_d-_en": en_dminus,
            "split_d-_stat": dminus_stat,
            "merged_d+": merged["d+"],
            "merged_d+_stat": merged["d+_stat"],
            "merged_d-": merged["d-"],
            "merged_d-_stat": merged["d-_stat"],
            "d+_gain_vs_merged": dplus - merged["d+"],
            "d-_gain_vs_merged": dminus - merged["d-"],
        })

    profile = pd.DataFrame(profile_rows)
    summary = pd.DataFrame(summary_rows)
    # the paper convention keeps the plain file names; the reference
    # convention is written next to it, never over it
    suffix = "" if args.convention == "model_form" else f"_{args.convention}"
    profile_path = out_dir / f"mavm_profile{suffix}.csv"
    summary_path = out_dir / f"mavm_summary{suffix}.csv"
    profile.to_csv(profile_path, index=False)
    summary.to_csv(summary_path, index=False)

    # -- report ------------------------------------------------------------
    print(80 * "-")
    print(f"{'sensor':>7} | {'split max d+':>12} {'@E_n':>4} | "
          f"{'split max d-':>12} {'@E_n':>4} | {'merged d+':>10} "
          f"{'merged d-':>10}")
    print(80 * "-")
    for _, r in summary.iterrows():
        print(f"{r['sensor']:>7} | {r['split_d+']:>12.4g} {int(r['split_d+_en']):>4} | "
              f"{r['split_d-']:>12.4g} {int(r['split_d-_en']):>4} | "
              f"{r['merged_d+']:>10.4g} {r['merged_d-']:>10.4g}")
    print(80 * "-")
    print("split = max over E_n of d+/d- (this method);  "
          "merged = paper-style single epistemic box, same samples")
    print(f"convention: {args.convention}")
    print(f"Files: {out_dir}")
    print(f"  design_meta.json | {profile_path.name} | {summary_path.name}")


if __name__ == "__main__":
    main()
