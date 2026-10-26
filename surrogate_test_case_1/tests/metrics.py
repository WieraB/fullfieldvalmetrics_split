"""Shared helpers and frozen expectations for the FMU verification suite.

Tolerances here were calibrated against an actual run of the suite:

* **training** points reproduce the vendor CSV to ``5.0e-8`` absolute -- that
  is pure CSV round-trip precision (9 significant digits), i.e. the Kriging
  surrogate interpolates its 209 training points exactly, which is the
  strongest possible proof that the Wine pipe evaluates the right model at
  the right inputs.
* **validation** points differ from the vendor FEA truth by up to
  ``5.73 degC`` on ``Sample Maximum Temperature`` with ``R2 >= 0.964`` --
  that residual is the surrogate's *modelling* error, not a pipe error.
"""

from __future__ import annotations

import numpy as np

# -- frozen expectations about the datasets -------------------------------
N_TRAIN = 209
N_VALID = 24
N_INPUTS = 18
N_OUTPUTS = 40

# Observed extremes of the training design space (regression guard against
# swapped/renamed CSVs).  Tests allow +/-1 % of the span.
DESIGN_SPACE = {
    "I_coil_rms (A)": (1409.0, 1691.0),
    "F_coil (Hz)": (1.229e5, 1.241e5),
    "alpha_tb (deg)": (-1.19, 2.585),
    "beta_tb (deg)": (-1.734, 1.089),
    "dx (mm)": (-2.468, -0.3725),
    "dy (mm)": (-0.6742, 1.677),
    "dz (mm)": (-1.25, 0.7869),
    "T_cool_sample (degC)": (140.0, 155.0),
    "HTC_sample": (3.714e4, 5.009e4),
    "St_alpha_scaling": (0.9005, 1.1),
    "St_sigma_scaling": (0.9005, 1.1),
    "St_k_scaling": (0.9005, 1.1),
    "St_E_scaling": (0.9005, 1.1),
    "St_nu_scaling": (0.9005, 1.1),
    "St_cp_scaling": (0.9005, 1.1),
    "Cu_sigma_scaling": (0.9005, 1.1),
    "Cu_k_scaling": (0.9005, 1.1),
    "Cu_rho_scaling": (0.9005, 1.1),
}

# -- tolerances -----------------------------------------------------------
TRAIN_ATOL = 1e-5        # observed 5.0e-8  (CSV round-trip precision)
TRAIN_NRMSE_MAX = 1e-3   # observed 2.2e-4, normalised by each output's range

VALID_R2_MIN = 0.95      # observed 0.9645
VALID_NRMSE_MAX = 0.06   # observed 0.0405
VALID_MAX_ABS = 10.0     # observed 5.73 degC
VALID_MEAN_ABS = 0.5     # observed 0.059

SENSITIVITY_SIGN_AGREEMENT = 0.95   # observed 1.00 (92/92 strong pairs)
SENSITIVITY_RHO_CUTOFF = 0.3        # pairs this strong in the FEA data


# -- metric helpers -------------------------------------------------------
def r2_scores(fmu: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Coefficient of determination per output column (guarded against
    constant reference columns)."""
    ss_res = ((fmu - ref) ** 2).sum(axis=0)
    ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-300)


def nrmse_by_range(fmu: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """RMSE per output, normalised by that output's range in the reference
    data (so big temperatures and ~1e-4 eigenvalues weigh alike)."""
    span = ref.max(axis=0) - ref.min(axis=0)
    rmse = np.sqrt(((fmu - ref) ** 2).mean(axis=0))
    return rmse / np.maximum(span, 1e-12)


def describe_worst(err: np.ndarray, names, label: str = "error") -> str:
    row, col = np.unravel_index(int(np.argmax(err)), err.shape)
    return (
        f"worst {label} = {err[row, col]:.6g} at output {names[col]!r} "
        f"(row {int(row)})"
    )
