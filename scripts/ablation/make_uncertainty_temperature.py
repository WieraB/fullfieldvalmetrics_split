"""Generate the ablation input ``pointsensors_total_uncertainty_temperature.csv``.

The ablation interpolation models need one MAVM uncertainty value per
thermocouple and convention. This script reproduces the calculation in
``scripts/main_thermocouples_pulses25X.py``:

* ``sim_plus`` / ``sim_minus`` — simulation epistemic uncertainty evaluated in
  the same E_n block that produced ``split_d+`` / ``split_d-``. Within that
  block the maximum (``sim_plus``) or minimum (``sim_minus``) CDF is selected
  by quantile sum, the nominal CDF is the pointwise mean of the sorted
  epistemic samples, and the uncertainty is the maximum horizontal CDF shift
  over ``p = 0.05..0.95``. This differs from the paper's pooled ensemble and
  keeps the simulation term consistent with the split workflow.
* ``model_form_plus`` / ``model_form_minus`` — ``split_d+`` / ``split_d-``
  from the split's ``mavm_summary.csv``.
* ``total_plus`` / ``total_minus`` — ``sim + model_form`` (the MAVM shifts are
  horizontal, so the paper's maximum total uncertainty is simply the sum).

Usage
-----
    python scripts/ablation/make_uncertainty_temperature.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# scripts/ablation/make_uncertainty_temperature.py -> <repo>/scripts
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = SCRIPTS_DIR / "mavm_nuisance_split" / "nen17_net250_na400"

# Same probability grid as main_thermocouples_pulses25X.py
P_GRID = np.linspace(0.05, 0.95, 181)

SPLIT_ROWS = {
    "model_form_plus": "split_d+",
    "model_form_minus": "split_d-",
}
OUTPUT_ROWS = [
    "sim_plus",
    "sim_minus",
    "model_form_plus",
    "model_form_minus",
    "total_plus",
    "total_minus",
]


def _load_blocks(
    results_dir: Path, sensors: list[str]
) -> tuple[dict[int, np.ndarray], int]:
    """Load each split block as an ``(epistemic, aleatory, sensor)`` array.

    Each ``block_*.csv`` contains ``n_et * n_a`` rows ordered
    ``et * n_a + a``, with the aleatory inputs repeated within each
    epistemic sample. Reading only the temperature columns keeps this fast.
    """
    meta = json.loads((results_dir / "design_meta.json").read_text())
    n_et, n_a = int(meta["n_et"]), int(meta["n_a"])

    columns = [f"Temperature (degC), {sensor}" for sensor in sensors]
    block_files = sorted(results_dir.glob("block_*.csv"))
    if not block_files:
        raise FileNotFoundError(f"No block_*.csv files found in {results_dir}")

    blocks: dict[int, np.ndarray] = {}
    for path in block_files:
        en_idx = int(path.stem.rsplit("_", 1)[-1])
        block = pd.read_csv(path, usecols=columns)
        if len(block) != n_et * n_a:
            raise ValueError(
                f"{path} has {len(block)} rows, expected n_et*n_a = "
                f"{n_et}*{n_a} = {n_et * n_a}"
            )
        blocks[en_idx] = block.to_numpy().reshape(n_et, n_a, len(sensors))

    return blocks, n_a


def _block_simulation_uncertainty(
    samples: np.ndarray,
    probabilities: np.ndarray,
    side: str,
) -> float:
    """Maximum horizontal CDF shift within one E_n block.

    ``samples`` has shape ``(n_et, n_a)``. The extreme CDF is the row with the
    largest (``side="plus"``) or smallest (``side="minus"``) quantile sum.
    ``np.sort`` plus ``np.arange(1, n_a + 1) / n_a`` is exactly
    ``scipy.stats.ecdf(x).cdf`` for a sample of length ``n_a``.
    """
    ordered = np.sort(samples, axis=1)
    quantile_sums = ordered.sum(axis=1)

    if side == "plus":
        extreme = ordered[np.argmax(quantile_sums)]
        shift = (
            np.interp(P_GRID, probabilities, extreme)
            - np.interp(P_GRID, probabilities, ordered.mean(axis=0))
        )
    elif side == "minus":
        extreme = ordered[np.argmin(quantile_sums)]
        shift = (
            np.interp(P_GRID, probabilities, ordered.mean(axis=0))
            - np.interp(P_GRID, probabilities, extreme)
        )
    else:
        raise ValueError(f"Unknown side {side!r}")

    return float(np.max(shift))


def _simulation_uncertainty(
    blocks: dict[int, np.ndarray],
    n_a: int,
    summary: pd.DataFrame,
    sensors: list[str],
) -> pd.DataFrame:
    """``sim_plus`` / ``sim_minus`` in the blocks selected by the split."""
    probabilities = np.arange(1, n_a + 1) / n_a
    rows: dict[str, dict[str, float]] = {}

    for j, sensor in enumerate(sensors):
        plus_en = int(summary.loc[sensor, "split_d+_en"])
        minus_en = int(summary.loc[sensor, "split_d-_en"])

        for label, en_idx in (("plus", plus_en), ("minus", minus_en)):
            if en_idx not in blocks:
                raise KeyError(
                    f"Block {en_idx} selected for {sensor} ({label}) is missing"
                )

        rows[sensor] = {
            "sim_plus": _block_simulation_uncertainty(
                blocks[plus_en][:, :, j], probabilities, side="plus"
            ),
            "sim_minus": _block_simulation_uncertainty(
                blocks[minus_en][:, :, j], probabilities, side="minus"
            ),
        }

    return pd.DataFrame(rows).T


def build_uncertainty_table(results_dir: Path) -> pd.DataFrame:
    """Build the per-sensor uncertainty table for the ablation study."""
    summary = pd.read_csv(results_dir / "mavm_summary.csv").set_index("sensor")
    sensors = [sensor for sensor in summary.index if sensor != "CV"]

    for column in SPLIT_ROWS.values():
        if column not in summary.columns:
            raise KeyError(f"Missing column {column!r} in mavm_summary.csv")
    for column in ("split_d+_en", "split_d-_en"):
        if column not in summary.columns:
            raise KeyError(f"Missing column {column!r} in mavm_summary.csv")

    blocks, n_a = _load_blocks(results_dir, sensors)
    sim = _simulation_uncertainty(blocks, n_a, summary, sensors)
    sim.index = sensors

    table = pd.DataFrame(index=OUTPUT_ROWS, columns=sensors, dtype=float)
    table.loc["sim_plus"] = sim["sim_plus"].to_numpy()
    table.loc["sim_minus"] = sim["sim_minus"].to_numpy()
    for convention, column in SPLIT_ROWS.items():
        table.loc[convention] = summary[column].reindex(sensors).to_numpy()

    # The MAVM d+ / d- values are horizontal CDF shifts, so the maximum total
    # uncertainty is max_p(sim(p) + d) = max_p sim(p) + d.
    table.loc["total_plus"] = table.loc["sim_plus"] + table.loc["model_form_plus"]
    table.loc["total_minus"] = (
        table.loc["sim_minus"] + table.loc["model_form_minus"]
    )

    table.index.name = ""
    return table.loc[OUTPUT_ROWS, sensors]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Split results directory containing mavm_summary.csv and block_*.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output CSV (default: "
            "<results-dir>/ablation/inputs/pointsensors_total_uncertainty_temperature.csv)"
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else results_dir
        / "ablation"
        / "inputs"
        / "pointsensors_total_uncertainty_temperature.csv"
    )

    table = build_uncertainty_table(results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)

    print(table.to_string())
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
