"""Generate the ablation input ``mean_sim_temperature.csv``.

The ablation interpolation models use one representative simulated
temperature per thermocouple. This script averages the thermocouple
temperatures over every row of the split's ``block_*.csv`` files and writes
the result in the format expected by ``ablation_funcs.load_data_temp``:

    ,TC1,TC2,...,TC10
    sim_nom_mean,<mean>,...

Usage
-----
    python scripts/ablation/make_mean_sim_temperature.py

    # explicit locations
    python scripts/ablation/make_mean_sim_temperature.py \
        --results-dir scripts/mavm_nuisance_split/nen17_net250_na400 \
        --output scripts/mavm_nuisance_split/nen17_net250_na400/ablation/inputs/mean_sim_temperature.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SENSORS = [f"TC{i}" for i in range(1, 11)]
TEMP_COLUMNS = {tc: f"Temperature (degC), {tc}" for tc in SENSORS}

# scripts/ablation/make_mean_sim_temperature.py -> <repo>/scripts
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = SCRIPTS_DIR / "mavm_nuisance_split" / "nen17_net250_na400"


def build_mean_temperature(results_dir: Path) -> pd.DataFrame:
    """Mean simulated temperature per thermocouple over all block rows."""
    block_files = sorted(results_dir.glob("block_*.csv"))
    if not block_files:
        raise FileNotFoundError(f"No block_*.csv files found in {results_dir}")

    blocks = [pd.read_csv(path) for path in block_files]
    sim_data = pd.concat(blocks, ignore_index=True)

    missing = [col for col in TEMP_COLUMNS.values() if col not in sim_data.columns]
    if missing:
        raise KeyError(f"Missing temperature columns: {missing}")

    means = sim_data[list(TEMP_COLUMNS.values())].mean()
    means.index = SENSORS

    return means.to_frame(name="sim_nom_mean").T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Split results directory containing block_*.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <results-dir>/ablation/inputs/mean_sim_temperature.csv)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else results_dir / "ablation" / "inputs" / "mean_sim_temperature.csv"
    )

    mean_temp = build_mean_temperature(results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    mean_temp.to_csv(output)

    print(mean_temp.to_string())
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
