#!/usr/bin/env python3
"""Compare pooled and split uncertainty-metric interpolation results from ablation studies.

The split study is here::

    scripts/mavm_nuisance_split/nen17_net250_na400/ablation/outputs/

The pooled study is in the corresponding ``interp_*`` / ``ipm_*``
directories of the main ``fullfieldvalmetrics_new`` repository.

Both studies contain one ``ablation_summary_<config>.csv`` per model
configuration.  Each summary has one row per ``d_type`` and columns such as
``mean_rel_error``, ``RMSE``, ``mean_interval_score`` and ``pi_coverage``.

This script pairs the model configurations that exist in both studies and
writes:

* the uncertainty values,
* paired model-performance statistics,
* per-family / per-model-type summary tables,
* best-model comparisons,
* a set of comparison figures.

Run from anywhere::

    ~/anaconda3/envs/pyvale/bin/python \
        scripts/ablation_plots/compare_pooled_vs_split.py

The default output directory is ``scripts/ablation_plots/comparison_outputs``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import warnings as warnings_module
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


# ---------------------------------------------------------------------------
# Study definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    key: str
    dirname: str
    model_type: str
    domain: str
    label: str
    short: str


FAMILIES: tuple[Family, ...] = (
    Family("reg_spatial", "interp_reg_spatial", "reg", "spatial", "Regression (spatial)", "Reg. spatial"),
    Family("reg_temp", "interp_reg_temp", "reg", "temp", "Regression (temperature)", "Reg. temp"),
    Family("gpr_spatial", "interp_gpr_spatial", "gpr", "spatial", "GPR (spatial)", "GPR spatial"),
    Family("gpr_temp", "interp_gpr_temp", "gpr", "temp", "GPR (temperature)", "GPR temp"),
    Family("ipm_spatial", "ipm_interp_spatial", "ipm", "spatial", "IPM (spatial)", "IPM spatial"),
    Family("ipm_temp", "ipm_interp_temp", "ipm", "temp", "IPM (temperature)", "IPM temp"),
)

FAMILY_BY_KEY = {family.key: family for family in FAMILIES}
FAMILY_ORDER = [family.short for family in FAMILIES]

D_TYPE_ORDER = [
    "sim_plus",
    "sim_minus",
    "model_form_plus",
    "model_form_minus",
    "total_plus",
    "total_minus",
]

D_TYPE_LABELS = {
    "sim_plus": "Simulation plus",
    "sim_minus": "Simulation minus",
    "model_form_plus": "Model-form plus",
    "model_form_minus": "Model-form minus",
    "total_plus": "Total plus",
    "total_minus": "Total minus",
}

D_TYPE_LABELS_SHORT = {
    "sim_plus": "sim +",
    "sim_minus": "sim -",
    "model_form_plus": "model form +",
    "model_form_minus": "model form -",
    "total_plus": "total +",
    "total_minus": "total -",
}


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    unit: str
    better: str  # "lower", "higher" or "coverage"
    pct_change: bool = False
    short: str = ""


METRICS: tuple[Metric, ...] = (
    Metric("mean_rel_error", "Mean relative error", "", "lower", True, "MRE"),
    Metric("RMSE", "RMSE", "°C", "lower", True, "RMSE"),
    Metric("MAE", "MAE", "°C", "lower", True, "MAE"),
    Metric("MAPE", "MAPE", "%", "lower", True, "MAPE"),
    Metric("mean_pi_error", "Mean PI error", "°C", "lower", True, "PI error"),
    Metric("mean_pi_width", "Mean PI width", "°C", "lower", True, "PI width"),
    Metric("mean_interval_score", "Mean interval score", "", "lower", True, "Interval score"),
    Metric("pi_coverage", "PI coverage", "", "coverage", False, "Coverage"),
    Metric("pi_coverage_error", "PI coverage error |c-0.95|", "", "lower", False, "Coverage error"),
    Metric("accuracy", "Accuracy (10% tolerance)", "", "higher", False, "Accuracy"),
)

METRIC_BY_NAME = {metric.name: metric for metric in METRICS}
METRIC_ORDER = [metric.name for metric in METRICS]
PRIMARY_METRICS = ["mean_rel_error", "RMSE", "mean_interval_score", "pi_coverage", "pi_coverage_error", "accuracy"]
ERROR_METRICS = ["mean_rel_error", "RMSE", "MAE", "MAPE", "mean_pi_error", "mean_pi_width", "mean_interval_score"]

COLUMN_ALIASES = {
    "accuracy": ("accuracy", "accuracy_2pct"),
    "accuracy_matrix": ("accuracy_matrix", "accuracy_2pct_matrix"),
}

POOLED_COLOR = "#4C78A8"
SPLIT_COLOR = "#F58518"
INCREASE_COLOR = "#D55E00"
DECREASE_COLOR = "#0072B2"
TIE_COLOR = "#9E9E9E"

FAMILY_PALETTE = {
    family.short: color
    for family, color in zip(
        FAMILIES,
        sns.color_palette("colorblind", n_colors=len(FAMILIES)),
    )
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the pooled-vs-split comparison.

    Parameters
    ----------
    None

    Returns
    -------
    args : argparse.Namespace
        Parsed paths and figure settings, including ``pooled_root``,
        ``split_root``, ``pooled_target``, ``split_target``, ``outdir``,
        ``formats`` and ``dpi``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    default_pooled_root = Path.home() / "Documents" / "fullfieldvalmetrics_new"
    default_split_root = (
        repo_root
        / "scripts"
        / "mavm_nuisance_split"
        / "nen17_net250_na400"
        / "ablation"
        / "outputs"
    )
    default_pooled_target = (
        default_pooled_root
        / "images_pointsensors_pulse25X_v4"
        / "pointsensors_total_uncertainty_temperature.csv"
    )
    default_split_target = (
        repo_root
        / "scripts"
        / "mavm_nuisance_split"
        / "nen17_net250_na400"
        / "ablation"
        / "inputs"
        / "pointsensors_total_uncertainty_temperature.csv"
    )
    default_outdir = repo_root / "scripts" / "ablation_plots" / "comparison_outputs"

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pooled-root", type=Path, default=default_pooled_root)
    parser.add_argument("--split-root", type=Path, default=default_split_root)
    parser.add_argument("--pooled-target", type=Path, default=default_pooled_target)
    parser.add_argument("--split-target", type=Path, default=default_split_target)
    parser.add_argument("--outdir", type=Path, default=default_outdir)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=("png", "pdf", "svg"),
        help="Figure formats to write (default: png).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def ensure_dirs(outdir: Path) -> tuple[Path, Path]:
    """Create the table and figure output directories.

    Parameters
    ----------
    outdir : pathlib.Path
        Root output directory for the comparison.

    Returns
    -------
    table_dir : pathlib.Path
        Created ``tables`` subdirectory.
    figure_dir : pathlib.Path
        Created ``figures`` subdirectory.
    """
    table_dir = outdir / "tables"
    figure_dir = outdir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir


def tc_sort_key(tc: str) -> tuple[int, str]:
    """Return a natural sort key for a thermocouple label.

    The first embedded integer determines the numeric ordering, so
    ``TC2`` sorts before ``TC10``.  The original string is used as a
    stable tie-breaker when labels contain the same number.

    Parameters
    ----------
    tc : str
        Thermocouple identifier such as ``TC1`` or ``TC12``.

    Returns
    -------
    key : tuple[int, str]
        Numeric component followed by the original label.
    """
    digits = "".join(ch for ch in str(tc) if ch.isdigit())
    return (int(digits) if digits else 10**9, str(tc))


def sorted_tcs(tcs: Iterable[str]) -> list[str]:
    """Clean, deduplicate and naturally sort thermocouple labels.

    Parameters
    ----------
    tcs : Iterable[str]
        Thermocouple identifiers, possibly with whitespace or duplicates.

    Returns
    -------
    sorted_labels : list[str]
        Unique stripped labels in natural numeric order.
    """
    return sorted({str(tc).strip() for tc in tcs}, key=tc_sort_key)


def d_type_sort_key(d_type: str) -> tuple[int, str]:
    """Return a sort key for an uncertainty-component name.

    Parameters
    ----------
    d_type : str
        Component name such as ``total_plus`` or ``sim_minus``.

    Returns
    -------
    key : tuple[int, str]
        Position in ``D_TYPE_ORDER`` followed by the component name.
        Unknown components are placed after the known ones.
    """
    try:
        return (D_TYPE_ORDER.index(d_type), d_type)
    except ValueError:
        return (len(D_TYPE_ORDER), d_type)


def metric_sort_key(metric: str) -> tuple[int, str]:
    """Return a sort key for a performance-metric name.

    Parameters
    ----------
    metric : str
        Metric column name such as ``RMSE`` or ``mean_interval_score``.

    Returns
    -------
    key : tuple[int, str]
        Position in ``METRIC_ORDER`` followed by the metric name.
        Unknown metrics are placed after the known ones.
    """
    try:
        return (METRIC_ORDER.index(metric), metric)
    except ValueError:
        return (len(METRIC_ORDER), metric)


def family_sort_key(family_short: str) -> tuple[int, str]:
    """Return a sort key for a model-family short label.

    Parameters
    ----------
    family_short : str
        Family label such as ``Reg. spatial`` or ``IPM temp``.

    Returns
    -------
    key : tuple[int, str]
        Position in ``FAMILY_ORDER`` followed by the family label.
        Unknown families are placed after the known ones.
    """
    try:
        return (FAMILY_ORDER.index(family_short), family_short)
    except ValueError:
        return (len(FAMILY_ORDER), family_short)


def fmt_num(value: float, digits: int = 3) -> str:
    """Format a number compactly for text reports.

    Parameters
    ----------
    value : float
        Value to format.  ``None`` and non-finite values become ``n/a``.
    digits : int, default=3
        Number of significant digits used for values below 100.

    Returns
    -------
    text : str
        Human-readable number, or ``n/a`` for missing input.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.{digits}g}"


def fmt_pct(value: float, digits: int = 0) -> str:
    """Format a value as a signed percentage for text reports.

    Parameters
    ----------
    value : float
        Percentage value to format.  ``None`` and non-finite values
        become ``n/a``.
    digits : int, default=0
        Number of decimal places.

    Returns
    -------
    text : str
        Signed percentage string, or ``n/a`` for missing input.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:+.{digits}f}%"


def pct_formatter(value: float, _position: float | None = None) -> str:
    """Format an axis tick as a percentage.

    Parameters
    ----------
    value : float
        Tick value in percent.
    _position : float | None, default=None
        Matplotlib tick position; unused by this formatter.

    Returns
    -------
    text : str
        Percentage label with no decimal places.
    """
    return f"{value:.0f}%"


def save_figure(
    fig: plt.Figure,
    figure_dir: Path,
    name: str,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Save a figure in each requested format and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write.
    figure_dir : pathlib.Path
        Destination directory.
    name : str
        Stem of the output filename, without an extension.
    formats : Iterable[str]
        File extensions such as ``png``, ``pdf`` or ``svg``.
    dpi : int
        Raster resolution used for raster formats.

    Returns
    -------
    None
        The figure is written to disk and closed.
    """
    for fmt in formats:
        path = figure_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def family_legend_handles(order: list[str]) -> list[Line2D]:
    """Build compact legend handles for model families.

    Parameters
    ----------
    order : list[str]
        Family short labels in the desired legend order.

    Returns
    -------
    handles : list[matplotlib.lines.Line2D]
        Marker-only handles using the configured family colours.
    """
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=FAMILY_PALETTE[family],
            markeredgecolor="none",
            markersize=8,
            label=family,
        )
        for family in order
    ]


def source_legend_handles() -> list[Line2D]:
    """Build legend handles for the pooled and split sources.

    Parameters
    ----------
    None

    Returns
    -------
    handles : list[matplotlib.lines.Line2D]
        Circular pooled handle and square split handle.
    """
    return [
        Line2D([0], [0], marker="o", linestyle="none", color=POOLED_COLOR, markersize=8, label="Pooled"),
        Line2D([0], [0], marker="s", linestyle="none", color=SPLIT_COLOR, markersize=8, label="Split"),
    ]


# ---------------------------------------------------------------------------
# Loading and normalising the ablation summaries
# ---------------------------------------------------------------------------


def parse_config(family: Family, config: str) -> dict[str, object]:
    """Parse a configuration token from an ablation-summary filename.

    The token layout depends on the model family: regularised models use a
    single integer, GPR models use ``lr_kernel_ard_nu`` and IPM models use
    ``lr_hidden_dim``.

    Parameters
    ----------
    family : Family
        Family metadata describing the model type.
    config : str
        Configuration token taken from the summary filename.

    Returns
    -------
    parsed : dict[str, object]
        Parsed values keyed by ``lr``, ``kernel``, ``ard``, ``nu``,
        ``hidden_dim`` and ``reg_model``.  Missing values are ``None`` or
        ``numpy.nan``.
    """
    parsed: dict[str, object] = {
        "lr": np.nan,
        "kernel": None,
        "ard": None,
        "nu": None,
        "hidden_dim": np.nan,
        "reg_model": None,
    }

    if family.model_type == "reg":
        parsed["reg_model"] = int(config) if config.isdigit() else config
        return parsed

    parts = str(config).split("_")
    if family.model_type == "gpr":
        if len(parts) == 4:
            lr_text, kernel, ard_text, nu_text = parts
            parsed["lr"] = float(lr_text)
            parsed["kernel"] = kernel
            parsed["ard"] = None if ard_text == "None" else int(ard_text)
            parsed["nu"] = None if nu_text == "None" else float(nu_text)
        return parsed

    if family.model_type == "ipm":
        if len(parts) == 2:
            parsed["lr"] = float(parts[0])
            parsed["hidden_dim"] = int(parts[1])
        return parsed

    return parsed


def normalise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the column schema of an ablation-summary table.

    Alias columns such as ``accuracy_2pct`` are mapped to their canonical
    names, the ``d_type`` column is validated, and all known metric columns
    are coerced to numeric values.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw summary table read from an ``ablation_summary_*.csv`` file.

    Returns
    -------
    normalised : pandas.DataFrame
        Copy of ``df`` with canonical metric columns.  The derived
        ``pi_coverage_error`` column is added later, after pairing.
    """
    out = df.copy()
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in out.columns:
            for alias in aliases:
                if alias in out.columns:
                    out[canonical] = out[alias]
                    break
        # Drop the alias column so that all study families end up with the
        # same canonical schema (IPM uses accuracy_2pct instead of accuracy).
        aliases_to_drop = [alias for alias in aliases if alias != canonical and alias in out.columns]
        if aliases_to_drop:
            out = out.drop(columns=aliases_to_drop)

    if "d_type" not in out.columns:
        raise ValueError("Summary is missing the 'd_type' column.")

    out["d_type"] = out["d_type"].astype(str).str.strip()
    for metric in METRICS:
        # ``pi_coverage_error`` is derived after pairing; keeping it out of the
        # raw frames avoids an all-NaN column during concatenation.
        if metric.name == "pi_coverage_error":
            continue
        if metric.name not in out.columns:
            out[metric.name] = np.nan
        out[metric.name] = pd.to_numeric(out[metric.name], errors="coerce")
    return out


def load_summaries(root: Path, source: str, warnings: list[str]) -> pd.DataFrame:
    """Load all ablation summaries for one study source.

    The loader walks every configured family directory below ``root``,
    reads each ``ablation_summary_*.csv`` file, parses its configuration
    token and attaches source, family and configuration metadata.

    Parameters
    ----------
    root : pathlib.Path
        Study root containing the ``interp_*`` and ``ipm_*`` directories.
    source : str
        Source label stored in the returned frame, usually ``pooled`` or
        ``split``.
    warnings : list[str]
        Mutable list to which missing directories or empty family folders
        are reported.

    Returns
    -------
    summaries : pandas.DataFrame
        Concatenated summary rows from all available families, with
        non-finite numeric values replaced by ``numpy.nan``.

    Raises
    ------
    FileNotFoundError
        If no summary files are found under ``root``.
    """
    frames: list[pd.DataFrame] = []

    for family in FAMILIES:
        folder = root / family.dirname
        if not folder.is_dir():
            warnings.append(f"Missing {source} directory: {folder}")
            continue

        files = sorted(folder.glob("ablation_summary_*.csv"))
        if not files:
            warnings.append(f"No ablation summaries found in {folder}")
            continue

        for path in files:
            config = path.stem.removeprefix("ablation_summary_")
            df = normalise_summary(pd.read_csv(path))
            parsed = parse_config(family, config)
            df = df.assign(
                source=source,
                family=family.key,
                family_label=family.label,
                family_short=family.short,
                model_type=family.model_type,
                domain=family.domain,
                config=config,
                config_label=config,
                **parsed,
            )
            df["d_type_order"] = df["d_type"].map(D_TYPE_ORDER.index).fillna(len(D_TYPE_ORDER)).astype(int)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No {source} ablation summaries were found under {root}")

    # The summaries have slightly different optional columns (for example the
    # IPM accuracy aliases).  Pandas warns about all-NA columns during concat, but
    # the resulting frame is fully aligned by column name, so the warning is
    # not relevant here.
    with warnings_module.catch_warnings():
        warnings_module.filterwarnings(
            "ignore",
            category=FutureWarning,
            message="The behavior of DataFrame concatenation.*",
        )
        combined = pd.concat(frames, ignore_index=True)
    numeric_columns = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_columns] = combined[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return combined


# ---------------------------------------------------------------------------
# Pairing and statistics
# ---------------------------------------------------------------------------


def build_pairs(
    pooled: pd.DataFrame,
    split: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Pair common model configurations from the pooled and split studies.

    Common configurations are determined per family.  The paired wide
    table keeps one row per family, configuration and component, with
    pooled and split metric columns distinguished by suffixes.  Coverage
    error is derived here because it requires both sides of the pair.

    Parameters
    ----------
    pooled : pandas.DataFrame
        Normalised summaries from the pooled study.
    split : pandas.DataFrame
        Normalised summaries from the split study.

    Returns
    -------
    paired : pandas.DataFrame
        Wide paired metrics with ``_pooled`` and ``_split`` suffixes.
    long : pandas.DataFrame
        Long-form paired metrics, one row per configuration, component and
        metric.
    family_overview : pandas.DataFrame
        Common, pooled-only and split-only configuration counts per family.
    excluded : pandas.DataFrame
        Configurations that are present in only one of the two studies.
    common_configs : dict[str, list[str]]
        Common configuration tokens keyed by family key.

    Raises
    ------
    ValueError
        If no paired metrics can be constructed.
    """
    overview_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    common_configs: dict[str, list[str]] = {}

    for family in FAMILIES:
        pooled_configs = set(pooled.loc[pooled["family"] == family.key, "config"].astype(str))
        split_configs = set(split.loc[split["family"] == family.key, "config"].astype(str))
        common = sorted(pooled_configs & split_configs, key=lambda value: str(value))
        common_configs[family.key] = common

        overview_rows.append(
            {
                "family": family.key,
                "family_label": family.label,
                "family_short": family.short,
                "model_type": family.model_type,
                "domain": family.domain,
                "n_pooled_configs": len(pooled_configs),
                "n_split_configs": len(split_configs),
                "n_common_configs": len(common),
                "n_pooled_only": len(pooled_configs - split_configs),
                "n_split_only": len(split_configs - pooled_configs),
            }
        )

        for config in sorted(pooled_configs - split_configs):
            excluded_rows.append(
                {
                    "family": family.key,
                    "family_label": family.label,
                    "config": config,
                    "present_in": "pooled",
                    "reason": "No matching split configuration",
                }
            )
        for config in sorted(split_configs - pooled_configs):
            excluded_rows.append(
                {
                    "family": family.key,
                    "family_label": family.label,
                    "config": config,
                    "present_in": "split",
                    "reason": "No matching pooled configuration",
                }
            )

    common_df = pd.DataFrame(
        [
            (family.key, config)
            for family in FAMILIES
            for config in common_configs[family.key]
        ],
        columns=["family", "config"],
    )

    pooled_common = pooled.merge(common_df, on=["family", "config"], how="inner")
    split_common = split.merge(common_df, on=["family", "config"], how="inner")

    # Keep configuration metadata from the pooled side only; otherwise the
    # merge below would create ``lr_pooled``/``lr_split`` style columns.
    split_common = split_common.drop(
        columns=[
            "family_label",
            "family_short",
            "config_label",
            "lr",
            "kernel",
            "ard",
            "nu",
            "hidden_dim",
            "reg_model",
            "d_type_order",
        ],
        errors="ignore",
    )

    paired = pooled_common.merge(
        split_common,
        on=["family", "model_type", "domain", "config", "d_type"],
        suffixes=("_pooled", "_split"),
        validate="one_to_one",
    )

    paired = paired.assign(
        family_label=paired["family"].map({family.key: family.label for family in FAMILIES}),
        family_short=paired["family"].map({family.key: family.short for family in FAMILIES}),
        d_type_label=paired["d_type"].map(D_TYPE_LABELS).fillna(paired["d_type"]),
    )

    # Derived metric: distance of PI coverage from the nominal 95 % level.
    for suffix in ("pooled", "split"):
        paired[f"pi_coverage_error_{suffix}"] = (paired[f"pi_coverage_{suffix}"] - 0.95).abs()

    paired["d_type_order"] = paired["d_type"].map(D_TYPE_ORDER.index).fillna(len(D_TYPE_ORDER)).astype(int)
    paired["family_order"] = paired["family_short"].map(FAMILY_ORDER.index).fillna(len(FAMILY_ORDER)).astype(int)

    paired = paired.sort_values(
        ["family_order", "config", "d_type_order"],
        kind="stable",
    ).reset_index(drop=True)

    long = build_long_table(paired)
    overview = pd.DataFrame(overview_rows)
    excluded = pd.DataFrame(excluded_rows)
    return paired, long, overview, excluded, common_configs


def _better_values(pooled: pd.Series, split: pd.Series, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """Classify which source wins for each paired metric value.

    Parameters
    ----------
    pooled : pandas.Series
        Pooled metric values.
    split : pandas.Series
        Split metric values aligned with ``pooled``.
    rule : str
        Winner rule: ``lower``, ``higher`` or ``coverage``.  The
        ``coverage`` rule compares distance to the nominal 0.95 level.

    Returns
    -------
    better : numpy.ndarray
        Object array containing ``pooled``, ``split``, ``tie`` or an empty
        string when either value is missing.
    flag : numpy.ndarray
        Float array with ``-1`` for a pooled win, ``0`` for a tie, ``1``
        for a split win and ``numpy.nan`` for missing comparisons.
    """
    pooled_arr = pd.to_numeric(pooled, errors="coerce").to_numpy(dtype=float)
    split_arr = pd.to_numeric(split, errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(pooled_arr) & np.isfinite(split_arr)
    better = np.full(len(pooled_arr), "", dtype=object)
    flag = np.full(len(pooled_arr), np.nan, dtype=float)

    if rule == "coverage":
        pooled_score = np.abs(pooled_arr - 0.95)
        split_score = np.abs(split_arr - 0.95)
        tie = finite & np.isclose(pooled_score, split_score, rtol=1e-9, atol=1e-12)
        split_win = finite & (split_score < pooled_score) & ~tie
        pooled_win = finite & (pooled_score < split_score) & ~tie
    elif rule == "higher":
        tie = finite & np.isclose(pooled_arr, split_arr, rtol=1e-9, atol=1e-12)
        split_win = finite & (split_arr > pooled_arr) & ~tie
        pooled_win = finite & (split_arr < pooled_arr) & ~tie
    else:
        tie = finite & np.isclose(pooled_arr, split_arr, rtol=1e-9, atol=1e-12)
        split_win = finite & (split_arr < pooled_arr) & ~tie
        pooled_win = finite & (split_arr > pooled_arr) & ~tie

    better[tie] = "tie"
    better[split_win] = "split"
    better[pooled_win] = "pooled"
    flag[tie] = 0
    flag[split_win] = 1
    flag[pooled_win] = -1
    return better, flag


def build_long_table(paired: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide paired-metric table into long form.

    Each metric becomes a row block with common metadata, pooled and split
    values, absolute and relative changes, and the winning source.

    Parameters
    ----------
    paired : pandas.DataFrame
        Wide table returned by :func:`build_pairs`.

    Returns
    -------
    long : pandas.DataFrame
        One row per family, configuration, component and metric.  Metrics
        whose paired columns are absent are skipped.

    Raises
    ------
    ValueError
        If no paired metric columns are available.
    """
    key_columns = [
        "family",
        "family_label",
        "family_short",
        "model_type",
        "domain",
        "config",
        "config_label",
        "lr",
        "kernel",
        "ard",
        "nu",
        "hidden_dim",
        "reg_model",
        "d_type",
        "d_type_label",
        "d_type_order",
        "family_order",
    ]

    frames: list[pd.DataFrame] = []
    for metric in METRICS:
        pooled_col = f"{metric.name}_pooled"
        split_col = f"{metric.name}_split"
        if pooled_col not in paired.columns or split_col not in paired.columns:
            continue

        frame = paired[key_columns].copy()
        frame["metric"] = metric.name
        frame["metric_label"] = metric.label
        frame["unit"] = metric.unit
        frame["better_rule"] = metric.better
        frame["metric_order"] = metric_sort_key(metric.name)[0]
        frame["pooled"] = pd.to_numeric(paired[pooled_col], errors="coerce")
        frame["split"] = pd.to_numeric(paired[split_col], errors="coerce")
        frame["delta"] = frame["split"] - frame["pooled"]
        frame["abs_delta"] = frame["delta"].abs()

        if metric.pct_change:
            pct = (frame["delta"] / frame["pooled"].replace(0, np.nan)) * 100.0
        else:
            pct = pd.Series(np.nan, index=frame.index)
        frame["pct_change"] = pct.replace([np.inf, -np.inf], np.nan)

        better, flag = _better_values(frame["pooled"], frame["split"], metric.better)
        frame["better"] = better
        frame["better_flag"] = flag
        frames.append(frame)

    if not frames:
        raise ValueError("No paired metrics could be constructed.")

    long = pd.concat(frames, ignore_index=True)
    numeric_columns = long.select_dtypes(include=[np.number]).columns
    long[numeric_columns] = long[numeric_columns].replace([np.inf, -np.inf], np.nan)
    return long


def summarise_groups(long: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Summarise paired metric differences for arbitrary grouping columns.

    For each group the function reports location and spread statistics,
    absolute and relative changes, win rates, and Cohen's dz.  The result is used for family, model-type and
    overall summary tables.

    Parameters
    ----------
    long : pandas.DataFrame
        Long-form paired metrics from :func:`build_long_table`.
    group_columns : list[str]
        Columns that define the summary groups, for example
        ``['d_type', 'metric']``.

    Returns
    -------
    stats : pandas.DataFrame
        One row per group with count, mean, median, standard deviation,
        delta statistics, percent changes, win rates, and Cohen's dz.  Metric and component labels are
        added for sorting and presentation.
    """
    def summarise(group: pd.DataFrame) -> pd.Series:
        """Compute paired summary statistics for one group.

        Parameters
        ----------
        group : pandas.DataFrame
            Long-form rows for one group, containing at least ``pooled``,
            ``split``, ``pct_change`` and ``better``.

        Returns
        -------
        summary : pandas.Series
            Counts, means, medians, standard deviations, delta statistics,
            percent changes, win rates, and Cohen's dz for the group.
        """
        valid = group.dropna(subset=["pooled", "split"])
        delta = valid["split"] - valid["pooled"]
        pct = valid["pct_change"].dropna()
        better = valid["better"].dropna()

        n_pairs = int(len(valid))
        n_split = int((better == "split").sum())
        n_pooled = int((better == "pooled").sum())
        n_tie = int((better == "tie").sum())
        n_better = int(len(better))

        delta_std = float(delta.std(ddof=1)) if len(delta) > 1 else np.nan
        mean_delta = float(delta.mean()) if len(delta) else np.nan
        cohens_dz = (
            mean_delta / delta_std
            if np.isfinite(delta_std) and delta_std > 0
            else np.nan
        )

        return pd.Series(
            {
                "n_pairs": n_pairs,
                "n_pct_pairs": int(len(pct)),
                "pooled_mean": float(valid["pooled"].mean()) if len(valid) else np.nan,
                "split_mean": float(valid["split"].mean()) if len(valid) else np.nan,
                "pooled_median": float(valid["pooled"].median()) if len(valid) else np.nan,
                "split_median": float(valid["split"].median()) if len(valid) else np.nan,
                "pooled_std": float(valid["pooled"].std(ddof=1)) if len(valid) > 1 else np.nan,
                "split_std": float(valid["split"].std(ddof=1)) if len(valid) > 1 else np.nan,
                "mean_delta": mean_delta,
                "median_delta": float(delta.median()) if len(delta) else np.nan,
                "mean_abs_delta": float(delta.abs().mean()) if len(delta) else np.nan,
                "median_abs_delta": float(delta.abs().median()) if len(delta) else np.nan,
                "delta_std": delta_std,
                "delta_min": float(delta.min()) if len(delta) else np.nan,
                "delta_max": float(delta.max()) if len(delta) else np.nan,
                "delta_q05": float(delta.quantile(0.05)) if len(delta) else np.nan,
                "delta_q95": float(delta.quantile(0.95)) if len(delta) else np.nan,
                "mean_pct_change": float(pct.mean()) if len(pct) else np.nan,
                "median_pct_change": float(pct.median()) if len(pct) else np.nan,
                "n_split_better": n_split,
                "n_pooled_better": n_pooled,
                "n_ties": n_tie,
                "win_rate_split": n_split / n_better if n_better else np.nan,
                "win_rate_pooled": n_pooled / n_better if n_better else np.nan,
                "tie_rate": n_tie / n_better if n_better else np.nan,
                "cohens_dz": cohens_dz,
            }
        )

    stats = (
        long.groupby(group_columns, dropna=False, observed=True)
        .apply(summarise, include_groups=False)
        .reset_index()
    )

    stats["metric_label"] = stats["metric"].map({metric.name: metric.label for metric in METRICS})
    stats["unit"] = stats["metric"].map({metric.name: metric.unit for metric in METRICS})
    stats["better_rule"] = stats["metric"].map({metric.name: metric.better for metric in METRICS})
    stats["metric_order"] = stats["metric"].map(lambda value: metric_sort_key(value)[0])
    if "d_type" in stats.columns:
        stats["d_type_label"] = stats["d_type"].map(D_TYPE_LABELS).fillna(stats["d_type"])
        stats["d_type_order"] = stats["d_type"].map(lambda value: d_type_sort_key(value)[0])
    if "family_short" in stats.columns:
        stats["family_order"] = stats["family_short"].map(lambda value: family_sort_key(value)[0])

    for column in ("n_pairs", "n_pct_pairs", "n_split_better", "n_pooled_better", "n_ties"):
        if column in stats.columns:
            stats[column] = stats[column].astype("Int64")

    sort_columns = [
        col
        for col in ("family_order", "model_type", "d_type_order", "metric_order")
        if col in stats.columns
    ]
    if sort_columns:
        stats = stats.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return stats


def best_by_source(paired: pd.DataFrame, metric: Metric, source: str) -> pd.DataFrame:
    """Select the best configuration for one source and metric.

    The best value is chosen per family and component using the metric's
    direction: lower is better for errors and interval scores, higher is
    better for accuracy, and closest to 0.95 is best for coverage.

    Parameters
    ----------
    paired : pandas.DataFrame
        Wide paired metrics from :func:`build_pairs`.
    metric : Metric
        Metric metadata, including its winner rule.
    source : str
        Source to select from, either ``pooled`` or ``split``.

    Returns
    -------
    best : pandas.DataFrame
        Best configuration and value per family and component, including
        the distance to nominal coverage where applicable.
    """
    value_col = f"{metric.name}_{source}"
    rows: list[dict[str, object]] = []

    for (family, d_type), group in paired.groupby(["family", "d_type"], dropna=False, observed=True):
        group = group.dropna(subset=[value_col])
        if group.empty:
            continue

        if metric.better == "coverage":
            score = (group[value_col] - 0.95).abs()
            index = score.idxmin()
        elif metric.better == "higher":
            index = group[value_col].idxmax()
        else:
            index = group[value_col].idxmin()

        row = group.loc[index]
        rows.append(
            {
                "family": family,
                "family_label": row["family_label"],
                "family_short": row["family_short"],
                "model_type": row["model_type"],
                "domain": row["domain"],
                "d_type": d_type,
                "d_type_label": row["d_type_label"],
                "metric": metric.name,
                "metric_label": metric.label,
                "better_rule": metric.better,
                "source": source,
                "config": row["config"],
                "value": float(row[value_col]),
                "distance_to_target": (
                    float(abs(row[value_col] - 0.95)) if metric.better == "coverage" else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_best_models(paired: pd.DataFrame) -> pd.DataFrame:
    """Compare the best pooled and split configuration for every metric.

    "Best" is evaluated separately for each model family and uncertainty
    component, using only the configurations shared by the pooled and split
    studies.  For each metric, the pooled and split studies are ranked
    independently across those configurations.  The winning configuration
    has the lowest value for error and interval-score metrics, the highest
    value for accuracy metrics, or the value closest to the nominal 0.95
    level for PI-coverage metrics.  The resulting pooled and split winners
    are then compared with the same rule; values that are equal within the
    tolerance of ``numpy.isclose`` are recorded as a tie.

    The function records which source wins, the value difference and the
    relative change from pooled to split.

    Parameters
    ----------
    paired : pandas.DataFrame
        Wide paired metrics from :func:`build_pairs`.

    Returns
    -------
    best : pandas.DataFrame
        One row per family, component and metric with the best pooled and
        split configurations, their values, the difference and the winning
        source.
    """
    rows: list[dict[str, object]] = []

    for metric in METRICS:
        pooled_best = best_by_source(paired, metric, "pooled")
        split_best = best_by_source(paired, metric, "split")
        if pooled_best.empty or split_best.empty:
            continue

        merge_keys = [
            "family",
            "family_label",
            "family_short",
            "model_type",
            "domain",
            "d_type",
            "d_type_label",
            "metric",
            "metric_label",
            "better_rule",
        ]
        merged = pooled_best.merge(
            split_best,
            on=merge_keys,
            suffixes=("_pooled", "_split"),
        )

        for _, row in merged.iterrows():
            if metric.better == "coverage":
                pooled_score = abs(row["value_pooled"] - 0.95)
                split_score = abs(row["value_split"] - 0.95)
                if np.isclose(pooled_score, split_score, rtol=1e-9, atol=1e-12):
                    best_source = "tie"
                elif split_score < pooled_score:
                    best_source = "split"
                else:
                    best_source = "pooled"
            elif metric.better == "higher":
                if np.isclose(row["value_pooled"], row["value_split"], rtol=1e-9, atol=1e-12):
                    best_source = "tie"
                elif row["value_split"] > row["value_pooled"]:
                    best_source = "split"
                else:
                    best_source = "pooled"
            else:
                if np.isclose(row["value_pooled"], row["value_split"], rtol=1e-9, atol=1e-12):
                    best_source = "tie"
                elif row["value_split"] < row["value_pooled"]:
                    best_source = "split"
                else:
                    best_source = "pooled"

            best_delta = row["value_split"] - row["value_pooled"]
            if metric.pct_change and row["value_pooled"] != 0:
                best_pct = best_delta / row["value_pooled"] * 100.0
            else:
                best_pct = np.nan

            rows.append(
                {
                    "family": row["family"],
                    "family_label": row["family_label"],
                    "family_short": row["family_short"],
                    "model_type": row["model_type"],
                    "domain": row["domain"],
                    "d_type": row["d_type"],
                    "d_type_label": row["d_type_label"],
                    "metric": row["metric"],
                    "metric_label": row["metric_label"],
                    "better_rule": row["better_rule"],
                    "pooled_best_config": row["config_pooled"],
                    "pooled_best_value": row["value_pooled"],
                    "split_best_config": row["config_split"],
                    "split_best_value": row["value_split"],
                    "delta_best": best_delta,
                    "pct_change_best": best_pct,
                    "best_source": best_source,
                }
            )

    best = pd.DataFrame(rows)
    if best.empty:
        return best

    best["metric_order"] = best["metric"].map(lambda value: metric_sort_key(value)[0])
    best["d_type_order"] = best["d_type"].map(lambda value: d_type_sort_key(value)[0])
    best["family_order"] = best["family_short"].map(lambda value: family_sort_key(value)[0])
    best = best.sort_values(
        ["family_order", "d_type_order", "metric_order"],
        kind="stable",
    ).reset_index(drop=True)
    return best


# ---------------------------------------------------------------------------
# Target-level comparison
# ---------------------------------------------------------------------------


def read_target_table(path: Path) -> pd.DataFrame:
    """Read and clean a target-uncertainty table.

    The CSV is expected to use uncertainty components as the row index and
    thermocouple labels as columns.  Labels are stripped, values are
    coerced to numeric and infinities are replaced by ``numpy.nan``.

    Parameters
    ----------
    path : pathlib.Path
        CSV file containing target uncertainty values.

    Returns
    -------
    target : pandas.DataFrame
        Numeric target table indexed by component with thermocouples as
        columns.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def build_target_comparison(
    pooled_target: pd.DataFrame,
    split_target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare pooled and split target-uncertainty values.

    Only thermocouples and components present in both target tables are
    compared.  The function returns a long table, a component-level
    summary and a wide table with one row per thermocouple.

    Parameters
    ----------
    pooled_target : pandas.DataFrame
        Pooled target table indexed by component, with thermocouples as
        columns.
    split_target : pandas.DataFrame
        Split target table in the same layout.

    Returns
    -------
    long : pandas.DataFrame
        One row per component and common thermocouple with pooled, split,
        delta, percentage change, ratio and direction.
    summary : pandas.DataFrame
        Per-component counts and summary statistics for the target changes.
    wide : pandas.DataFrame
        One row per thermocouple with pooled, split, delta and percentage
        change columns for every component.

    Raises
    ------
    ValueError
        If the two target tables have no thermocouples in common.
    """
    tcs = sorted_tcs(set(pooled_target.columns) & set(split_target.columns))
    if not tcs:
        raise ValueError("The pooled and split target tables have no common thermocouples.")

    d_types = sorted(
        set(pooled_target.index) & set(split_target.index),
        key=d_type_sort_key,
    )

    rows: list[dict[str, object]] = []
    for d_type in d_types:
        for tc in tcs:
            pooled = pooled_target.loc[d_type, tc]
            split = split_target.loc[d_type, tc]
            delta = split - pooled
            if pd.notna(pooled) and pooled != 0:
                pct_change = delta / pooled * 100.0
                ratio = split / pooled
            else:
                pct_change = np.nan
                ratio = np.nan

            if pd.isna(delta):
                direction = "n/a"
            elif np.isclose(delta, 0.0, rtol=1e-9, atol=1e-12):
                direction = "unchanged"
            elif delta > 0:
                direction = "increase"
            else:
                direction = "decrease"

            # A percentage change is mathematically defined for any non-zero
            # pooled value, but values below 0.1 °C are dominated by rounding
            # and produce unhelpful percentages (especially model-form terms).
            pct_change_reliable = bool(pd.notna(pooled) and abs(pooled) >= 0.1)

            rows.append(
                {
                    "d_type": d_type,
                    "d_type_label": D_TYPE_LABELS.get(d_type, d_type),
                    "tc": tc,
                    "tc_order": tc_sort_key(tc)[0],
                    "pooled": pooled,
                    "split": split,
                    "delta": delta,
                    "abs_delta": abs(delta) if pd.notna(delta) else np.nan,
                    "pct_change": pct_change,
                    "pct_change_reliable": pct_change_reliable,
                    "ratio": ratio,
                    "direction": direction,
                }
            )

    long = pd.DataFrame(rows)
    long["d_type_order"] = long["d_type"].map(lambda value: d_type_sort_key(value)[0])
    long = long.sort_values(["d_type_order", "tc_order"], kind="stable").reset_index(drop=True)

    summary_rows: list[dict[str, object]] = []
    for d_type, group in long.groupby("d_type", sort=False, observed=True):
        delta = group["delta"].dropna()
        pct = group["pct_change"].dropna()
        pct_reliable = group.loc[group["pct_change_reliable"], "pct_change"].dropna()
        max_abs_index = group["abs_delta"].idxmax() if group["abs_delta"].notna().any() else None
        max_abs_tc = group.loc[max_abs_index, "tc"] if max_abs_index is not None else "n/a"
        summary_rows.append(
            {
                "d_type": d_type,
                "d_type_label": D_TYPE_LABELS.get(d_type, d_type),
                "n_tc": int(len(group)),
                "pooled_mean": group["pooled"].mean(),
                "split_mean": group["split"].mean(),
                "pooled_median": group["pooled"].median(),
                "split_median": group["split"].median(),
                "pooled_min": group["pooled"].min(),
                "pooled_max": group["pooled"].max(),
                "split_min": group["split"].min(),
                "split_max": group["split"].max(),
                "mean_delta": delta.mean(),
                "median_delta": delta.median(),
                "mean_abs_delta": delta.abs().mean(),
                "max_abs_delta": delta.abs().max(),
                "tc_max_abs_delta": max_abs_tc,
                "mean_pct_change": pct.mean() if len(pct) else np.nan,
                "median_pct_change": pct.median() if len(pct) else np.nan,
                "n_pct_defined": int(pct.notna().sum()),
                "n_pct_reliable": int(pct_reliable.notna().sum()),
                "mean_pct_change_reliable": pct_reliable.mean() if len(pct_reliable) else np.nan,
                "median_pct_change_reliable": pct_reliable.median() if len(pct_reliable) else np.nan,
                "n_increase": int((group["direction"] == "increase").sum()),
                "n_decrease": int((group["direction"] == "decrease").sum()),
                "n_unchanged": int((group["direction"] == "unchanged").sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["d_type_order"] = summary["d_type"].map(d_type_sort_key)
    summary = summary.sort_values("d_type_order", kind="stable").drop(columns="d_type_order").reset_index(drop=True)

    # Convenient wide table: one row per TC.
    wide_frames: list[pd.DataFrame] = []
    for d_type in d_types:
        sub = long[long["d_type"] == d_type].set_index("tc")
        sub = sub.reindex(tcs)
        wide_frames.append(
            sub[["pooled", "split", "delta", "pct_change"]].rename(
                columns={
                    "pooled": f"{d_type}_pooled",
                    "split": f"{d_type}_split",
                    "delta": f"{d_type}_delta",
                    "pct_change": f"{d_type}_pct_change",
                }
            )
        )
    wide = pd.concat(wide_frames, axis=1)
    wide.index.name = "tc"
    wide = wide.reset_index()

    return long, summary, wide


def validate_target_against_results(
    pooled_root: Path,
    split_root: Path,
    pooled_target: pd.DataFrame,
    split_target: pd.DataFrame,
    common_configs: dict[str, list[str]],
) -> pd.DataFrame:
    """Check target tables against measured values in ablation result files.

    For each source and family, the first common configuration is used to
    compare the ``total_plus`` target values with the ``measured`` column of
    the corresponding ``total_plus_ablation_<config>.csv`` file.

    Parameters
    ----------
    pooled_root : pathlib.Path
        Root of the pooled study outputs.
    split_root : pathlib.Path
        Root of the split study outputs.
    pooled_target : pandas.DataFrame
        Pooled target table.
    split_target : pandas.DataFrame
        Split target table.
    common_configs : dict[str, list[str]]
        Common configuration tokens keyed by family key.

    Returns
    -------
    validation : pandas.DataFrame
        One row per source and family with the number of compared
        thermocouples, maximum absolute difference and status.
    """
    rows: list[dict[str, object]] = []

    for source, root, target in (
        ("pooled", pooled_root, pooled_target),
        ("split", split_root, split_target),
    ):
        if "total_plus" not in target.index:
            continue
        target_total_plus = target.loc["total_plus"]

        for family in FAMILIES:
            configs = common_configs.get(family.key, [])
            if not configs:
                continue
            config = configs[0]
            path = root / family.dirname / "ablation_results" / f"total_plus_ablation_{config}.csv"
            if not path.exists():
                rows.append(
                    {
                        "source": source,
                        "family": family.key,
                        "config": config,
                        "n_compared": 0,
                        "max_abs_diff": np.nan,
                        "status": "result file missing",
                    }
                )
                continue

            result = pd.read_csv(path)
            if "excluded_tc" not in result.columns or "measured" not in result.columns:
                rows.append(
                    {
                        "source": source,
                        "family": family.key,
                        "config": config,
                        "n_compared": 0,
                        "max_abs_diff": np.nan,
                        "status": "unexpected result columns",
                    }
                )
                continue

            target_series = target_total_plus.rename("target")
            merged = result[["excluded_tc", "measured"]].merge(
                target_series,
                left_on="excluded_tc",
                right_index=True,
                how="inner",
            )
            if merged.empty:
                rows.append(
                    {
                        "source": source,
                        "family": family.key,
                        "config": config,
                        "n_compared": 0,
                        "max_abs_diff": np.nan,
                        "status": "no matching TCs",
                    }
                )
                continue

            max_abs_diff = float((merged["measured"] - merged["target"]).abs().max())
            rows.append(
                {
                    "source": source,
                    "family": family.key,
                    "config": config,
                    "n_compared": int(len(merged)),
                    "max_abs_diff": max_abs_diff,
                    "status": "ok" if max_abs_diff <= 1e-5 else "MISMATCH",
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def add_zero_line(ax: plt.Axes) -> None:
    """Add a horizontal zero reference line to an axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate.

    Returns
    -------
    None
        The axes is modified in place.
    """
    ax.axhline(0.0, color="0.35", linewidth=0.9, zorder=0)


def set_box_limits(ax: plt.Axes, values: pd.Series) -> None:
    """Set y limits for a box plot from robust quantiles.

    The limits are based on the 1st and 99th percentiles with 18 percent
    padding, which keeps extreme outliers from compressing the boxes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose y limits are updated.
    values : pandas.Series
        Values used to derive the limits.

    Returns
    -------
    None
        The axes limits are modified in place.  Empty input leaves the axes
        unchanged.
    """
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return
    low = float(values.quantile(0.01))
    high = float(values.quantile(0.99))
    if np.isclose(low, high):
        pad = max(abs(high) * 0.1, 1.0)
    else:
        pad = 0.18 * (high - low)
    ax.set_ylim(low - pad, high + pad)


def plot_relative_change_boxplots(
    long: pd.DataFrame,
    figure_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Plot relative error changes for the total components.

    The figure shows box plots of ``split - pooled`` percentage changes for
    mean relative error, RMSE and mean interval score, split by model family.

    Parameters
    ----------
    long : pandas.DataFrame
        Long-form paired metrics from :func:`build_long_table`.
    figure_dir : pathlib.Path
        Destination directory for the figure.
    formats : Iterable[str]
        File formats to write.
    dpi : int
        Raster resolution.

    Returns
    -------
    None
        The figure is saved as ``fig_relative_change_boxplots``.
    """
    metrics = ["mean_rel_error", "RMSE", "mean_interval_score"]
    d_types = ["total_plus", "total_minus"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for row_index, d_type in enumerate(d_types):
        for col_index, metric_name in enumerate(metrics):
            metric = METRIC_BY_NAME[metric_name]
            ax = axes[row_index, col_index]
            sub = long[(long["d_type"] == d_type) & (long["metric"] == metric_name)].copy()

            sns.boxplot(
                data=sub,
                x="family_short",
                y="pct_change",
                order=FAMILY_ORDER,
                hue="family_short",
                hue_order=FAMILY_ORDER,
                palette=FAMILY_PALETTE,
                legend=False,
                showfliers=False,
                width=0.7,
                linewidth=1.0,
                ax=ax,
            )
            add_zero_line(ax)
            set_box_limits(ax, sub["pct_change"])
            ax.set_xlabel("")
            ax.set_ylabel("Split vs pooled [%]")
            ax.set_title(f"{D_TYPE_LABELS[d_type]} | {metric.label}")
            ax.yaxis.set_major_formatter(FuncFormatter(pct_formatter))
            ax.tick_params(axis="x", rotation=22)
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

    fig.suptitle(
        "Relative change in interpolation error: split vs pooled\n"
        "Negative values indicate lower split error; outliers are not shown in the boxes",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.94))
    save_figure(fig, figure_dir, "fig_relative_change_boxplots", formats, dpi)


def plot_component_heatmap(
    family_stats: pd.DataFrame,
    figure_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Plot uncertainty component-level median relative changes as heatmaps.

    The figure shows RMSE and mean interval score changes for every
    uncertainty component and model family.

    Parameters
    ----------
    family_stats : pandas.DataFrame
        Family-level summary from :func:`summarise_groups`.
    figure_dir : pathlib.Path
        Destination directory for the figure.
    formats : Iterable[str]
        File formats to write.
    dpi : int
        Raster resolution.

    Returns
    -------
    None
        The figure is saved as ``fig_component_heatmap``.
    """
    metrics = ["RMSE", "mean_interval_score"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))

    d_type_labels = [D_TYPE_LABELS[d_type] for d_type in D_TYPE_ORDER]
    for ax, metric_name in zip(axes, metrics):
        metric = METRIC_BY_NAME[metric_name]
        sub = family_stats[family_stats["metric"] == metric_name]
        pivot = sub.pivot_table(
            index="family_short",
            columns="d_type",
            values="median_pct_change",
            aggfunc="first",
        )
        pivot = pivot.reindex(index=FAMILY_ORDER, columns=D_TYPE_ORDER)
        pivot.columns = d_type_labels

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="RdBu_r",
            center=0,
            robust=True,
            linewidths=0.6,
            linecolor="white",
            cbar_kws={"label": "Median change [%]"},
            ax=ax,
        )
        ax.set_title(metric.label)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=28)

    fig.suptitle(
        "Component-level median relative change (positive = split worse)\n",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.0, 1, 0.90))
    save_figure(fig, figure_dir, "fig_component_heatmap", formats, dpi)


def plot_best_models(
    best_models: pd.DataFrame,
    figure_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    """Plot the best pooled and split configuration per model family.

    Parameters
    ----------
    best_models : pandas.DataFrame
        Best-model table from :func:`build_best_models`.
    figure_dir : pathlib.Path
        Destination directory for the figure.
    formats : Iterable[str]
        File formats to write.
    dpi : int
        Raster resolution.

    Returns
    -------
    None
        The figure is saved as ``fig_best_models``.
    """
    d_types = ["total_plus", "total_minus"]
    metrics = ["mean_rel_error", "mean_interval_score"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    for row_index, d_type in enumerate(d_types):
        for col_index, metric_name in enumerate(metrics):
            metric = METRIC_BY_NAME[metric_name]
            ax = axes[row_index, col_index]
            sub = best_models[
                (best_models["d_type"] == d_type) & (best_models["metric"] == metric_name)
            ].set_index("family_short")
            sub = sub.reindex(FAMILY_ORDER)

            x = np.arange(len(FAMILY_ORDER))
            pooled = pd.to_numeric(sub["pooled_best_value"], errors="coerce").to_numpy(dtype=float)
            split = pd.to_numeric(sub["split_best_value"], errors="coerce").to_numpy(dtype=float)

            for x_value, pooled_value, split_value in zip(x, pooled, split):
                if np.isfinite(pooled_value) and np.isfinite(split_value):
                    ax.plot(
                        [x_value, x_value],
                        [pooled_value, split_value],
                        color="0.75",
                        linewidth=1.1,
                        zorder=1,
                    )

            ax.plot(
                x - 0.08,
                pooled,
                marker="o",
                linestyle="none",
                markersize=7,
                color=POOLED_COLOR,
                label="Pooled best",
                zorder=3,
            )
            ax.plot(
                x + 0.08,
                split,
                marker="s",
                linestyle="none",
                markersize=7,
                color=SPLIT_COLOR,
                label="Split best",
                zorder=3,
            )

            finite_pooled = pooled[np.isfinite(pooled)]
            finite_split = split[np.isfinite(split)]
            if (
                metric_name == "mean_interval_score"
                and np.all(finite_pooled > 0)
                and np.all(finite_split > 0)
            ):
                ax.set_yscale("log")

            ax.set_xticks(x, FAMILY_ORDER)
            ax.tick_params(axis="x", rotation=22)
            ax.set_ylabel(metric.label, fontsize=11)
            ax.set_title(f"{D_TYPE_LABELS[d_type]} | best {metric.label}")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

    fig.legend(
        handles=source_legend_handles(),
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=11
    )
    fig.suptitle(
        "Best configuration per model family",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(fig, figure_dir, "fig_best_models", formats, dpi)


# ---------------------------------------------------------------------------
# Text reports
# ---------------------------------------------------------------------------


def write_metric_definitions(table_dir: Path) -> None:
    """Write the metric-definition reference table.

    Parameters
    ----------
    table_dir : pathlib.Path
        Directory in which to write ``metric_definitions.csv``.

    Returns
    -------
    None
        The CSV file is written to disk.
    """
    rows = []
    descriptions = {
        "mean_rel_error": "Mean absolute relative error over the leave-one-out thermocouples.",
        "RMSE": "Root mean squared error in °C.",
        "MAE": "Mean absolute error in °C.",
        "MAPE": "Mean absolute percentage error over non-zero targets.",
        "mean_pi_error": "Mean distance of the target outside the 95 % prediction interval.",
        "mean_pi_width": "Mean 95 % prediction-interval width.",
        "mean_interval_score": "Mean strictly proper interval score; lower is better.",
        "pi_coverage": "Fraction of targets inside the 95 % prediction interval; nominal value 0.95.",
        "pi_coverage_error": "Derived metric |pi_coverage - 0.95|; lower is better.",
        "accuracy": "Fraction of predictions within 10 % relative tolerance.",
    }
    for metric in METRICS:
        rows.append(
            {
                "metric": metric.name,
                "label": metric.label,
                "unit": metric.unit,
                "better_rule": metric.better,
                "percent_change_defined": metric.pct_change,
                "description": descriptions.get(metric.name, ""),
            }
        )
    pd.DataFrame(rows).to_csv(table_dir / "metric_definitions.csv", index=False)


def write_summary_text(
    path: Path,
    target_summary: pd.DataFrame,
    overall_stats: pd.DataFrame,
    family_overview: pd.DataFrame,
    excluded: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Write the human-readable comparison summary.

    Parameters
    ----------
    path : pathlib.Path
        Destination text file.
    target_summary : pandas.DataFrame
        Component-level target comparison from
        :func:`build_target_comparison`.
    overall_stats : pandas.DataFrame
        Overall paired-performance summary from
        :func:`summarise_groups`.
    family_overview : pandas.DataFrame
        Per-family configuration counts from :func:`build_pairs`.
    excluded : pandas.DataFrame
        Configurations excluded because they occur in only one study.
    validation : pandas.DataFrame
        Target-table validation rows from
        :func:`validate_target_against_results`.

    Returns
    -------
    None
        The summary text is written to ``path``.
    """
    lines: list[str] = []
    lines.append("POOLED vs SPLIT UNCERTAINTY-METRIC INTERPOLATION")
    lines.append("=" * 62)
    lines.append("")
    lines.append("Common model configurations")
    lines.append("-" * 62)
    for _, row in family_overview.iterrows():
        lines.append(
            f"  {row['family_label']:<31} "
            f"{row['n_common_configs']:>3} common / "
            f"{row['n_pooled_configs']:>3} pooled / "
            f"{row['n_split_configs']:>3} split"
        )
    lines.append("")

    lines.append("Target uncertainty changes (split vs pooled)")
    lines.append("-" * 62)
    for d_type in ("sim_plus", "sim_minus", "model_form_plus", "model_form_minus", "total_plus", "total_minus"):
        row = target_summary[target_summary["d_type"] == d_type]
        if row.empty:
            continue
        row = row.iloc[0]
        count_text = f"{int(row['n_increase'])} up / {int(row['n_decrease'])} down"
        if d_type.startswith("model_form"):
            # Pooled model-form values are often zero or near zero, so percent
            # changes are not meaningful.  Report the absolute change instead.
            change_text = f"median Δ {fmt_num(row['median_delta'])} °C"
        else:
            change_text = f"median {fmt_pct(row['median_pct_change'])}"
        lines.append(
            f"  {D_TYPE_LABELS[d_type]:<18} "
            f"{change_text:>20}   "
            f"mean Δ {fmt_num(row['mean_delta'])} °C   "
            f"({count_text})"
        )
    lines.append("  Model-form percentages are omitted because pooled values are often near zero.")
    lines.append("")

    lines.append("Interpolation performance across all common configurations")
    lines.append("-" * 62)
    for d_type in ("total_plus", "total_minus"):
        lines.append(f"  {D_TYPE_LABELS[d_type]}")
        for metric_name in ("mean_rel_error", "mean_interval_score", "pi_coverage_error", "accuracy"):
            row = overall_stats[
                (overall_stats["d_type"] == d_type)
                & (overall_stats["metric"] == metric_name)
            ]
            if row.empty:
                continue
            row = row.iloc[0]
            win_rate = 100.0 * row["win_rate_split"] if pd.notna(row["win_rate_split"]) else np.nan
            if metric_name in {"mean_rel_error", "mean_interval_score"}:
                change_text = fmt_pct(row["median_pct_change"])
            else:
                change_text = fmt_num(row["median_delta"])
            lines.append(
                f"    {METRIC_BY_NAME[metric_name].label:<28} "
                f"median change {change_text:>9}   "
                f"split win rate {win_rate:5.1f} %"
            )
        lines.append("")

    lines.append("Component pattern (median change in RMSE / interval score)")
    lines.append("-" * 62)
    for d_type in D_TYPE_ORDER:
        rmse_row = overall_stats[
            (overall_stats["d_type"] == d_type)
            & (overall_stats["metric"] == "RMSE")
        ]
        score_row = overall_stats[
            (overall_stats["d_type"] == d_type)
            & (overall_stats["metric"] == "mean_interval_score")
        ]
        if rmse_row.empty or score_row.empty:
            continue
        lines.append(
            f"  {D_TYPE_LABELS[d_type]:<18} "
            f"RMSE {fmt_pct(rmse_row.iloc[0]['median_pct_change']):>10}   "
            f"interval score {fmt_pct(score_row.iloc[0]['median_pct_change']):>10}"
        )
    lines.append("  Negative = split lower (better) for these error/interval metrics.")
    lines.append("")

    if not excluded.empty:
        lines.append("Excluded configurations (not present in both studies)")
        lines.append("-" * 62)
        for _, row in excluded.iterrows():
            lines.append(
                f"  {row['family']:<18} {row['config']:<28} only in {row['present_in']}"
            )
        lines.append("")

    if not validation.empty:
        failures = validation[validation["status"] != "ok"]
        lines.append("Target-table validation")
        lines.append("-" * 62)
        if failures.empty:
            lines.append("  All checked target tables match the result-file 'measured' values.")
        else:
            for _, row in failures.iterrows():
                lines.append(
                    f"  {row['source']:<6} {row['family']:<18} {row['config']:<28} {row['status']}"
                )
        lines.append("")

    lines.append("Definitions")
    lines.append("-" * 62)
    lines.append("  Positive change = split value is higher than pooled value.")
    lines.append("  Error/interval metrics: lower is better; negative change means split improves.")
    lines.append("  Coverage: better means closer to 0.95; coverage error is |coverage - 0.95|.")
    lines.append("  Coverage-error changes are absolute changes in that distance.")
    lines.append("  Accuracy: higher is better; all studies use a 10 % relative tolerance.")
    lines.append("  Accuracy changes are absolute accuracy points (0-1).")
    lines.append("  Win rates are over common model configurations.")
    lines.append("  independent-sample test.")

    path.write_text("\n".join(lines) + "\n")


def write_run_info(
    path: Path,
    args: argparse.Namespace,
    warnings: list[str],
    family_overview: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    """Write run provenance and validation information.

    Parameters
    ----------
    path : pathlib.Path
        Destination text file.
    args : argparse.Namespace
        Parsed command-line options for the comparison.
    warnings : list[str]
        Warnings collected while loading the study summaries.
    family_overview : pandas.DataFrame
        Per-family configuration counts from :func:`build_pairs`.
    validation : pandas.DataFrame
        Target-table validation rows from
        :func:`validate_target_against_results`.

    Returns
    -------
    None
        The run-information text is written to ``path``.
    """
    lines = [
        "Pooled vs split comparison run",
        "=" * 40,
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Repository root: {Path(__file__).resolve().parents[2]}",
        f"Pooled root: {args.pooled_root}",
        f"Split root: {args.split_root}",
        f"Pooled target table: {args.pooled_target}",
        f"Split target table: {args.split_target}",
        f"Output directory: {args.outdir}",
        f"Figure formats: {', '.join(args.formats)}",
        f"DPI: {args.dpi}",
        "",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"pandas: {pd.__version__}",
        f"numpy: {np.__version__}",
        f"matplotlib: {matplotlib.__version__}",
        f"seaborn: {sns.__version__}",
        "",
        "Common configurations:",
    ]
    for _, row in family_overview.iterrows():
        lines.append(
            f"  {row['family']:<18} "
            f"{row['n_common_configs']:>3} common "
            f"(pooled {row['n_pooled_configs']}, split {row['n_split_configs']})"
        )
    lines.append("")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    else:
        lines.append("Warnings: none")
    lines.append("")
    if not validation.empty:
        failures = validation[validation["status"] != "ok"]
        if failures.empty:
            lines.append("Target validation: all checked tables match result 'measured' values.")
        else:
            lines.append("Target validation failures:")
            for _, row in failures.iterrows():
                lines.append(
                    f"  - {row['source']} {row['family']} {row['config']}: {row['status']}"
                )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete pooled-versus-split comparison.

    The function parses the command line, loads both studies, builds the
    paired statistics and target comparison, writes all tables, figures
    and text reports, and prints a short console summary.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Outputs are written below the configured output directory.
    """
    args = parse_args()
    args.pooled_root = args.pooled_root.expanduser().resolve()
    args.split_root = args.split_root.expanduser().resolve()
    args.pooled_target = args.pooled_target.expanduser().resolve()
    args.split_target = args.split_target.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()

    table_dir, figure_dir = ensure_dirs(args.outdir)
    warnings: list[str] = []

    print(f"Pooled root: {args.pooled_root}")
    print(f"Split root : {args.split_root}")
    print(f"Output     : {args.outdir}")

    pooled = load_summaries(args.pooled_root, "pooled", warnings)
    split = load_summaries(args.split_root, "split", warnings)

    paired, long, family_overview, excluded, common_configs = build_pairs(pooled, split)
    family_stats = summarise_groups(
        long,
        ["family", "family_label", "family_short", "model_type", "domain", "d_type", "metric"],
    )
    model_type_stats = summarise_groups(long, ["model_type", "d_type", "metric"])
    overall_stats = summarise_groups(long, ["d_type", "metric"])
    best = build_best_models(paired)

    pooled_target = read_target_table(args.pooled_target)
    split_target = read_target_table(args.split_target)
    target_long, target_summary, target_wide = build_target_comparison(pooled_target, split_target)
    validation = validate_target_against_results(
        args.pooled_root,
        args.split_root,
        pooled_target,
        split_target,
        common_configs,
    )

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    # pooled.to_csv(table_dir / "model_summaries_pooled.csv", index=False)
    # split.to_csv(table_dir / "model_summaries_split.csv", index=False)
    # paired.to_csv(table_dir / "paired_model_metrics_wide.csv", index=False)
    long.to_csv(table_dir / "paired_model_metrics_long.csv", index=False)
    family_stats.to_csv(table_dir / "family_metric_summary.csv", index=False)
    # model_type_stats.to_csv(table_dir / "model_type_metric_summary.csv", index=False)
    # overall_stats.to_csv(table_dir / "overall_metric_summary.csv", index=False)
    best.to_csv(table_dir / "best_models.csv", index=False)
    # family_overview.to_csv(table_dir / "family_overview.csv", index=False)
    # excluded.to_csv(table_dir / "excluded_configs.csv", index=False)
    # target_long.to_csv(table_dir / "target_values_by_tc.csv", index=False)
    # target_summary.to_csv(table_dir / "target_summary_by_dtype.csv", index=False)
    # target_wide.to_csv(table_dir / "target_values_wide.csv", index=False)
    # validation.to_csv(table_dir / "target_validation.csv", index=False)
    write_metric_definitions(table_dir)

    # Smaller, easier-to-read extracts for the headline total-uncertainty case.
    key_metrics = ["mean_rel_error", "RMSE", "MAE", "mean_interval_score", "pi_coverage_error", "accuracy"]
    total_d_types = ["total_plus", "total_minus"]
    family_stats_total = family_stats[
        family_stats["d_type"].isin(total_d_types)
        & family_stats["metric"].isin(key_metrics)
    ]
    best_total = best[
        best["d_type"].isin(total_d_types)
        & best["metric"].isin(key_metrics)
    ]
    # target_summary_total = target_summary[target_summary["d_type"].isin(total_d_types)]
    # family_stats_total.to_csv(table_dir / "total_metric_summary.csv", index=False)
    # best_total.to_csv(table_dir / "total_best_models.csv", index=False)
    # target_summary_total.to_csv(table_dir / "target_total_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    
    plot_relative_change_boxplots(long, figure_dir, args.formats, args.dpi)
    plot_component_heatmap(family_stats, figure_dir, args.formats, args.dpi)
    plot_best_models(best, figure_dir, args.formats, args.dpi)

    # ------------------------------------------------------------------
    # Text reports
    # ------------------------------------------------------------------
    write_summary_text(
        args.outdir / "comparison_summary.txt",
        target_summary,
        overall_stats,
        family_overview,
        excluded,
        validation,
    )
    write_run_info(
        args.outdir / "run_info.txt",
        args,
        warnings,
        family_overview,
        validation,
    )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\nCommon configurations:")
    print(family_overview[["family_label", "n_common_configs"]].to_string(index=False))
    print("\nTarget changes (median split vs pooled):")
    target_cols = ["d_type_label", "median_delta", "mean_delta", "n_increase", "n_decrease"]
    print(target_summary[target_cols].to_string(index=False, float_format=lambda value: f"{value:+.3f}"))
    print("\nWritten tables:", table_dir)
    print("Written figures:", figure_dir)
    print("Summary:", args.outdir / "comparison_summary.txt")


if __name__ == "__main__":
    main()
