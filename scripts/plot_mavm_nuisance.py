"""Plots for the E_n x E_t x A MAVM pilot (please refer to mavm_nuisance_split.py).

Reads a results directory produced by ``mavm_nuisance_split.py`` and writes
these figures into ``<res-dir>/plots/``:

1. ``01_profile_per_sensor.png`` -- d+/d- vs E_n block, one panel per sensor
2. ``02_dpm_vs_current.png``     -- d+/d- vs coil current, all sensors
3. ``03_ecdfs.png``              -- p-box picture: sub-boxes, merged box,
                                    experiment box (chosen sensors)
4. ``04_distributions.png``      -- per-block aleatory spreads vs experiment
5. ``05_split_vs_paper.png``     -- split metric vs the paper's published
                                    model-form MAVM values, same-sample
                                    merged value in between
6. ``06_ecdfs_vs_paper.png``     -- split p-boxes vs pooled and experimental
                                    p-boxes (subset selectable)
7. ``10_dextremes_simcdfs.png``  -- simulation limits with the d+/d- MAVM
                                    CDF shifts; split blocks and the paper's
                                    pooled p-box by default (no experimental
                                    p-box)

``main()`` also writes the presentation subsets ``07``--``09`` of the
split-vs-paper p-box figure.

Usage (pyvale conda env):

    ~/anaconda3/envs/pyvale/bin/python scripts/plot_mavm_nuisance.py \\
        [--res-dir scripts/mavm_nuisance_split/nen9_net16_na100] \\
        [--sensors TC2 TC3] \\
        [--selected split_d+ split_d- pooled experimental] \\
        [--dextremes-selected split_d+ split_d- pooled]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "axes.edgecolor": "0.35",
            "axes.labelcolor": "0.15",
            "axes.titleweight": "semibold",
            "grid.color": "0.88",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        },
    )
    _PALETTE = sns.color_palette("deep")
except Exception:
    sns = None
    _PALETTE = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mavm_nuisance_split as mns  # noqa: E402
import valmetrics as vm  # noqa: E402

SIM_C = _PALETTE[0] if _PALETTE else "tab:blue"
SIM_C2 = _PALETTE[2] if _PALETTE else "tab:cyan"
EXP_C = _PALETTE[1] if _PALETTE else "tab:orange"
PLUS_C = _PALETTE[3] if _PALETTE else "tab:red"
MINUS_C = _PALETTE[0] if _PALETTE else "tab:blue"

# P-boxes drawn by ``fig_ecdfs_vs_paper``; also used by the CLI.
PBOX_NAMES = ("split_d+", "split_d-", "pooled", "experimental")
# Simulation p-boxes selectable in ``fig_dextremes_simcdfs``.  "pooled" is
# the paper's original simulation p-box; the split entries are the E_n blocks
# that maximise d+ and d-.
DEXTREME_NAMES = ("split_d+", "split_d-", "pooled")


def _style_axis(ax):
    """Apply the shared seaborn cosmetics to one axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to restyle in place.

    Returns
    -------
    None
    """
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if sns is not None:
        sns.despine(ax=ax, trim=False)

# Model-form tables written by main_thermocouples_pulses25X.py (paper
# pipeline).  These hold the MAVM values the paper reports per sensor:
# model_form_plus/minus are measured against the outer simulation bound
# only (sim-max bound row for d+, sim-min bound row for d-).  The
# temperature table holds the thermocouples, the all-sensor table adds CV.
PAPER_DIR = SCRIPT_DIR.parent / "images_pointsensors_pulse25X_v4"
PAPER_MODEL_FORM = PAPER_DIR / "pointsensors_total_uncertainty_temperature.csv"
PAPER_MODEL_FORM_ALL = PAPER_DIR / "pointsensors_total_uncertainty.csv"


def load_results(res_dir: Path, convention: str = "model_form"):
    """Load a results directory written by ``mavm_nuisance_split.py``.

    Parameters
    ----------
    res_dir : pathlib.Path
        Results directory holding ``mavm_profile.csv``,
        ``mavm_summary.csv``, ``design_meta.json`` and one ``block_XX.csv``
        per E_n block, numbered from ``block_00.csv``.  The reference
        convention, if it was computed, sits next to them as
        ``mavm_profile_reference.csv``/``mavm_summary_reference.csv``.
    convention : str
        Which convention to read, ``"model_form"`` (the paper's own rule,
        the default) or ``"reference"``.

    Returns
    -------
    profile : pandas.DataFrame
        MAVM d+/d- per sensor and E_n block, with the E_n input coordinates
        and the winning bound combination of each value.  The values are in
        the selected convention.
    summary : pandas.DataFrame
        Per sensor, the split metric (max over E_n) and the merged-box
        baseline, with the E_n index attaining each maximum and the name of
        the convention used.
    meta : dict
        Contents of ``design_meta.json``: seed, design sizes and partition.
    data : dict[str, numpy.ndarray]
        Per sensor, the simulated outputs of all blocks as an array of
        shape=(n_en_total, n_et, n_a).
    """
    suffix = "" if convention == "model_form" else f"_{convention}"
    profile = pd.read_csv(res_dir / f"mavm_profile{suffix}.csv")
    summary = pd.read_csv(res_dir / f"mavm_summary{suffix}.csv")
    meta = json.loads((res_dir / "design_meta.json").read_text())
    n_en = meta["n_en_total"]
    n_et = meta["n_et"]
    n_a = meta["n_a"]
    data = {}
    for kk in mns.SENSORS:
        col = mns.SIM_COL[kk]
        flat = np.concatenate(
            [pd.read_csv(res_dir / f"block_{en:02d}.csv")[col].to_numpy()
             for en in range(n_en)])
        data[kk] = flat.reshape(n_en, n_et, n_a)
    return profile, summary, meta, data


def convention_label(convention: str) -> str:
    """Describe a bound-selection convention in words.

    Parameters
    ----------
    convention : str
        Name of the convention, ``"model_form"`` or ``"reference"``.

    Returns
    -------
    str
        A short label for figure titles and axes.
    """
    if convention == "model_form":
        return ("model-form convention")
    return "reference convention"


def summary_convention(summary) -> str:
    """Read the bound-selection convention a results directory was built with.

    Parameters
    ----------
    summary : pandas.DataFrame
        Summary frame from :func:`load_results`.

    Returns
    -------
    str
        The convention name, ``"model_form"`` when the frame predates the
        ``convention`` column.
    """
    if "convention" in summary.columns and len(summary):
        return str(summary["convention"].iloc[0])
    return "model_form"


def unit_label(sensor: str, latex: bool = False) -> str:
    """Return the physical unit of a sensor's output.

    Parameters
    ----------
    sensor : str
        Sensor tag, e.g. ``"TC3"`` or ``"CV"``.
    latex : bool, optional
        Use mathtext for the degree symbol (``"[$^\\circ$C]"``) instead of
        the plain ``"[degC]"``.

    Returns
    -------
    str
        ``"[V]"`` for the coil-voltage channel ``"CV"``, otherwise
        ``"[degC]"`` (or its mathtext form when `latex` is True).
    """
    if sensor == "CV":
        return "[V]"
    return r"[$^\circ$C]" if latex else "[degC]"


def fig_profile(profile, summary, res_dir: Path):
    """Plot d+ / d- across the E_n blocks, one panel per sensor.

    Each panel shows the profile of both metric directions against the E_n
    block index (annotated with the block's coil current), marks the
    best-estimate centre block and stars the block attaining each maximum.

    Parameters
    ----------
    profile : pandas.DataFrame
        Profile frame from :func:`load_results`.
    summary : pandas.DataFrame
        Summary frame from :func:`load_results`; supplies the argmax values
        and blocks starred on each panel.
    res_dir : pathlib.Path
        Results directory; the figure is written to ``res_dir / "plots"``.

    Returns
    -------
    pathlib.Path
        Path of the saved ``01_profile_per_sensor.png``.
    """
    sensors = summary["sensor"].tolist()
    ncol = 4
    nrow = int(np.ceil(len(sensors) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(14, 3.4 * nrow),
                            layout="constrained")
    axs = np.atleast_1d(axs).ravel()

    for ax, kk in zip(axs, sensors):
        prof = profile[profile["sensor"] == kk].sort_values("en_idx")
        x = prof["en_idx"].to_numpy()
        dplus = prof["d+"].to_numpy()
        dminus = prof["d-"].to_numpy()

        ax.plot(x, dplus, "o-", color=PLUS_C, lw=1.4, ms=5,
                label=r"max $d^+$")
        ax.plot(x, dminus, "s--", color=MINUS_C, lw=1.4, ms=5,
                label=r"max $d^-$")
        # headroom for the annotations, which always sit above their point
        ax.margins(y=0.15)

        # best-estimate centre block
        ax.axvline(0, color="grey", lw=0.8, ls=":",
                   label=r"centre (nominal $E_n$)")
        # ax.annotate("centre", (0, ax.get_ylim()[1]), fontsize=7,
        #             color="grey", ha="left", va="top")

        # argmax markers
        row = summary[summary["sensor"] == kk].iloc[0]
        for val, en, col, txt in ((row["split_d+"], int(row["split_d+_en"]),
                                   PLUS_C, r"overall max $d^+$"),
                                  (row["split_d-"], int(row["split_d-_en"]),
                                   MINUS_C, r"overall max $d^-$")):
            ax.plot(en, val, "*", color=col, ms=14, zorder=5, label=txt)
            ax.annotate(f"{val:.1f}", (en, val), fontsize=7,
                        color=col, ha="center", va="bottom",
                        xytext=(0, 8), textcoords="offset points")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{en}" for en in x], fontsize=7)
        # coil current, rotated so that neighbouring values do not collide
        for en, ii in zip(x, prof["I_coil_rms (A)"]):
            ax.annotate(f"{ii:.0f} A", (en, 0),
                        xycoords=ax.get_xaxis_transform(),
                        xytext=(0, -22), textcoords="offset points",
                        rotation=45, ha="right", va="top", fontsize=7)
        ax.set_title(kk, fontsize=10)
        ax.set_xlabel(r"$E_n$ box (index, coil current)", fontsize=8,
                      labelpad=30)
        ax.set_ylabel(rf"MAVM $d^\pm$ {unit_label(kk, latex=True)}",
                      fontsize=8)
        ax.tick_params(labelsize=8)
        _style_axis(ax)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, ncol=5,
               loc="outside lower center")
    # fig.suptitle("MAVM profile over E_n boxes "
    #              "(E_t x A per box)\n"
    #              + convention_label(summary_convention(summary)),
    #              fontsize=11)
    fig.suptitle(r"MAVM profile over $E_n$ boxes "
                 r"($E_t \times A$ per box)"
                 "\n",
                 fontsize=11)
    out = res_dir / "plots" / "01_profile_per_sensor.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_vs_current(profile, summary, res_dir: Path):
    """Plot d+ and d- against the nuisance coil current, all sensors.

    Two panels share the coil-current axis and overlay every sensor, showing
    the metric direction that dominates at low current (experiment hotter
    than simulation) and at high current (simulation hotter than
    experiment). The spread about the trend comes from F_coil and
    T_cool_sample co-varying in the Sobol design.

    Parameters
    ----------
    profile : pandas.DataFrame
        Profile frame from :func:`load_results`.
    summary : pandas.DataFrame
        Summary frame from :func:`load_results`; supplies the sensor list.
    res_dir : pathlib.Path
        Results directory; the figure is written to ``res_dir / "plots"``.

    Returns
    -------
    pathlib.Path
        Path of the saved ``02_dpm_vs_current.png``.
    """
    sensors = summary["sensor"].tolist()
    colors = (sns.color_palette("tab10", len(sensors)) if sns is not None
              else plt.cm.tab10(np.linspace(0, 1, 10)))

    # Ensure seaborn context is applied for the rest of the figure styling
    if sns is not None:
        sns.set_context("notebook")

    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            rc={
                "axes.edgecolor": "0.35",
                "axes.labelcolor": "0.15",
                "axes.titleweight": "semibold",
                "grid.color": "0.88",
                "grid.linewidth": 0.6,
                "legend.frameon": False,
            },
        )
        try:
            _palette = sns.color_palette("deep")
        except Exception:
            _palette = None
    else:
        _palette = None

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    _sim_c = _palette[0] if _palette else SIM_C
    _exp_c = _palette[1] if _palette else EXP_C
    for ax, dcol, dmath, name in (
            (axs[0], "d+", r"$d^+$", "experiment hotter than sim"),
            (axs[1], "d-", r"$d^-$", "sim hotter than experiment")):
        _cols = colors
        for isc, kk in enumerate(sensors):
            prof = profile[profile["sensor"] == kk].sort_values("I_coil_rms (A)")
            ax.plot(prof["I_coil_rms (A)"], prof[dcol], "o-", lw=1.2, ms=4,
                    color=_cols[isc], alpha=0.9, label=kk)
        ax.axvline(1550.0, color="grey", lw=1.0, ls="--",
                   label=r"Nominal $I$ = 1550 A")
        ax.set_xlabel(r"Coil current $I$ [A]  ($E_n$)", fontsize=10)
        ax.set_ylabel(rf"MAVM {dmath} {unit_label('TC', latex=True)}",
                      fontsize=9)
        ax.set_title(f"{dmath}", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.margins(x=0.02)
        _style_axis(ax)
        ax.margins(x=0.03)
        _style_axis(ax)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=5,
               loc="outside lower center")
    # fig.suptitle("Nuisance dependence of MAVM "
    #              "(varying F and T_cool)\n"
    #              + convention_label(summary_convention(summary)), fontsize=11)
    fig.suptitle(r"MAVM dependence on nuisance (note varying coil frequency and "
                 r"monoblock coolant temperature)", fontsize=11)
    out = res_dir / "plots" / "02_dpm_vs_current.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out

def load_paper_mavm(paper_csv: Path = PAPER_MODEL_FORM,
                    extra_csv: Path = PAPER_MODEL_FORM_ALL) -> pd.DataFrame:
    """Read the paper pipeline's published model-form MAVM values.

    ``main_thermocouples_pulses25X.py`` writes the model-form part of its
    uncertainty budget to ``pointsensors_total_uncertainty_temperature.csv``:
    the rows ``model_form_plus`` and ``model_form_minus`` hold the MAVM
    ``d+``/``d-`` measured against the *outer* simulation bound only (the
    sim-max bound row for ``d+``, the sim-min bound row for ``d-``). These
    are the per-sensor metric values the paper reports.

    Parameters
    ----------
    paper_csv : pathlib.Path
        Path of the paper's temperature model-form table.
    extra_csv : pathlib.Path
        Path of the paper's all-sensor model-form table, used to fill in
        sensors the temperature table omits (CV).

    Returns
    -------
    pandas.DataFrame
        Indexed by sensor tag with columns ``"paper_d+"`` and ``"paper_d-"``.
    """
    if not Path(paper_csv).is_file():
        raise FileNotFoundError(
            f"paper model-form table not found: {paper_csv}\n"
            f"Run scripts/main_thermocouples_pulses25X.py first.")
    table = pd.read_csv(paper_csv, index_col=0)
    out = pd.DataFrame({
        "paper_d+": table.loc["model_form_plus"],
        "paper_d-": table.loc["model_form_minus"],
    })
    extra = Path(extra_csv)
    if extra.is_file():
        full = pd.read_csv(extra, index_col=0)
        gap = [kk for kk in full.columns if kk not in table.columns]
        if gap:
            out = out.reindex(list(out.index) + gap)
            for col, row in (("paper_d+", "model_form_plus"),
                             ("paper_d-", "model_form_minus")):
                out.loc[gap, col] = full.loc[row, gap].to_numpy()
    return out


def fig_split_vs_paper(summary, res_dir: Path,
                       paper_csv: Path = PAPER_MODEL_FORM,
                       extra_csv: Path = PAPER_MODEL_FORM_ALL):
    """Plot the split metric against the paper's published MAVM values.

    All three bars use the paper's model-form convention: the metric is
    measured against the outer simulation bound only (``d+`` against the
    sim-max bound row, ``d-`` against the sim-min bound row), which is the
    quantity the paper reports as ``model_form_plus``/``model_form_minus``.
    The first bar is the split metric (max over the E_n blocks), the second
    the same samples pooled into one epistemic box, and the third the paper's
    own published value.

    The summary frame must have been produced with the same convention
    (``--convention model_form``); a summary built with the reference
    convention is flagged in the title, because the paper's published
    values are model-form numbers.

    Note that the split bars are measured against much narrower per-block
    p-boxes (E_t only) than the paper's single pooled box, so their
    magnitude is not directly comparable with the paper's value; the
    merged bar is the like-for-like one.

    Parameters
    ----------
    summary : pandas.DataFrame
        Summary frame from :func:`load_results`.
    res_dir : pathlib.Path
        Results directory; the figure is written to ``res_dir / "plots"``.
    paper_csv : pathlib.Path
        Path of the paper's temperature model-form table.
    extra_csv : pathlib.Path
        Path of the paper's all-sensor model-form table (for CV).

    Returns
    -------
    pathlib.Path
        Path of the saved ``05_split_vs_paper.png``.
    """
    paper = load_paper_mavm(paper_csv, extra_csv)
    sensors = summary["sensor"].tolist()
    paper = paper.reindex(sensors)

    x = np.arange(len(sensors))
    w = 0.27

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    for ax, (scol, mcol, pcol, name) in zip(axs, (
            ("split_d+", "merged_d+", "paper_d+", r"max $d^+$"),
            ("split_d-", "merged_d-", "paper_d-", r"max $d^-$"))):
        ax.bar(x - w, summary[scol], w, color=SIM_C, edgecolor="white",
               linewidth=0.6, label=r"Split (max over $E_n$)")
        ax.bar(x, summary[mcol], w, color="grey", alpha=0.55,
               edgecolor="white", linewidth=0.6,
               label="Merged (same samples)")
        ax.bar(x + w, paper[pcol], w, color=EXP_C, edgecolor="white",
               linewidth=0.6, label="Test Case 1 paper")

        for i, kk in enumerate(sensors):
            pv = paper[pcol].iloc[i]
            sv = summary[scol].iloc[i]
            if np.isfinite(pv) and pv > 0 and sv > 0:
                diff = 100 * (sv - pv) / pv
                if abs(diff) >= 5:
                    ax.annotate(f"{diff:+.0f}%", (i + w, pv),
                                xytext=(0, 14), textcoords="offset points",
                                ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(sensors, fontsize=10)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel(rf"MAVM $d^\pm$ {unit_label('TC', latex=True)}",
                      fontsize=10)
        ax.tick_params(labelsize=10)
        ax.margins(y=0.12)
        ax.legend(fontsize=10, loc="best")
        _style_axis(ax)

    missing = [kk for kk in sensors
               if not np.isfinite(
                   paper.loc[kk, ["paper_d+", "paper_d-"]]).all()]
    note = ""
    if missing:
        note = f"  (absent from paper table: {', '.join(missing)})"
    conv = summary_convention(summary)
    if conv != "model_form":
        note += (f"  (WARNING: the summary uses the '{conv}' convention, "
                 f"the paper's values are model-form)")
    fig.suptitle("Split metric vs Test Case 1 paper results" + note, fontsize=11)
    out = res_dir / "plots" / "05_split_vs_paper.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def _ecdf_xy(arr):
    """Convert a sample into plottable empirical-CDF coordinates.

    Parameters
    ----------
    arr : array_like
        One-dimensional sample; NaNs must already have been removed, since
        they would sort to the end and corrupt the CDF.

    Returns
    -------
    q : numpy.ndarray
        Sorted sample values, shape=(n,).
    p : numpy.ndarray
        Plotting positions ``k / n`` for ``k = 1..n``, shape=(n,).
    """
    q = np.sort(np.asarray(arr, dtype=float))
    p = np.arange(1, len(q) + 1) / len(q)
    return q, p


def fig_ecdfs(data, summary, res_dir: Path, sensors):
    """Plot the p-box picture: block sub-boxes, merged box and experiment.

    For each requested sensor this draws the min/max bound CDFs of every E_n
    sub-box (thin blue), of the merged box (black dashed) and of the
    experiment (orange nominal plus shaded error box), with the block that
    maximises d+ drawn bold.

    Parameters
    ----------
    data : dict[str, numpy.ndarray]
        Per sensor, simulated outputs of shape=(n_en_total, n_et, n_a).
    summary : pandas.DataFrame
        Summary frame from :func:`load_results`; supplies the E_n block that
        maximises d+, used for the highlighted box.
    res_dir : pathlib.Path
        Results directory; the figure is written to ``res_dir / "plots"``.
    sensors : list[str]
        Sensor tags to plot, one panel each, e.g. ``["TC2", "TC3"]``.

    Returns
    -------
    pathlib.Path
        Path of the saved ``03_ecdfs.png``.
    """
    exp_bounds, _ = mns.load_experiment()
    n_en = data[sensors[0]].shape[0]

    fig, axs = plt.subplots(1, len(sensors), figsize=(6.2 * len(sensors), 5),
                            layout="constrained")
    axs = np.atleast_1d(axs)

    for ax, kk in zip(axs, sensors):
        exp = exp_bounds[kk]
        n_a = data[kk].shape[2]

        # experiment box + nominal
        for bound, lw, ls in (("min", 1.0, ":"), ("max", 1.0, ":")):
            q, p = _ecdf_xy(exp[bound])
            ax.plot(q, p, color=EXP_C, lw=lw, ls=ls, alpha=0.8)
        qn, pn = _ecdf_xy(exp["nom"])
        q_lo, p_lo = _ecdf_xy(exp["min"])
        q_hi, p_hi = _ecdf_xy(exp["max"])
        ax.plot(qn, pn, color=EXP_C, lw=2.0, label="Experiment mean")
        ax.fill_betweenx(p_lo, q_lo, q_hi, color=EXP_C, alpha=0.15,
                         label="Experiment limits")

        # per-block sub-box bounds (thin)
        first_sub = True
        for en in range(n_en):
            bnd = mns.sim_bounds(data[kk][en])
            for bound in ("min", "max"):
                q, p = _ecdf_xy(bnd[bound])
                if first_sub and bound == "min":
                    ax.plot(q, p, color=SIM_C, lw=0.8, alpha=0.3,
                            label="Sub-box")
                    first_sub = False
                else:
                    ax.plot(q, p, color=SIM_C, lw=0.8, alpha=0.3)

        # merged box bounds
        merged = mns.sim_bounds(data[kk].reshape(-1, n_a))
        for bound in ("min", "max"):
            q, p = _ecdf_xy(merged[bound])
            ax.plot(q, p, color="black", lw=1.5, ls="--",
                    label="Merged box" if bound == "min" else None)

        # argmax block bounds (highlighted)
        row = summary[summary["sensor"] == kk].iloc[0]
        en_star = int(row["split_d+_en"])
        bnd = mns.sim_bounds(data[kk][en_star])
        for bound in ("min", "max"):
            q, p = _ecdf_xy(bnd[bound])
            ax.plot(q, p, color=SIM_C, lw=2.2,
                    label=rf"$E_n$ block - Box {en_star} - maximises $d^+$"
                    if bound == "min" else None)

        ax.set_title(f"{kk}", fontsize=10)
        xlabel = ("Coil voltage [V]" if kk == "CV"
                  else f"Temperature {unit_label(kk, latex=True)}")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.margins(x=0.02)
        _style_axis(ax)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, ncol=5,
               loc="outside lower center")
    fig.suptitle(r"Split p-boxes vs merged box vs experiment ", 
    fontsize=11)
    out = res_dir / "plots" / "03_ecdfs.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_ecdfs_vs_paper(data, summary, res_dir: Path, sensors: list, selected: list = None, plot_name: str = None):
    """Compare the split maximising block with the paper's p-boxes.

    For each requested sensor this draws the selected p-box envelopes as
    min/max empirical-CDF bounds:

    * the E_n block that maximises the split-metric ``d+``,
    * the E_n block that maximises the split-metric ``d-``,
    * the original paper's full simulation p-box, and
    * the original paper's experimental p-box.

    All four are drawn by default; pass ``selected`` to draw a subset.

    The paper simulation p-box is rebuilt from the full
    ``SamplingResultsOnlyPointSensors.csv`` sample matrix using the paper's
    quantile-sum bound-selection rule.  The experimental p-box is the
    nominal experiment CDF expanded by the paper's epistemic sensor errors.

    Parameters
    ----------
    data : dict[str, numpy.ndarray]
        Per sensor, simulated outputs of shape=(n_en_total, n_et, n_a).
    summary : pandas.DataFrame
        Summary frame from :func:`load_results`; supplies the E_n block that
        maximises d+ for each sensor.
    res_dir : pathlib.Path
        Results directory; the figure is written to ``res_dir / "plots"``.
    sensors : list[str]
        Sensor tags to plot, one panel each, e.g. ``["TC2", "TC3"]``.
    selected : list[str], optional
        Names of p-boxes to plot. If None, plot all. Valid names are
        ``"split_d+"`` (the E_n block maximising d+), ``"split_d-"`` (the
        E_n block maximising d-), ``"pooled"`` (the paper's pooled p-box)
        and ``"experimental"``.

    Returns
    -------
    pathlib.Path
        Path of the saved ``06_ecdfs_vs_paper.png``.
    """
    if selected is None:
        selected = list(PBOX_NAMES)
    else:
        selected = list(dict.fromkeys(selected))
    unknown = [name for name in selected if name not in PBOX_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown p-box name(s) {unknown}; valid names are "
            f"{list(PBOX_NAMES)}."
        )
    if not selected:
        raise ValueError("selected must contain at least one p-box name.")

    paper_sim_csv = (
        SCRIPT_DIR.parent
        / "STC_ProbSim_FieldsFull_25X_v4"
        / "SamplingResultsOnlyPointSensors.csv"
    )
    paper_epis_n = 250
    paper_alea_n = 400
    paper = pd.read_csv(
        paper_sim_csv,
        usecols=[mns.SIM_COL[kk] for kk in sensors],
    )
    exp_bounds, _ = mns.load_experiment()

    def draw_pbox(ax, bounds, color, label, linestyle="-", linewidth=1.7):
        """Draw min/max CDF bounds and their shaded envelope."""
        lo = np.asarray(bounds["min"], dtype=float)
        hi = np.asarray(bounds["max"], dtype=float)
        q_lo, p_lo = _ecdf_xy(lo[np.isfinite(lo)])
        q_hi, p_hi = _ecdf_xy(hi[np.isfinite(hi)])
        ax.plot(q_lo, p_lo, color=color, lw=linewidth, ls=linestyle,
                label=label)
        ax.plot(q_hi, p_hi, color=color, lw=linewidth, ls=linestyle)
        if len(p_lo) == len(p_hi) and np.allclose(p_lo, p_hi):
            ax.fill_betweenx(p_lo, q_lo, q_hi, color=color, alpha=0.13,
                             linewidth=0)
    
    fig, axs = plt.subplots(
        1,
        len(sensors),
        figsize=(6.2 * len(sensors), 5),
        layout="constrained",
    )
    axs = np.atleast_1d(axs)
    
    for ax, kk in zip(axs, sensors):
        paper_values = paper[mns.SIM_COL[kk]].to_numpy(dtype=float)
        expected_size = paper_epis_n * paper_alea_n
        if paper_values.size != expected_size:
            raise ValueError(
                f"{kk}: paper simulation has {paper_values.size} rows, "
                f"expected {expected_size} "
                f"({paper_epis_n} epistemic x {paper_alea_n} aleatory)"
            )
        paper_block = paper_values.reshape(paper_epis_n, paper_alea_n)
    
        row = summary[summary["sensor"] == kk].iloc[0]
        en_star_plus = int(row["split_d+_en"])
        en_star_minus = int(row["split_d-_en"])
        split_bounds_plus = mns.sim_bounds(data[kk][en_star_plus])
        split_bounds_minus = mns.sim_bounds(data[kk][en_star_minus])
        paper_bounds = mns.sim_bounds(paper_block)
        exp = exp_bounds[kk]
    
        # Calculate MAVM d+/d- for each p-box
        convention = mns.CONVENTIONS.get(
            row.get("convention", "model_form"),
            mns.model_form_bounds,
        )
        split_mavm_plus = convention(split_bounds_plus, exp)
        split_mavm_minus = convention(split_bounds_minus, exp)
        paper_mavm = convention(paper_bounds, exp)
    
        pboxes = {
            "split_d+": (
                split_bounds_plus,
                SIM_C,
                rf"Split p-box - $E_n$ block {en_star_plus} ($\max d^+$): "
                rf"$d^+={split_mavm_plus['d+']:.3g}$, ",
                "-",
            ),
            "split_d-": (
                split_bounds_minus,
                #SIM_C2,
                SIM_C,
                rf"Split p-box - $E_n$ block {en_star_minus} ($\max d^-$): "
                rf"$d^-={split_mavm_minus['d-']:.3g}$",
                "-",
            ),
            "pooled": (
                paper_bounds,
                #"black",
                SIM_C,
                rf"Pooled p-box (Test Case 1 paper): "
                rf"$d^+={paper_mavm['d+']:.3g}$, "
                rf"$d^-={paper_mavm['d-']:.3g}$",
                "--",
            ),
            "experimental": (
                exp,
                EXP_C,
                "Experimental p-box",
                ":",
            ),
        }
        for name in selected:
            bounds, color, label, linestyle = pboxes[name]
            draw_pbox(
                ax,
                bounds,
                color,
                label,
                linestyle=linestyle,
            )
    
        ax.set_title(f"{kk}", fontsize=10)
        xlabel = (
            "Coil voltage [V]"
            if kk == "CV"
            else f"Temperature {unit_label(kk, latex=True)}"
        )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.margins(x=0.02)
        _style_axis(ax)

        # ax.legend(fontsize=8, loc="outside lower center", framealpha=0.9)

        ax.legend(
            fontsize=10,
            loc="upper left",
            bbox_to_anchor=(0.2, -0.20),
            borderaxespad=0,
            framealpha=0.9,
        )
    
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, fontsize=9, ncol=2,
    #            loc="outside lower center")
    title_names = {
        "split_d+": r"split $d^+$ block",
        "split_d-": r"split $d^-$ block",
        "pooled": "pooled (Test Case 1 paper)",
        "experimental": "experimental",
    }
    if selected == list(PBOX_NAMES):
        title = (
            "Split maximising block vs pooled (Test Case 1 paper) vs "
            "experimental p-boxes"
        )
    else:
        title = " vs ".join(title_names[name] for name in selected) + " p-boxes"
    fig.suptitle(title, fontsize=11)

    if plot_name == None:
        out = res_dir / "plots" / "06_ecdfs_vs_paper.png"
    else:
        out = res_dir / "plots" / plot_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_dextremes_simcdfs(data, summary, res_dir: Path, sensors: list,
                          selected: list = None, plot_name: str = None):
    """Plot simulation limits with the MAVM d+/d- CDF shifts.

    This is the split analogue of the paper pipeline's
    ``*_dextremes_simcdfsonly_*.png`` figures.  For each sensor, simulation
    p-boxes are drawn as min/max empirical-CDF envelopes, together with the
    shifted CDFs implied by the metric.  If only one split p-box is selected,
    both of its ``d+``/``d-`` shifts are drawn so the metric band remains
    visible.  The experimental p-box is used to compute ``d+/d-`` but is not
    drawn.

    Parameters
    ----------
    data : dict[str, numpy.ndarray]
        Per sensor, simulated outputs of shape=(n_en_total, n_et, n_a).
    summary : pandas.DataFrame
        Summary frame from :func:`load_results`; supplies the E_n blocks that
        maximise d+ and d- for each sensor.
    res_dir : pathlib.Path
        Results directory; the figure is written to ``res_dir / "plots"``.
    sensors : list[str]
        Sensor tags to plot, one panel each, e.g. ``["TC2", "TC3"]``.
    selected : list[str], optional
        Names of simulation p-boxes to plot. If None, plot all. Valid names
        are ``"split_d+"`` (the E_n block maximising d+), ``"split_d-"``
        (the E_n block maximising d-) and ``"pooled"`` (the paper's original
        full simulation p-box).
    plot_name : str, optional
        Output filename.  Defaults to ``10_dextremes_simcdfs.png``.

    Returns
    -------
    pathlib.Path
        Path of the saved figure.
    """
    if selected is None:
        selected = list(DEXTREME_NAMES)
    else:
        selected = list(dict.fromkeys(selected))
    unknown = [name for name in selected if name not in DEXTREME_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown simulation p-box name(s) {unknown}; valid names are "
            f"{list(DEXTREME_NAMES)}."
        )
    if not selected:
        raise ValueError("selected must contain at least one p-box name.")

    exp_bounds, _ = mns.load_experiment()

    pooled_bounds = {}
    if "pooled" in selected:
        paper_sim_csv = (
            SCRIPT_DIR.parent
            / "STC_ProbSim_FieldsFull_25X_v4"
            / "SamplingResultsOnlyPointSensors.csv"
        )
        paper_epis_n = 250
        paper_alea_n = 400
        paper = pd.read_csv(
            paper_sim_csv,
            usecols=[mns.SIM_COL[kk] for kk in sensors],
        )
        for kk in sensors:
            paper_values = paper[mns.SIM_COL[kk]].to_numpy(dtype=float)
            expected_size = paper_epis_n * paper_alea_n
            if paper_values.size != expected_size:
                raise ValueError(
                    f"{kk}: paper simulation has {paper_values.size} rows, "
                    f"expected {expected_size} "
                    f"({paper_epis_n} epistemic x {paper_alea_n} aleatory)"
                )
            pooled_bounds[kk] = mns.sim_bounds(
                paper_values.reshape(paper_epis_n, paper_alea_n)
            )

    def draw_pbox(ax, bounds, color, label, linestyle="-", linewidth=1.7):
        """Draw min/max CDF bounds and their shaded envelope."""
        lo = np.asarray(bounds["min"], dtype=float)
        hi = np.asarray(bounds["max"], dtype=float)
        q_lo, p_lo = _ecdf_xy(lo[np.isfinite(lo)])
        q_hi, p_hi = _ecdf_xy(hi[np.isfinite(hi)])
        ax.plot(q_lo, p_lo, color=color, lw=linewidth, ls=linestyle,
                label=label)
        ax.plot(q_hi, p_hi, color=color, lw=linewidth, ls=linestyle)
        if len(p_lo) == len(p_hi) and np.allclose(p_lo, p_hi):
            ax.fill_betweenx(p_lo, q_lo, q_hi, color=color, alpha=0.2,
                             linewidth=0)

    def winning_shift(bounds, exp, stat_key):
        """Recompute vm.mavm for a convention's winning sim/exp pair."""
        sim_key, exp_key = (part.split("-")[1]
                            for part in stat_key.split("_"))
        return vm.mavm(bounds[sim_key], exp[exp_key])

    fig, axs = plt.subplots(
        1,
        len(sensors),
        figsize=(6.2 * len(sensors), 5),
        layout="constrained",
    )
    axs = np.atleast_1d(axs)

    for ax, kk in zip(axs, sensors):
        row = summary[summary["sensor"] == kk].iloc[0]
        exp = exp_bounds[kk]

        convention = mns.CONVENTIONS.get(
            row.get("convention", "model_form"),
            mns.model_form_bounds,
        )

        split_plus_shift = None
        split_minus_shift = None
        if "split_d+" in selected:
            en_star_plus = int(row["split_d+_en"])
            bounds_plus = mns.sim_bounds(data[kk][en_star_plus])
            plus = convention(bounds_plus, exp)
            draw_pbox(
                ax,
                bounds_plus,
                SIM_C,
                rf"Simulation limits - $E_n$ block {en_star_plus} "
                rf"($\max d^+$)",
            )
            if plus["d+_stat"] is not None:
                split_plus_shift = winning_shift(
                    bounds_plus, exp, plus["d+_stat"]
                )
                ax.plot(
                    split_plus_shift["F_"] + split_plus_shift["d+"],
                    split_plus_shift["F_Y"],
                    ls=":",
                    lw=2.0,
                    color="black",
                    label=rf"$d^+={split_plus_shift['d+']:.3g}$",
                )
            # With only this split p-box selected, draw its own d- as well
            # so the d+/d- band is still visible.
            if ("split_d-" not in selected
                    and plus["d-_stat"] is not None):
                split_minus_shift = winning_shift(
                    bounds_plus, exp, plus["d-_stat"]
                )
                ax.plot(
                    split_minus_shift["F_"] - split_minus_shift["d-"],
                    split_minus_shift["F_Y"],
                    ls="--",
                    lw=2.0,
                    color="black",
                    label=rf"$d^-={split_minus_shift['d-']:.3g}$",
                )

        if "split_d-" in selected:
            en_star_minus = int(row["split_d-_en"])
            bounds_minus = mns.sim_bounds(data[kk][en_star_minus])
            minus = convention(bounds_minus, exp)
            draw_pbox(
                ax,
                bounds_minus,
                SIM_C,
                rf"Simulation limits - $E_n$ block {en_star_minus} "
                rf"($\max d^-$)",
            )
            if minus["d-_stat"] is not None:
                split_minus_shift = winning_shift(
                    bounds_minus, exp, minus["d-_stat"]
                )
                ax.plot(
                    split_minus_shift["F_"] - split_minus_shift["d-"],
                    split_minus_shift["F_Y"],
                    ls="--",
                    lw=2.0,
                    color="black",
                    label=rf"$d^-={split_minus_shift['d-']:.3g}$",
                )
            # With only this split p-box selected, draw its own d+ as well
            # so the d+/d- band is still visible.
            if ("split_d+" not in selected
                    and minus["d+_stat"] is not None):
                split_plus_shift = winning_shift(
                    bounds_minus, exp, minus["d+_stat"]
                )
                ax.plot(
                    split_plus_shift["F_"] + split_plus_shift["d+"],
                    split_plus_shift["F_Y"],
                    ls=":",
                    lw=2.0,
                    color="black",
                    label=rf"$d^+={split_plus_shift['d+']:.3g}$",
                )

        if split_plus_shift is not None and split_minus_shift is not None:
            ax.fill_betweenx(
                split_plus_shift["F_Y"],
                split_minus_shift["F_"] - split_minus_shift["d-"],
                split_plus_shift["F_"] + split_plus_shift["d+"],
                color="black",
                alpha=0.1,
                ls=":",
            )

        if "pooled" in selected:
            bounds_pooled = pooled_bounds[kk]
            pooled = convention(bounds_pooled, exp)
            draw_pbox(
                ax,
                bounds_pooled,
                SIM_C,
                "Pooled p-box (Test Case 1 paper)",
                linestyle="--",
            )

            pooled_plus_shift = None
            pooled_minus_shift = None
            if pooled["d+_stat"] is not None:
                pooled_plus_shift = winning_shift(
                    bounds_pooled, exp, pooled["d+_stat"]
                )
                ax.plot(
                    pooled_plus_shift["F_"] + pooled_plus_shift["d+"],
                    pooled_plus_shift["F_Y"],
                    ls=":",
                    lw=2.0,
                    color="black",
                    label=rf"pooled $d^+={pooled_plus_shift['d+']:.3g}$",
                )
            if pooled["d-_stat"] is not None:
                pooled_minus_shift = winning_shift(
                    bounds_pooled, exp, pooled["d-_stat"]
                )
                ax.plot(
                    pooled_minus_shift["F_"] - pooled_minus_shift["d-"],
                    pooled_minus_shift["F_Y"],
                    ls="--",
                    lw=2.0,
                    color="black",
                    label=rf"pooled $d^-={pooled_minus_shift['d-']:.3g}$",
                )
            if (pooled_plus_shift is not None
                    and pooled_minus_shift is not None):
                ax.fill_betweenx(
                    pooled_plus_shift["F_Y"],
                    pooled_minus_shift["F_"] - pooled_minus_shift["d-"],
                    pooled_plus_shift["F_"] + pooled_plus_shift["d+"],
                    color="black",
                    alpha=0.1,
                    ls=":",
                )

        ax.set_title(f"{kk}", fontsize=10)
        xlabel = (
            "Coil voltage [V]"
            if kk == "CV"
            else f"Temperature {unit_label(kk, latex=True)}"
        )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Probability", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.margins(x=0.02)
        _style_axis(ax)
        ax.legend(
            fontsize=10,
            loc="upper left",
            bbox_to_anchor=(0.2, -0.20),
            borderaxespad=0,
            framealpha=0.9,
        )

    title_names = {
        "split_d+": r"split $d^+$ block",
        "split_d-": r"split $d^-$ block",
        "pooled": "pooled (Test Case 1 paper)",
    }
    if selected == list(DEXTREME_NAMES):
        title = r"Split and pooled simulation limits with $d^+$/$d^-$ shifts"
    elif selected == ["pooled"]:
        title = ("Pooled simulation limits with $d^+$/$d^-$ shifts "
                 "(Test Case 1 paper)")
    else:
        title = (" vs ".join(title_names[name] for name in selected)
                 + r" simulation limits with $d^+$/$d^-$ shifts")
    fig.suptitle(title, fontsize=11)
    if plot_name is None:
        out = res_dir / "plots" / "10_dextremes_simcdfs.png"
    else:
        out = res_dir / "plots" / plot_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_distributions(data, summary, res_dir: Path, sensors):
    """Plot per-box aleatory spreads against the experiment band.

    Each box contributes all n_et x n_a outputs as one box plot; boxes are
    ordered by their sample mean so the nuisance-driven shift is visible,
    with the experimental mean and its epistemic uncertainty band overlaid.

    Parameters
    ----------
    data : dict[str, numpy.ndarray]
        Per sensor, simulated outputs of shape=(n_en_total, n_et, n_a).
    summary : pandas.DataFrame
        Summary frame from :func:`load_results`. Accepted for a uniform call
        signature with the other figure functions; not used here.
    res_dir : pathlib.Path
        Results directory; the figure is written to ``res_dir / "plots"``.
    sensors : list[str]
        Sensor tags to plot, one panel each.

    Returns
    -------
    pathlib.Path
        Path of the saved ``04_distributions.png``.
    """
    exp_bounds, exp_err = mns.load_experiment()
    n_en, n_et, n_a = data[sensors[0]].shape

    fig, axs = plt.subplots(1, len(sensors), figsize=(6.2 * len(sensors), 4.6),
                            layout="constrained")
    axs = np.atleast_1d(axs)

    for ax, kk in zip(axs, sensors):
        # order blocks by their mean (proxy for current-driven shift)
        means = [data[kk][en].mean() for en in range(n_en)]
        order = np.argsort(means)
        samples = [data[kk][en].ravel() for en in order]
        labels = [f"{data[kk][en].reshape(n_et, n_a)[:, 0].mean():.0f}"
                  for en in order]

        if sns is not None:
            sns.boxplot(data=samples, ax=ax, width=0.62, showfliers=False,
                        palette=sns.color_palette("Blues", n_en),
                        linewidth=0.8, saturation=0.9)
        else:
            ax.boxplot(samples, tick_labels=labels, showfliers=False,
                       widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor="lightblue", alpha=0.7))
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)

        exp = exp_bounds[kk]
        nom = exp["nom"]
        nom = nom[~np.isnan(nom)]
        err = exp_err[kk]
        ax.axhline(nom.mean(), color=EXP_C, lw=1.8)
        ax.axhspan(nom.mean() - err, nom.mean() + err, color=EXP_C,
                   alpha=0.2, label="Experimental data")
        ax.set_xlabel(r"$E_n$ box, ordered by sim. mean (label = mean output)",
                      fontsize=10)
        ax.set_ylabel(rf"{kk} {unit_label(kk, latex=True)}", fontsize=10)
        ax.set_title(f"{kk}: aleatory spread per nuisance box", fontsize=10)
        ax.tick_params(labelsize=10)
        ax.margins(x=0.02)
        ax.legend(fontsize=10, loc="upper left")
        _style_axis(ax)

    fig.suptitle("Sub-box contents vs experimental data", fontsize=11)
    out = res_dir / "plots" / "04_distributions.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out





def main():
    """Generate all figures for the results directory given on the CLI.

    Returns
    -------
    None
        The figures are written to ``<res-dir>/plots/`` and their paths are
        printed to stdout.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", type=Path,
                    default=SCRIPT_DIR / "mavm_nuisance_split" / "nen9_net16_na100")
    ap.add_argument("--sensors", nargs="+", default=["TC2", "TC3"],
                    help="sensors for the CDF/distribution figures")
    ap.add_argument("--selected", nargs="+", default=None, choices=PBOX_NAMES,
                    help="p-boxes to draw in 06_ecdfs_vs_paper "
                         "(default: all)")
    ap.add_argument("--dextremes-selected", nargs="+", default=None,
                    choices=DEXTREME_NAMES,
                    help="simulation p-boxes to draw in 10_dextremes_simcdfs "
                         "(default: all)")
    ap.add_argument("--convention", choices=("model_form", "reference"),
                    default="model_form",
                    help="which convention to read from the results "
                         "directory (default: the paper's model-form one)")
    args = ap.parse_args()

    res_dir = args.res_dir.resolve()
    out_dir = res_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile, summary, meta, data = load_results(res_dir, args.convention)
    print(f"Loaded {res_dir}  (E_n={meta['n_en_total']}, E_t={meta['n_et']}, "
          f"A={meta['n_a']})")
    print(f"  convention: {summary_convention(summary)}")

    outs = [
        fig_profile(profile, summary, res_dir),
        fig_vs_current(profile, summary, res_dir),
        fig_split_vs_paper(summary, res_dir),
        fig_ecdfs(data, summary, res_dir, args.sensors),
        fig_ecdfs_vs_paper(data, summary, res_dir, args.sensors, args.selected),
        
        # For the presentation
        fig_ecdfs_vs_paper(data, summary, res_dir, sensors = ["TC2"], 
                           selected=["pooled", "experimental"], 
                           plot_name="07_ecdfs_vs_paper.png"),
        fig_ecdfs_vs_paper(data, summary, res_dir, sensors = ["TC2"], 
                           selected=["split_d+", "experimental"], 
                           plot_name="08_ecdfs_vs_paper.png"),
        fig_ecdfs_vs_paper(data, summary, res_dir, sensors = ["TC2"], 
                           selected=["split_d-", "experimental"], 
                           plot_name="09_ecdfs_vs_paper.png"),

        fig_dextremes_simcdfs(data, summary, res_dir, args.sensors,
                             args.dextremes_selected),

        # For the presentation
        fig_dextremes_simcdfs(data, summary, res_dir, sensors = ["TC2"],
                             selected=["pooled"],
                             plot_name="11_dextremes_simcdfs.png"),
        fig_dextremes_simcdfs(data, summary, res_dir, sensors = ["TC2"],
                             selected=["split_d+"],
                             plot_name="12_dextremes_simcdfs.png"),
        fig_dextremes_simcdfs(data, summary, res_dir, sensors = ["TC2"],
                             selected=["split_d-"],
                             plot_name="13_dextremes_simcdfs.png"),
        fig_dextremes_simcdfs(data, summary, res_dir, sensors = ["TC2"],
                             selected=["split_d-", "split_d+"],
                             plot_name="14_dextremes_simcdfs.png"),


        fig_distributions(data, summary, res_dir, args.sensors),
    ]
    for o in outs:
        print("saved", o)


if __name__ == "__main__":
    main()
