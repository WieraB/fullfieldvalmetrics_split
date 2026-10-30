import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def error_scatter(df_all, tol, tag):
    """Plot measured versus predicted values with tolerance bounds.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Result rows containing at least ``measured``, ``predicted`` and
        ``model`` columns.
    tol : float
        Relative tolerance used for the ``+/- tol`` reference lines.
    tag : str
        Label appended to the plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the scatter plot.
    ax : matplotlib.axes.Axes
        Axes of the scatter plot.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Scatter points
    sns.scatterplot(
        data=df_all,
        x="measured",
        y="predicted",
        hue="model",
        alpha=0.7,
        ax=ax
    )
    
    lims = [
        min(df_all["measured"].min(), df_all["predicted"].min()) - 2,
        max(df_all["measured"].max(), df_all["predicted"].max()) + 2
    ]
    
    x = np.linspace(lims[0], lims[1], 100)
    
    # Ideal line
    ax.plot(x, x, "k-", linewidth=2, label="Ideal")
    
    # ±% tolerance
    ax.plot(x, (1.0 + tol) * x, "k--", alpha=0.7)
    ax.plot(x, (1.0 - tol) * x, "k--", alpha=0.7)
    
    ax.axis("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    ax.set_xlabel("Measured [°C]")
    ax.set_ylabel("Predicted [°C]")
    
    ax.set_title(f"Errors {tag}")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True
    )

    return fig, ax


def interval_score(df_all, models, tag):
    """Plot interval-score distributions for a set of models.

    The ``model`` column in ``df_all`` is converted in place to an ordered
    categorical so that the requested model order is respected.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Result rows containing ``model`` and ``interval_score`` columns.
    models : sequence[str]
        Model names in the desired plotting order.
    tag : str
        Label appended to the plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the box plot.
    ax : matplotlib.axes.Axes
        Axes of the box plot.
    """
    df_all["model"] = pd.Categorical(df_all["model"], categories=models, ordered=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        data=df_all,
        x="model",
        y="interval_score",
        order=models,
        ax=ax
    )

    # Compute medians in the same order as the models
    medians = (
        df_all.groupby("model", observed=False)["interval_score"]
        .median()
        .reindex(models)
    )

    # Add median labels aligned to x positions
    for i, model in enumerate(models):
        y = medians.loc[model]
        ax.text(
            i,
            y,
            f"{y:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_ylabel("Interval score")
    ax.set_xlabel("Model")
    ax.set_title(f"Interval score {tag}")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    return fig, ax


def coverage(df_all, tag):
    """Plot prediction-interval coverage by model.

    Coverage is the percentage of rows whose ``within_pi`` value is true.
    A dashed line marks the nominal 95 percent level.

    Parameters
    ----------
    df_all : pandas.DataFrame
        Result rows containing ``model`` and ``within_pi`` columns.
    tag : str
        Label appended to the plot title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the coverage bars.
    ax : matplotlib.axes.Axes
        Axes of the coverage plot.
    """
    coverage = (
        df_all.groupby("model")["within_pi"]
        .mean()              # True=1, False=0
        .mul(100)            # convert to %
        .reset_index(name="coverage")
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    
    sns.barplot(
        data=coverage,
        x="model",
        y="coverage",
        ax=ax
    )
    
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontweight="bold", fontsize=6)
    
    ax.set_ylabel("Coverage [%]")
    ax.set_xlabel("Model")
    ax.set_title(f"Coverage {tag}")

    ax.set_ylim(0, 100)
    ax.axhline(95, color="red", linestyle="--", label="Nominal 95%")
    ax.legend()

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f"Coverage {tag}", pad=25)
    fig.tight_layout()

    return fig, ax



def performance_dist(results, metrics_to_plot, tag):
    """Plot performance distributions for several metrics.

    The function creates one box-and-strip panel per metric in
    ``metrics_to_plot`` and groups the values by ``d_type``.

    Parameters
    ----------
    results : pandas.DataFrame
        Result table containing ``d_type`` and the requested metric columns.
    metrics_to_plot : sequence[str]
        Metric column names to plot, one per panel.  The current layout
        provides three panels, so at most the first three are shown.
    tag : str
        Label appended to the figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the performance panels.
    axes : numpy.ndarray
        Array of axes, one for each plotted metric.
    """
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    for ax, metric in zip(axes, metrics_to_plot):

        sns.boxplot(
            data=results,
            x="d_type",
            y=metric,
            ax=ax,
            color="lightblue"
        )

        sns.stripplot(
            data=results,
            x="d_type",
            y=metric,
            ax=ax,
            color="black",
            alpha=0.6,
            size=4
        )

        ax.set_xlabel("d_type")
        ax.set_ylabel(metric)
        ax.set_title(metric)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(
        f"Performance distribution for {tag}",
        fontsize=16
    )

    return fig, axes

def table(df_to_plot):
    """Render a DataFrame as a Matplotlib table.

    Parameters
    ----------
    df_to_plot : pandas.DataFrame
        Values and column labels to display.  Numeric values are rounded
        to three decimal places for rendering.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the table.
    ax : matplotlib.axes.Axes
        Axis used to draw the table.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    
    table = ax.table(
        cellText=df_to_plot.round(3).values,
        colLabels=df_to_plot.columns,
        cellLoc="center",
        loc="center"
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    return fig, ax
