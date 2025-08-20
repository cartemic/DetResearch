import warnings
from typing import TYPE_CHECKING

import seaborn as sns

from plotting.plots import formatting

if TYPE_CHECKING:
    from pandas import DataFrame


def __plot_gamma(
    data: "DataFrame",
    y: str,
    title: str,
    hue: str | None = None,
    color: str | None = None,
) -> sns.FacetGrid:
    n_cols = 2
    grid = sns.relplot(
        x="progress",
        y=y,
        data=data,
        col="dil_condition",
        row="phi_nom",
        hue=hue,
        color=color,
        kind="line",
        col_order=["low", "high"],
        height=formatting.AX_HEIGHT,
        aspect=formatting.FIG_ASPECT / n_cols,
        palette=formatting.DiluentColor.palette(),
    )
    if grid.legend is not None:
        formatting.format_diluent_legend(grid.legend)
    for ax in grid.axes.flatten():
        ax.set_ylabel(r"$\gamma$")
        ax.set_xlabel("Induction Progress")
        ax.set_title(
            ax.get_title().replace("phi_nom", r"$\phi$").replace(" |", ",").replace("dil_condition", "Dilution"),
            weight="normal",
        )
    grid.fig.suptitle(title)
    grid.fig.subplots_adjust(top=0.875)
    return grid


def plot(data: "DataFrame") -> sns.FacetGrid:
    return __plot_gamma(data, y="gamma", title="Specific Heat Ratios Through Induction Zone", hue="diluent")


def plot_ratios(data: "DataFrame") -> sns.FacetGrid:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # yea, we're not assigning hue
        return __plot_gamma(data, y="gamma", title="CO$_{2}$/N$_{2}$ Gamma Ratio Through Induction Zone", color=formatting.BLACK)
