from typing import TYPE_CHECKING

import seaborn as sns

from plotting.plots import formatting
from plotting.plots.formatting import DiluentColor

if TYPE_CHECKING:
    from pandas import DataFrame


def scatter_error(
    data: "DataFrame",
    x: str,
    y: str,
    xerr: str | None,
    yerr: str | None,
    col: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> sns.FacetGrid:
    n_cols = 3
    grid = sns.relplot(
        data=data,
        x=x,
        y=y,
        hue="diluent",
        col=col,
        height=formatting.AX_HEIGHT,
        aspect=formatting.FIG_ASPECT / n_cols,
        palette=formatting.DiluentColor.palette(),
    )
    formatting.format_diluent_legend(grid.legend)
    for ax, (_, grouped_data) in zip(grid.axes.flatten(), data.groupby(col)):
        # startswith allows for inerts, e.g. CO2i
        co2 = grouped_data[grouped_data["diluent"].str.startswith("CO2")]
        n2 = grouped_data[grouped_data["diluent"].str.startswith("N2")]
        for dil_data, color in ((co2, DiluentColor.co2), (n2, DiluentColor.n2)):
            ax.errorbar(
                x=dil_data[x],
                y=dil_data[y],
                xerr=dil_data[xerr] if isinstance(xerr, str) else None,
                yerr=dil_data[yerr] if isinstance(yerr, str) else None,
                ecolor=color,
                capsize=2,
                elinewidth=1,
                ls="",
                zorder=-1,
            )
        ax.set_title(
            ax.get_title()
            .replace("dil_mf_nom", r"$X_{dil, CO_{2}e}$")
            .replace("phi_nom", r"$\phi_{nom}$")
            .replace("dil_condition", "Dilution"),
            weight="normal",
        )
    grid.set_ylabels(ylabel)
    grid.set_xlabels(xlabel)
    grid.fig.suptitle(title)
    grid.fig.subplots_adjust(top=0.825)
    grid.despine()
    return grid
