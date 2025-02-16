from typing import TYPE_CHECKING

import numpy as np
import seaborn as sns

from scripts.final_manuscript.plotting.data import combine_inert_and_active
from scripts.final_manuscript.plotting.plots import formatting
from scripts.final_manuscript.plotting.plots.formatting import DiluentColor

if TYPE_CHECKING:
    from matplotlib.legend import Legend
    from pandas import DataFrame


def vs_active(data: "DataFrame"):
    col = "dil_mf_nom"
    x = "phi_nom"
    y = "cell_size"
    title = "Simulated Cell Size, Inert and Active Diluents"
    n_cols = 3
    grid = sns.relplot(
        data=data,
        x=x,
        y=y,
        hue="diluent_kind",
        row="diluent",
        style="diluent",
        col=col,
        height=formatting.AX_HEIGHT,
        aspect=formatting.FIG_ASPECT / n_cols,
        palette=formatting.DiluentKindColor.palette(),
    )
    _format_legend(grid.legend)
    for ax in grid.axes[0].flatten():
        ax.set_title(
            ax.get_title()
            .split("|")[-1]
            .replace("dil_mf_nom", r"$X_{dil, CO_{2}e}$")
            .replace("phi_nom", r"$\phi_{nom}$")
            .replace("dil_condition", "Dilution"),
            weight="normal",
        )
    for ax in grid.axes[1].flatten():
        ax.set_title("")
    grid.set_ylabels("Cell Size (mm)")
    grid.set_xlabels(r"$\phi_{nom}$")
    grid.tight_layout()
    grid.fig.suptitle(title)
    grid.fig.subplots_adjust(top=0.9)
    grid.despine()


def vs_active_2(data: "DataFrame"):
    col = "dil_mf_nom"
    x = "phi_nom"
    y = "cell_size"
    title = "Simulated Cell Size, Inert and Active Diluents"
    n_cols = 3
    grid = sns.relplot(
        data=data,
        x=x,
        y=y,
        hue="diluent",
        row="diluent_kind",
        style="diluent_kind",
        col=col,
        height=formatting.AX_HEIGHT,
        aspect=formatting.FIG_ASPECT / n_cols,
        palette=formatting.DiluentColor.palette(),
    )
    _format_legend(grid.legend)
    for ax in grid.axes[0].flatten():
        ax.set_title(
            ax.get_title()
            .split("|")[-1]
            .replace("dil_mf_nom", r"$X_{dil, CO_{2}e}$")
            .replace("phi_nom", r"$\phi_{nom}$")
            .replace("dil_condition", "Dilution"),
            weight="normal",
        )
    for ax in grid.axes[1].flatten():
        ax.set_title("")
    grid.set_ylabels("Cell Size (mm)")
    grid.set_xlabels(r"$\phi_{nom}$")
    grid.tight_layout()
    grid.fig.suptitle(title)
    grid.fig.subplots_adjust(top=0.9)
    grid.despine()


# Matplotlib sometimes requires access to private members
# ruff: noqa: SLF001
# noinspection PyProtectedMember,PyUnresolvedReferences
def _format_legend(leg: "Legend") -> None:
    for t in leg.texts:
        if (lbl := t.get_text()) == "diluent":
            t.set_text("Diluent")
            t.set_weight("bold")
        elif lbl == "diluent_kind":
            t.set_text("Diluent Kind")
            t.set_weight("bold")
        else:
            t.set_text(t._text.replace("2", "$_{2}$").replace("active", "Active").replace("inert", "Inert"))


def vs_measured(active: "DataFrame", inert: "DataFrame") -> None:
    data = combine_inert_and_active(inert, active)
    col = "dil_mf_nom"
    x = "phi"
    y = "cell_size"
    title = "Cell Size Ratio vs. Equivalence Ratio"
    yerr = "u_cell_size"
    n_cols = 3
    grid = sns.relplot(
        data=data,
        x=x,
        y=y,
        hue="diluent",
        style="diluent_kind",
        row="diluent_kind",
        col=col,
        height=formatting.AX_HEIGHT,
        aspect=formatting.FIG_ASPECT / n_cols,
        palette=DiluentColor.palette(),
    )
    _format_legend(grid.legend)

    active_mask = data["diluent_kind"] == "active"
    inert_mask = np.invert(active_mask)
    dil_low_mask = data["dil_mf_nom"] == 0.1
    dil_med_mask = data["dil_mf_nom"] == 0.15
    dil_high_mask = np.invert(dil_low_mask | dil_med_mask)
    grouped_data = (
        data[active_mask & dil_low_mask],
        data[active_mask & dil_med_mask],
        data[active_mask & dil_high_mask],
        data[inert_mask & dil_low_mask],
        data[inert_mask & dil_med_mask],
        data[inert_mask & dil_high_mask],
    )
    for ax, grouped_data in zip(grid.axes.flatten(), grouped_data):
        co2 = grouped_data[grouped_data["diluent"].str.startswith("CO2")]
        n2 = grouped_data[grouped_data["diluent"].str.startswith("N2")]
        for dil_data, color in ((co2, DiluentColor.co2), (n2, DiluentColor.n2)):
            ax.errorbar(
                x=dil_data[x],
                y=dil_data[y],
                yerr=dil_data[yerr],
                ecolor=color,
                capsize=2,
                elinewidth=1,
                ls="",
                zorder=-1,
            )
        ax.set_title(
            ax.get_title()
            .split("|")[-1]
            .replace("dil_mf_nom", r"$X_{dil, CO_{2}e}$")
            .replace("phi_nom", r"$\phi_{nom}$")
            .replace("dil_condition", "Dilution"),
            weight="normal",
        )
        # todo: add error bars
    for ax in grid.axes[1].flatten():
        ax.set_title("")
    grid.set_ylabels("Simulated / Measured")
    grid.set_xlabels(r"$\phi$")
    grid.tight_layout()
    grid.fig.suptitle(title)
    grid.fig.subplots_adjust(top=0.9)
    grid.despine()
