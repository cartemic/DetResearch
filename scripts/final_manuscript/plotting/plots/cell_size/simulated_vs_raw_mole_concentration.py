from typing import TYPE_CHECKING

from scripts.final_manuscript.plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame


def plot(data: "DataFrame"):
    by_diluent.scatter_error(
        data=data,
        x="dil_mf",
        y="cell_size",
        xerr="u_dil_mf",
        yerr="u_cell_size",
        col="phi_nom",
        ylabel="Cell Size (mm)",
        xlabel=r"$X_{dil}$",
        title="Simulated Cell Size vs. Diluent Mole Fraction",
    )
