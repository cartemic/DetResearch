from typing import TYPE_CHECKING

from scripts.final_manuscript.plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame


def plot(data: "DataFrame"):
    by_diluent.scatter_error(
        data=data,
        x="dil_mf_co2e",
        y="cell_size",
        xerr="u_dil_mf_co2e",
        yerr="u_cell_size",
        col="phi_nom",
        xlabel=r"$X_{dil, CO_{2}e}$",
        ylabel="Cell Size (mm)",
        title="Measured Cell Size vs. T$_{ad}$ Matched Diluent Mole Fraction",
    )
