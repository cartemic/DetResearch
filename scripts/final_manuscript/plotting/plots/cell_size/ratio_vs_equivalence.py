from typing import TYPE_CHECKING

from scripts.final_manuscript.plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame


def plot(data: "DataFrame"):
    by_diluent.scatter_error(
        data=data,
        x="phi",
        y="cell_size",
        xerr="u_phi",
        yerr="u_cell_size",
        col="dil_mf_nom",
        xlabel=r"$\phi$",
        ylabel="Simulated / Measured",
        title="Cell Size Ratio vs. Equivalence Ratio",
    )
