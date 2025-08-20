from typing import TYPE_CHECKING

from plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame
    from seaborn import FacetGrid


def plot(data: "DataFrame") -> "FacetGrid":
    return by_diluent.scatter_error(
        data=data,
        x="dil_mf",
        y="cell_size",
        xerr="u_dil_mf",
        yerr="u_cell_size",
        col="phi_nom",
        ylabel="Cell Size (mm)",
        xlabel=r"$X_{dil}$",
        title="Measured Cell Size vs. Diluent Mole Fraction",
    )
