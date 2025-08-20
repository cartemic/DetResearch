from typing import TYPE_CHECKING

from plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame
    from seaborn import FacetGrid


def plot(data: "DataFrame") -> "FacetGrid":
    return by_diluent.scatter_error(
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
