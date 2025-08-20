from typing import TYPE_CHECKING

from plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame
    from seaborn import FacetGrid


def plot(data: "DataFrame") -> "FacetGrid":
    return by_diluent.scatter_error(
        data=data,
        x="phi_nom",
        y="cj_speed",
        xerr=None,
        yerr=None,
        col="dil_mf_nom",
        xlabel=r"$\phi$",
        ylabel="CJ Speed (m/s)",
        title="CJ Speed vs. Equivalence Ratio",
    )
