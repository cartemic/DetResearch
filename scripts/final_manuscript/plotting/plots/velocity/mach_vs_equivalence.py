from typing import TYPE_CHECKING

from plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame
    from seaborn import FacetGrid


def plot(data: "DataFrame") -> "FacetGrid":
    return by_diluent.scatter_error(
        data=data,
        x="phi_nom",
        y="mach",
        xerr=None,
        yerr=None,
        col="dil_mf_nom",
        xlabel=r"$\phi$",
        ylabel="Measured Mach Number",
        title="Mach Number vs. Equivalence Ratio",
    )
