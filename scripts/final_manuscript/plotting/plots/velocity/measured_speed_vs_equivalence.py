from typing import TYPE_CHECKING

from plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame
    from seaborn import FacetGrid


def plot(data: "DataFrame") -> "FacetGrid":
    return by_diluent.scatter_error(
        data=data,
        x="phi_nom",
        y="wave_speed",
        xerr=None,
        yerr="u_wave_speed",
        col="dil_mf_nom",
        xlabel=r"$\phi$",
        ylabel="Wave Speed (m/s)",
        title="Measured Speed vs. Equivalence Ratio",
    )
