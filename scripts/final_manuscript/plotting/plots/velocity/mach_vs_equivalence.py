from typing import TYPE_CHECKING

from scripts.final_manuscript.plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame


def plot(data: "DataFrame"):
    by_diluent.scatter_error(
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
