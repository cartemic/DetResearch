from typing import TYPE_CHECKING

from scripts.final_manuscript.plotting.plots import by_diluent

if TYPE_CHECKING:
    from pandas import DataFrame


def plot(data: "DataFrame"):
    by_diluent.scatter_error(
        data=data,
        x="phi_nom",
        y="cj_ratio",
        xerr=None,
        yerr="u_cj_ratio",
        col="dil_mf_nom",
        xlabel=r"$\phi$",
        ylabel="Speed Ratio",
        title="CJ/Measured Speed vs. Equivalence Ratio",
    )
