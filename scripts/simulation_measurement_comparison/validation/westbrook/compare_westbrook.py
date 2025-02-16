import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("darkgrid")

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def load_my_data(data_file: str = "simulated_and_measured.h5", column: str = "cell_size_westbrook") -> pd.DataFrame:
    with pd.HDFStore(
        os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), data_file),
        "r",
    ) as store:
        plot_data = pd.DataFrame()
        for _, row in store.data_fixed_uncert.iterrows():
            plot_data = pd.concat(
                [
                    plot_data,
                    pd.DataFrame(
                        {
                            "fuel": ["CH4"] * 2,
                            "oxidizer": ["N2O"] * 2,
                            "diluent": [row["diluent"]] * 2,
                            "phi_nom": [row["phi_nom"]] * 2,
                            "dil_mf_nom": [row["dil_mf_nom"]] * 2,
                            "cell_size": [row["cell_size_measured"], row[column]],
                            "source": ["measurement", "simulation"],
                            "uncertainty": [row["u_cell_size_measured"], np.nan],
                        }
                    ),
                ]
            )
        return plot_data


def load_westbrook_data() -> pd.DataFrame:
    with pd.HDFStore(os.path.join(SCRIPT_DIR, "westbrook_validation.h5"), "r") as store:
        plot_data = pd.DataFrame()
        for _, row in store.data.iterrows():
            plot_data = pd.concat(
                [
                    plot_data,
                    pd.DataFrame(
                        {
                            "mixture": [row["mixture"]] * 2,
                            "fuel": [row["fuel"]] * 2,
                            "oxidizer": [row["oxidizer"]] * 2,
                            "phi": [row["phi"]] * 2,
                            "pressure": [row["pressure"]] * 2,
                            "cell_size": [row["cell_size"], row["cell_size_westbrook"]],
                            "source": ["literature", "simulation"],
                        }
                    ),
                ]
            )
    return plot_data


def main():
    my_data = load_my_data()
    westbrook_data = load_westbrook_data()
    print(my_data.head())
    print(westbrook_data.head())

    deviation = (
        (
            westbrook_data[westbrook_data["source"] == "simulation"]
            .set_index(["mixture", "fuel", "oxidizer", "phi", "pressure"])["cell_size"]
            .div(
                westbrook_data[westbrook_data["source"] == "literature"].set_index(
                    ["mixture", "fuel", "oxidizer", "phi", "pressure"]
                )["cell_size"]
            )
        )
        .rename("simulated/measured")
        .reset_index()
    )

    grid = sns.relplot(x="pressure", y="cell_size", style="source", row="mixture", data=westbrook_data, kind="scatter")
    grid.set(xscale="log", yscale="log")
    for ax in grid.axes[0]:
        ax.invert_xaxis()

    grid2 = sns.relplot(
        x="dil_mf_nom", y="cell_size", style="source", row="phi_nom", col="diluent", data=my_data, kind="scatter"
    )
    grid2.set(xscale="log", yscale="log")
    for ax in grid2.axes[0]:
        ax.invert_xaxis()

    grid3 = sns.relplot(x="pressure", y="simulated/measured", row="mixture", data=deviation, kind="scatter")
    grid3.set(xscale="log", yscale="linear")
    for ax in grid3.axes:
        ax[0].invert_xaxis()
        ax[0].axhline(3, c="k", alpha=0.5)

    plt.figure()
    westbrook_error_pct = (
        (
            westbrook_data[westbrook_data["source"] == "simulation"].set_index(["mixture", "fuel", "oxidizer", "phi"])[
                "cell_size"
            ]
            - westbrook_data[westbrook_data["source"] == "literature"].set_index(
                ["mixture", "fuel", "oxidizer", "phi"]
            )["cell_size"]
        )
        .abs()
        .div(
            westbrook_data[westbrook_data["source"] == "literature"].set_index(["mixture", "fuel", "oxidizer", "phi"])[
                "cell_size"
            ]
        )
        .mul(100)
    )
    my_error_pct = (
        (
            my_data[my_data["source"] == "simulation"].set_index(["diluent", "dil_mf_nom", "phi_nom"])["cell_size"]
            - my_data[my_data["source"] == "measurement"].set_index(["diluent", "dil_mf_nom", "phi_nom"])["cell_size"]
        )
        .abs()
        .div(my_data[my_data["source"] == "measurement"].set_index(["diluent", "dil_mf_nom", "phi_nom"])["cell_size"])
        .mul(100)
    )

    # todo: finalize plots, summarize, and email Kyle
    simulated_key = "simulation error (abs % measured)"
    violin = pd.concat(
        (
            pd.DataFrame({simulated_key: my_error_pct.to_numpy(), "source": "mine"}),
            pd.DataFrame({simulated_key: westbrook_error_pct.to_numpy(), "source": "literature"}),
        )
    )
    sns.violinplot(violin, x="source", y=simulated_key)
    plt.show()


if __name__ == "__main__":
    main()
