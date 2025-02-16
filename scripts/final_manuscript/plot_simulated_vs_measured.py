import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from scripts.final_manuscript.plot_settings import PlotKind, set_palette, set_style

SCRIPT_DIR = os.path.dirname(__file__)
# CELL_SIZE_DATA_PATH = os.path.join(
#     os.path.dirname(SCRIPT_DIR), "simulation_measurement_comparison", "simulated_and_measured_2024_03_02.h5"
# )
# CELL_SIZE_DATA_PATH = os.path.join(
#     os.path.dirname(SCRIPT_DIR), "simulation_measurement_comparison", "simulated_and_measured_mevel2015.h5"
# )
CELL_SIZE_DATA_PATH = os.path.join(
    os.path.dirname(SCRIPT_DIR), "simulation_measurement_comparison", "simulated_and_measured_inert_co2.h5"
)
CO2 = "CO2i"
WAVE_SPEED_DATA_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "cj_study", "cj_tad_ss_results.csv")


def load_data():
    cell_size_data = pd.DataFrame()
    with pd.HDFStore(
        CELL_SIZE_DATA_PATH,
        "r",
    ) as store:
        for _, row in store.data_fixed_uncert.iterrows():
            cell_size_data = pd.concat(
                [
                    cell_size_data,
                    pd.DataFrame(
                        {
                            "fuel": ["CH4"] * 2,
                            "oxidizer": ["N2O"] * 2,
                            "diluent": [row["diluent"]] * 2,
                            "phi_nom": [row["phi_nom"]] * 2,
                            "dil_mf_nom": [row["dil_mf_nom"]] * 2,
                            "cell_size": [row["cell_size_measured"], row["cell_size_westbrook_2"]],
                            "source": ["measurement", "simulation"],
                            "uncertainty": [row["u_cell_size_measured"], np.nan],
                        }
                    ),
                ]
            )

    speed_data = pd.read_csv(WAVE_SPEED_DATA_PATH)[
        [
            "fuel",
            "oxidizer",
            "diluent",
            "phi_nom",
            "phi",
            "u_phi",
            "dil_mf_nom",
            "dil_mf",
            "u_dil_mf",
            "wave_speed",
            "u_wave_speed",
            "cj_speed",
            "sound_speed",
            "t_ad",
        ]
    ]
    # Filter out unsuccessful detonations
    speed_data = speed_data[
        (speed_data["cj_speed"].notna()) & (speed_data["sound_speed"].notna()) & (speed_data["wave_speed"].notna())
    ]
    # Ignore propane/air data
    speed_data = speed_data[speed_data["fuel"] == "CH4"].reset_index(drop=True)

    return cell_size_data, speed_data


def plot_cell_size(data: pd.DataFrame, show_title: bool, save_plot: bool):
    grid = sns.relplot(
        x="dil_mf_nom",
        y="cell_size",
        row="source",
        col="phi_nom",
        hue="diluent",
        data=data,
        kind="scatter",
        zorder=2,
        facet_kws={"sharey": "row"},
    )
    grid.fig.set_size_inches((7, 6))
    if show_title:
        grid.fig.suptitle("Diluent Comparison", weight="bold")

    measurements = data[data["source"] == "measurement"]
    plot_args = {
        "x": "dil_mf_nom",
        "y": "cell_size",
        "yerr": "uncertainty",
        "ls": "None",
        "linewidth": 0.5,
        "zorder": 1,
        "capsize": 4,
        "alpha": 0.4,
    }
    er_low = 0.4
    er_med = 0.7
    er_high = 1.0
    grid.axes[0][0].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == er_low) & (measurements["diluent"] == CO2)]
    )
    grid.axes[0][0].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == er_low) & (measurements["diluent"] == "N2")]
    )
    grid.axes[0][1].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == er_med) & (measurements["diluent"] == CO2)]
    )
    grid.axes[0][1].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == er_med) & (measurements["diluent"] == "N2")]
    )
    grid.axes[0][2].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == er_high) & (measurements["diluent"] == CO2)]
    )
    grid.axes[0][2].errorbar(
        **plot_args, data=measurements[(measurements["phi_nom"] == er_high) & (measurements["diluent"] == "N2")]
    )

    for ax in grid.axes.flatten():
        title = ax.get_title()
        ax.set_title(
            title.replace("source = ", "")
            .replace("measurement ", "Measured")
            .replace("simulation ", "Simulated")
            .replace("| ", "\n")
            .replace("phi_nom", r"$\phi_{nom}$")
        )
        if ax.get_xlabel():
            ax.set_xlabel("$x_{CO_{2},nom}$")
        if ax.get_ylabel():
            ax.set_ylabel("Cell size (mm)")
    grid.tight_layout()

    if save_plot:
        grid.fig.savefig("plots/simulated_vs_measured.pdf")

    ratio = (
        data[data["source"] == "simulation"].set_index(["dil_mf_nom", "phi_nom", "fuel", "oxidizer", "diluent"])[
            "cell_size"
        ]
        / data[data["source"] == "measurement"].set_index(["dil_mf_nom", "phi_nom", "fuel", "oxidizer", "diluent"])[
            "cell_size"
        ]
    )
    ratio.name = "ratio"
    ratio = ratio.reset_index()
    grid = sns.relplot(x="phi_nom", y="ratio", col="dil_mf_nom", hue="diluent", data=ratio)
    grid.fig.set_size_inches((7, 3))
    if show_title:
        grid.fig.suptitle("Simulated / Measured Cell Size Ratios", weight="bold")
    for ax in grid.axes.flatten():
        title = ax.get_title()
        ax.set_title(title.replace("dil_mf_nom", "$x_{CO_{2},nom}$"))
        if ax.get_xlabel():
            ax.set_xlabel(r"$\phi_{nom}$")
        if ax.get_ylabel():
            ax.set_ylabel("Simulated / Measured")
    grid.tight_layout()

    if save_plot:
        grid.fig.savefig("plots/simulated_measured_ratio.pdf")


def plot_wave_speed(data: pd.DataFrame, show_title: bool, save_plot: bool):
    data["measured_cj_ratio"] = data["wave_speed"] / data["cj_speed"]
    data["measured_mach"] = data["wave_speed"] / data["sound_speed"]
    hue_order = [CO2, "N2"]
    # CJ
    grid = sns.displot(data, x="measured_cj_ratio", hue="diluent", kind="kde", hue_order=hue_order)
    grid.fig.set_size_inches((7, 2))
    if show_title:
        grid.fig.suptitle("Measured / CJ Speed Ratio", weight="bold")
    for ax in grid.axes.flatten():
        if ax.get_xlabel():
            ax.set_xlabel("D/D$_{CJ}$")
        if ax.get_ylabel():
            ax.set_ylabel("Density")
    grid.tight_layout()

    if save_plot:
        grid.fig.savefig("plots/measured_cj_ratio.pdf")

    # Mach
    grid = sns.displot(data, x="measured_mach", hue="diluent", kind="kde", hue_order=hue_order)
    grid.fig.set_size_inches((7, 2))
    if show_title:
        grid.fig.suptitle("Measured Mach Number", weight="bold")
    for ax in grid.axes.flatten():
        if ax.get_xlabel():
            ax.set_xlabel("Ma")
        if ax.get_ylabel():
            ax.set_ylabel("Density")
    grid.tight_layout()

    if save_plot:
        grid.fig.savefig("plots/measured_mach.pdf")


def main():
    show_plot = True
    save_plot = False
    show_title = True

    set_palette(plot_kind=PlotKind.CONDITION)
    set_style()
    sns.set_style(
        {
            "font.scale": 0.75,
            "font.family": "serif",
            "font.serif": "Computer Modern",
        }
    )

    cell_size_data, wave_speed_data = load_data()
    plot_cell_size(data=cell_size_data, show_title=show_title, save_plot=save_plot)
    plot_wave_speed(data=wave_speed_data, show_title=show_title, save_plot=save_plot)

    if show_plot:
        plt.show()


if __name__ == "__main__":
    main()
