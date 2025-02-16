import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import linregress
from uncertainties import unumpy as unp

import funcs

d_drive = funcs.dir.d_drive
PLOT_FILETYPE = "pdf"
PAPER_BASE_DIR = os.path.join(d_drive, "Final-Manuscript")
CSV_LOCATION = os.path.join(PAPER_BASE_DIR, "data")
IMAGE_LOCATION = os.path.join(PAPER_BASE_DIR, "images")
DPI = 200

# ibm color-blind safe palette
# https://lospec.com/palette-list/ibm-color-blind-safe
# https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
PALETTE = sns.color_palette(["#648fff", "#fe6100", "#dc267f", "#785ef0", "#ffb000", "#000000", "#ffffff"])

# sns.set_style("darkgrid")
sns.set_palette(PALETTE)
PLOT_FONT_SIZE = 20
sns.set_color_codes("deep")
sns.set_context(
    # "paper",
    "talk",
    rc={
        "font.size": PLOT_FONT_SIZE,
        "axes.titlesize": PLOT_FONT_SIZE + 1.5,
        "axes.titleweight": "bold",
        "axes.labelsize": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE,
        "ytick.labelsize": PLOT_FONT_SIZE,
    },
)
sns.set_style(
    {
        "font.family": "serif",
        "font.serif": "Computer Modern",
    }
)
# plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.dpi"] = DPI


def load_cell_size_data():
    with pd.HDFStore(os.path.join("..", "simulation_measurement_comparison", "simulated_and_measured.h5")) as store:
        data = store.data_fixed_uncert_with_co2e[
            [
                "diluent",
                "phi_nom",
                "phi",
                "u_phi",
                "dil_mf_nom",
                "dil_mf",
                "u_dil_mf",
                "dil_mf_co2e",
                "u_dil_mf_co2e",
                "wave_speed",
                "u_wave_speed",
                "cell_size_measured",
                "u_cell_size_measured",
                "cell_size_westbrook",
            ]
        ]

    data.rename(
        {
            "cell_size_measured": "measured",
            "cell_size_westbrook": "simulated",
            "u_cell_size_measured": "u_cell_size",
        },
        axis=1,
        inplace=True,
    )
    data = data.melt(
        id_vars=[
            "diluent",
            "phi_nom",
            "phi",
            "u_phi",
            "dil_mf_nom",
            "dil_mf",
            "u_dil_mf",
            "dil_mf_co2e",
            "u_dil_mf_co2e",
            "wave_speed",
            "u_wave_speed",
            "u_cell_size",
        ],
        value_vars=["measured", "simulated"],
        var_name="method",
        value_name="cell_size",
    )
    data.loc[data["method"] == "simulated", "u_cell_size"] = np.nan

    return data


def load_wave_speed_data():
    data = pd.read_csv(os.path.join("..", "cj_study", "cj_tad_ss_results_with_co2e.csv"))[
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
            "dil_mf_co2e",
            "u_dil_mf_co2e",
            "wave_speed",
            "u_wave_speed",
            "cj_speed",
            "sound_speed",
            "t_ad",
        ]
    ]

    # Filter out unsuccessful detonations
    data = data[(data["cj_speed"].notna()) & (data["sound_speed"].notna()) & (data["wave_speed"].notna())]

    # Ignore propane/air data
    data = data[data["fuel"] == "CH4"]

    return data


def plot_cell_sizes(cell_size_data: pd.DataFrame):
    plt.figure()
    cell_size_plots = sns.relplot(
        x="phi_nom",
        y="cell_size",
        col="dil_mf_nom",
        hue="diluent",
        style="diluent",
        row="method",
        data=cell_size_data,
        zorder=4,
    )
    for idx, ax in enumerate(cell_size_plots.axes.flatten()):
        which = "Measured" if idx < 3 else "Simulated"
        ax.set_ylabel(f"{which}\nCell Size (mm)")
        if ax.get_xlabel() is not None:
            ax.set_xlabel(r"$\phi_{nominal}$")
        # I don't remember what this magic number means...
        # ruff: noqa: PLR2004
        if idx < 3:
            title = ax.get_title()
            dil_mf_nom = float(title.split(" = ")[2])
            sub_df = cell_size_data[(cell_size_data["dil_mf_nom"] == dil_mf_nom)]
            for _, sub_df_diluent in sub_df.groupby("diluent"):
                ax.errorbar(
                    x=sub_df_diluent["phi_nom"],
                    y=sub_df_diluent["cell_size"],
                    yerr=sub_df_diluent["u_cell_size"],
                    ls="None",
                    lw=0.5,
                    zorder=3,
                    capsize=3,
                    marker=None,
                )
            ax.set_title(
                ax.get_title()
                .replace("dil_mf_nom", "$X_{dil, nominal}$ (CO$_2$e)")
                .replace("method = ", "")
                .replace("simulated", "")
                .replace("measured", "")
                .replace(" | ", "")
            )
        else:
            ax.set_title(None)
    cell_size_plots.tight_layout()
    plt.savefig(os.path.join(IMAGE_LOCATION, f"cell_sizes.{PLOT_FILETYPE}"), format=PLOT_FILETYPE)

    # Measured only
    plt.figure()
    measured = cell_size_data[cell_size_data["method"] == "measured"]
    cell_size_plots = sns.relplot(
        x="phi_nom",
        y="cell_size",
        col="dil_mf_nom",
        hue="diluent",
        style="diluent",
        data=measured,
        zorder=4,
    )
    for ax in cell_size_plots.axes.flatten():
        ax.set_ylabel("Cell Size (mm)")
        if ax.get_xlabel() is not None:
            ax.set_xlabel(r"$\phi_{nominal}$")
        title = ax.get_title()
        dil_mf_nom = float(title.split(" = ")[1])
        sub_df = measured[(measured["dil_mf_nom"] == dil_mf_nom)]
        for _, sub_df_diluent in sub_df.groupby("diluent"):
            ax.errorbar(
                x=sub_df_diluent["phi_nom"],
                y=sub_df_diluent["cell_size"],
                yerr=sub_df_diluent["u_cell_size"],
                ls="None",
                lw=0.5,
                zorder=3,
                capsize=3,
                marker=None,
            )
        ax.set_title(title.replace("dil_mf_nom", "$X_{dil, nominal}$ (CO2e)"))
    cell_size_plots.tight_layout()
    plt.savefig(os.path.join(IMAGE_LOCATION, f"cell_sizes_measured_x_nom_co2e.{PLOT_FILETYPE}"), format=PLOT_FILETYPE)

    # Measured only -- vs. x, not co2e
    plt.figure()
    measured = cell_size_data[cell_size_data["method"] == "measured"]
    cell_size_plots = sns.relplot(
        x="dil_mf",
        y="cell_size",
        col="phi_nom",
        hue="diluent",
        style="diluent",
        data=measured,
        zorder=4,
    )
    for ax in cell_size_plots.axes.flatten():
        ax.set_ylabel("Cell Size (mm)")
        if ax.get_xlabel() is not None:
            ax.set_xlabel("$X_{dil}$")
        title = ax.get_title()
        phi_nom = float(title.split(" = ")[1])
        sub_df = measured[(measured["phi_nom"] == phi_nom)]
        for _, sub_df_diluent in sub_df.groupby("diluent"):
            ax.errorbar(
                x=sub_df_diluent["dil_mf"],
                y=sub_df_diluent["cell_size"],
                yerr=sub_df_diluent["u_cell_size"],
                ls="None",
                lw=0.5,
                zorder=3,
                capsize=3,
                marker=None,
            )
        ax.set_title(title.replace("phi_nom", r"$\phi_{nominal}$"))
    cell_size_plots.tight_layout()
    plt.savefig(os.path.join(IMAGE_LOCATION, f"cell_sizes_measured_x_actual.{PLOT_FILETYPE}"), format=PLOT_FILETYPE)

    # Measured only -- vs. x, co2e
    plt.figure()
    measured = cell_size_data[cell_size_data["method"] == "measured"]
    cell_size_plots = sns.relplot(
        x="dil_mf_co2e",
        y="cell_size",
        col="phi_nom",
        hue="diluent",
        style="diluent",
        data=measured,
        zorder=4,
    )
    for ax in cell_size_plots.axes.flatten():
        ax.set_ylabel("Cell Size (mm)")
        if ax.get_xlabel() is not None:
            ax.set_xlabel("$X_{dil}$ (CO$_2$e)")
        title = ax.get_title()
        phi_nom = float(title.split(" = ")[1])
        sub_df = measured[(measured["phi_nom"] == phi_nom)]
        for _, sub_df_diluent in sub_df.groupby("diluent"):
            ax.errorbar(
                x=sub_df_diluent["dil_mf_co2e"],
                y=sub_df_diluent["cell_size"],
                yerr=sub_df_diluent["u_cell_size"],
                ls="None",
                lw=0.5,
                zorder=3,
                capsize=3,
                marker=None,
            )
        ax.set_title(title.replace("phi_nom", r"$\phi_{nominal}$"))
    cell_size_plots.tight_layout()
    plt.savefig(os.path.join(IMAGE_LOCATION, f"cell_sizes_measured_x_co2e.{PLOT_FILETYPE}"), format=PLOT_FILETYPE)

    # ratio of simulated to measured
    plt.figure()
    simulated = cell_size_data[cell_size_data["method"] == "simulated"].copy()
    sim_meas_ratio = simulated["cell_size"] / unp.uarray(measured["cell_size"], measured["u_cell_size"])
    simulated["sim_meas"] = unp.nominal_values(sim_meas_ratio)
    simulated["u_sim_meas"] = unp.std_devs(sim_meas_ratio)
    cell_size_plots = sns.relplot(
        x="phi_nom",
        y="sim_meas",
        col="dil_mf_nom",
        hue="diluent",
        style="diluent",
        data=simulated,
        zorder=4,
    )
    for ax in cell_size_plots.axes.flatten():
        ax.set_ylabel("Simulated / Measured")
        if ax.get_xlabel() is not None:
            ax.set_xlabel(r"$\phi_{nominal}$")
        title = ax.get_title()
        dil_mf_nom = float(title.split(" = ")[1])
        sub_df = simulated[(simulated["dil_mf_nom"] == dil_mf_nom)]
        for _, sub_df_diluent in sub_df.groupby("diluent"):
            ax.errorbar(
                x=sub_df_diluent["phi_nom"],
                y=sub_df_diluent["sim_meas"],
                yerr=sub_df_diluent["u_sim_meas"],
                ls="None",
                lw=0.5,
                zorder=3,
                capsize=3,
                marker=None,
            )
        ax.set_title(
            ax.get_title()
            .replace("dil_mf_nom", "$X_{dil, nominal}$ (CO$_2$e)")
            .replace("method = ", "")
            .replace("simulated", "Simulated")
            .replace("measured", "Measured")
            .replace(" | ", ", ")
        )
    cell_size_plots.tight_layout()
    plt.savefig(os.path.join(IMAGE_LOCATION, f"cell_size_ratios.{PLOT_FILETYPE}"), format=PLOT_FILETYPE)


# Allow private member access for plot stuff
# ruff: noqa: SLF001
def plot_measured_wave_speeds(speed_data: pd.DataFrame):
    plt.figure()
    measured_speed_plots = sns.relplot(
        x="dil_mf",
        y="wave_speed",
        hue="phi_nom",
        style="phi_nom",
        col="diluent",
        data=speed_data,
        palette=PALETTE[:3],
    )
    measured_speed_plots.tight_layout()
    measured_speed_plots.set_xlabels("$X_{dil}$")
    measured_speed_plots.set_ylabels("Wave speed (m/s)")
    # noinspection PyProtectedMember
    measured_speed_plots._legend.set_title(r"$\phi_{nominal}$")
    for ax in measured_speed_plots.axes.flatten():
        ax.set_title(ax.get_title().replace("diluent", "Diluent"))
    measured_speed_plots.tight_layout()
    plt.savefig(os.path.join(IMAGE_LOCATION, f"wave_speeds_measured.{PLOT_FILETYPE}"), format=PLOT_FILETYPE)

    plt.figure()
    measured_speed_plots = sns.relplot(
        x="dil_mf_co2e",
        y="wave_speed",
        hue="phi_nom",
        style="phi_nom",
        col="diluent",
        data=speed_data,
        palette=PALETTE[:3],
    )
    measured_speed_plots.tight_layout()
    measured_speed_plots.set_xlabels("$X_{dil}$ (CO$_2$e)")
    measured_speed_plots.set_ylabels("Wave speed (m/s)")
    # noinspection PyProtectedMember
    measured_speed_plots._legend.set_title(r"$\phi_{nominal}$")
    for ax in measured_speed_plots.axes.flatten():
        ax.set_title(ax.get_title().replace("diluent", "Diluent"))
    measured_speed_plots.tight_layout()
    plt.savefig(os.path.join(IMAGE_LOCATION, f"wave_speeds_measured_co2e.{PLOT_FILETYPE}"), format=PLOT_FILETYPE)


def plot_ss_normalized_wave_speeds(wave_speed_data: pd.DataFrame):
    # use dil_mf or dil_mf_nom everywhere
    mol_frac = "dil_mf"

    # Normalize
    wave_speed_data["normalized"] = wave_speed_data["wave_speed"] / wave_speed_data["sound_speed"]

    plt.figure()
    sns.scatterplot(
        x=mol_frac,
        y="normalized",
        hue="diluent",
        data=wave_speed_data,
    )

    # Perform regressions
    n2_data = wave_speed_data[wave_speed_data["diluent"] == "N2"]
    co2_data = wave_speed_data[wave_speed_data["diluent"] == "CO2"]
    n2_regression = linregress(n2_data[mol_frac], n2_data["normalized"])
    co2_regression = linregress(co2_data[mol_frac], co2_data["normalized"])
    total_regression = linregress(wave_speed_data[mol_frac], wave_speed_data["normalized"])

    # x = np.array([wave_speed_data[mol_frac].min(), wave_speed_data[mol_frac].max()])
    # plt.plot(x, total_regression.slope * x + total_regression.intercept, "k--", alpha=0.5)

    for i, (regression, data) in enumerate(zip([n2_regression, co2_regression], [n2_data, co2_data])):
        x = np.array([data[mol_frac].min(), data[mol_frac].max()])
        plt.plot(x, regression.slope * x + regression.intercept, f"C{i}")

    plot_center = wave_speed_data["normalized"].mean()
    window_height = 1
    plt.ylim(plot_center - window_height / 2, plot_center + window_height / 2)
    plt.xlabel("$X_{dil}$")
    plt.ylabel("Mach Number")

    # Regression summary table
    # todo: show how close these curve fits are statistically
    pd.DataFrame(
        {
            "CO2": {
                "slope": co2_regression.slope,
                "intercept": co2_regression.intercept,
                "R$^{2}$": co2_regression.rvalue**2,
            },
            "N2": {
                "slope": n2_regression.slope,
                "intercept": n2_regression.intercept,
                "R$^{2}$": n2_regression.rvalue**2,
            },
            "Combined": {
                "slope": total_regression.slope,
                "intercept": total_regression.intercept,
                "R$^{2}$": total_regression.rvalue**2,
            },
        }
    ).to_csv(os.path.join(CSV_LOCATION, "normalized_curve_fit_summary.csv"))

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_LOCATION, f"wave_speeds_ss_normalized.{PLOT_FILETYPE}"), format=PLOT_FILETYPE)


def main():
    cell_size_data = load_cell_size_data()
    plot_cell_sizes(cell_size_data)

    wave_speed_data = load_wave_speed_data()
    plot_measured_wave_speeds(wave_speed_data)
    plot_ss_normalized_wave_speeds(wave_speed_data)


if __name__ == "__main__":
    main()
