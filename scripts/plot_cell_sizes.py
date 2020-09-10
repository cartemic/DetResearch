###
# script for measuring cell size from triple point locations and post
# processed tube data
###

import os

import numpy as np
import pandas as pd
import seaborn as sns
import uncertainties as un
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter, EngFormatter
from scipy.stats import t
from uncertainties import unumpy as unp

import funcs

sns.set_palette("colorblind")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Experimental data
dir_data = os.path.join(
    "/d/",
    "Data",
    "Processed",
    "Data"
)
loc_processed = os.path.join(dir_data, "data_fffff.h5")
loc_schlieren = os.path.join(dir_data, "schlieren_fffff.h5")

# Simulated data
simulated_data_loc = "/d/DetResearch/scripts/cell_size_simulated.h5"
with pd.HDFStore(simulated_data_loc, "r") as store:
    df_sim = store.results
df_sim.loc[pd.isna(df_sim["diluent"]), "diluent"] = "None"
df_sim["phi"] = 1
df_sim.reset_index(drop=True, inplace=True)

# Choose estimator
estimator = np.median

# Read in data
with pd.HDFStore(loc_processed, "r") as store:
    df_day = store["data"]
with pd.HDFStore(loc_schlieren, "r") as store:
    df_schlieren = store.data

# set dates to process. currently set to all dates
dates_to_process = sorted(list(set(df_day["date"].values)))
df_day = df_day[df_day["date"].isin(dates_to_process)]
# df_day["success"] = df_day["schlieren"].notna()


# cut down data
df_day = df_day[
    df_day["schlieren"].notna() &  # only shots with schlieren data
    (df_day["fuel"] == "C3H8") &  # propane
    (df_day["oxidizer"] == "air") &  # air
    (df_day["phi_nom"] > 0.85) &  # Throw out low ER data
    ~(  # Get rid of high dilution at low ER
            (df_day["diluent"] != "None") &
            (np.isclose(df_day["phi_nom"], 0.9)) &
            (np.isclose(df_day["dil_mf_nom"], 0.02))
    )
    ]

df_schlieren_uncert = funcs.uncertainty.df_merge_uncert(
    df_schlieren.dropna(),
    ["loc_px", "delta_px", "spatial_centerline"]
)

# turn locations into cell sizes
df_schlieren_uncert["measured"] = df_schlieren_uncert["delta_px"] * \
                                  df_schlieren_uncert["spatial_centerline"] * 2
red_columns = ["p_0", "t_0", "phi", "dil_mf", "wave_speed"]
all_sep_columns = ["date", "shot", ] + red_columns
all_out_columns = ["diluent"] + red_columns + ["cell_size"]
df_out = pd.DataFrame(columns=all_out_columns)
for (_dil, _phi, _mf), df_day_red in df_day.groupby(
        ["diluent", "phi_nom", "dil_mf_nom"]
):
    if len(df_day_red):

        # collect population standard deviation of measurements and turn into CI
        t_stat = t.ppf(0.975, len(df_day_red) - 1)
        series_pop_std = pd.Series(
            (np.nanstd(df_day_red[c].values) * t_stat for c in red_columns),
            index=red_columns
        )

        # convert columns to uncertainty for convenience and calculate means
        df_day_red = funcs.uncertainty.df_merge_uncert(
            df_day_red,
            red_columns,
        )
        df_day_red = df_day_red[all_sep_columns]
        series_mean = pd.Series(
            [
                _dil,
                *(np.nanmean(df_day_red[c].values) for c in red_columns),
                np.NaN
            ],
            index=all_out_columns
        )
        series_mean["phi_nom"] = _phi

        # add population and measurement uncertainty
        for c in red_columns:
            series_mean[c] += un.ufloat(0, series_pop_std[c])

        # collect schlieren measurements
        df_schlieren_red = df_schlieren_uncert[
            (df_schlieren_uncert["date"].isin(df_day_red["date"])) &
            (df_schlieren_uncert["shot"].isin(df_day_red["shot"]))
            ]
        series_mean["cell_size"] = estimator(
            df_schlieren_red["measured"].values)
        #         print(np.median(df_schlieren_red["measured"].values))
        legend_label = "ϕ={:1.1f}, dil={:s}, CO2 equivalent={:3.2f}".format(
            _phi,
            _dil,
            _mf
        )

        # EXPERIMENTING WITH DISTRIBUTIONS
        plt.figure()
        sns.distplot(unp.nominal_values(df_schlieren_red["measured"]))
        plt.axvline(np.median(unp.nominal_values(df_schlieren_red["measured"])),
                    ls=":")
        plt.axvline(np.mean(unp.nominal_values(df_schlieren_red["measured"])),
                    ls="--")

        plt.figure()
        medians = df_schlieren_red["measured"].apply(
            lambda x: x.nominal_value
        ).rolling(
            len(df_schlieren_red),
            min_periods=0
        ).median().reset_index(drop=True)
        (
                (medians.iloc[:-1] - medians.iloc[-1]).abs() /
                medians.iloc[-1] * 100
        ).plot(
            style="-",
            alpha=0.5,
            mew=0,
            label=legend_label,
            figsize=(18, 4)
        )

        df_out = df_out.append(
            series_mean,
            ignore_index=True
        )

        percent_cutoff = 2
        plt.axhline(
            percent_cutoff,
            ls="--",
            c="k",
            alpha=0.25,
            zorder=-1,
            label="{:0.2f} %".format(percent_cutoff)
        )
        plt.xlabel("Measurement #")
        plt.ylabel("Absolute Estimator\nError (%)")
        plt.title(
            f"Convergence of measured cell sizes\n"
            fr"$\phi$={_phi}, diluent={_dil}, CO2e mf={_mf}",
            weight="bold"
        )

# plt.show()
plt.close("all")
print(df_out)

df_plot_base = funcs.uncertainty.df_split_uncert(
    df_out,
    ["p_0", "t_0", "phi", "dil_mf", "wave_speed", "cell_size"]
)


def convert_to_co2(row):
    if row["diluent"] == "None":
        return 0., 0.
    elif row["diluent"] == "CO2":
        if "u_dil_mf" in row.keys():
            return row["dil_mf"], row["u_dil_mf"]
        else:
            return row["dil_mf"], np.NaN
    else:
        dil_mf_co2 = funcs.specific_heat_matching.match_adiabatic_temp(
            "gri30.cti",
            "C3H8",
            "O2:1 N2:3.76",
            row["phi"],
            row["diluent"],
            row["dil_mf"],
            "CO2",
            300,
            101325,
        )
        if "u_dil_mf" in row.keys():
            dil_mf_co2_perturbed = funcs.specific_heat_matching.\
                match_adiabatic_temp(
                    "gri30.cti",
                    "C3H8",
                    "O2:1 N2:3.76",
                    row["phi"],
                    row["diluent"],
                    row["dil_mf"] + row["u_dil_mf"],
                    "CO2",
                    300,
                    101325,
                )
            dil_mf_pert = np.abs(dil_mf_co2_perturbed - dil_mf_co2)
        else:
            dil_mf_pert = np.NaN
        return dil_mf_co2, dil_mf_pert


for i, r in df_plot_base.iterrows():
    new_values = convert_to_co2(r)
    df_plot_base.loc[i, "dil_mf_CO2e"] = new_values[0]
    df_plot_base.loc[i, "u_dil_mf_CO2e"] = new_values[1]

for i, r in df_sim.iterrows():
    new_values = convert_to_co2(r)
    df_sim.loc[i, "dil_mf_CO2e"] = new_values[0]


def split_diluent_dataframes(df):
    _df_co2 = df[
        (df["diluent"] == "CO2") |
        (df["diluent"] == "None")
        ].copy()
    _df_n2 = df[
        (df["diluent"] == "N2") |
        (df["diluent"] == "None")
        ].copy()
    _df_co2["diluent"] = "CO2"
    _df_n2["diluent"] = "N2"
    return _df_co2, _df_n2


df_plot_diluted_exp = pd.concat(split_diluent_dataframes(df_plot_base))
f = sns.FacetGrid(
    data=df_plot_diluted_exp,
    col="diluent",
)
f.map_dataframe(
    sns.scatterplot,
    x="dil_mf_CO2e",
    y="cell_size",
    hue="phi_nom",
    legend="full",
    palette=sns.color_palette("colorblind", n_colors=2),
)
ax = plt.gca()
plt.xticks([0, 0.01, 0.02])
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(EngFormatter("mm"))
f.set_axis_labels("Mole Fraction (CO2e)", "Cell Size")
plt.tight_layout()
sns.despine()

print(df_sim)

df_plot_diluted_exp = pd.concat(split_diluent_dataframes(df_sim))
f = sns.FacetGrid(
    data=df_plot_diluted_exp,
    col="diluent",
)
f.map_dataframe(
    sns.scatterplot,
    x="dil_mf_CO2e",
    y="westbrook",
    hue="mechanism",
    legend="full",
    palette=sns.color_palette("colorblind", n_colors=3),
)
ax = plt.gca()
plt.legend()
plt.xticks([0, 0.01, 0.02])
ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
ax.yaxis.set_major_formatter(EngFormatter("mm"))
f.set_axis_labels("Mole Fraction (CO2e)", "Cell Size")
plt.tight_layout()
sns.despine()

plt.show()
