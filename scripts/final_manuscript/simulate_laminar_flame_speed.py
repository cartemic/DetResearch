import datetime
import multiprocessing as mp
import os
import sys
import traceback
import warnings
import zoneinfo

import cantera as ct
import numpy as np
import pandas as pd
import tqdm
from uncertainties import unumpy as unp

from funcs.simulation import thermo
from sdtoolbox.postshock import CJspeed, PostShock_fr
from sdtoolbox.thermo import soundspeed_fr

FUEL = "CH4"
OXIDIZER = "N2O"
MECH = "gri30_highT.yaml"
LOCAL_TZ = zoneinfo.ZoneInfo("US/Pacific")


def main():
    df_measured = read_in_measured_data()
    df_result = simulate_all_lfs(df_measured)
    today = datetime.date.today().isoformat().replace("-", "_")
    data_file = f"laminar_flame_speeds_{today}.h5"
    with pd.HDFStore(os.path.join(os.path.dirname(__file__), data_file), "w") as store:
        store["data"] = df_result


def get_column_mean_with_uncertainty(df: pd.DataFrame, column: str):
    mean = np.mean(unp.uarray(df[column], df[f"u_{column}"]))

    # noinspection PyUnresolvedReferences
    return mean.nominal_value, mean.std_dev


def read_in_measured_data():
    data_loc = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "simulation_measurement_comparison",
        "measurements.h5",
    )
    with pd.HDFStore(data_loc, "r") as store:
        data = store.data

    data = data[(data["fuel"] == FUEL) & (data["oxidizer"] == OXIDIZER)]
    df_out = pd.DataFrame()
    for idx, ((diluent, phi_nom, dil_mf_nom), df_grouped) in enumerate(
        data.groupby(["diluent", "phi_nom", "dil_mf_nom"])
    ):
        p_0, u_p_0 = get_column_mean_with_uncertainty(df_grouped, "p_0")
        t_0, u_t_0 = get_column_mean_with_uncertainty(df_grouped, "t_0")
        phi, u_phi = get_column_mean_with_uncertainty(df_grouped, "phi")
        dil_mf, u_dil_mf = get_column_mean_with_uncertainty(df_grouped, "dil_mf")
        wave_speed, u_wave_speed = get_column_mean_with_uncertainty(df_grouped, "wave_speed")
        cell_size, u_cell_size = get_column_mean_with_uncertainty(df_grouped, "cell_size")
        this_row = pd.DataFrame(
            data={
                "diluent": diluent,
                "phi_nom": phi_nom,
                "dil_mf_nom": dil_mf_nom,
                "p_0": p_0,
                "u_p_0": u_p_0,
                "t_0": t_0,
                "phi": phi,
                "u_phi": u_phi,
                "dil_mf": dil_mf,
                "u_dil_mf": u_dil_mf,
                "wave_speed": wave_speed,
                "u_wave_speed": u_wave_speed,
                "cell_size_measured": cell_size,
                "u_cell_size_measured": u_cell_size,
            },
            index=[idx],
        )
        df_out = pd.concat((df_out, this_row), axis=0)

    return df_out


def simulate_all_lfs(df_measured: pd.DataFrame):
    with mp.Pool() as p:
        # noinspection PyTypeChecker
        df_result = pd.DataFrame(
            tqdm.tqdm(
                p.imap(simulate_single_lfs, df_measured.iterrows()),
                total=len(df_measured),
                file=sys.stdout,
                colour="green",
                unit="simulation",
            )
        )

    return df_result.sort_index()


def simulate_single_lfs(idx_and_row: tuple[int, pd.Series]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        row = idx_and_row[1].copy()
        # noinspection PyBroadException
        try:
            t0 = row["t_0"]
            p0 = row["p_0"]
            phi = row["phi"]
            dil = row["diluent"]
            dil_mf = row["dil_mf"]

            gas = ct.Solution(MECH)
            gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
            q = thermo.diluted_species_dict(gas.mole_fraction_dict(), dil, dil_mf)

            gas.TPX = t0, p0, q
            sound_speed_reactants = soundspeed_fr(gas)

            gas.TPX = t0, p0, q
            cj = CJspeed(P1=p0, T1=t0, q=gas.mole_fraction_dict(), mech=MECH)
            gas_post_shock_frozen = PostShock_fr(cj, p0, t0, q, MECH)
            sound_speed_products = soundspeed_fr(gas)
            gas.TPX = t0, p0, q

            lfs = thermo.calculate_laminar_flame_speed(gas)
            gas.TPX = t0, p0, q

            row["cj"] = cj
            row["lfs"] = lfs
            row["cp_initial"] = gas.cp
            row["cv_initial"] = gas.cv
            row["spec_heat_ratio_initial"] = gas.cp / gas.cv
            row["cp_ps"] = gas_post_shock_frozen.cp
            row["cv_ps"] = gas_post_shock_frozen.cv
            row["spec_heat_ratio_ps"] = gas_post_shock_frozen.cp / gas_post_shock_frozen.cv
            row["sound_speed_reactants"] = sound_speed_reactants
            row["sound_speed_products"] = sound_speed_products
            row["mach_reactants"] = row["wave_speed"] / sound_speed_reactants
            row["mach_cj_reactants"] = cj / sound_speed_reactants
            row["mach_products"] = row["wave_speed"] / sound_speed_products
            row["mach_cj_products"] = cj / sound_speed_products

        except Exception:
            problem = (
                f"{datetime.datetime.now(LOCAL_TZ)}\n"
                "=================\n"
                f"Conditions:\n{row}\n"
                f"Traceback:\n{traceback.format_exc()}"
                "\n\n"
            )
            with open("error_log_lfs_simulation", "a") as f:
                f.write(problem)

    return row


if __name__ == "__main__":
    main()
