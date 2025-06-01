import psycopg

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp

from funcs.simulation.interpolate import interpolate
from scripts.final_manuscript.plotting import directory


def load_cell_size_data() -> pd.DataFrame:
    with pd.HDFStore(directory.SIM_MEAS_COMPARISON / "simulated_and_measured.h5", "r") as store:
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
    data = data.rename(
        {
            "cell_size_measured": "measured",
            "cell_size_westbrook": "simulated",
            "u_cell_size_measured": "u_cell_size",
        },
        axis=1,
    ).melt(
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


def calculate_simulated_measured_ratio(simulated: pd.DataFrame, measured: pd.DataFrame) -> pd.DataFrame:
    simulated_prep = simulated.set_index(["dil_mf_nom", "phi_nom", "diluent"]).sort_index()
    measured_prep = measured.set_index(["dil_mf_nom", "phi_nom", "diluent"]).sort_index()
    if simulated_prep.shape[0] != measured_prep.shape[0]:
        raise ValueError("Simulated and measured data must have same length")
    # yes, it's fine, wtf
    # noinspection PyTypeChecker
    if any(simulated_prep.index != measured_prep.index):
        raise ValueError("Simulated and measured data must have same index")

    simulated_cell_size = simulated_prep["cell_size"].to_numpy()
    measured_cell_size = unp.uarray(
        measured_prep["cell_size"].to_numpy(),
        measured_prep["u_cell_size"].to_numpy(),
    )
    cell_size_ratio = simulated_cell_size / measured_cell_size
    simulated_prep["cell_size"] = unp.nominal_values(cell_size_ratio)
    simulated_prep["u_cell_size"] = unp.std_devs(cell_size_ratio)
    return simulated_prep.reset_index()


def load_cp_cv_sim_bulk_properties() -> pd.DataFrame:
    conn_str = "postgresql://postgres@localhost:5432/simulated_and_measured_2025-06-01_gri30_highT.yaml"
    with psycopg.connect(conn_str) as con:
        bulk_properties = pd.read_sql_query(
            """
            select
                diluent,
                dil_condition,
                "time",
                progress,
                phi_nom,
                cp,
                cv,
                gamma
            from
                (select *, (bp."time" / c.t_ind) progress
                from 
                bulk_properties bp
                join conditions c
                on bp.condition_id = c.id
                )
            where
                sim_type = 'cv'
                and mech = 'gri30_highT.yaml'
                and match = 'tad'
                and progress <= 1
                and dil_condition != 'medium'
                and not phi_nom = 0.4
            order by
                diluent,
                dil_condition,
                phi_nom,
                progress
            ;
            """,
            con=con,
        )
    return bulk_properties


def calculate_cp_cv_gamma_ratios(data: pd.DataFrame) -> pd.DataFrame:
    interp_columns = ["cp", "cv", "gamma"]
    data_interp = interpolate(
        data=data,
        time_column="progress",
        interp_columns=interp_columns,
        saturate=False,
        co2_label="CO2",
        n2_label="N2",
    ).reset_index(drop=True)
    co2_diluted = (
        data_interp[data_interp["diluent"].str.startswith("CO2")]
        .set_index(["dil_condition", "phi_nom"])
        .drop("diluent", axis=1)
    )
    n2_diluted = (
        data_interp[data_interp["diluent"].str.startswith("N2")]
        .set_index(["dil_condition", "phi_nom"])
        .drop("diluent", axis=1)
    )
    for column in interp_columns:
        co2_diluted.loc[:, column] = co2_diluted[column] / n2_diluted[column]
    return co2_diluted


def load_sim_conditions() -> pd.DataFrame:
    conn_str = "postgresql://postgres@localhost:5432/simulated_and_measured_2025-06-01_gri30_highT.yaml"
    with psycopg.connect(conn_str) as con:
        conditions = pd.read_sql_query(
            """
            select
                diluent,
                case
                    when dil_condition = 'high' then 0.2
                    when dil_condition = 'medium' then 0.15
                    else 0.1 end as dil_mf_nom,
                phi_nom,
                equivalence phi,
                u_cj cj_speed,
                t_ind * 1e6 t_ind,
                cell_size * 1000 cell_size
            from conditions
            where
                sim_type = 'cv'
                and mech = 'gri30_highT.yaml'
                and match = 'tad'
                -- and dil_condition != 'medium'
                ;
            """,
            con=con,
        )
    test_data = pd.read_hdf(
        (directory.SIM_MEAS_COMPARISON / conn_str.split("/")[-1]).with_suffix(".h5"),
        key="data_fixed_uncert",
    )[["diluent", "phi_nom", "dil_mf_nom", "wave_speed", "u_wave_speed"]]
    mach_data = pd.read_csv(directory.SCRIPTS / "cj_study" / "cj_tad_ss_results.csv")[[
        "diluent",
        "phi_nom",
        "dil_mf_nom",
        "sound_speed",
    ]]

    conditions = conditions.join(
        test_data.set_index(["diluent", "dil_mf_nom", "phi_nom"]),
        ["diluent", "dil_mf_nom", "phi_nom"],
    ).join(
        mach_data.groupby(["diluent", "dil_mf_nom", "phi_nom"]).mean(),
        ["diluent", "dil_mf_nom", "phi_nom"],
    ).sort_values(["diluent", "dil_mf_nom", "phi_nom"]).reset_index(drop=True)

    cj = conditions["cj_speed"].to_numpy()
    wave_speed = conditions["wave_speed"].to_numpy()
    measured_speed = unp.uarray(wave_speed, conditions["u_wave_speed"].to_numpy())
    cj_ratio = cj / measured_speed
    conditions["cj_ratio"] = unp.nominal_values(cj_ratio)
    conditions["u_cj_ratio"] = unp.std_devs(cj_ratio)
    sound_speed = conditions["sound_speed"].to_numpy()
    conditions["cj_mach"] = cj / sound_speed
    conditions["mach"] = wave_speed / sound_speed

    return conditions


def load_sim_inert() -> pd.DataFrame:
    conn_str = "postgresql://postgres@localhost:5432/simulated_and_measured_2025-06-01_gri30_highT_inerts.yaml_inerts"
    with psycopg.connect(conn_str) as con:
        conditions = pd.read_sql_query(
            """
            select
                diluent,
                case
                    when dil_condition = 'high' then 0.2
                    when dil_condition = 'medium' then 0.15
                    else 0.1 end as dil_mf_nom,
                phi_nom,
                equivalence phi,
                u_cj cj_speed,
                t_ind * 1e6 t_ind,
                cell_size * 1000 cell_size
            from conditions
            where
                sim_type = 'cv'
                and match = 'tad'
                -- and dil_condition != 'medium'
                ;
            """,
            con=con,
        )
    test_data = pd.read_hdf(
        (directory.SIM_MEAS_COMPARISON / conn_str.split("/")[-1]).with_suffix(".h5"),
        key="data_fixed_uncert",
    )[["diluent", "phi_nom", "dil_mf_nom", "wave_speed", "u_wave_speed"]]
    conditions = conditions.join(
        test_data.set_index(["diluent", "dil_mf_nom", "phi_nom"]),
        ["diluent", "dil_mf_nom", "phi_nom"],
    ).sort_values(["diluent", "dil_mf_nom", "phi_nom"]).reset_index(drop=True)

    return conditions


def combine_inert_and_active(inert: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    inert = inert.copy()
    active = active.filter(inert.columns).copy()
    inert["diluent_kind"] = "inert"
    inert["diluent"] = inert["diluent"].str.replace("i", "")
    active["diluent_kind"] = "active"
    combined = pd.concat((active, inert), ignore_index=True)

    return combined
