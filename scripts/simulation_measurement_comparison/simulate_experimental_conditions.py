import datetime
import functools
import multiprocessing as mp
import os
import sys
import traceback
import warnings
import zoneinfo
from pathlib import Path
from typing import Optional

import cantera as ct
import numpy as np
import pandas as pd
import tqdm
from pandas.errors import PerformanceWarning
from scipy.stats import t

from simulation.cell_size import CvConfig
from uncertainties import unumpy as unp

from funcs.simulation import cell_size as cs
from funcs.simulation import thermo
from scripts.final_manuscript.co2_reaction_simulations import (
    get_important_reaction_indices,
    get_important_species_indices,
)
from sdtoolbox import output
from sdtoolbox.postshock import CJspeed

# these are simulation parameters, but will never change in this context
FUEL = "CH4"
OXIDIZER = "N2O"
LOCAL_TZ = zoneinfo.ZoneInfo("US/Pacific")
TODAY = datetime.datetime.now(LOCAL_TZ).date().isoformat()

THIS_SCRIPT_DIR = Path(__file__).absolute().parent


def main(with_inerts: bool = False, westbrook_only: bool = True):
    mech = "gri30_highT_inerts.yaml" if with_inerts else "gri30_highT.yaml"
    maybe_inerts = "_inerts" if with_inerts else ""
    df_measured = read_in_measured_data()

    mech_name = Path(mech).name
    output_base = THIS_SCRIPT_DIR / f"simulated_and_measured_{TODAY}_{mech_name}{maybe_inerts}"
    df_result = simulate_measured_conditions(
        df_measured,
        mech=mech,
        with_inerts=with_inerts,
        westbrook_only=westbrook_only,
        db_path=output_base.with_suffix(".sqlite") if westbrook_only else None,
    )
    h5_path = output_base.with_suffix(".h5")
    with warnings.catch_warnings():
        # Yes, yes, PyTables will pickle stuff, I really don't care here
        warnings.simplefilter("ignore", PerformanceWarning)
        df_result.to_hdf(h5_path, key="data")
        add_fixed_uncert(h5_path)


def add_fixed_uncert(h5_path: Path) -> None:
    # wack pandas type hinting
    # noinspection PyTypeChecker
    meas: pd.DataFrame = pd.read_hdf(THIS_SCRIPT_DIR / "measurements.h5")
    # wack pandas type hinting
    # noinspection PyTypeChecker
    needs_fixin: pd.DataFrame = pd.read_hdf(h5_path, key="data")
    for (phi, dil_mf, diluent), data in meas.groupby(["phi_nom", "dil_mf_nom", "diluent"]):
        cell_size = unp.uarray(data["cell_size"], data["u_cell_size"]).mean()
        fixed_uncert = cell_size.std_dev * t.ppf(0.975, len(data) - 1)
        needs_fixin.loc[
            (needs_fixin["phi_nom"] == phi)
            & (needs_fixin["dil_mf_nom"] == dil_mf)
            & (needs_fixin["diluent"] == diluent),
            "u_cell_size_measured",
        ] = fixed_uncert
    needs_fixin.to_hdf(h5_path, key="data_fixed_uncert")


def get_column_mean_with_uncertainty(df: pd.DataFrame, column: str):
    mean = np.mean(unp.uarray(df[column], df[f"u_{column}"]))

    return mean.nominal_value, mean.std_dev


def read_in_measured_data():
    data_loc = os.path.join(os.path.join(os.path.dirname(__file__), "measurements.h5"))
    with pd.HDFStore(data_loc, "r") as store:
        measured = store.data

    measured = measured[(measured["fuel"] == FUEL) & (measured["oxidizer"] == OXIDIZER)]
    df_out = pd.DataFrame()
    for idx, ((diluent, phi_nom, dil_mf_nom), df_grouped) in enumerate(
        measured.groupby(["diluent", "phi_nom", "dil_mf_nom"])
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


def simulate_measured_conditions(
    df_measured: pd.DataFrame,
    mech: str,
    with_inerts: bool,
    westbrook_only: bool,
    db_path: Optional[Path],
):
    if westbrook_only and db_path is not None and db_path.exists():
        output.clear_simulation_database(db_path)

    base_gas = ct.Solution(mech)
    rxn_indices = get_important_reaction_indices(gas=base_gas)
    spec_indices = get_important_species_indices(gas=base_gas)

    _simulate = functools.partial(
        simulate_single_condition,
        mech=mech,
        with_inerts=with_inerts,
        westbrook_only=westbrook_only,
        db_path=db_path,
        rxn_indices=rxn_indices,
        spec_indices=spec_indices,
    )
    with mp.Pool() as p:
        # noinspection PyTypeChecker
        df_result = pd.DataFrame(
            tqdm.tqdm(
                p.imap(_simulate, df_measured.iterrows()),
                total=len(df_measured),
                file=sys.stdout,
                colour="green",
                unit="simulation",
            )
        )

    return df_result.sort_index()


def simulate_single_condition(
    idx_and_row: tuple[int, pd.Series],
    *,
    mech: str,
    with_inerts: bool,
    westbrook_only: bool,
    db_path: Optional[Path],
    rxn_indices: list[int],
    spec_indices: list[int],
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        row = idx_and_row[1].copy()
        cv_config = CvConfig(max_tries=3, max_step=1e-6, end_time=12e-6, solver_method="Radau")
        cj_speed = None
        dil = ""

        try:
            dil = row["diluent"]
            if with_inerts:
                if dil == "CO2":
                    dil = "CO2i"
                elif dil == "N2":
                    dil = "N2i"
                row["diluent"] = dil

            phi = row["phi"]
            phi_nom = row["phi_nom"]
            dil_mf = row["dil_mf"]
            p_0 = row["p_0"]
            t_0 = row["t_0"]
            dil_mf_nom = row["dil_mf_nom"]
            if np.isclose(dil_mf_nom, 0.10):
                dil_condition = "low"
            elif np.isclose(dil_mf_nom, 0.20):
                dil_condition = "high"
            else:
                dil_condition = "medium"
            gas = ct.Solution(mech)
            gas.set_equivalence_ratio(phi, FUEL, OXIDIZER)
            q = thermo.diluted_species_dict(gas.mole_fraction_dict(), dil, dil_mf)
            cj_speed = CJspeed(p_0, t_0, q, mech)

            if westbrook_only:
                simulated = cs.calculate_westbrook_only(
                    mechanism=mech,
                    initial_temp=t_0,
                    initial_press=p_0,
                    fuel=FUEL,
                    oxidizer=OXIDIZER,
                    equivalence=phi,
                    diluent=dil,
                    diluent_mol_frac=dil_mf,
                    cj_speed=cj_speed,
                    cv_config=cv_config,
                    match="tad",
                    dil_condition=dil_condition,
                    rxn_indices=rxn_indices,
                    spec_indices=spec_indices,
                    db_path=db_path,
                    phi_nom=phi_nom,
                )
            else:
                # database write not implemented (yet)
                simulated = cs.calculate(
                    mechanism=mech,
                    initial_temp=t_0,
                    initial_press=p_0,
                    fuel=FUEL,
                    oxidizer=OXIDIZER,
                    equivalence=phi,
                    diluent=dil,
                    diluent_mol_frac=dil_mf,
                    cj_speed=cj_speed,
                    # cv_end_time=12e-6,
                    # max_step_cv=1e-6,
                    # max_tries_cv=1,
                    max_step_znd=1e-4,
                    # max_tries_znd=5,
                    znd_end_time=5e-4,
                )
                row["cell_size_gavrikov"] = simulated.cell_size.gavrikov * 1000  # m -> mm
                row["cell_size_ng"] = simulated.cell_size.ng * 1000  # m -> mm
            row["cell_size_westbrook"] = simulated.cell_size.westbrook * 1000  # m -> mm
            row["cell_size_westbrook_2"] = simulated.cell_size.westbrook_2 * 1000  # m -> mm
            row["gavrikov_criteria_met"] = simulated.gavrikov_criteria_met

            row["znd_step"] = simulated.znd_step
            row["znd_end_time"] = simulated.znd_end_time
            row["znd_tries"] = simulated.znd_tries
            row["znd_max_temp_time"] = simulated.znd_max_temp_time

        except Exception as e:
            problem = (
                f"{datetime.datetime.now(LOCAL_TZ)}\n"
                f"Caught: {e}\n"
                "=================\n"
                f"Conditions:\n{row}\n"
                "INERT CO2"
                if dil == "CO2i"
                else ""
                f"CV Config:\n{cv_config}\n"
                f"CJ Speed:\n{cj_speed}\n"
                f"Traceback:\n{traceback.format_exc()}"
                "\n\n"
            )
            with open(f"error_log_experimental_conditions_simulation_{TODAY}", "a") as f:
                f.write(problem)

    return row


if __name__ == "__main__":
    main(with_inerts=True)
