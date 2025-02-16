import multiprocessing as mp

import cantera as ct
import numpy as np
import pandas as pd

from funcs.simulation import thermo
from sdtoolbox import postshock
from sdtoolbox import thermo as sdt_thermo

MECH = "gri30.xml"


def calculate_single_row(_: int, row: pd.Series) -> pd.Series:
    row = row.copy()
    oxidizer = row["oxidizer"] if row["oxidizer"] != "air" else "O2:1,N2:3.76"
    gas = ct.Solution(MECH)
    gas.set_equivalence_ratio(
        row["phi"],
        row["fuel"],
        oxidizer,
    )
    if row["diluent"].lower() == "none":  # diluted_species_dict doesn't handle str "None", so do it here instead
        q = gas.mole_fraction_dict()
    else:
        q = thermo.diluted_species_dict(
            spec=gas.mole_fraction_dict(),
            diluent=row["diluent"],
            diluent_mol_frac=row["dil_mf"],
        )
    tad = thermo.get_adiabatic_temp(
        mech=MECH,
        fuel=row["fuel"],
        oxidizer=oxidizer,
        phi=row["phi"],
        diluent=row["diluent"],  # diluent is checked for str "None"
        diluent_mol_frac=row["dil_mf"],
        init_temp=row["t_0"],
        init_press=row["p_0"],
    )
    try:
        cj = postshock.CJspeed(
            P1=row["p_0"],
            T1=row["t_0"],
            q=q,
            mech=MECH,
        )
    except ct.CanteraError:
        cj = np.nan
    try:
        gas = postshock.PostShock_eq(
            U1=cj,
            P1=row["p_0"],
            T1=row["t_0"],
            q=q,
            mech=MECH,
        )
        ss = sdt_thermo.soundspeed_eq(gas)
    except ct.CanteraError:
        ss = np.nan

    row["t_ad"] = tad
    row["cj_speed"] = cj
    row["sound_speed"] = ss

    return row


if __name__ == "__main__":
    with pd.HDFStore("/d/Data/Processed/Data/all_tube_data.h5", "r") as store:
        data = store.data

    with mp.Pool() as pool:
        results = pool.starmap(calculate_single_row, data.iterrows())

    results = pd.DataFrame(results)
    results.to_csv("cj_tad_ss_results.csv", index=False)
