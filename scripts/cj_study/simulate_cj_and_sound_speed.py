import multiprocessing as mp

import cantera as ct
import get_mole_fractions as mf
import pandas as pd

from funcs.simulation import thermo
from sdtoolbox import postshock
from sdtoolbox import thermo as sd_thermo


def get_all_conditions():
    df_conditions = pd.read_csv("conditions.csv")
    all_conditions = []
    for _, row in df_conditions.iterrows():
        all_conditions.extend(
            [
                (
                    mf.FUEL,
                    mf.OXIDIZER,
                    "CO2",
                    "-",
                    row["x_co2"],
                    mf.MECH,
                    mf.INIT_PRESS,
                    mf.INIT_TEMP,
                    row["phi"],
                    row["x_co2"],
                ),
                (
                    mf.FUEL,
                    mf.OXIDIZER,
                    "N2",
                    "aft",
                    row["x_co2"],
                    mf.MECH,
                    mf.INIT_PRESS,
                    mf.INIT_TEMP,
                    row["phi"],
                    row["x_n2_flame_temp_match"],
                ),
                (
                    mf.FUEL,
                    mf.OXIDIZER,
                    "N2",
                    "ss",
                    row["x_co2"],
                    mf.MECH,
                    mf.INIT_PRESS,
                    mf.INIT_TEMP,
                    row["phi"],
                    row["x_n2_sound_speed_match"],
                ),
            ]
        )

    return pd.DataFrame(
        all_conditions,
        columns=["fuel", "oxidizer", "diluent", "match", "dil_mf_co2", "mech", "p0", "t0", "phi", "dil_mf"],
    )


def simulate(_, conditions: pd.Series):
    gas = ct.Solution(conditions["mech"])
    gas.set_equivalence_ratio(
        conditions["phi"],
        conditions["fuel"],
        conditions["oxidizer"],
    )
    q = thermo.diluted_species_dict(gas.mole_fraction_dict(), conditions["diluent"], conditions["dil_mf"])
    conditions["cj"] = postshock.CJspeed(
        conditions["p0"],
        conditions["t0"],
        q,
        conditions["mech"],
    )
    gas = postshock.PostShock_eq(
        conditions["cj"],
        conditions["p0"],
        conditions["t0"],
        q,
        conditions["mech"],
    )
    conditions["sound_speed"] = sd_thermo.soundspeed_eq(gas)
    return conditions


if __name__ == "__main__":
    simulation_conditions = get_all_conditions()
    with mp.Pool() as pool:
        test = pool.starmap(simulate, simulation_conditions.iterrows())

    pd.DataFrame(test).to_csv("simulations.csv", index=False)
