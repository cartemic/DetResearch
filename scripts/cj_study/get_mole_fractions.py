import itertools
import multiprocessing as mp

from funcs.simulation import thermo

FUEL = "CH4"
OXIDIZER = "N2O"
INIT_TEMP = 300
INIT_PRESS = 101325
MECH = "gri30_highT.xml"


def gather_conditions_row(phi, x_co2):
    n2_aft = thermo.match_adiabatic_temp(
        mech=MECH,
        fuel=FUEL,
        oxidizer=OXIDIZER,
        phi=phi,
        dil_original="CO2",
        dil_original_mol_frac=x_co2,
        dil_new="N2",
        init_temp=INIT_TEMP,
        init_press=INIT_PRESS,
    )
    n2_ss = thermo.match_sound_speed(
        mech=MECH,
        fuel=FUEL,
        oxidizer=OXIDIZER,
        phi=phi,
        dil_original="CO2",
        dil_original_mol_frac=x_co2,
        dil_new="N2",
        init_temp=INIT_TEMP,
        init_press=INIT_PRESS,
    )
    return f"{phi},{x_co2},{n2_aft},{n2_ss}"


if __name__ == "__main__":
    conditions = ["phi,x_co2,x_n2_flame_temp_match,x_n2_sound_speed_match"]

    equivalence_ratios = (0.4, 0.7, 1.0)
    co2_mol_fracs = (0.1, 0.15, 0.2)
    with mp.Pool() as pool:
        results = pool.starmap(gather_conditions_row, itertools.product(equivalence_ratios, co2_mol_fracs))

    pool.join()

    with open("conditions.csv", "w") as f:
        f.write("\n".join(conditions + results))
