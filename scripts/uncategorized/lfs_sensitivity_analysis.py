import pandas as pd

from funcs.simulation.sensitivity.flame_speed.analysis import perform_study
from funcs.simulation.thermo import match_adiabatic_temp


def combine_results(df_in, ser_in):
    for key, val in ser_in.items():
        df_in[key] = val

    return df_in


if __name__ == "__main__":
    df_result = pd.DataFrame()
    use_multiprocessing = True
    n_cores = None  # None for all cores
    mech = "gri30_highT.yaml"
    fuel = "CH4"
    oxidizer = "N2O"
    dil_base = "CO2"
    t0 = 300
    p0 = 101325
    diluents = ("CO2", "N2O", None)
    dil_mfs_base = (0.05, 0.1, 0.15)
    equivs = (1, 0.4)
    for dil in diluents:
        for phi in equivs:
            if dil is not None:
                for dil_mf_base in dil_mfs_base:
                    if dil != dil_base:
                        # adjust mole fraction
                        dil_mf = match_adiabatic_temp(mech, fuel, oxidizer, phi, dil_base, dil_mf_base, dil, t0, p0)
                    else:
                        dil_mf = dil_mf_base
                    # run analysis and save results
                    df_new, ser_info_new = perform_study(
                        mech,
                        t0,
                        p0,
                        phi,
                        fuel,
                        oxidizer,
                        dil,
                        dil_mf,
                        use_multiprocessing=use_multiprocessing,
                        n_cores=n_cores,
                    )
                    df_new = combine_results(df_new, ser_info_new)
                    df_result = pd.concat((df_result, df_new), axis=0)
            else:
                # run analysis and save results
                df_new, ser_info_new = perform_study(
                    mech,
                    t0,
                    p0,
                    phi,
                    fuel,
                    oxidizer,
                    dil,
                    0,  # undiluted mixture has no dil_mf :)
                    use_multiprocessing=use_multiprocessing,
                )
                df_new = combine_results(df_new, ser_info_new)
                df_result = pd.concat((df_result, df_new), axis=0)

    with pd.HDFStore("lfs_sensitivity.h5", "w") as store:
        store.put("data", df_result)
