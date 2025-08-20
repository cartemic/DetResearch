import multiprocessing as mp

import pandas as pd
from numpy import nan
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from funcs.simulation.sensitivity.gas import build as build_gas
from funcs.simulation.thermo import calculate_laminar_flame_speed as flame_speed
from funcs.simulation.thermo import ct


def calculate_flame_speed(
    mech,
    init_temp,
    init_press,
    equivalence,
    fuel,
    oxidizer,
    diluent,
    diluent_mol_frac,
    perturbation_fraction,
    perturbed_reaction_no=-1,
):
    gas = build_gas(
        mech,
        init_temp,
        init_press,
        equivalence,
        fuel,
        oxidizer,
        diluent,
        diluent_mol_frac,
    )
    if perturbed_reaction_no > 0:
        gas.set_multiplier(1 + perturbation_fraction, perturbed_reaction_no)
    return (
        perturbed_reaction_no,
        flame_speed(gas),
    )


def _run_with_mp(
    mech,
    init_temp,
    init_press,
    equivalence,
    fuel,
    oxidizer,
    diluent,
    diluent_mol_frac,
    perturbation_fraction,
    df_result,
    n_cores,
):
    with mp.Pool(n_cores) as p:
        for idx, lfs in tqdm(
            p.istarmap(
                calculate_flame_speed,
                [
                    [
                        mech,
                        init_temp,
                        init_press,
                        equivalence,
                        fuel,
                        oxidizer,
                        diluent,
                        diluent_mol_frac,
                        perturbation_fraction,
                        rxn_no,
                    ]
                    for rxn_no in df_result.index.to_numpy()
                ],
            ),
            total=len(df_result),
        ):
            df_result.at[idx, "flame_speed"] = lfs
    return df_result


def _run_without_mp(
    mech,
    init_temp,
    init_press,
    equivalence,
    fuel,
    oxidizer,
    diluent,
    diluent_mol_frac,
    perturbation_fraction,
    df_result,
    _,  # n_cores not used
):
    for idx in tqdm(df_result.index.values):
        _, lfs = calculate_flame_speed(
            mech,
            init_temp,
            init_press,
            equivalence,
            fuel,
            oxidizer,
            diluent,
            diluent_mol_frac,
            perturbation_fraction,
            idx,
        )
        df_result.at[idx, "flame_speed"] = lfs
    return df_result


def perform_study(
    mech,
    init_temp,
    init_press,
    equivalence,
    fuel,
    oxidizer,
    diluent,
    diluent_mol_frac,
    perturbation_fraction=1e-2,
    use_multiprocessing=False,
    n_cores=None,
):
    arg_info = locals()
    # noinspection PyArgumentList
    n_rxns = len(ct.Reaction.listFromFile(mech))
    pert_rxn = [-1, *list(range(n_rxns))]
    # this shouldn't take up too much memory, so I'm going to do it all at once.
    ser_info = pd.Series(
        index=list(arg_info.keys()),
        data=list(arg_info.values()),
    )
    df_result = pd.DataFrame(index=pert_rxn, columns=["flame_speed", "sensitivity"])
    run_func = _run_with_mp if use_multiprocessing else _run_without_mp
    df_result = run_func(
        mech,
        init_temp,
        init_press,
        equivalence,
        fuel,
        oxidizer,
        diluent,
        diluent_mol_frac,
        perturbation_fraction,
        df_result,
        n_cores,
    )
    lfs_base = df_result["flame_speed"][-1]
    df_result["sensitivity"] = (df_result["flame_speed"] - lfs_base) / (lfs_base * perturbation_fraction)
    df_result["sensitivity"][-1] = nan

    return df_result, ser_info
