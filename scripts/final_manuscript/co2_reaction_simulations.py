import concurrent.futures
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import cantera as ct
import simulation.thermo
from rich.pretty import install
from tqdm import tqdm

import sdtoolbox as sdt
from funcs.simulation.cell_size import CvConfig, build_gas_object, calculate_westbrook_only
from sdtoolbox.output import clear_simulation_database

install()


def get_important_reaction_indices(gas: ct.Solution) -> list[int]:
    rxn_indices = []
    for (
        i,
        rxn,
    ) in enumerate(gas.reactions()):
        reactants_and_products = [*rxn.reactants.keys(), *rxn.products.keys()]
        if "CO2" in reactants_and_products and "OH" in reactants_and_products:
            rxn_indices.append(i)
    return rxn_indices


def get_important_species_indices(gas: ct.Solution) -> list[int]:
    species_indices = []
    for spec in ["O2", "O", "N2", "CO2", "CO", "OH", "H"]:
        try:
            species_indices.append(gas.species_index(spec))
        except ValueError:
            warnings.warn(f"Species not found: {spec}")
    return species_indices


def simulate(
    mech: str,
    match: Optional[str],
    dil_condition: str,
    fuel: str,
    oxidizer: str,
    diluent: str,
    t0: float,
    p0: float,
    phi: float,
    dil_mf: float,
    db_path: str,
) -> None:
    base_gas = build_gas_object(
        mechanism=mech,
        equivalence=phi,
        fuel=fuel,
        oxidizer=oxidizer,
        diluent=diluent,
        diluent_mol_frac=dil_mf,
        initial_temp=t0,
        initial_press=p0,
    )
    q = base_gas.mole_fraction_dict()
    cj_speed = sdt.postshock.CJspeed(p0, t0, q, mech)
    with warnings.catch_warnings():
        # don't fuck up my tqdm counter >:(
        warnings.simplefilter("ignore")
        calculate_westbrook_only(
            mechanism=mech,
            match=match,
            dil_condition=dil_condition,
            initial_temp=t0,
            initial_press=p0,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=phi,
            diluent=diluent,
            diluent_mol_frac=dil_mf,
            cj_speed=cj_speed,
            rxn_indices=get_important_reaction_indices(gas=base_gas),
            spec_indices=get_important_species_indices(gas=base_gas),
            db_path=db_path,
            cv_config=CvConfig(max_tries=1, max_step=1e-6, end_time=12e-6, solver_method="Radau"),
            # znd_config=ZndConfig(max_tries=1, max_step=1e-4, end_time=3e-5),
        )


def cell_size_serial(
    co2_dil_mfs: list[float],
    n2_dil_mfs: list[float],
    n2_match: list[str],
    dil_conditions: list[str],
    mech: str,
    fuel: str,
    oxidizer: str,
    t0: float,
    p0: float,
    phi: float,
    db_path: str,
    counter: tqdm,
):
    for dil_mf, dil_condition in zip(co2_dil_mfs, dil_conditions):
        simulate(
            mech=mech,
            match=None,
            dil_condition=dil_condition,
            fuel=fuel,
            oxidizer=oxidizer,
            diluent="CO2",
            t0=t0,
            p0=p0,
            phi=phi,
            dil_mf=dil_mf,
            db_path=db_path,
        )
        counter.update()
    for dil_mf, match, dil_condition in zip(n2_dil_mfs, n2_match, dil_conditions * 2):
        simulate(
            mech=mech,
            match=match,
            dil_condition=dil_condition,
            fuel=fuel,
            oxidizer=oxidizer,
            diluent="N2",
            t0=t0,
            p0=p0,
            phi=phi,
            dil_mf=dil_mf,
            db_path=db_path,
        )
        counter.update()


def cell_size_parallel(
    co2_dil_mfs: list[float],
    n2_dil_mfs: list[float],
    n2_match: list[str],
    dil_conditions: list[str],
    mech: str,
    fuel: str,
    oxidizer: str,
    t0: float,
    p0: float,
    phi: float,
    db_path: str,
    counter: tqdm,
):
    with ProcessPoolExecutor() as executor:
        futures = set()
        for dil_mf, dil_condition in zip(co2_dil_mfs, dil_conditions):
            # noinspection PyTypeChecker
            f = executor.submit(
                simulate,
                mech=mech,
                match=None,
                dil_condition=dil_condition,
                fuel=fuel,
                oxidizer=oxidizer,
                diluent="CO2",
                t0=t0,
                p0=p0,
                phi=phi,
                dil_mf=dil_mf,
                db_path=db_path,
            )
            futures.add(f)
        for dil_mf, match, dil_condition in zip(n2_dil_mfs, n2_match, dil_conditions * 2):
            # noinspection PyTypeChecker
            f = executor.submit(
                simulate,
                mech=mech,
                match=match,
                dil_condition=dil_condition,
                fuel=fuel,
                oxidizer=oxidizer,
                diluent="N2",
                t0=t0,
                p0=p0,
                phi=phi,
                dil_mf=dil_mf,
                db_path=db_path,
            )
            futures.add(f)
        results = []

        for done in concurrent.futures.as_completed(futures):
            results.append(done.result())
            counter.update()


def match_dil_mf_aft_parallel(
    mech: str,
    fuel: str,
    oxidizer: str,
    phi: float,
    co2_dil_mfs: list[float],
    t0: float,
    p0: float,
    counter: tqdm,
) -> list[float]:
    with ProcessPoolExecutor() as executor:
        futures = set()
        for mf_co2 in co2_dil_mfs:
            # noinspection PyTypeChecker
            f = executor.submit(
                match_dil_mf_aft,
                mech=mech,
                fuel=fuel,
                oxidizer=oxidizer,
                phi=phi,
                mf_co2=mf_co2,
                t0=t0,
                p0=p0,
            )
            futures.add(f)
        results = []

        for done in concurrent.futures.as_completed(futures):
            results.append(done.result())
            counter.update()

    return results


def match_dil_mf_aft(
    mech: str,
    fuel: str,
    oxidizer: str,
    phi: float,
    mf_co2: float,
    t0: float,
    p0: float,
) -> float:
    return simulation.thermo.match_adiabatic_temp(
        mech=mech,
        fuel=fuel,
        oxidizer=oxidizer,
        phi=phi,
        dil_original="CO2",
        dil_original_mol_frac=mf_co2,
        dil_new="N2",
        init_temp=t0,
        init_press=p0,
    )


def main():
    mech = "Blanquart2018.cti"
    fuel = "CH4"
    oxidizer = "N2O"
    t0 = 300
    p0 = 101325
    phi = 1.0
    db_path = "/home/mick/DetResearch/scripts/final_manuscript/co2_reaction_study.sqlite"
    clear_existing_data = True

    if clear_existing_data:
        clear_simulation_database(path=db_path)

    parallelize = True
    parallel_or_series = "parallel" if parallelize else "series"
    print(f"Matching CO2 adiabatic flame temperature in {parallel_or_series}")
    co2_dil_mfs = [0.1, 0.2]
    with tqdm(total=len(co2_dil_mfs), unit="calculation", file=sys.stdout, colour="green", desc="Running") as counter:
        if parallelize:
            n2_aft_matched_dil_mfs = match_dil_mf_aft_parallel(
                mech=mech,
                fuel=fuel,
                oxidizer=oxidizer,
                phi=phi,
                co2_dil_mfs=co2_dil_mfs,
                t0=t0,
                p0=p0,
                counter=counter,
            )
        else:
            n2_aft_matched_dil_mfs = [match_dil_mf_aft(
                mech=mech,
                fuel=fuel,
                oxidizer=oxidizer,
                phi=phi,
                mf_co2=mf_co2,
                t0=t0,
                p0=p0,
            ) for mf_co2 in co2_dil_mfs]
        counter.set_description_str("Done")

    n2_dil_mfs = n2_aft_matched_dil_mfs + co2_dil_mfs  # Tad matched + mole fraction matched
    n2_match = ["tad" for _ in co2_dil_mfs] + ["mf" for _ in co2_dil_mfs]
    dil_conditions = ["low", "high"]

    print(f"Running cell size simulations in {parallel_or_series}")
    n_conditions = len(co2_dil_mfs) + len(n2_dil_mfs)
    with tqdm(total=n_conditions, unit="simulation", file=sys.stdout, colour="green", desc="Running") as counter:
        if parallelize:
            cell_size_parallel(
                co2_dil_mfs=co2_dil_mfs,
                n2_dil_mfs=n2_dil_mfs,
                n2_match=n2_match,
                dil_conditions=dil_conditions,
                mech=mech,
                fuel=fuel,
                oxidizer=oxidizer,
                t0=t0,
                p0=p0,
                phi=phi,
                counter=counter,
                db_path=db_path,
            )
        else:
            cell_size_serial(
                co2_dil_mfs=co2_dil_mfs,
                n2_dil_mfs=n2_dil_mfs,
                n2_match=n2_match,
                dil_conditions=dil_conditions,
                mech=mech,
                fuel=fuel,
                oxidizer=oxidizer,
                t0=t0,
                p0=p0,
                phi=phi,
                counter=counter,
                db_path=db_path,
            )
        counter.set_description_str("Done")


if __name__ == "__main__":
    main()
