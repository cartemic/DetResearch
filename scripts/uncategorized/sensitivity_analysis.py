import datetime
import multiprocessing as mp
import pprint
import sys
import time
import traceback
from concurrent import futures

import cantera as ct
from tqdm import tqdm

import funcs.simulation.sensitivity.detonation.database as db
from funcs.simulation import thermo
from funcs.simulation.sensitivity.detonation import analysis

if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")

    # Inputs
    perturbation_fraction = 1e-2
    max_step_znd = 1e-4  # default 1e-4
    db_path = "sensitivity_3.sqlite"
    mechanism = "gri30_highT.yaml"
    initial_temp = 300
    initial_press = 101325
    equivalence = 1
    fuel = "H2"
    oxidizer = "O2"
    diluent_to_match = "CO2"
    diluent = "N2"
    diluent_mol_frac_to_match = 0.05
    overwrite_existing_perturbed_results = False

    # Preparations
    if diluent_to_match == diluent:
        diluent_mol_frac = diluent_mol_frac_to_match
    else:
        diluent_mol_frac = thermo.match_adiabatic_temp(
            mechanism,
            fuel,
            oxidizer,
            equivalence,
            diluent_to_match,
            diluent_mol_frac_to_match,
            diluent,
            initial_temp,
            initial_press,
        )
    print("Initializing study... ", end="", flush=True)
    test_conditions = analysis.initialize_study(
        db_path=db_path,
        test_conditions=db.TestConditions(
            mechanism=mechanism,
            initial_temp=initial_temp,
            initial_press=initial_press,
            fuel=fuel,
            oxidizer=oxidizer,
            equivalence=equivalence,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
        ),
        max_step_znd=max_step_znd,
    )
    print("Done!")
    reactions = ct.Reaction().listFromFile(mechanism)

    # Calculations
    _lock = mp.Lock()
    n_rxns = len(reactions)

    n_errors = 0
    error_log = f"error_log_{datetime.datetime.now().isoformat()}"
    with (
        futures.ProcessPoolExecutor(initializer=analysis.init, initargs=(_lock,)) as executor,
        tqdm(total=n_rxns, unit="calc", file=sys.stdout, colour="green", desc="Running") as counter,
    ):
        futures = []
        inputs = []  # track these for exception logs
        for reaction_number in range(n_rxns):
            kwargs = {
                "test_conditions": test_conditions,
                "perturbation_fraction": perturbation_fraction,
                "perturbed_reaction_no": reaction_number,
                "db_path": db_path,
                "max_step_znd": max_step_znd,
                "overwrite_existing": overwrite_existing_perturbed_results,
            }
            future = executor.submit(analysis.perform_study, **kwargs)
            futures.append(future)
            inputs.append(kwargs)

        while len(futures):
            for i, future in enumerate(futures):
                if future.done():
                    # noinspection PyBroadException
                    try:
                        # There aren't results, but we try anyway so that we can catch and log any exceptions
                        # that pop up.
                        future.result()
                    except Exception:
                        with open(error_log, "a") as f:
                            n_errors += 1
                            f.write(
                                f"Error logged at {datetime.datetime.now().isoformat()}\n"
                                "Inputs:\n"
                                f"{pprint.pformat(kwargs, indent=4, width=120)}\n\n"
                                "Stack Trace:\n"
                                f"{traceback.format_exc()}\n==================================================\n\n"
                            )
                            f.flush()
                    inputs.pop(i)
                    futures.pop(i)
                    counter.update()

            time.sleep(1.0)
        counter.set_description_str("Done")

    if n_errors:
        print(f"Encountered {n_errors} errors -- see f{error_log}")
