from __future__ import annotations
import dataclasses
import glob
import itertools
import warnings
from multiprocessing import Lock, Pool
import os.path
import time
from enum import Enum
from typing import Optional, Tuple, List, Any, Dict, Callable

import cantera as ct
import numpy as np
import pandas as pd
import tqdm
from sdtoolbox.postshock import CJspeed

from funcs.simulation.sensitivity.istarmap import istarmap  # noqa: F401
from funcs.simulation import cell_size
from funcs.simulation import thermo


DATA_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mechanism_comparison_results.h5")
lock = Lock()


class MechanismValidationError(Exception):
    pass


class DataColumn(Enum):
    mechanism = "mech"
    fuel = "fuel"
    oxidizer = "oxidizer"
    diluent = "diluent"
    co2e_diluent_mole_fraction = "dil_mf_co2e"
    actual_diluent_mole_fraction = "dil_mf"
    cj_speed = "cj_speed"
    cj_time = "cj_time"
    gavrikov = "cs_gavrikov"
    ng = "cs_ng"
    westbrook = "cs_westbrook"
    cell_size_time = "cs_time"

    @classmethod
    def all(cls) -> List[str]:
        return [item.value for _, item in cls.__members__.items()]


@dataclasses.dataclass
class DataUpdate:
    column: DataColumn
    value: Any


class Mechanism(Enum):
    GRI3 = "gri30.cti"
    GRI3HighT = "gri30_highT.cti"
    GRI3Ion = "gri30_ion.cti"
    # Mevel2015 = "Mevel2015.cti"  # contains undeclared duplicate reactions
    # Mevel2018 = "Mevel2018.cti"  # contains undeclared duplicate reactions
    # HexanePartial = "hexanePartial.cti"  # contains undeclared duplicate reactions
    SanDiego = "sandiego20161214.cti"
    JetSurf = "JetSurf2.cti"
    Blanquart = "Blanquart2018.cti"
    Aramco = "aramco2.cti"
    FFCM = "ffcm1.cti"

    @classmethod
    def all(cls) -> List[Mechanism]:
        gri = []
        non_gri = []
        for _, mech in cls.__members__.items():
            if "gri3" in mech.value:
                gri.append(mech)
            else:
                non_gri.append(mech)

        def by_value(m):
            return m.value

        # run the GRI mechs first, then do the rest alphabetically
        return sorted(gri, key=by_value, reverse=True) + sorted(non_gri, key=by_value, reverse=True)

    @classmethod
    def min_itemsize(cls):
        longest = 0
        for _, mech in cls.__members__.items():
            longest = max(len(mech.value), longest)

        return longest

    @classmethod
    def validate_all(cls):
        print("Validating mechanisms")
        ctis = [mech.value for _, mech in cls.__members__.items()]
        with Pool() as p:
            results = p.map(cls._validate, ctis)
        p.join()
        cls._sort_validations_and_print(results)

    @staticmethod
    def _sort_validations_and_print(results: list[tuple[str, Optional[bool]]]):
        good_mechanisms = []
        bad_mechanisms = []
        unparseable_mechanisms = []
        for (cti, success) in results:
            if success:
                good_mechanisms.append(cti)
            elif success is None:
                unparseable_mechanisms.append(cti)
            else:
                bad_mechanisms.append(cti)
        sep = "\n    - "
        if good_mechanisms:
            msg = f"\u001b[32mCONTAINS REACTANTS: {sep}{sep.join(sorted(good_mechanisms))}\u001b[0m"
            print(msg)
        if bad_mechanisms:
            msg = f"\u001b[31mDOES NOT CONTAIN REACTANTS: {sep}{sep.join(sorted(bad_mechanisms))}\u001b[0m"
            print(msg)
        if unparseable_mechanisms:
            msg = f"\u001b[31mUNPARSEABLE: {sep}{sep.join(sorted(unparseable_mechanisms))}\u001b[0m"
            print(msg)

    @staticmethod
    def _validate(cti: str) -> Tuple[str, Optional[bool]]:
        try:
            with warnings.catch_warnings():
                # I don't care about deprecations in cantera 3
                warnings.simplefilter("ignore")
                gas = ct.Solution(cti)
            good_species = gas.species_names
            check_species = [d.value for d in Diluent.all() if d is not Diluent.NONE]
            # noinspection PyTypeChecker
            check_species.extend([InitialConditions.fuel, InitialConditions.oxidizer])
            for s in check_species:
                if s not in good_species:
                    return cti, False
            return cti, True
        except:
            return cti, None

    @classmethod
    def validate_all_cantera_mechanisms(cls):
        mechs_to_validate = set()
        for d in ct.get_data_directories():
            mechs_to_validate.update([
                os.path.split(mech)[1] for mech in
                glob.glob(os.path.join(d, "*.cti"))
                + glob.glob(os.path.join(d, "*.xml"))
                + glob.glob(os.path.join(d, "*.yaml"))
            ])

        print("Validating _all_ mechanisms")
        with Pool() as p:
            results = p.map(cls._validate, mechs_to_validate)
        p.join()
        cls._sort_validations_and_print(results)


class Diluent(Enum):
    NONE = None
    CO2 = "CO2"
    N2 = "N2"

    @classmethod
    def all(cls) -> List[Diluent]:
        return [item for _, item in cls.__members__.items()]

    def as_string(self):
        if self is Diluent.NONE:
            return "None"
        else:
            return self.value


@dataclasses.dataclass(frozen=True)
class InitialConditions:
    p0: int = 101325  # Pa
    t0: int = 300  # K
    phi: int = 1
    fuel: str = "CH4"
    oxidizer: str = "N2O"

    @classmethod
    def undiluted_gas(cls, mechanism: Mechanism) -> ct.Solution:
        gas = ct.Solution(mechanism.value)
        gas.TP = cls.t0, cls.p0
        gas.set_equivalence_ratio(cls.phi, cls.fuel, cls.oxidizer)

        return gas

    @classmethod
    def diluted_gas(cls, mechanism: Mechanism, diluent: Diluent, diluent_mol_frac: float) -> ct.Solution:
        gas = cls.undiluted_gas(mechanism)
        if diluent is not Diluent.NONE:
            if diluent is not Diluent.CO2:
                # adjust mole fraction to be CO2e
                diluent_mol_frac = thermo.match_adiabatic_temp(
                    mechanism.value,
                    InitialConditions.fuel,
                    InitialConditions.oxidizer,
                    InitialConditions.phi,
                    Diluent.CO2.value,
                    diluent_mol_frac,
                    diluent.value,
                    InitialConditions.t0,
                    InitialConditions.p0
                )

            new_mole_fractions = thermo.diluted_species_dict(
                spec=gas.mole_fraction_dict(),
                diluent=diluent.as_string(),
                diluent_mol_frac=diluent_mol_frac,
            )
            gas.X = new_mole_fractions

        return gas


def calculate_all_new(getter_func: Callable, diluent_mol_fracs: Tuple[float, ...], force_calc: bool):
    arg_combinations = tuple(itertools.product(
        Mechanism.all(),
        Diluent.all(),
        diluent_mol_fracs,
        (force_calc,)
    ))
    # cut down number of processes to conserve RAM
    with Pool(initializer=init, initargs=(lock,), processes=8) as pool:
        # noinspection PyUnresolvedReferences,PyTestUnpassedFixture
        for _ in tqdm.tqdm(pool.istarmap(getter_func, arg_combinations), total=len(arg_combinations)):
            pass
    pool.join()


def init(l):  # noqa: E741
    global lock
    lock = l


def _new_data_row(
    mech: Mechanism,
    diluent: Diluent,
    actual_dil_mf: float,
    co2e_dil_mf: float,
    updates_dict: Dict[str, str],
):
    data = pd.DataFrame({
        DataColumn.mechanism.value: mech.value,
        DataColumn.fuel.value: InitialConditions.fuel,
        DataColumn.oxidizer.value: InitialConditions.oxidizer,
        DataColumn.diluent.value: diluent.as_string(),
        DataColumn.co2e_diluent_mole_fraction.value: co2e_dil_mf,
        DataColumn.actual_diluent_mole_fraction.value: actual_dil_mf,
        **updates_dict
    }, index=[0], columns=DataColumn.all())
    object_keys = (
        DataColumn.mechanism.value,
        DataColumn.fuel.value,
        DataColumn.oxidizer.value,
        DataColumn.diluent.value,
    )
    numeric_keys = list(set(data.keys()).difference(object_keys))
    data[numeric_keys] = data[numeric_keys].astype(np.float64)

    return data


def update_results(
    mech: Mechanism,
    diluent: Diluent,
    actual_dil_mf: float,
    co2e_dil_mf: float,
    updates: List[DataUpdate],
):
    with lock:
        with pd.HDFStore(DATA_FILE, "a") as store:
            updates_dict = {update.column.value: update.value for update in updates}
            if "/data" in store.keys():
                for update in updates:
                    mask = (
                        (store["data"][DataColumn.mechanism.value] == mech.value)
                        & (store["data"][DataColumn.diluent.value] == diluent.as_string())
                        & np.isclose(store["data"][DataColumn.co2e_diluent_mole_fraction.value], co2e_dil_mf)
                    )
                    if mask.any():
                        update_row = store["data"][mask]
                        update_row[update.column.value] = update.value
                        store.remove("data", where=mask)
                        store.append("data", update_row, format="table")
                    else:
                        store.append(
                            "data", _new_data_row(mech, diluent, actual_dil_mf, co2e_dil_mf, updates_dict)
                        )
                        break
            else:
                df = _new_data_row(mech, diluent, actual_dil_mf, co2e_dil_mf, updates_dict)
                store.put(
                    "data", df, format="table", min_itemsize={DataColumn.mechanism.value: Mechanism.min_itemsize()}
                )


def get_cj_speed(mech: Mechanism, diluent: Diluent, diluent_mol_frac: float, force_calc: bool = False,) -> float:
    if diluent is Diluent.NONE:
        diluent_mol_frac = 0.0
    elif np.isclose(diluent_mol_frac, 0.0):
        diluent = Diluent.NONE

    if force_calc:
        cj = _calculate_cj_speed(mech, diluent, diluent_mol_frac)
    else:
        cj = _try_loading_value(mech, diluent, diluent_mol_frac, DataColumn.cj_speed)
        if cj is None:
            cj = _calculate_cj_speed(mech, diluent, diluent_mol_frac)

    return cj


def _calculate_cj_speed(mech: Mechanism, diluent: Diluent, co2e_dil_mf: float) -> float:
    if diluent is not Diluent.NONE and not np.isclose(co2e_dil_mf, 0) and co2e_dil_mf > 0:
        gas = InitialConditions.diluted_gas(mech, diluent, co2e_dil_mf)
        actual_dil_mf = gas.mole_fraction_dict()[diluent.value]
    else:
        gas = InitialConditions.undiluted_gas(mech)
        actual_dil_mf = co2e_dil_mf  # = 0

    t0 = time.time()
    cj = CJspeed(
        P1=InitialConditions.p0,
        T1=InitialConditions.t0,
        q=gas.mole_fraction_dict(),
        mech=mech.value
    )
    t1 = time.time()
    updates = [
        DataUpdate(DataColumn.cj_speed, cj),
        DataUpdate(DataColumn.cj_time, t1-t0)
    ]
    update_results(mech, diluent, actual_dil_mf, co2e_dil_mf, updates)

    return cj


def _try_loading_value(mech: Mechanism, diluent: Diluent, co2e_dil_mf: float, col: DataColumn) -> Optional[float]:
    val = None
    with lock:
        with pd.HDFStore(DATA_FILE, "a") as store:
            if "/data" in store.keys():
                maybe_data = store["data"][
                    (store["data"][DataColumn.mechanism.value] == mech.value)
                    & (store["data"][DataColumn.diluent.value] == diluent.as_string())
                    & np.isclose(store["data"][DataColumn.co2e_diluent_mole_fraction.value], co2e_dil_mf)
                ]
                if len(maybe_data):
                    val = maybe_data[col.value].values[0]

    return val


def get_cell_sizes(
    mech: Mechanism,
    diluent: Diluent,
    diluent_mol_frac: float,
    force_calc: bool = False,
) -> Tuple[float, float, float]:
    if diluent is Diluent.NONE:
        diluent_mol_frac = 0.0
    elif np.isclose(diluent_mol_frac, 0.0):
        diluent = Diluent.NONE

    cj_speed = get_cj_speed(mech, diluent, diluent_mol_frac, force_calc)

    if force_calc:
        gavrikov, ng, westbrook = _simulate_cell_sizes(mech, diluent, diluent_mol_frac, cj_speed)
    else:
        gavrikov = _try_loading_value(mech, diluent, diluent_mol_frac, DataColumn.gavrikov)
        ng = _try_loading_value(mech, diluent, diluent_mol_frac, DataColumn.ng)
        westbrook = _try_loading_value(mech, diluent, diluent_mol_frac, DataColumn.westbrook)
        if pd.isna([gavrikov, ng, westbrook]).any:
            gavrikov, ng, westbrook = _simulate_cell_sizes(mech, diluent, diluent_mol_frac, cj_speed)

    return gavrikov, ng, westbrook


def _simulate_cell_sizes(
    mech: Mechanism,
    diluent: Diluent,
    co2e_dil_mf: float,
    cj_speed: float,
) -> Tuple[float, float, float]:  # todo: figure out timeout
    if diluent is not Diluent.NONE and not np.isclose(co2e_dil_mf, 0) and co2e_dil_mf > 0:
        gas = InitialConditions.diluted_gas(mech, diluent, co2e_dil_mf)
        actual_dil_mf = gas.mole_fraction_dict()[diluent.value]
    else:
        actual_dil_mf = co2e_dil_mf  # = 0

    # noinspection PyBroadException
    try:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cell_sizes = cell_size.calculate(
                mech.value,
                InitialConditions.t0,
                InitialConditions.p0,
                InitialConditions.fuel,
                InitialConditions.oxidizer,
                InitialConditions.phi,
                diluent.value,
                actual_dil_mf,
                cj_speed,
            )
        t1 = time.time()
        dt = t1 - t0
    except Exception:
        cell_sizes = cell_size.CellSizeResults.empty()
        dt = np.NaN

    gavrikov = cell_sizes.cell_size.gavrikov
    ng = cell_sizes.cell_size.ng
    westbrook = cell_sizes.cell_size.westbrook
    updates = [
        DataUpdate(DataColumn.gavrikov, gavrikov),
        DataUpdate(DataColumn.ng, ng),
        DataUpdate(DataColumn.westbrook, westbrook),
        DataUpdate(DataColumn.cell_size_time, dt)
    ]
    update_results(mech, diluent, actual_dil_mf, co2e_dil_mf, updates)

    return gavrikov, ng, westbrook


def main():
    # Mechanism.validate_all()
    Mechanism.validate_all_cantera_mechanisms()
    # print(get_cj_speed(Mechanism.GRI3HighT, Diluent.NONE, 0.0))
    # calculate_all_new(get_cj_speeds, (0.1, 0.2), False)
    # calculate_all_new(get_cell_sizes, (0.1,), False)
    # with pd.HDFStore(DATA_FILE, "r") as store:
    #     df = store["data"]
    # pd.set_option("display.max_columns", None)
    # print(df)


if __name__ == "__main__":
    main()
