import cantera as ct
import numpy as np
from scipy.optimize import minimize


def diluted_species_dict(
        spec,
        diluent,
        diluent_mol_frac
):
    """
    Creates a dictionary of mole fractions diluted by a given amount with a
    given gas mixture

    Parameters
    ----------
    spec : dict
        Mole fraction dictionary (gas.mole_fraction_dict() from undiluted)
    diluent : str
        String of diluents using cantera's format, e.g. "CO2" or "N2:1 NO:0.01"
    diluent_mol_frac : float
        mole fraction of diluent to add

    Returns
    -------
    dict
        new mole_fraction_dict to be inserted into the cantera solution object
    """
    # collect total diluent moles
    moles_dil = 0.
    diluent_dict = dict()
    split_diluents = diluent.split(" ")
    if len(split_diluents) > 1:
        for d in split_diluents:
            key, value = d.split(":")
            value = float(value)
            diluent_dict[key] = value
            moles_dil += value
    else:
        diluent_dict[diluent] = 1
        moles_dil = 1

    for key in diluent_dict.keys():
        diluent_dict[key] /= moles_dil

    for key, value in diluent_dict.items():
        if key not in spec.keys():
            spec[key] = 0

        if diluent_mol_frac != 0:
            spec[key] += diluent_dict[key] / (1 / diluent_mol_frac - 1)

    new_total_moles = sum(spec.values())
    for s in spec.keys():
        spec[s] /= new_total_moles
    return spec


def spec_heat_error(
        inert_mol_frac,
        active_mol_frac,
        inert_diluent,
        active_diluent,
        undiluted_mixture,
        init_temp,
        init_press,
        mech
):
    inert_gas = ct.Solution(mech)
    active_gas = ct.Solution(mech)
    inert_gas.TPX = (
        init_temp,
        init_press,
        diluted_species_dict(
            undiluted_mixture,
            inert_diluent,
            inert_mol_frac
        )
    )
    active_gas.TPX = (
        init_temp,
        init_press,
        diluted_species_dict(
            undiluted_mixture,
            active_diluent,
            active_mol_frac
        )
    )
    err_gamma = (inert_gas.cp / inert_gas.cv - active_gas.cp / active_gas.cv) \
        / (inert_gas.cp / inert_gas.cv)
    err_cp = 1 - inert_gas.cp / active_gas.cp
    err_cv = 1 - inert_gas.cv / active_gas.cv

    error = np.sqrt(np.sum(np.square([err_cp, err_gamma, err_cv])))

    return error


def match_specific_heat(
        initial_mol_frac,
        initial_diluent,
        new_diluent,
        undiluted_mixture,
        init_temp,
        init_press,
        mech,
        bounds=(0, 0.9999)
):
    best = minimize(
        spec_heat_error,
        np.array([initial_mol_frac]),
        args=(
            initial_mol_frac,
            new_diluent,
            initial_diluent,
            undiluted_mixture,
            init_temp,
            init_press,
            mech
        ),
        bounds=np.array([bounds])
    )
    new_mol_fraction = best.x[0]
    return new_mol_fraction


def get_adiabatic_temp(
        mech,
        fuel,
        oxidizer,
        phi,
        diluent,
        diluent_mol_frac,
        init_temp,
        init_press
):
    """
    Calculates the adiabatic flame temperature of a given mixture using
    Cantera

    Parameters
    ----------
    mech : str
        Mechanism to use
    fuel : str
        Fuel to use; must be in `mech`
    oxidizer : str
        Oxidizer to use; must be in `mech`
    phi : float
        Equivalence ratio
    diluent: str
        Species with which to dilute the mixture; must be in `mech`
    diluent_mol_frac : float
        Mole fraction of active diluent to apply to the undiluted mixture
    init_temp : float
        Mixture initial temperature in Kelvin
    init_press : float
        Mixture initial pressure in Pascals

    Returns
    -------
    float
        Adiabatic flame temperature of the input mixture in Kelvin
    """
    gas = ct.Solution(mech)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    gas.TPX = (
        init_temp,
        init_press,
        diluted_species_dict(
            gas.mole_fraction_dict(),
            diluent,
            diluent_mol_frac
        )
    )
    gas.equilibrate("HP")
    return gas.T


def temp_error(
        diluent_mol_frac,
        target_temp,
        mech,
        fuel,
        oxidizer,
        phi,
        diluent,
        init_temp,
        init_press
):
    """
    Compares the adiabatic flame temperature from a given combination of
    inputs to a target temperature and returns the absolute value of the
    resulting difference.

    Parameters
    ----------
    diluent_mol_frac : float
        Mole fraction of active diluent to apply to the undiluted mixture
    target_temp : float
        Adiabatic flame temperature to match, in Kelvin
    mech : str
        Mechanism to use
    fuel : str
        Fuel to use; must be in `mech`
    oxidizer : str
        Oxidizer to use; must be in `mech`
    phi : float
        Equivalence ratio
    diluent: str
        Diluent with which to evaluate the new adiabatic flame temperature;
        must be in `mech`
    init_temp : float
        Mixture initial temperature in Kelvin
    init_press : float
        Mixture initial pressure in Pascals

    Returns
    -------
    float
        Absolute difference between the target temperature and the adiabatic
        flame temperature of the input mixture, in Kelvin
    """
    return abs(
        get_adiabatic_temp(
            mech=mech,
            fuel=fuel,
            oxidizer=oxidizer,
            phi=phi,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
            init_temp=init_temp,
            init_press=init_press
        ) - target_temp
    )


def match_adiabatic_temp(
        mech,
        fuel,
        oxidizer,
        phi,
        dil_active,
        dil_active_mol_frac,
        dil_inert,
        init_temp,
        init_press,
        tol=1e-6
):
    """
    This function returns the **additional** mole fraction of a diluent gas
    required to match the adiabatic flame temperature of another diluent. If
    the diluent is *not* in the original mixture (e.g. H2/O2 diluted with N2)
    this will be the **total** mole fraction; if the diluent **is** in the
    original mixture (e.g. H2/air diluted with N2) then the **total** mole
    fraction can be seen by calling:

    diluted_species_dict(
        gas.mole_fraction_dict(),
        dil_inert,
        inert_mol_frac
    )

    The **additional** mole fraction is returned because, in this application,
    air is being added as a single component, and thus the partial pressure
    of the **additional** nitrogen is a parameter of interest.

    Parameters:
    -----------
    mech : str
        Mechanism to use
    fuel : str
        Fuel to use; must be in `mech`
    oxidizer : str
        Oxidizer to use; must be in `mech`
    phi : float
        Equivalence ratio of undiluted mixture
    dil_active : str
        Active diluent, which gives the target adiabatic flame temperature
        to be matched; must be in `mech`
    dil_active_mol_frac : float
        Mole fraction of active diluent to apply to the undiluted mixture
    dil_inert : str
        Inert diluent to match to the active diluent; must be in `mech`
    init_temp : float
        Mixture initial temperature in Kelvin
    init_press : float
        Mixture initial pressure in Pascals
    tol : float
        Tolerance for adiabatic flame temperature matching, in Kelvin

    Returns
    -------
    float
        Additional mole fraction of diluent gas needed to match the adiabatic
        flame temperature to within the specified tolerance
    """
    target_temp = get_adiabatic_temp(
        mech,
        fuel,
        oxidizer,
        phi,
        dil_active,
        dil_active_mol_frac,
        init_temp,
        init_press
    )
    best = minimize(
        temp_error,
        np.array([dil_active_mol_frac]),
        args=(
            target_temp,
            mech,
            fuel,
            oxidizer,
            phi,
            dil_inert,
            init_temp,
            init_press
        ),
        method="Nelder-Mead",
        tol=tol
    )
    return best.x[0]


if __name__ == '__main__':
    mechanism = "gri30.cti"
    initial_temperature = 300
    initial_pressure = ct.one_atm
    mixture_fuel = "C3H8"
    mixture_oxidizer = "O2:1 N2:3.76"
    mixture_phi = 1
    mixture_active_diluent_mol_frac = 0.02
    mixture_active_diluent = "CO2"
    mixture_inert_diluent = "N2"

    matched_inert_mol_frac = match_adiabatic_temp(
        mechanism,
        mixture_fuel,
        mixture_oxidizer,
        mixture_phi,
        mixture_active_diluent,
        mixture_active_diluent_mol_frac,
        mixture_inert_diluent,
        initial_temperature,
        initial_pressure
    )
    print(matched_inert_mol_frac)
