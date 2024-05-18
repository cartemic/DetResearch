import cantera as ct
import numpy as np
from scipy.optimize import minimize

import sdtoolbox


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
    try:
        gas = ct.Solution(mech)
        gas.set_equivalence_ratio(phi, fuel, oxidizer)
        if (diluent.lower() in ("none", "")) or np.isclose(diluent_mol_frac, 0):
            spec = gas.mole_fraction_dict()
        else:
            spec = diluted_species_dict(
                gas.mole_fraction_dict(),
                diluent,
                diluent_mol_frac
            )

        gas.TPX = (
            init_temp,
            init_press,
            spec
        )
        gas.equilibrate("HP")
        temp = gas.T
    except ct.CanteraError:
        temp = np.NaN

    return temp


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
        dil_original,
        dil_original_mol_frac,
        dil_new,
        init_temp,
        init_press,
        tol=1e-6
):
    """
    This function returns the mole fraction of a diluent gas
    required to match the adiabatic flame temperature of another diluent. If
    the diluent is *not* in the original mixture (e.g. H2/O2 diluted with N2)
    this will be the **total** mole fraction; if the diluent **is** in the
    original mixture (e.g. H2/air diluted with N2) then the **total** mole
    fraction can be seen by calling:

    diluted_species_dict(
        gas.mole_fraction_dict(),
        dil_new,
        new_mol_frac,
    )

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
    dil_original : str
        Original diluent, which gives the target adiabatic flame temperature
        to be matched; must be in `mech`
    dil_original_mol_frac : float
        Mole fraction of original diluent to apply to the undiluted mixture
    dil_new : str
        New diluent to match to the original diluent; must be in `mech`
    init_temp : float
        Mixture initial temperature in Kelvin
    init_press : float
        Mixture initial pressure in Pascals
    tol : float
        Tolerance for adiabatic flame temperature matching, in Kelvin

    Returns
    -------
    float
        Mole fraction of diluent gas needed to match the adiabatic
        flame temperature to within the specified tolerance
    """
    target_temp = get_adiabatic_temp(
        mech,
        fuel,
        oxidizer,
        phi,
        dil_original,
        dil_original_mol_frac,
        init_temp,
        init_press
    )
    best = minimize(
        temp_error,
        np.array([dil_original_mol_frac]),
        args=(
            target_temp,
            mech,
            fuel,
            oxidizer,
            phi,
            dil_new,
            init_temp,
            init_press
        ),
        method="Nelder-Mead",
        tol=tol
    )
    return best.x[0]


def get_f_a_st(
        fuel="C3H8",
        oxidizer="O2:1 N2:3.76",
        mech="gri30.cti"
):
    """
    Calculate the stoichiometric fuel/air ratio of an undiluted mixture using
    Cantera. Calculates using only x_fuel to allow for compound oxidizer
    (e.g. air)

    Parameters
    ----------
    fuel : str
    oxidizer : str
    mech : str
        mechanism file to use

    Returns
    -------
    float
        stoichiometric fuel/air ratio
    """
    if oxidizer.lower() == "air":
        oxidizer = "O2:1 N2:3.76"

    gas = ct.Solution(mech)
    gas.set_equivalence_ratio(
        1,
        fuel,
        oxidizer
    )
    x_fuel = gas.mole_fraction_dict()[fuel]
    return x_fuel / (1 - x_fuel)


def get_dil_mol_frac(
        p_fuel,
        p_oxidizer,
        p_diluent
):
    """
    Parameters
    ----------
    p_fuel : float or un.ufloat
        Fuel partial pressure
    p_oxidizer : float or un.ufloat
        Oxidizer partial pressure
    p_diluent : float or un.ufloat
        Diluent partial pressure

    Returns
    -------
    float or un.ufloat
        Diluent mole fraction
    """
    return p_diluent / (p_fuel + p_oxidizer + p_diluent)


def get_equivalence_ratio(
        p_fuel,
        p_oxidizer,
        f_a_st
):
    """
    Simple equivalence ratio function

    Parameters
    ----------
    p_fuel : float or un.ufloat
        Partial pressure of fuel
    p_oxidizer : float or un.ufloat
        Partial pressure of oxidizer
    f_a_st : float or un.ufloat
        Stoichiometric fuel/air ratio

    Returns
    -------
    float or un.ufloat
        Mixture equivalence ratio
    """
    return p_fuel / p_oxidizer / f_a_st


def calculate_laminar_flame_speed(
        gas
):
    """
    Calculates the laminar flame speed of a gas mixture.
    Based on:
    https://www.cantera.org/examples/jupyter/flames/flame_speed_with_sensitivity_analysis.ipynb.html

    Parameters
    ----------
    gas : ct.Solution
        Cantera solution object with input conditions

    Returns
    -------
    float
        Laminar flame speed (m/s)
    """
    flame = ct.FreeFlame(gas)
    flame.set_refine_criteria(ratio=3, slope=0.1, curve=0.1)
    flame.solve(loglevel=0)

    return flame.velocity[0]


def get_sound_speed(
    mech,
    fuel,
    oxidizer,
    phi,
    diluent,
    diluent_mol_frac,
    init_temp,
    init_press,
):
    gas = ct.Solution(mech)
    gas.set_equivalence_ratio(phi, fuel, oxidizer)
    q_diluted = diluted_species_dict(
        spec=gas.mole_fraction_dict(),
        diluent=diluent,
        diluent_mol_frac=diluent_mol_frac,
    )
    cj_dil = sdtoolbox.postshock.CJspeed(
        P1=init_press,
        T1=init_temp,
        q=q_diluted,
        mech=mech,
    )
    gas_fr = sdtoolbox.postshock.PostShock_eq(
        U1=cj_dil,
        P1=init_press,
        T1=init_temp,
        q=q_diluted,
        mech=mech,
    )
    return sdtoolbox.thermo.soundspeed_fr(gas_fr)


def sound_speed_error(
    diluent_mol_frac,
    target_speed,
    mech,
    fuel,
    oxidizer,
    phi,
    diluent,
    init_temp,
    init_press,
):
    return abs(
        get_sound_speed(
            mech=mech,
            fuel=fuel,
            oxidizer=oxidizer,
            phi=phi,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac,
            init_temp=init_temp,
            init_press=init_press,
        ) - target_speed
    )


def match_sound_speed(
    mech,
    fuel,
    oxidizer,
    phi,
    dil_original,
    dil_original_mol_frac,
    dil_new,
    init_temp,
    init_press,
    tol=1e-6,
):
    target_speed = get_sound_speed(
        mech=mech,
        fuel=fuel,
        oxidizer=oxidizer,
        phi=phi,
        diluent=dil_original,
        diluent_mol_frac=dil_original_mol_frac,
        init_temp=init_temp,
        init_press=init_press,
    )
    best = minimize(
        sound_speed_error,
        np.array([dil_original_mol_frac]),
        args=(
            target_speed,
            mech,
            fuel,
            oxidizer,
            phi,
            dil_new,
            init_temp,
            init_press,
        ),
        method="Nelder-Mead",
        tol=tol,
        bounds=[(0, 1)],
    )
    return best.x[0]
