from funcs.simulation import thermo


def build(
    mech,
    init_temp,
    init_press,
    equivalence,
    fuel,
    oxidizer,
    diluent=None,
    diluent_mol_frac=0,
):
    """
    Build a gas object for simulations
    Perhaps this should be in thermo

    Returns
    -------
    ct.Solution
    """
    gas = thermo.ct.Solution(mech)
    gas.TP = init_temp, init_press
    gas.set_equivalence_ratio(equivalence, fuel, oxidizer)
    if diluent is not None and diluent_mol_frac > 0:
        spec = thermo.diluted_species_dict(gas.mole_fraction_dict(), diluent, diluent_mol_frac)
        gas.X = spec

    return gas
