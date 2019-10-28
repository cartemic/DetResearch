import cantera as ct
import numpy as np
from scipy.optimize import minimize


def diluted_species_dict(spec, diluent, diluent_mol_frac):
    if diluent not in spec.keys():
        spec = {k: v * (1 - diluent_mol_frac) for k, v in spec.items()}
        spec[diluent] = diluent_mol_frac
        return spec
    else:
        spec[diluent] += 1 / (1 / diluent_mol_frac - 1)
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
        initial_temperature,
        initial_pressure,
        mechanism
):
    inert_gas = ct.Solution(mechanism)
    active_gas = ct.Solution(mechanism)
    inert_gas.TPX = (
        initial_temperature,
        initial_pressure,
        diluted_species_dict(
            undiluted_mixture,
            inert_diluent,
            inert_mol_frac
        )
    )
    active_gas.TPX = (
        initial_temperature,
        initial_pressure,
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
        initial_temperature,
        initial_pressure,
        mechanism
):
    best = minimize(
        spec_heat_error,
        np.array([initial_mol_frac]),
        args=(
            initial_mol_frac,
            new_diluent,
            initial_diluent,
            undiluted_mixture,
            initial_temperature,
            initial_pressure,
            mechanism
        ),
        bounds=np.array([(0, 1)])
    )
    new_mol_fraction = best.x[0]
    return new_mol_fraction


if __name__ == '__main__':
    mech = 'gri30.cti'
    initial_gas = ct.Solution(mech)
    new_gas = ct.Solution(mech)

    mol_frac_to_match = 0.1
    diluent_to_match = 'CO2'
    replacement_diluent = 'AR'
    initial_mixture = {'CH4': 0.2, 'N2O': 0.8}
    temp_init = 300
    press_init = 101325

    initial_gas.TPX = (
        temp_init,
        press_init,
        diluted_species_dict(
            initial_mixture,
            diluent_to_match,
            mol_frac_to_match
        )
    )

    new_mole_fraction = match_specific_heat(
        mol_frac_to_match,
        diluent_to_match,
        replacement_diluent,
        initial_mixture,
        temp_init,
        press_init,
        mech,
    )

    new_gas.TPX = (
        temp_init,
        press_init,
        diluted_species_dict(
            initial_mixture,
            replacement_diluent,
            new_mole_fraction
        )
    )

    print(
        'Mole fraction of {:s}: {:0.6f}\n'.format(
            replacement_diluent,
            new_mole_fraction
        )
    )

    gamma_err = initial_gas.cp / initial_gas.cv - new_gas.cp / new_gas.cv
    gamma_err_pct = gamma_err / (initial_gas.cp / initial_gas.cv) * 100
    cp_err = initial_gas.cp - new_gas.cp
    cp_err_pct = cp_err / initial_gas.cp * 100
    cv_err = initial_gas.cv - new_gas.cv
    cv_err_pct = cv_err / initial_gas.cv * 100
    rss_err = np.sqrt(np.sum(np.square([cp_err, cv_err, gamma_err])))
    rss_err_pct = np.sqrt(np.sum(np.square(
        [cp_err_pct, cv_err_pct, gamma_err_pct]
    )))
    print(
        'gamma err: {:+0.6f} ({:+0.2f}%)'.format(
            gamma_err,
            gamma_err_pct
        )
    )
    print(
        '   cp err: {:+0.6f} ({:+0.2f}%)'.format(
            cp_err,
            cp_err_pct
        )
    )
    print(
        '   cv err: {:+0.6f} ({:+0.2f}%)'.format(
            cv_err,
            cv_err_pct
        )
    )
    print(
        '  RSS err: {:+0.6f} ({:+0.2f}%)'.format(
            rss_err,
            rss_err_pct
        )
    )
