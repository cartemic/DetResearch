import cantera as ct
import os
from funcs import cti_solution_writer as cti
from numpy import isclose, allclose


def test_cti_writer():
    global FILE_LOC
    mech = 'gri30.cti'
    test_name = 'TEST'
    gas = ct.Solution(mech)
    test_file_name, ct_data_dir = cti.write(gas, test_name)
    test_file_loc = os.path.join(
        ct_data_dir,
        test_file_name
    )
    test_gas = ct.Solution(test_file_name)
    os.remove(test_file_loc)
    checks = [
        test_gas.n_species == gas.n_species,
        test_gas.n_reactions == gas.n_reactions,
        test_gas.n_elements == gas.n_elements
    ]
    for good, test in zip(gas.reaction_equations(),
                          test_gas.reaction_equations()):
        checks.append(test == good)

    for good, test in zip(gas.forward_rate_constants,
                          test_gas.forward_rate_constants):
        checks.append(isclose(good, test))

    for good, test in zip(gas.species(),
                          test_gas.species()):
        checks.append(test.composition == good.composition)
        checks.append(test.name == good.name)
        checks.append(isclose(test.charge, good.charge))
        checks.append(isclose(test.size, good.size))
        checks.append(allclose(test.thermo.coeffs, good.thermo.coeffs))
        checks.append(test.thermo.derived_type == good.thermo.derived_type)
        checks.append(isclose(test.thermo.max_temp, good.thermo.max_temp))
        checks.append(isclose(test.thermo.min_temp, good.thermo.min_temp))
        checks.append(test.thermo.n_coeffs == good.thermo.n_coeffs)
        checks.append(isclose(test.thermo.reference_pressure,
                              good.thermo.reference_pressure))
        checks.append(isclose(test.transport.acentric_factor,
                              good.transport.acentric_factor))
        checks.append(isclose(test.transport.diameter,
                              good.transport.diameter))
        checks.append(isclose(test.transport.dipole,
                              good.transport.dipole))
        checks.append(isclose(test.transport.dispersion_coefficient,
                              good.transport.dispersion_coefficient))
        checks.append(test.transport.geometry == good.transport.geometry)
        checks.append(isclose(test.transport.polarizability,
                              good.transport.polarizability))
        checks.append(isclose(test.transport.quadrupole_polarizability,
                              good.transport.quadrupole_polarizability))
        checks.append(isclose(test.transport.rotational_relaxation,
                              good.transport.rotational_relaxation))
        checks.append(isclose(test.transport.well_depth,
                              good.transport.well_depth))

    assert(all(checks))


if __name__ == '__main__':  # pragma: no cover
    import subprocess
    subprocess.check_call(
        'pytest test_cti_solution_Writer.py -vv --noconftest --cov'
    )
