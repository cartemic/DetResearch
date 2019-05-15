from funcs.cell_size import CellSize
import numpy as np
from time import time
import funcs.database as db
from uuid import uuid4

relTol = 1e-4
absTol = 1e-6

# Test calculated values against demo script
original_cell_sizes = {
    'Gavrikov': 1.9316316546518768e-02,
    'Ng': 6.5644825968914763e-03,
    'Westbrook': 3.4852910825972942e-03
}

original_induction_lengths = {
    'Gavrikov': 0.0001137734347788197,
    'Ng': 0.00014438224858385156,
    'Westbrook': 0.00012018245112404462
}

cj_speed = 1968.121482986247


# noinspection PyProtectedMember
class TestAgainstDemo:
    cells = CellSize()

    init_press = 100000
    init_temp = 300
    mechanism = 'Mevel2017.cti'

    def test_induction_lengths(self):
        self.cells(
            base_mechanism=self.mechanism,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            fuel='H2',
            oxidizer='O2:1, N2:3.76',
            equivalence=1,
            diluent='None',
            diluent_mol_frac=0,
            cj_speed=cj_speed
        )
        assert (
            all([[
                abs(length - self.cells.induction_length[correlation]) / length
                < relTol
                for correlation, length in original_induction_lengths.items()
            ], [
                abs(length - self.cells.induction_length[correlation]) < absTol
                for correlation, length in original_induction_lengths.items()
            ]
            ])
        )

    def test_cell_sizes(self):
        test = self.cells(
            base_mechanism=self.mechanism,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            fuel='H2',
            oxidizer='O2:1, N2:3.76',
            equivalence=1,
            diluent='None',
            diluent_mol_frac=0,
            cj_speed=cj_speed
        )
        assert (
            all([[
                abs(cell - test[correlation]) / cell < relTol
                for correlation, cell in original_cell_sizes.items()
            ], [
                abs(cell - test[correlation]) < absTol
                for correlation, cell in original_cell_sizes.items()
            ]]
            )
        )

    def test_build_gas_no_dilution(self):
        undiluted = {'H2': 2 / 3, 'O2': 1 / 3}

        # should not dilute with diluent=None
        test = self.cells._build_gas_object(
            fuel='H2',
            oxidizer='O2',
            phi=1,
            diluent=None,
            diluent_mol_frac=0.5
        ).mole_fraction_dict()
        check_none = [
            np.isclose(undiluted[key], value) for key, value in test.items()
        ]

        # should not dilute with diluent_mol_frac=0
        test = self.cells._build_gas_object(
            fuel='H2',
            oxidizer='O2',
            phi=1,
            diluent='AR',
            diluent_mol_frac=0
        ).mole_fraction_dict()
        check_zero = [
            np.isclose(undiluted[key], value) for key, value in test.items()
        ]
        assert all([check_none, check_zero])

    def test_build_gas_with_dilution(self):
        test = self.cells._build_gas_object(
            fuel='H2',
            oxidizer='O2',
            phi=1,
            diluent='AR',
            diluent_mol_frac=0.1
        ).mole_fraction_dict()
        check = [
            np.isclose(test['H2'] / test['O2'], 2),
            np.isclose(test['AR'], 0.1)
        ]
        assert all(check)


if __name__ == '__main__':  # pragma: no cover
    import subprocess
    from funcs.tests.test_database import remove_stragglers
    try:
        subprocess.check_call(
            'pytest test_cell_size.py -vv --noconftest --cov '
            '--cov-report html'
        )
    except subprocess.CalledProcessError as e:
        # clean up in case of an unexpected error cropping up
        remove_stragglers()
        raise e

    remove_stragglers()
