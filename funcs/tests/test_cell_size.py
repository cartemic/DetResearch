print(__package__)

from funcs.cell_size import CellSize
import numpy as np

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


class TestAgainstDemo:
    cells = CellSize()

    init_press = 100000
    init_temp = 300
    mechanism = 'Mevel2017.cti'

    def test_induction_lengths(self):
        self.cells(
            P1=self.init_press,
            T1=self.init_temp,
            fuel='H2',
            oxidizer='O2:1, N2:3.76',
            phi=1,
            mech=self.mechanism
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
            P1=self.init_press,
            T1=self.init_temp,
            fuel='H2',
            oxidizer='O2:1, N2:3.76',
            phi=1,
            mech=self.mechanism
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

    def test_no_dilution(self):
        undiluted = {'H2': 2 / 3, 'O2': 1 / 3}

        # should not dilute with diluent=None
        test = self.cells._build_gas_object(
            mech='gri30.cti',
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
            mech='gri30.cti',
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

    def test_with_dilution(self):
        test = self.cells._build_gas_object(
            mech='gri30.cti',
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

