import numpy as np

import funcs.simulation.cell_size as cs

# Had to loosen tolerances after cantera version upgrade
rel_tol = 1e-4
abs_tol = 1e-4


# noinspection PyProtectedMember
class TestAgainstDemo:
    cj_speed = 1967.8454767711942
    init_press = 100000
    init_temp = 300
    mechanism = "Mevel2017.yaml"

    def test_calculations_are_correct(self):
        results = cs.calculate(
            mechanism=self.mechanism,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            fuel="H2",
            oxidizer="O2:1, N2:3.76",
            equivalence=1,
            diluent=None,
            diluent_mol_frac=0,
            cj_speed=self.cj_speed,
        )
        # Test calculated values against demo script
        original_cell_sizes = cs.ModelResults(
            gavrikov=1.9316316546518768e-02,
            ng=6.5644825968914763e-03,
            westbrook=3.4852910825972942e-03,
        )
        original_induction_lengths = cs.ModelResults(
            gavrikov=0.0001137734347788197,
            ng=0.00014438224858385156,
            westbrook=0.00012018245112404462,
        )

        assert np.allclose(results.cell_size.values(), original_cell_sizes.values(), rtol=rel_tol, atol=abs_tol)
        assert np.allclose(
            results.induction_length.values(), original_induction_lengths.values(), rtol=rel_tol, atol=abs_tol
        )

    def test_build_gas_no_dilution(self):
        undiluted = {"H2": 2 / 3, "O2": 1 / 3}

        # should not dilute with diluent=None
        test = cs.build_gas_object(
            mechanism="Mevel2017.yaml",
            equivalence=1,
            fuel="H2",
            oxidizer="O2",
            diluent=None,
            diluent_mol_frac=0.5,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            perturbed_reaction=None,
            perturbation_fraction=0,
        ).mole_fraction_dict()
        assert all(np.isclose(undiluted[key], value) for key, value in test.items())

        # should not dilute with diluent_mol_frac=0
        test = cs.build_gas_object(
            mechanism="Mevel2017.yaml",
            equivalence=1,
            fuel="H2",
            oxidizer="O2",
            diluent="Ar",
            diluent_mol_frac=0,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            perturbed_reaction=None,
            perturbation_fraction=0,
        ).mole_fraction_dict()
        assert all(np.isclose(undiluted[key], value) for key, value in test.items())

    def test_build_gas_with_dilution(self):
        test = cs.build_gas_object(
            mechanism="Mevel2017.yaml",
            equivalence=1,
            fuel="H2",
            oxidizer="O2",
            diluent="Ar",
            diluent_mol_frac=0.1,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            perturbed_reaction=None,
            perturbation_fraction=0,
        ).mole_fraction_dict()
        assert np.isclose(test["H2"] / test["O2"], 2)
        assert np.isclose(test["AR"], 0.1)

    def test_perturbed_gas_object_undiluted(self):
        pert = 3
        pert_frac = 0.01
        test = cs.build_gas_object(
            mechanism=self.mechanism,
            initial_temp=self.init_temp,
            initial_press=self.init_press,
            fuel="H2",
            oxidizer="O2:1, N2:3.76",
            equivalence=1,
            diluent=None,
            diluent_mol_frac=0,
            perturbed_reaction=pert,
            perturbation_fraction=pert_frac,
        )
        n_rxns = test.n_reactions
        correct_multipliers = np.ones(n_rxns)
        correct_multipliers[pert] = 1 + pert_frac
        multipliers = [test.multiplier(i) for i in range(n_rxns)]
        assert np.allclose(multipliers, correct_multipliers)

    def test_perturbed_gas_object_diluted(self):
        pert = 3
        pert_frac = 0.01
        test = cs.build_gas_object(
            mechanism="Mevel2017.yaml",
            initial_temp=300,
            initial_press=101325,
            fuel="H2",
            oxidizer="O2",
            equivalence=1,
            diluent="AR",
            diluent_mol_frac=0.02,
            perturbed_reaction=pert,
            perturbation_fraction=pert_frac,
        )
        n_rxns = test.n_reactions
        correct_multipliers = np.ones(n_rxns)
        correct_multipliers[pert] = 1 + pert_frac
        multipliers = [test.multiplier(i) for i in range(n_rxns)]
        assert np.allclose(multipliers, correct_multipliers)


if __name__ == "__main__":  # pragma: no cover
    import subprocess

    subprocess.check_call("pytest test_cell_size.py -vv --noconftest --cov " "--cov-report html")
