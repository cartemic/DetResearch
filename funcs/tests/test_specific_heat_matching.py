import cantera as ct
import numpy as np

from .. import specific_heat_matching as shm


class TestDilutedSpeciesDict:
    def test_single_species_diluent(self):
        dil_frac = 0.1
        gas = ct.Solution("gri30.cti")
        gas.set_equivalence_ratio(1, "H2", "O2")
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = shm.diluted_species_dict(
            gas.mole_fraction_dict(),
            "CO2",
            dil_frac
        )

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                dil_frac  # correct diluent fraction
            ],
            [
                spec_dil["H2"] / spec_dil["O2"],
                spec_dil["CO2"]
            ]
        )

    def test_multi_species_diluent(self):
        mol_co2 = 5
        mol_ar = 3
        dil_frac = 0.1
        gas = ct.Solution("gri30.cti")
        gas.set_equivalence_ratio(1, "H2", "O2")
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = shm.diluted_species_dict(
            gas.mole_fraction_dict(),
            "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar),
            dil_frac
        )

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac  # correct diluent fraction
            ],
            [
                spec_dil["H2"] / spec_dil["O2"],
                spec_dil["CO2"] / spec_dil["AR"],
                spec_dil["CO2"] + spec_dil["AR"]
            ]
        )

    def test_single_species_diluent_plus_ox(self):
        mol_co2 = 0
        mol_ar = 3
        ox_diluent = 10
        dil_frac = 0.1
        gas = ct.Solution("gri30.cti")
        gas.set_equivalence_ratio(1, "H2:1", "O2:1 AR:{:d}".format(ox_diluent))
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = shm.diluted_species_dict(
            gas.mole_fraction_dict(),
            "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar),
            dil_frac
        )
        # adjust argon to account for only the portion in the diluent mixture
        ar_adjusted = spec_dil["AR"] - spec["AR"] * spec_dil["O2"] / spec["O2"]

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac  # correct diluent fraction
            ],
            [
                spec_dil["H2"] / spec_dil["O2"],
                spec_dil["CO2"] / ar_adjusted,
                spec_dil["CO2"] + ar_adjusted
            ]
        )

    def test_multi_species_diluent_plus_ox(self):
        mol_co2 = 1
        mol_ar = 3
        ox_diluent = 10
        dil_frac = 0.1
        gas = ct.Solution("gri30.cti")
        gas.set_equivalence_ratio(1, "H2:1", "O2:1 AR:{:d}".format(ox_diluent))
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = shm.diluted_species_dict(
            gas.mole_fraction_dict(),
            "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar),
            dil_frac
        )
        # adjust argon to account for only the portion in the diluent mixture
        ar_adjusted = spec_dil["AR"] - spec["AR"] * spec_dil["O2"] / spec["O2"]

        assert np.allclose(
            [
                f_a_orig,          # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac           # correct diluent fraction
            ],
            [
                spec_dil["H2"] / spec_dil["O2"],
                spec_dil["CO2"] / ar_adjusted,
                spec_dil["CO2"] + ar_adjusted
            ]
        )
