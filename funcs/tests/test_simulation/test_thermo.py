import cantera as ct
import numpy as np

from funcs.simulation import thermo


class TestDilutedSpeciesDict:
    def test_single_species_diluent(self):
        dil_frac = 0.1
        gas = ct.Solution("gri30.yaml")
        gas.set_equivalence_ratio(1, "H2", "O2")
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = thermo.diluted_species_dict(gas.mole_fraction_dict(), "CO2", dil_frac)

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                dil_frac,  # correct diluent fraction
            ],
            [spec_dil["H2"] / spec_dil["O2"], spec_dil["CO2"]],
        )

    def test_multi_species_diluent(self):
        mol_co2 = 5
        mol_ar = 3
        dil_frac = 0.1
        gas = ct.Solution("gri30.yaml")
        gas.set_equivalence_ratio(1, "H2", "O2")
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = thermo.diluted_species_dict(
            gas.mole_fraction_dict(), "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar), dil_frac
        )

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac,  # correct diluent fraction
            ],
            [spec_dil["H2"] / spec_dil["O2"], spec_dil["CO2"] / spec_dil["AR"], spec_dil["CO2"] + spec_dil["AR"]],
        )

    def test_single_species_diluent_plus_ox(self):
        mol_co2 = 0
        mol_ar = 3
        ox_diluent = 10
        dil_frac = 0.1
        gas = ct.Solution("gri30.yaml")
        gas.set_equivalence_ratio(1, "H2:1", "O2:1 AR:{:d}".format(ox_diluent))
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = thermo.diluted_species_dict(
            gas.mole_fraction_dict(), "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar), dil_frac
        )
        # adjust argon to account for only the portion in the diluent mixture
        ar_adjusted = spec_dil["AR"] - spec["AR"] * spec_dil["O2"] / spec["O2"]

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac,  # correct diluent fraction
            ],
            [spec_dil["H2"] / spec_dil["O2"], spec_dil["CO2"] / ar_adjusted, spec_dil["CO2"] + ar_adjusted],
        )

    def test_multi_species_diluent_plus_ox(self):
        mol_co2 = 1
        mol_ar = 3
        ox_diluent = 10
        dil_frac = 0.1
        gas = ct.Solution("gri30.yaml")
        gas.set_equivalence_ratio(1, "H2:1", "O2:1 AR:{:d}".format(ox_diluent))
        spec = gas.mole_fraction_dict()
        f_a_orig = spec["H2"] / spec["O2"]
        spec_dil = thermo.diluted_species_dict(
            gas.mole_fraction_dict(), "CO2:{:d} AR:{:d}".format(mol_co2, mol_ar), dil_frac
        )
        # adjust argon to account for only the portion in the diluent mixture
        ar_adjusted = spec_dil["AR"] - spec["AR"] * spec_dil["O2"] / spec["O2"]

        assert np.allclose(
            [
                f_a_orig,  # fuel/air ratio preserved
                mol_co2 / mol_ar,  # ratio preserved within diluent mixture
                dil_frac,  # correct diluent fraction
            ],
            [spec_dil["H2"] / spec_dil["O2"], spec_dil["CO2"] / ar_adjusted, spec_dil["CO2"] + ar_adjusted],
        )


class TestGetFASt:
    def test_single_species_oxidizer(self):
        assert np.isclose(thermo.get_f_a_st("H2", "O2"), 2)

    def test_compound_oxidizer(self):
        assert np.isclose(thermo.get_f_a_st("CH4", "O2:1 N2:3.76"), 1 / 9.52)

    def test_air(self):
        assert np.isclose(thermo.get_f_a_st("CH4", "air"), 1 / 9.52)


def test_get_dil_mol_frac():
    # 1 F + 2 O + 4 D
    assert np.isclose(thermo.get_dil_mol_frac(1, 2, 4), 4 / 7)


class TestGetEquivalenceRatio:
    def test_lean(self):
        # H2 + O2
        assert np.isclose(thermo.get_equivalence_ratio(1, 1, 2), 0.5)

    def test_unity(self):
        # H2 + 1/2 O2
        assert np.isclose(thermo.get_equivalence_ratio(1, 0.5, 2), 1)

    def test_rich(self):
        # 4 H2 + 1/2 O2
        assert np.isclose(thermo.get_equivalence_ratio(4, 0.5, 2), 4)


def test_get_adiabatic_temp():
    aft = thermo.get_adiabatic_temp("gri30.yaml", "H2", "O2:1 N2:3.76", 1, "", 0, 300, 101325)
    assert np.isclose(aft, 2380.8062780784453)


def test_match_adiabatic_temp():
    mech = "gri30.yaml"
    fuel = "H2"
    oxidizer = "O2"
    dil_original = "CO2"
    dil_new = "AR"
    phi = 1
    dil_mf_original = 0.1
    t_0 = 300
    p_0 = 101325
    dil_mf_new = thermo.match_adiabatic_temp(
        mech,
        fuel,
        oxidizer,
        phi,
        dil_original,
        dil_mf_original,
        dil_new,
        t_0,
        p_0,
    )
    t_ad_original = thermo.get_adiabatic_temp(mech, fuel, oxidizer, phi, dil_original, dil_mf_original, t_0, p_0)
    t_ad_new = thermo.get_adiabatic_temp(mech, fuel, oxidizer, phi, dil_new, dil_mf_new, t_0, p_0)
    assert np.isclose(t_ad_original, t_ad_new)
