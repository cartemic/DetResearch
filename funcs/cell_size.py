"""
Detonation cell size estimation functions, based on the work of Shepherd et al:
http://shepherd.caltech.edu/EDL/PublicResources/sdt/nb/sdt_intro.slides.html

A description of the theory, numerical methods, and applications are described
in the following report:

    Numerical Solution Methods for Shock and Detonation Jump Conditions, S.
    Browne, J. Ziegler, and J. E. Shepherd, GALCIT Report FM2006.006 - R3,
    California Institute of Technology Revised September, 2018.

This script uses SDToolbox, which can be found at
http://shepherd.caltech.edu/EDL/PublicResources/sdt/
"""
import cantera as ct
import numpy as np

from .specific_heat_matching import diluted_species_dict

OriginalSolution = ct.Solution


def _enforce_species_list(species):
    if isinstance(species, str):
        species = [species.upper()]
    elif hasattr(species, '__iter__') and \
            all([isinstance(s, str) for s in species]):
        species = [s.upper() for s in species]
    else:
        if hasattr(species, '__iter__'):
            bad_type = [
                type(item) for item in species if not isinstance(item, str)
            ]
        else:
            bad_type = type(species)

        raise TypeError('Bad species type: %s' % bad_type)

    return species


# noinspection PyArgumentList
def solution_with_inerts(
        mech,
        inert_species
):
    inert_species = _enforce_species_list(inert_species)
    species = ct.Species.listFromFile(mech)
    reactions = []
    for rxn in ct.Reaction.listFromFile(mech):
        if not any([
            s in list(rxn.reactants) + list(rxn.products)
            for s in inert_species
        ]):
            reactions.append(rxn)

    return OriginalSolution(
        thermo='IdealGas',
        species=species,
        reactions=reactions,
        kinetics='GasKinetics'
    )


def wrapped_cvsolve(
        gas,
        sd,
        max_tries=5,
        t_end=1e-6
):
    """
    Look jack, I don't have time for your `breaking` malarkey

    Try cvsolve a handful of times with increasing max time lengths to see if
    maybe it's just being mean.

    Parameters
    ----------
    gas : ct.Solution
        gas object to work on
    sd : module
        locally modified sdtoolbox
    max_tries : int
        how motivated are you
    t_end : float
        initial end time, which is doubled each iteration

    Returns
    -------
    idk whatever was supposed to come out in the first place, it's late
    """
    tries = 0
    # if you don't reset the Solution's properties it'll go crazy during the
    # iteration process. Iterations are supposed to be independent, so that's...
    # you know, pretty bad
    init_tpx = gas.TPX
    out = None
    while tries <= max_tries:
        gas.TPX = init_tpx
        tries += 1
        if tries < max_tries:
            # this exception is broad on purpose.
            # noinspection PyBroadException
            try:
                out = sd.cv.cvsolve(
                    gas,
                    t_end=t_end
                )
                break
            except:  # noqa: E722
                pass
        else:
            # let it break if it's gonna break after max tries
            out = sd.cv.cvsolve(
                gas,
                t_end=t_end
            )
        t_end *= 2
    return out


def wrapped_zndsolve(
        gas,
        base_gas,
        cj_speed,
        t_end,
        max_step,
        sd,
        max_tries=5
):
    tries = 0
    init_tpx = gas.TPX
    init_tpx_base = base_gas.TPX
    out = None
    while tries <= max_tries:
        # retry the simulation if the initial time step doesn't work
        # drop time step by one order of magnitude every failure
        gas.TPX = init_tpx
        base_gas.TPX = init_tpx_base
        tries += 1
        if tries < max_tries:
            try:
                out = sd.znd.zndsolve(
                    gas,
                    base_gas,
                    cj_speed,
                    advanced_output=True,
                    t_end=t_end,
                    max_step=max_step
                )
                break
            except ct.CanteraError:
                pass
        else:
            # let it break if it's gonna break after max tries
            out = sd.znd.zndsolve(
                gas,
                base_gas,
                cj_speed,
                advanced_output=True,
                t_end=t_end,
                max_step=max_step
            )
        max_step /= 10.
    return out


class CellSize:
    """
    A class that, when called, calculates detonation cell sizes using the
    methods of Westbrook, Ng, and Gavrikov. Based on the script
    demo_ZND_CJ_cell.m which can be found at
    http://shepherd.caltech.edu/EDL/PublicResources/sdt/nb/sdt_intro.slides.html

    Inputs
    ------
    base_mechanism : str
        Base chemical mechanism for Cantera calculations
    initial_temp : float
        Initial mixture temperature in Kelvin
    initial_press : float
        Initial mixture temperature in Pascals
    fuel : str
        Fuel
    oxidizer : str
        Oxidizer
    equivalence : float
        Equivalence ratio
    diluent : str
        Diluent
    diluent_mol_frac : float
        Mole fraction of diluent
    cj_speed : float
        Chapman-Jouguet wave speed of the mixture specified above
    inert : str or None
        Specie to deactivate (must be deactivated in CJ speed calculation as
        well). Defaults to None
    perturbed_reaction : int
        Reaction number to perturb. Defaults to -1, meaning that no reaction is
        perturbed
    perturbation_fraction : float
        Fraction by which to perturb the specified reaction's forward and
        reverse rate constants, i.e. dk/k. Defaults to 1e-2 (1%)
    """
    def __call__(
            self,
            base_mechanism,
            initial_temp,
            initial_press,
            fuel,
            oxidizer,
            equivalence,
            diluent,
            diluent_mol_frac,
            cj_speed,
            inert=None,
            perturbed_reaction=-1,
            perturbation_fraction=1e-2,
            max_tries_znd=5,
            max_step_znd=1e-4,
            max_tries_cv=10
    ):
        # sdt import is here to avoid any module-level weirdness stemming from
        # Solution object modification
        import sdtoolbox as sd

        # self.mechanism will change with inert species, but base will not
        self.mechanism = base_mechanism
        self.base_mechanism = base_mechanism
        self.initial_temp = initial_temp
        self.initial_press = initial_press
        self.fuel = fuel
        self.oxidizer = oxidizer
        self.equivalence = equivalence
        self.diluent = diluent
        self.diluent_mol_frac = diluent_mol_frac
        self.inert = inert
        self.T1 = initial_temp
        self.perturbed_reaction = perturbed_reaction
        self.perturbation_fraction = perturbation_fraction

        if perturbed_reaction != -1:
            # alter ct.Solution within sdtoolbox so that it returns perturbed
            # reaction results
            def my_solution(mech):
                if inert is None:
                    original_gas = OriginalSolution(mech)
                else:
                    original_gas = solution_with_inerts(
                        mech,
                        inert
                    )
                original_gas.set_multiplier(
                    1 + perturbation_fraction,
                    perturbed_reaction
                )
                return original_gas
            sd.postshock.ct.Solution = my_solution

        self.base_gas = self._build_gas_object()
        q = self.base_gas.X

        if perturbed_reaction >= 0:
            assert (self.base_gas.multiplier(perturbed_reaction) ==
                    1 + perturbation_fraction)
        for rxn in range(self.base_gas.n_reactions):
            if rxn != perturbed_reaction:
                assert self.base_gas.multiplier(rxn) == 1

        # FIND EQUILIBRIUM POST SHOCK STATE FOR GIVEN SPEED
        gas = sd.postshock.PostShock_eq(
            cj_speed,
            initial_press,
            initial_temp,
            q,
            self.mechanism
        )

        if perturbed_reaction >= 0:
            assert (gas.multiplier(perturbed_reaction) ==
                    1 + perturbation_fraction)
        for rxn in range(gas.n_reactions):
            if rxn != perturbed_reaction:
                assert gas.multiplier(rxn) == 1

        u_cj = cj_speed * self.base_gas.density / gas.density
        self.cj_speed = u_cj

        # FIND FROZEN POST SHOCK STATE FOR GIVEN SPEED
        gas = sd.postshock.PostShock_fr(
            cj_speed,
            initial_press,
            initial_temp,
            q,
            self.mechanism
        )

        if perturbed_reaction >= 0:
            assert (gas.multiplier(perturbed_reaction) ==
                    1 + perturbation_fraction)

        # SOLVE ZND DETONATION ODES
        out = wrapped_zndsolve(
            gas=gas,
            base_gas=self.base_gas,
            cj_speed=cj_speed,
            t_end=2e-3,
            max_step=max_step_znd,
            sd=sd,
            max_tries=max_tries_znd
        )

        # Find CV parameters including effective activation energy
        gas.TPX = initial_temp, initial_press, q
        gas = sd.postshock.PostShock_fr(
            cj_speed,
            initial_press,
            initial_temp,
            q,
            self.mechanism
        )

        if perturbed_reaction >= 0:
            assert (gas.multiplier(perturbed_reaction) ==
                    1 + perturbation_fraction)

        self.Ts, Ps = gas.TP
        temp_a = self.Ts * 1.02
        gas.TPX = temp_a, Ps, q

        base_t_end = 1e-6
        # cv_out_0 = sd.cv.cvsolve(gas)
        cv_out_0 = wrapped_cvsolve(
            gas,
            sd,
            max_tries_cv,
            base_t_end
        )

        temp_b = self.Ts * 0.98
        gas.TPX = temp_b, Ps, q
        # cv_out_1 = sd.cv.cvsolve(gas, t_end=10e-6)
        cv_out_1 = wrapped_cvsolve(
            gas,
            sd,
            max_tries_cv,
            base_t_end
        )

        # Approximate effective activation energy for CV explosion
        tau_a = cv_out_0['ind_time']
        tau_b = cv_out_1['ind_time']
        if tau_a == 0 or tau_b == 0:
            self.activation_energy = 0
        else:
            self.activation_energy = 1 / self.Ts * (
                    np.log(tau_a / tau_b) / ((1 / temp_a) - (1 / temp_b))
            )

        #  Find Gavrikov induction time based on 50% limiting species
        #  consumption, fuel for lean mixtures, oxygen for rich mixtures
        if equivalence <= 1:
            limit_species = fuel
        else:
            limit_species = 'O2'
        limit_species_loc = gas.species_index(limit_species)
        gas.TPX = self.Ts, Ps, q
        mf_initial = gas.mole_fraction_dict()[limit_species]
        gas.equilibrate('UV')
        mf_final = gas.mole_fraction_dict()[limit_species]
        temp_final = gas.T
        mf_gav = 0.5*(mf_initial - mf_final) + mf_final
        t_gav = np.nanmax(
            np.concatenate([
                cv_out_0['time'][
                    cv_out_0['speciesX'][limit_species_loc] > mf_gav
                ],
                [0]
            ])
        )

        #  Westbrook time based on 50% temperature rise
        temp_west = 0.5*(temp_final - self.Ts) + self.Ts
        t_west = np.nanmax(
            np.concatenate([
                cv_out_0['time'][cv_out_0['T'] < temp_west],
                [0]
            ])
        )

        # Ng et al definition of max thermicity width
        # Equation 2
        self.chi_ng = self.activation_energy * out['ind_len_ZND'] / \
            (u_cj / max(out['thermicity']))

        self.induction_length = {
            'Westbrook': t_west*out['U'][0],
            'Gavrikov': t_gav*out['U'][0],
            'Ng': out['ind_len_ZND']
        }

        # calculate and return cell size results
        self.cell_size = {
            'Westbrook': self._cell_size_westbrook(),
            'Gavrikov': self._cell_size_gavrikov(),
            'Ng': self._cell_size_ng()
        }

        return self.cell_size

    def _build_gas_object(self):
        if self.inert is None:
            gas = ct.Solution(self.mechanism)
        else:
            # this will change self.mechanism to the new one with inerts
            gas = solution_with_inerts(self.mechanism, self.inert)

        gas.set_equivalence_ratio(
            self.equivalence,
            self.fuel,
            self.oxidizer
        )
        if self.diluent is not None and self.diluent_mol_frac > 0:
            # # dilute the gas!
            # mole_fractions = gas.mole_fraction_dict()
            # new_fuel_fraction = (1 - self.diluent_mol_frac) * \
            #     mole_fractions[self.fuel]
            # new_oxidizer_fraction = (1 - self.diluent_mol_frac) * \
            #     mole_fractions[self.oxidizer]
            # gas.X = '{0}: {1} {2}: {3} {4}: {5}'.format(
            #     self.diluent,
            #     self.diluent_mol_frac,
            #     self.fuel,
            #     new_fuel_fraction,
            #     self.oxidizer,
            #     new_oxidizer_fraction
            # )
            # no need to write this twice. Use the function written for
            # building test matrices
            gas.X = diluted_species_dict(
                gas.mole_fraction_dict(),
                self.diluent,
                self.diluent_mol_frac
            )

        gas.TP = self.initial_temp, self.initial_press
        if self.perturbed_reaction >= 0:
            gas.set_multiplier(
                1 + self.perturbation_fraction,
                self.perturbed_reaction
            )
        return gas

    def _cell_size_ng(self):
        """
        Calculates cell size using the correlation given by Ng, H. D., Ju, Y.,
        & Lee, J. H. S. (2007). Assessment of detonation hazards in
        high-pressure hydrogen storage from chemical sensitivity analysis.
        International Journal of Hydrogen Energy, 32(1), 93–99.
        https://doi.org/10.1016/j.ijhydene.2006.03.012 using equations (1) and
        (2) along with the coefficients given in Table 1.

        Parameters
        ----------

        Returns
        -------
        cell_size : float
            Estimated cell size (m)
        """
        # Coefficients from Table 1
        a_0 = 30.465860763763
        a = np.array([
            89.55438805808153,
            -130.792822369483,
            42.02450507117405
        ])
        b = np.array([
            -0.02929128383850,
            1.026325073064710e-5,
            -1.031921244571857e-9
        ])

        # Equation 1
        chi_pow = np.power(self.chi_ng, range(1, len(a)+1))
        cell_size = (
            a_0 + np.sum(a / chi_pow + b * chi_pow)
        ) * self.induction_length['Ng']

        return cell_size

    def _cell_size_gavrikov(self):
        """
        Calculates cell size using the correlation given by Gavrikov, A.I.,
        Efimenko, A.A., & Dorofeev, S.B. (2000). A model for detonation cell
        size prediction from chemical kinetics. Combustion and Flame, 120(1–2),
        19–33. https://doi.org/10.1016/S0010-2180(99)00076-0 using equation (5)
        along with coefficients given in Table 1

        Parameters
        ----------

        Returns
        -------
        cell_size : float
            Estimated cell size (m)
        """
        # Load parameters
        temp_0 = self.T1  # initial reactant temperature (K)
        temp_vn = self.Ts

        # Coefficients from Table 1
        a = -0.007843787493
        b = 0.1777662961
        c = 0.02371845901
        d = 1.477047968
        e = 0.1545112957
        f = 0.01547021569
        g = -1.446582357
        h = 8.730494354
        i = 4.599907939
        j = 7.443410379
        k = 0.4058325462
        m = 1.453392165

        # Equation 5
        gav_y = temp_vn / temp_0
        cell_size = np.power(
            10,
            gav_y * (a * gav_y - b) + self.activation_energy *
            (c * self.activation_energy - d + (e - f * gav_y) * gav_y) +
            g * np.log(gav_y) + h * np.log(self.activation_energy) +
            gav_y * (i / self.activation_energy - k *
                     gav_y / np.power(self.activation_energy, m)) - j
        ) * self.induction_length['Gavrikov']

        return cell_size

    def _cell_size_westbrook(self):
        """
        Calculates cell size using the correlation given by Westbrook, C. K., &
        Urtiew, P. A. (1982). Chemical kinetic prediction of critical parameters
        in gaseous detonations. Symposium (International) on Combustion, 19(1),
        615–623. https://doi.org/10.1016/S0082-0784(82)80236-1

        Parameters
        ----------

        Returns
        -------
        cell_size : float
            Estimated cell size (m)
        """
        return 29 * self.induction_length['Westbrook']


if __name__ == '__main__':  # pragma: no cover
    import subprocess
    subprocess.check_call('pytest -vv tests/test_cell_size.py')
