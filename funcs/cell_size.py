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
import numpy as np
import sdtoolbox as sd
import cantera as ct
from . import database as db


class CellSize:
    """
    todo: docstring pls
    """
    def __call__(
            self,
            mechanism,
            initial_temp,
            initial_press,
            fuel,
            oxidizer,
            equivalence,
            diluent,
            diluent_mol_frac,
            inert=None,
            perturbed_reaction=-1,
            perturbation_fraction=1e-2,
            store_data=False,
            database='sensitivity.sqlite',
            table_name='test_data'
    ):
        data_table = db.Table(database, table_name)
        self.T1 = initial_temp
        gas1 = self._build_gas_object(
            mechanism,
            fuel,
            oxidizer,
            equivalence,
            diluent,
            diluent_mol_frac,
            inert
        )
        q = gas1.X
        cj_speed = sd.postshock.CJspeed(initial_press, initial_temp, q, mechanism)
        gas1.TP = initial_temp, initial_press

        # FIND EQUILIBRIUM POST SHOCK STATE FOR GIVEN SPEED
        # todo: need to bypass mechanism
        gas = sd.postshock.PostShock_eq(
            cj_speed,
            initial_press,
            initial_temp,
            q,
            mechanism
        )
        u_cj = cj_speed * gas1.density / gas.density
        self.cj_speed = u_cj

        # FIND FROZEN POST SHOCK STATE FOR GIVEN SPEED
        # todo: need to bypass mechanism
        gas = sd.postshock.PostShock_fr(
            cj_speed,
            initial_press,
            initial_temp,
            q,
            mechanism
        )

        # SOLVE ZND DETONATION ODES
        out = sd.znd.zndsolve(
            gas,
            gas1,
            cj_speed,
            advanced_output=True,
            t_end=2e-3
        )

        # Find CV parameters including effective activation energy
        gas.TPX = initial_temp, initial_press, q
        # todo: need to bypass mechanism
        gas = sd.postshock.PostShock_fr(
            cj_speed,
            initial_press,
            initial_temp,
            q,
            mechanism
        )
        self.Ts, Ps = gas.TP
        Ta = self.Ts * 1.02
        gas.TPX = Ta, Ps, q
        CVout1 = sd.cv.cvsolve(gas)
        Tb = self.Ts * 0.98
        gas.TPX = Tb, Ps, q
        CVout2 = sd.cv.cvsolve(gas)

        # Approximate effective activation energy for CV explosion
        taua = CVout1['ind_time']
        taub = CVout2['ind_time']
        if taua == 0 or taub == 0:
            self.activation_energy = 0
        else:
            self.activation_energy = 1 / self.Ts * (
                    np.log(taua / taub) / ((1 / Ta) - (1 / Tb))
            )

        #  Find Gavrikov induction time based on 50% limiting species
        #  consumption, fuel for lean mixtures, oxygen for rich mixtures
        if equivalence >= 1:
            limit_species = fuel
        else:
            limit_species = 'O2'
        limit_species_loc = gas.species_index(limit_species)
        gas.TPX = self.Ts, Ps, q
        X_initial = gas.mole_fraction_dict()[limit_species]
        gas.equilibrate('UV')
        X_final = gas.mole_fraction_dict()[limit_species]
        T_final = gas.T
        X_gav = 0.5*(X_initial - X_final) + X_final
        t_gav = np.nanmax(
            np.concatenate([
                CVout1['time'][CVout1['speciesX'][limit_species_loc] > X_gav],
                [0]
            ])
        )

        #  Westbrook time based on 50% temperature rise
        T_west = 0.5*(T_final - self.Ts) + self.Ts
        t_west = np.nanmax(
            np.concatenate([
                CVout1['time'][CVout1['T'] < T_west],
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

        # store calculations
        if perturbed_reaction == -1:
            # unperturbed data, store in test condition table with initial
            # conditions and store all reactions
            table_id = data_table.store_test_row(
                mechanism=mechanism,
                initial_temp=initial_temp,
                initial_press=initial_press,
                fuel=fuel,
                oxidizer=oxidizer,
                equivalence=equivalence,
                diluent=diluent,
                diluent_mol_frac=diluent_mol_frac,
                inert=inert,
                cj_speed=cj_speed,
                ind_len_west=self.induction_length['Westbrook'],
                ind_len_gav=self.induction_length['Gavrikov'],
                ind_len_ng=self.induction_length['Ng'],
                cell_size_west=self.cell_size['Westbrook'],
                cell_size_gav=self.cell_size['Gavrikov'],
                cell_size_ng=self.cell_size['Ng'],
            )
            data_table.store_base_rxn_table(
                rxn_table_id=table_id,
                gas=gas
            )
        else:
            # perturbed data, store in perturbed table
            # todo: make sure a base is stored before doing this
            [table_id] = data_table.fetch_test_rows(
                mechanism=mechanism,
                initial_temp=initial_temp,
                initial_press=initial_press,
                fuel=fuel,
                oxidizer=oxidizer,
                equivalence=equivalence,
                diluent=diluent,
                diluent_mol_frac=diluent_mol_frac,
                inert=inert
            )['rxn_table_id']
            # todo: fill out all inputs correctly
            data_table.store_perturbed_row(
                rxn_table_id=table_id,
                rxn_no=perturbed_reaction,
                rxn='',
                k_i='ki',
                cj_speed='cj_speed',
                ind_len_west=self.induction_length['Westbrook'],
                ind_len_gav=self.induction_length['Gavrikov'],
                ind_len_ng=self.induction_length['Ng'],
                cell_size_west=self.cell_size['Westbrook'],
                cell_size_gav=self.cell_size['Gavrikov'],
                cell_size_ng=self.cell_size['Ng'],
                sens_cj_speed='',
                sens_ind_len_west='',
                sens_ind_len_gav='',
                sens_ind_len_ng='',
                sens_cell_size_west='',
                sens_cell_size_gav='',
                sens_cell_size_ng=''
            )

        return self.cell_size

    def _build_solution_with_inerts(
            self,
            mech,
            inert_species
    ):
        inert_species = self._enforce_species_list(inert_species)
        species = ct.Species.listFromFile(mech)
        reactions = []
        for rxn in ct.Reaction.listFromFile(mech):
            if not any([
                s in list(rxn.reactants) + list(rxn.products)
                for s in inert_species
            ]):
                reactions.append(rxn)

        return ct.Solution(
            thermo='IdealGas',
            species=species,
            reactions=reactions,
            kinetics='GasKinetics'
        )

    @staticmethod
    def _enforce_species_list(species):
        if isinstance(species, str):
            species = [species.upper()]
        elif hasattr(species, '__iter__'):
            species = [s.upper() for s in species]
        else:
            raise TypeError('Bad species type: %s' % type(species))

        return species

    def _build_gas_object(
            self,
            mech,
            fuel,
            oxidizer,
            phi,
            diluent,
            diluent_mol_frac,
            inert=None
    ):
        if inert is not None:
            gas = ct.Solution(mech)
        else:
            gas = self._build_solution_with_inerts(mech, inert)

        gas.set_equivalence_ratio(
            phi,
            fuel,
            oxidizer
        )
        if diluent is not None and diluent_mol_frac > 0:
            # dilute the gas!
            mole_fractions = gas.mole_fraction_dict()
            new_fuel_fraction = (1 - diluent_mol_frac) * \
                mole_fractions[fuel]
            new_oxidizer_fraction = (1 - diluent_mol_frac) * \
                mole_fractions[oxidizer]
            gas.X = '{0}: {1} {2}: {3} {4}: {5}'.format(
                diluent,
                diluent_mol_frac,
                fuel,
                new_fuel_fraction,
                oxidizer,
                new_oxidizer_fraction
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
        T_0 = self.T1  # initial reactant temperature (K)
        T_vn = self.Ts

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
        gav_y = T_vn / T_0
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
