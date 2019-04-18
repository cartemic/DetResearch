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


class CellSize:
    """
    todo:
     1: znd calcs
     2: change calculation functions to use self properties
    """
    def calculate_ng(
            self,
            activation_energy,
            induction_length,
            max_thermicity,
            cj_speed
    ):
        """
        Calculates cell size using the correlation given by Ng, H. D., Ju, Y.,
        & Lee, J. H. S. (2007). Assessment of detonation hazards in
        high-pressure hydrogen storage from chemical sensitivity analysis.
        International Journal of Hydrogen Energy, 32(1), 93–99.
        https://doi.org/10.1016/j.ijhydene.2006.03.012 using equations (1) and
        (2) along with the coefficients given in Table 1.

        Parameters
        ----------
        activation_energy : float
            reduced activation energy (Ea/RT_ps) (unitless)
        induction_length : float
            induction length determined by peak thermicity (m)
        max_thermicity : float
            peak thermicity value from ZND calculation (1/s)
        cj_speed : float
            CJ particle velocity (m/s)

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

        # Equation 2
        chi = activation_energy * induction_length * max_thermicity / cj_speed

        # Equation 1
        chi_pow = np.power(chi, range(1, len(a)+1))
        cell_size = (
            a_0 + np.sum(a / chi_pow + b * chi_pow)
        ) * induction_length

        return cell_size

    def calculate_gavrikov(
            self,
            induction_length,
            activation_energy,
            T_vn,
            T_0
    ):
        """
        Calculates cell size using the correlation given by Gavrikov, A.I.,
        Efimenko, A.A., & Dorofeev, S.B. (2000). A model for detonation cell
        size prediction from chemical kinetics. Combustion and Flame, 120(1–2),
        19–33. https://doi.org/10.1016/S0010-2180(99)00076-0 using equation (5)
        along with coefficients given in Table 1

        Parameters
        ----------
        induction_length : float
            induction length determined by time to 50% consumption of limiting
            species and ZND initial velocity
        activation_energy : float
            reduced activation energy (Ea/RT_ps) (unitless)
        T_vn : float
            von Neumann temperature (just behind a CJ wave) (K)
        T_0 : float
            initial reactant temperature (K)

        Returns
        -------
        cell_size : float
            Estimated cell size (m)
        """
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
            gav_y * (a * gav_y - b) + activation_energy *
            (c * activation_energy - d + (e - f * gav_y) * gav_y) +
            g * np.log(gav_y) + h * np.log(activation_energy) +
            gav_y * (i / activation_energy - k *
                     gav_y / np.power(activation_energy, m)) - j
        ) * induction_length

        return cell_size

    def calculate_westbrook(
            self,
            induction_length
    ):
        """
        Calculates cell size using the correlation given by Westbrook, C. K., &
        Urtiew, P. A. (1982). Chemical kinetic prediction of critical parameters
        in gaseous detonations. Symposium (International) on Combustion, 19(1),
        615–623. https://doi.org/10.1016/S0082-0784(82)80236-1

        Parameters
        ----------
        induction_length : float
            todo: get friendly with this

        Returns
        -------
        cell_size : float
            Estimated cell size (m)
        """
        return 29 * induction_length
