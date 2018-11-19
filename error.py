# -*- coding: utf-8 -*-
"""
PURPOSE:
    Error propagation using sequential perturbation

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import numpy as np
import warnings


class ErrorAnalysis:
    @staticmethod
    def add_uncertainty(
            uncertainties
    ):
        """
        Performs a RSS addition of uncertainty components

        Parameters
        ----------
        uncertainties : Union[Tuple, Dict, Iterable]

        Returns
        -------
        float
        """
        if not isinstance(uncertainties, np.ndarray):
            uncertainties = np.array(uncertainties)

        if len(uncertainties.shape) > 1:
            # warn the user if array isn't one dimensional so they aren't caught
            # by surprise when it is flattened
            warnings.warn(
                'Uncertainties array is not 1-D and will be flattened',
                Warning
            )

        return np.sqrt(np.sum(np.square(uncertainties)))

    @staticmethod
    def perturb(
            func,
            values,
            return_importance=False
    ):
        """
        Perturbs a function and returns the total uncertainty, calculated using
        sequential perturbation

        Parameters
        ----------
        func : function
            The function to be evaluated
        values : Dict[str, Tuple[float, float]]
            Dictionary with keys matching each function input. Values should
            be tuples where the first entry is the nominal value of the input
            and the second entry is its uncertainty value.
        return_importance : bool
            If this is true, the relative importance of each input variable is
            calculated and returned in a dictionary

        Returns
        -------
        float or Tuple[float, Dict[str, float]
        """
        nominal_values = {var: values[var][0] for var in values}
        f_nominal = func(**nominal_values)

        deltas = dict()
        for var in values:
            # get f_i plus
            perturbed_values = nominal_values.copy()
            perturbed_values[var] += values[var][1]
            f_plus = func(**perturbed_values)

            # calculate delta_i
            deltas[var] = (f_plus-f_nominal)

        uncertainty = np.sqrt(
            np.sum(
                np.square(
                    np.array(
                        list(
                            deltas.values()
                        )
                    )
                )
            )
        )

        if return_importance:
            if uncertainty == 0:
                # avoid a possible zero division error
                importance = {var: np.NaN for var in deltas}
            else:
                importance = {var: (deltas[var]/uncertainty)**2
                              for var in deltas}
            return uncertainty, importance

        else:
            return uncertainty
