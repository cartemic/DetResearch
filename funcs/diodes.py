# -*- coding: utf-8 -*-
"""
PURPOSE:
    Tools for measuring detonation wave speeds with photodiodes

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""

import os
from tkinter import Tk
from tkinter import filedialog as fd

import numpy as np
import pandas as pd
import scipy.signal as signal
import uncertainties as un
from nptdms import TdmsFile
from uncertainties import unumpy as unp


def find_diode_data(
        data_directory='',
        base_file_name='diodes.tdms',
        ignore_hidden=True
):
    """
    Finds all diode data for a day of testing

    Parameters
    ----------
    data_directory : str
        Directory containing data. If none is given, the user will be asked to
        browse to the desired directory.
    base_file_name : str
        Common file name for diode data (default is diodes.tdms)
    ignore_hidden : bool
        If true, ignore files/directories preceded by .

    Returns
    -------
    list
        List of diode data file locations
    """
    # have user browse for directory if none was given
    if not len(data_directory):
        Tk().withdraw()
        data_directory = fd.askdirectory(
            initialdir=os.getcwd()
        )

    # get a list of directories containing base file name
    diode_data_locations = []
    for location in os.walk(data_directory):
        if base_file_name in location[2]:
            if ignore_hidden and "." in location[0]:
                pass
            else:
                diode_data_locations.append(
                    os.path.join(location[0], base_file_name)
                )

    # raise an error if no folders contain diode output
    if not len(diode_data_locations):
        raise FileNotFoundError(
            'No instances of ' + base_file_name + ' found'
        )

    return sorted(diode_data_locations)


def _velocity_calculator(
        diode_data_file,
        sample_specific_velocity,
        apply_lowpass,
        multiprocess=False,
        instance=0
):
    """

    Parameters
    ----------
    diode_data_file : str
        Location of photo diode data file
    sample_specific_velocity : float
        meters/second/sample
    instance : int
        Solution instance (only matters for multiprocessing)

    Returns
    -------
    numpy.ndarray
        Array of inter-diode velocities in m/s, calculated using the
        maximum gradient.
    """
    bad_value = np.array([0]) * un.ufloat(np.NaN, np.NaN)
    diode_dataframe = load_diode_data(
        diode_data_file,
        apply_lowpass
    )

    try:
        arrival_times = unp.uarray(
            diode_dataframe.diff(axis=0).idxmax(axis=0, skipna=True).values,
            [0.5, 0.5]
        )
        arrival_diff = np.diff(arrival_times)

    except ValueError:
        # empty array, return zero velocity
        arrival_diff = bad_value

    if arrival_diff > 0:
        calculated_velocity = sample_specific_velocity / arrival_diff
    else:
        calculated_velocity = bad_value

    if calculated_velocity > 3000:
        # obvious garbage
        calculated_velocity = bad_value

    if multiprocess:
        return instance, calculated_velocity

    else:
        return calculated_velocity


def load_diode_data(
        diode_data_file,
        apply_lowpass
):
    """
    Loads in diode data from a tdms file given the location

    Parameters
    ----------
    diode_data_file : str
        Location of diode data file
    apply_lowpass : bool
        Determines whether or not a lowpass filter is applied

    Returns
    -------
    pandas.core.frame.DataFrame
        Dataframe of diode data, with time stamps removed if they are present
    """
    # import data
    tf = TdmsFile(diode_data_file)
    diode_channels = tf.group_channels("diodes")
    if len(diode_channels) == 0 or \
            len(diode_channels[0].data) > 0:
        # diodes.tdms has data
        data = TdmsFile(diode_data_file).as_dataframe()
        for key in data.keys():
            # remove time column
            if "diode" not in key.replace("diodes", "").lower():
                data = data.drop(key, axis=1)

        if apply_lowpass:
            # noinspection PyTypeChecker
            data = data.apply(_diode_filter)

    else:
        # empty tdms file
        data = pd.DataFrame(
            columns=[c.path for c in diode_channels],
            data=np.array(
                [[np.NaN] * 50 for _ in diode_channels]).T
        )

    return data


def _diode_filter(
        data,
        n=3,
        wn=0.01
):
    """
    A butterworth lowpass filter for diode data

    Parameters
    ----------
    data : numpy.ndarray or pandas.core.frame.DataFrame
        Data to be filtered
    n : int
        Order of butterworth filter
    wn : float
        Relative cutoff frequency (cutoff/Nyquist)

    Returns
    -------
    numpy.ndarray or pandas.core.frame.DataFrame
        Filtered signal
    """
    butter = signal.butter(n, wn, btype='lowpass', output='ba')
    return signal.filtfilt(butter[0], butter[1], data)


# noinspection PyTypeChecker
def calculate_velocity(
        diode_data_location,
        diode_spacing=0.3048,
        spacing_uncert=0.00079375,
        sample_frequency=1e6,
        lowpass=True,
        multiprocess=False
):
    """
    Calculates wave speed from photo diode data

    Parameters
    ----------
    diode_data_location : list or str
        Location or list of locations of photo diode data files
    diode_spacing : float
        Distance between the diodes, in meters (assumed to be uniform between
        all diodes)
    spacing_uncert : float
        Uncertainty in diode spacing, in meters
    sample_frequency : float
        Sampling frequency of diode data in Hz
    lowpass : bool
        Applies a 3rd order low-pass Butterworth filter if True
    multiprocess : bool
        Uses multiple processes to calculate velocities to speed up analysis

    Returns
    -------
    numpy.ndarray
        Array of inter-diode wave speeds
    """
    specific_velocity = un.ufloat(
        diode_spacing * sample_frequency,
        spacing_uncert
    )

    if isinstance(diode_data_location, str):
        # a single path was input as a string
        return _velocity_calculator(
            diode_data_location,
            specific_velocity,
            lowpass,
            multiprocess
        )

    elif np.isnan(diode_data_location):
        # this means that there was a diode crash during the test, and no data
        # was written to the shot directory. Return a unp.uarray of NaN so that
        # NaNs propagate correctly during post-processing
        return unp.uarray([np.NaN], np.NaN)

    else:
        if multiprocess:
            import multiprocessing as mp
            pool = mp.Pool()
            mp_result = pool.starmap(
                _velocity_calculator,
                [[location,
                  specific_velocity,
                  lowpass,
                  multiprocess,
                  instance
                  ] for instance, location in enumerate(diode_data_location)
                 ]
            )
            pool.close()

            # sort result by instance (to match inputs) and build array of
            # the results for output
            mp_result.sort()
            mp_result = np.array(
                [item for _, item in mp_result]
            )
            return mp_result

        else:
            return np.array(list(map(
                _velocity_calculator,
                [location for location in diode_data_location],
                [specific_velocity for _ in range(len(diode_data_location))],
                [lowpass for _ in range(len(diode_data_location))],
                [multiprocess for _ in range(len(diode_data_location))]
            )))
