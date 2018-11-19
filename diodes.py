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
import numpy as np
from nptdms import TdmsFile
from tkinter import filedialog as fd
from tkinter import Tk
import scipy.signal as signal

def find_diode_data(
        data_directory='',
        base_file_name='diodes.tdms'
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
    diode_data_locations = [
        os.path.join(location[0], base_file_name)
        for location in os.walk(data_directory)
        if base_file_name in location[2]
    ]

    # raise an error if no folders contain diode output
    if not len(diode_data_locations):
        raise FileNotFoundError(
            'No instances of ' + base_file_name + ' found'
        )

    return diode_data_locations


def _velocity_calculator(
        diode_data_file,
        sample_rate,
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
    diode_dataframe = _load_diode_data(
        diode_data_file,
        apply_lowpass
    )

    # make sure something happened during the test by averaging over a single
    # period of 60Hz noise. If only 60Hz noise is present, the rolling average
    # will be less than zero.
    noise_period = int(sample_rate/60)
    data_is_useful = all(
        [value > 0 for value in
         diode_dataframe.rolling(noise_period).mean().max().values]
    )

    if data_is_useful:
        calculated_velocity = sample_specific_velocity / np.diff(
                diode_dataframe.diff(axis=0).idxmax(
                    axis=0,
                    skipna=True
                ).values
            )
    else:
        calculated_velocity = np.zeros(len(diode_dataframe.keys())-1)

    if multiprocess:
        return instance, calculated_velocity

    else:
        return calculated_velocity


def _load_diode_data(
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
    data = TdmsFile(diode_data_file).as_dataframe()
    for key in data.keys():
        # remove time column
        if 'diode' not in key.replace('diodes', '').lower():
            data = data.drop(key, axis=1)

        if apply_lowpass:
            data = data.apply(_diode_filter)

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
    specific_velocity = diode_spacing * sample_frequency

    if isinstance(diode_data_location, str):
        # a single path was input as a string
        return _velocity_calculator(
            diode_data_location,
            sample_frequency,
            specific_velocity,
            lowpass,
            multiprocess
        )

    else:
        if multiprocess:
            import multiprocessing as mp
            pool = mp.Pool()
            mp_result = pool.starmap(
                _velocity_calculator,
                [[location,
                  sample_frequency,
                  specific_velocity,
                  lowpass,
                  multiprocess,
                  instance
                  ] for instance, location in enumerate(diode_data_location)
                 ]
            )

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
                [sample_frequency for _ in range(len(diode_data_location))],
                [specific_velocity for _ in range(len(diode_data_location))],
                [lowpass for _ in range(len(diode_data_location))],
                [multiprocess for _ in range(len(diode_data_location))]
            )))
