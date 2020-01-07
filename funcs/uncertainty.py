import os
import pint
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import uncertainties as un
from uncertainties import unumpy as unp
from scipy.stats import t, sem

ureg = pint.UnitRegistry()
quant = ureg.Quantity

# Temperature -- T type thermocouple in manifold is used for temp calculations.
# Old datasets use K type tube thermocouple.
# All values here are in degrees Celsius
# ==============================================================================
# standard: https://www.omega.com/en-us/resources/thermocouple-types
# daq: NI 9211 operating instructions and specifications p 26
#      using Typ (Autozero on)
DF_9211 = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "data", "tc_err.csv")
)
TEMP_STD = {
    "T": (1.0, 0.0075),
    "K": (2.2, 0.0075),
    "E": (1.7, 0.0050),
    "J": (2.2, 0.0075)
}


def u_temperature(
        measured,
        units="K",
        u_thermocouple=None,
        tc_type="T",
        collapse=False
):
    """
    Calculate uncertainty in a temperature measurement

    Parameters
    ----------
    measured : float or np.ndarray
        Nominal measured temperature values
    units : str
        Pressure units of `measured` array. Defaults to K.
    u_thermocouple : float or None
        Thermocouple uncertainty. Pass None to use standard limits.
    tc_type : str
        Type of thermocouple used in measured value.
        Currently only supports T and K types.
    collapse : bool
        Whether or not to collapse array to a single value. This will return
        the scalar population uncertainty of the input array rather than an
        array of uncertainties.

    Returns
    -------
    float or np.ndarray
        Calculated uncertainty values corresponding to input measurements, in
        the same units as the input array.
    """
    if units in {"degC", "K"}:
        units_out = "delta_degC"
    elif units in {"degF", "degR"}:
        units_out = "delta_degF"
    else:
        raise ValueError("bad temperature units")

    if tc_type not in {"T", "K"}:
        return ValueError("u_temperature currently only supports T or K type")

    measured = quant(measured, units).to("degC").magnitude

    if u_thermocouple is None:
        # apply standard limits of error
        u_thermocouple = np.array([
            TEMP_STD[tc_type][0] * np.ones_like(measured),
            TEMP_STD[tc_type][1] * measured
        ]).max(axis=0)

    daq_err = DF_9211[DF_9211["type"] == tc_type]

    uncert = np.sqrt(
        np.sum(
            np.square(
                [
                    np.ones_like(measured) * u_thermocouple,
                    interp1d(
                        daq_err["temp_C"],
                        daq_err["err_C"],
                        kind="cubic"
                    )(measured)
                ]
            ),
            axis=0
        )
    )

    measured = quant(measured, "degC").to(units).magnitude
    uncert = quant(uncert, "delta_degC").to(units_out).magnitude

    measured = unp.uarray(measured, uncert)

    if collapse:
        t_95 = t.ppf(0.975, len(measured))
        return (
            un.ufloat(
                0,
                sem([m.nominal_value for m in measured]) * t_95
            ) +
            un.ufloat(
                0,
                np.max([m.std_dev for m in measured]) * t_95
            )
        ).std_dev
    else:
        return np.array([m.std_dev for m in measured])


# Pressure
# current measurements in Amps,
# ==============================================================================
PRESSURE_SOURCES = {
    "calibration": {
        # These values come from TEST Lab Rosemount calibration
        # Slope and intercept are from 95% CI on curve fit using scipy.stats
        "accuracy": 110000 * 0.00055,
        "slope": 75.21235963329673,
        "intercept": 0.6137811639346182
    },
    "daq": {
        "current_gain_pct_rdg": 0.0087,
        # convert to pressure using calibration constants given in u_pressure
        "current_offset_A": 0.0005 * 0.00022
    }
}
PRESSURE_CAL = {
    "slope": 13005886.223474432,
    "intercept": -51985.514384049835
}


def u_pressure(
        measured,
        units="Pa",
        slope=None,
        u_slope=None,
        intercept=None,
        u_intercept=None,
        u_cal_accuracy=None,
        daq_err=True,
        collapse=False
):
    """
    Calculate uncertainty in a pressure measurement

    Parameters
    ----------
    measured : float or np.ndarray
        Nominal measured pressure values
    units : str
        Pressure units of `measured` array. Defaults to Pa
    slope : float or None
        Calibration curve slope in Pa/A. Pass None to use latest value.
    u_slope : float or None
        Uncertainty in calibration slope (Pa/A). Pass None to use latest value.
    intercept : float or None
        Calibration curve intercept in Pa. Pass None to use latest value.
    u_intercept : float or None
        Uncertainty in calibration intercept (Pa). Pass None to use latest
        value.
    u_cal_accuracy : float or None
        Calibration transducer accuracy uncertainty (Pa). Pass None to use
        latest value.
    daq_err : bool
        Whether or not to apply DAQ error, mostly for answering questions
    collapse : bool
        Whether or not to collapse array to a single value. This will return
        the scalar population uncertainty of the input array rather than an
        array of uncertainties.

    Returns
    -------
    float or np.ndarray
        Calculated uncertainty values corresponding to input measurements, in
        the same units as the input array.
    """
    if slope is None:
        slope = PRESSURE_CAL["slope"]
    if u_slope is None:
        u_slope = PRESSURE_SOURCES["calibration"]["slope"]
    if intercept is None:
        intercept = PRESSURE_CAL["intercept"]
    if u_intercept is None:
        u_intercept = PRESSURE_SOURCES["calibration"]["intercept"]
    if u_cal_accuracy is None:
        u_cal_accuracy = PRESSURE_SOURCES["calibration"]["accuracy"]

    measured = quant(measured, units).to("Pa").magnitude
    measured = _u_pressure_daq_current(
        measured,
        slope,
        intercept,
        daq_err
    )
    measured = _u_pressure_fit(
        measured,
        slope,
        u_slope,
        intercept,
        u_intercept,
        u_cal_accuracy
    )
    measured = quant(measured, "Pa").to(units).magnitude
    if collapse:
        t_95 = t.ppf(0.975, len(measured))
        return (
            un.ufloat(
                0,
                sem([m.nominal_value for m in measured]) * t_95
            ) +
            un.ufloat(
                0,
                np.max([m.std_dev for m in measured]) * t_95
            )
        ).std_dev
    else:
        return np.array([m.std_dev for m in measured])


def _u_pressure_daq_current(
        measured,
        slope,
        intercept,
        daq_err
):
    """
    Calculate daq uncertainty from pressure measurements

    Parameters
    ----------
    measured : float or np.ndarray
        Float or array of a measured pressure in Pa
    slope : float
        Nominal value of calibration curve slope (Pa/A)
    intercept : float
        Nominal value of calibration curve intercept (Pa)
    daq_err : bool
        Whether or not to apply DAQ error, mostly for answering questions

    Returns
    -------
    measured : un.ufloat or np.ndarray
        Float or array of current values with added uncertainty, in A
    """
    measured = (measured - intercept) / slope

    if not daq_err:
        return measured

    else:
        u_gain = measured * PRESSURE_SOURCES["daq"]["current_gain_pct_rdg"]
        u_offset = unp.uarray(
            np.zeros_like(measured),
            np.ones_like(measured) * PRESSURE_SOURCES["daq"]["current_offset_A"]
        )
        measured = unp.uarray(measured, u_gain) + u_offset
        return measured


def _u_pressure_fit(
        measured,
        slope,
        u_slope,
        intercept,
        u_intercept,
        u_cal_accuracy
):
    """
    Calculate uncertainty of pressure measurements after current-pressure
    calibration curve has been applied

    Parameters
    ----------
    measured : float or np.ndarray
        Float or array of a measured pressure in Pa
    slope : float
        Nominal value of calibration curve slope (Pa/A)
    u_slope : float
        Uncertainty of calibration curve slope (Pa/A)
    intercept : float
        Nominal value of calibration curve intercept (Pa)
    u_intercept : float
        Uncertainty of calibration curve intercept (Pa)
    u_cal_accuracy : float
        Calibration transducer accuracy uncertainty (Pa)

    Returns
    -------
    measured : un.ufloat or np.ndarray
        Float or array of pressure values with added uncertainty, in Pa
    """
    slope = un.ufloat(slope, u_slope)
    intercept = un.ufloat(intercept, u_intercept)
    cal_transducer = un.ufloat(0, u_cal_accuracy)
    return measured * slope + intercept + cal_transducer
