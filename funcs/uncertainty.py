import os
import pint
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import uncertainties as un
from uncertainties import unumpy as unp

ureg = pint.UnitRegistry()
quant = ureg.Quantity

# Temperature -- T type thermocouple in manifold is used for temp calculations.
# All values here are in degrees Celsius
# ==============================================================================
# standard: https://www.omega.com/en-us/resources/thermocouple-types
# daq: NI 9211 operating instructions and specifications p 26
#      using Typ (Autozero on)
df_9211 = pd.read_csv(
    os.path.join("data", "tc_err.txt"),
    skiprows=1,
    names=["measured", "error"]
).sort_values("measured")
temperature_sources = {
    "standard": 1.0,
    "daq": interp1d(
        df_9211["measured"],
        df_9211["error"],
        kind="cubic"
    )
}


def u_temperature(
        measured,
        units="K",
        u_thermocouple=temperature_sources["standard"],
        tc_type="T",
):
    """
    Calculate uncertainty in a temperature measurement

    Parameters
    ----------
    measured : float or np.ndarray
        Nominal measured temperature values
    units : str
        Pressure units of `measured` array. Defaults to K.
    u_thermocouple : float
        Thermocouple uncertainty. Defaults to standard limits.
    tc_type : str
        Type of thermocouple used in measured value.
        Currently only supports T type.

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

    if tc_type != "T":
        return ValueError("u_temperature currently only supports T type")

    measured = quant(measured, units).to("degC").magnitude

    uncert = np.sqrt(
        np.sum(
            np.square(
                [
                    np.ones_like(measured) * u_thermocouple,
                    # this will have to be changed to support different TC types
                    temperature_sources["daq"](measured)
                ]
            ),
            axis=0
        )
    )
    return quant(uncert, "delta_degC").to(units_out).magnitude


# Pressure
# current measurements in Amps,
# ==============================================================================
pressure_sources = {
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
pressure_cal = {
    "slope": 13005886.223474432,
    "intercept": -51985.514384049835
}


def u_pressure(
        measured,
        units="Pa",
        slope=pressure_cal["slope"],
        u_slope=pressure_sources["calibration"]["slope"],
        intercept=pressure_cal["intercept"],
        u_intercept=pressure_sources["calibration"]["intercept"],
        u_cal_accuracy=pressure_sources["calibration"]["accuracy"],
        daq_err=True
):
    """
    Calculate uncertainty in a pressure measurement

    Parameters
    ----------
    measured : float or np.ndarray
        Nominal measured pressure values
    units : str
        Pressure units of `measured` array. Defaults to Pa
    slope : float
        Calibration curve slope in Pa/A
    u_slope : float
        Uncertainty in calibration slope (Pa/A)
    intercept : float
        Calibration curve intercept in Pa
    u_intercept
        Uncertainty in calibration intercept (Pa)
    u_cal_accuracy : float
        Calibration transducer accuracy uncertainty (Pa)
    daq_err : bool
        Whether or not to apply DAQ error, mostly for answering questions

    Returns
    -------
    float or np.ndarray
        Calculated uncertainty values corresponding to input measurements, in
        the same units as the input array.
    """
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
        u_gain = measured * pressure_sources["daq"]["current_gain_pct_rdg"]
        u_offset = unp.uarray(
            np.zeros_like(measured),
            np.ones_like(measured) * pressure_sources["daq"]["current_offset_A"]
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
