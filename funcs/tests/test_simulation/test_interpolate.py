import numpy as np
import pandas as pd

from funcs.simulation.interpolate import interpolate


def test_interpolate_times_overlap() -> None:
    time_column = "t"
    interp_column = "a"
    co2_label = "CO2"
    n2_label = "N2"
    # Give n2 an extra step -- return should have the co2 same time steps only, so n2 actual shape shouldn't matter
    data = pd.DataFrame({
        "diluent": [co2_label, co2_label, n2_label, n2_label, n2_label],
        "dil_condition": ["low", "low", "low", "low", "low"],
        "phi_nom": [1, 1, 1, 1, 1],
        time_column: [1.75, 1.25, 1, 2, 7],
        interp_column: [0, 1, 2, 3, 4],
    })
    expected = pd.DataFrame({
        "diluent": [co2_label, co2_label, n2_label, n2_label],
        "dil_condition": ["low", "low", "low", "low"],
        "phi_nom": [1, 1, 1, 1],
        # make sure order is correct!
        time_column: [1.25, 1.75, 1.75, 1.25],
        interp_column: [1, 0, 2.75, 2.25],
    }).set_index(["diluent", "dil_condition", "phi_nom", time_column])  # actual index doesn't matter
    result = interpolate(
        data=data,
        time_column=time_column,
        interp_columns=[interp_column],
        saturate=True,
        co2_label=co2_label,
        n2_label=n2_label,
    ).set_index(["diluent", "dil_condition", "phi_nom", time_column])

    pd.testing.assert_frame_equal(result, expected, check_like=True, check_dtype=False)


def test_interpolate_no_overlap() -> None:
    time_column = "t"
    interp_column = "a"
    co2_label = "CO2"
    n2_label = "N2"
    data = pd.DataFrame({
        "diluent": [co2_label, co2_label, n2_label, n2_label],
        "dil_condition": ["low", "low", "low", "low"],
        "phi_nom": [1, 1, 1, 1],
        time_column: [1.75, 1.25, 7, 9],
        interp_column: [0, 1, 2, 3],
    })
    saturated_expected = pd.DataFrame({
        "diluent": [co2_label, co2_label, n2_label, n2_label],
        "dil_condition": ["low", "low", "low", "low"],
        "phi_nom": [1, 1, 1, 1],
        time_column: [1.25, 1.75, 1.75, 1.25],
        interp_column: [1.0, 0.0, 2.0, 2.0],
    }).set_index(["diluent", "dil_condition", "phi_nom", time_column])
    unsaturated_expected = pd.DataFrame({
        "diluent": [co2_label, co2_label, n2_label, n2_label],
        "dil_condition": ["low", "low", "low", "low"],
        "phi_nom": [1, 1, 1, 1],
        time_column: [1.25, 1.75, 1.75, 1.25],
        interp_column: [1.0, 0.0, np.nan, np.nan],
    }).set_index(["diluent", "dil_condition", "phi_nom", time_column])
    saturated_result = interpolate(
        data=data,
        time_column=time_column,
        interp_columns=[interp_column],
        saturate=True,
        co2_label=co2_label,
        n2_label=n2_label,
    ).set_index(["diluent", "dil_condition", "phi_nom", time_column])
    unsaturated_result = interpolate(
        data=data,
        time_column=time_column,
        interp_columns=[interp_column],
        saturate=False,
        co2_label=co2_label,
        n2_label=n2_label,
    ).set_index(["diluent", "dil_condition", "phi_nom", time_column])

    pd.testing.assert_frame_equal(unsaturated_result, unsaturated_expected, check_like=True, check_dtype=False)
    pd.testing.assert_frame_equal(saturated_result, saturated_expected, check_like=True, check_dtype=False)
