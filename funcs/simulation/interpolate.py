from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def interpolate(
    data: pd.DataFrame,
    time_column: str,
    interp_columns: list[str],
    saturate: bool,
    co2_label: str = "CO2",
    n2_label: str = "N2",
) -> pd.DataFrame:
    """
    Interpolate N2-diluted values into CO2-diluted time steps to allow for further analysis.

    Parameters
    ----------
    data: Dataframe with columns ["diluent", "dil_condition", "phi_nom", `time_column`, **`interp_columns`],
        where "diluent" is ``"CO2"`` or ``"N2"``.
    time_column: Name of column containing time data
    interp_columns: Columns whose values need to be interpolated
    co2_label: Label for CO2 diluted data, probably ``"CO2"`` or ``"CO2i"``
    n2_label: Label for N2 diluted data, probably ``"N2"`` or ``"N2i"``
    saturate: Whether to use first and last column values in the case of out-of-range interpolation times.

    Returns
    -------
    Dataframe containing unchanged CO2 data, along with N2 data interpolated to CO2 data timestamps.
    """
    # Times need to be sorted for interpolation to work correctly
    interp_data = data.sort_values(time_column, ascending=True)
    out = pd.DataFrame(columns=interp_data.columns)
    for _, group in interp_data.groupby(["dil_condition", "phi_nom"]):
        n2 = group[group["diluent"] == n2_label]
        co2 = group[group["diluent"] == co2_label]
        n2_new = co2.copy()
        n2_new.loc[:, "diluent"] = n2_label

        # Extend interp range to accommodate induction time shift
        n2_interp_times = [-np.inf, *n2[time_column].to_numpy(), np.inf]
        for column in interp_columns:
            n2_low = n2[column].iloc[0] if saturate else np.nan
            n2_high = n2[column].iloc[-1] if saturate else np.nan
            fit = interp1d(
                n2_interp_times,
                [n2_low, *n2[column].to_numpy(), n2_high],
            )
            n2_new.loc[:, column] = fit(n2_new[time_column].to_numpy())
        out = pd.concat((out, co2, n2_new))
    return out
