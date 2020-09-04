################################################################################
#       Measure, calibrate, and store all schlieren shots on a given day       #
################################################################################

from funcs import schlieren
import pandas as pd
import numpy as np
import uncertainties as un


def get_frame_info(frame_data_loc):
    _, _, this_date, this_shot, this_frame = frame_data_loc.split("/")
    this_date = this_date.replace("d", "").replace("_", "-")
    this_shot = int(this_shot[-2:])
    this_frame = int(this_frame[-2:])
    return this_date, this_shot, this_frame


def collect_single_replicate(data_store, dates):
    if isinstance(dates, str):
        dates = [dates]

    frame_columns = [
        "date",
        "shot",
        "frame",
        "loc_px",
        "u_loc_px",
        "delta_px",
        "u_delta_px",
        "spatial_near",
        "u_spatial_near",
        "spatial_near_estimated",
        "spatial_far",
        "u_spatial_far",
        "spatial_far_estimated",
        "spatial_centerline",
        "u_spatial_centerline",
    ]
    df_meas = pd.DataFrame(columns=frame_columns)

    for current in np.random.permutation(data_store.keys()[1:]):
        this_date, this_shot, this_frame = get_frame_info(current)
        if this_date in dates:
            image = data_store.get(current).values
            locs = schlieren.measure_single_frame(image)
            if len(locs) > 0:
                deltas = np.concatenate(
                    ([un.ufloat(np.NaN, np.NaN)], np.diff(locs)))
                df_current = pd.DataFrame(columns=frame_columns)
                df_current["loc_px"] = [ll.nominal_value for ll in locs]
                df_current["u_loc_px"] = [ll.std_dev for ll in locs]
                df_current["delta_px"] = [d.nominal_value for d in deltas]
                df_current["u_delta_px"] = [d.std_dev for d in deltas]
                df_current["date"] = this_date
                df_current["shot"] = this_shot
                df_current["frame"] = this_frame
                df_meas = pd.concat((df_meas, df_current), ignore_index=True)

                df_meas[["shot", "frame"]] = df_meas[["shot", "frame"]].astype(
                    int)

                # ensure good dtypes
                df_meas["date"] = df_meas["date"].astype(str)
                df_meas[[
                    "loc_px",
                    "u_loc_px",
                    "delta_px",
                    "u_delta_px",
                    "spatial_near",
                    "u_spatial_near",
                    "spatial_far",
                    "u_spatial_far",
                    "spatial_centerline",
                    "u_spatial_centerline",
                ]] = df_meas[[
                    "loc_px",
                    "u_loc_px",
                    "delta_px",
                    "u_delta_px",
                    "spatial_near",
                    "u_spatial_near",
                    "spatial_far",
                    "u_spatial_far",
                    "spatial_centerline",
                    "u_spatial_centerline",
                ]].astype(float)
                df_meas[[
                    "spatial_near_estimated",
                    "spatial_far_estimated"
                ]] = df_meas[[
                    "spatial_near_estimated",
                    "spatial_far_estimated"
                ]].astype(bool)

    return df_meas


def process_all_schlieren(
        loc_processed,
        loc_schlieren,
        dates
):
    with pd.HDFStore(loc_processed, "r") as store:
        df_initial_meas = collect_single_replicate(store, dates)

    with pd.HDFStore(loc_schlieren, "a") as f:
        if hasattr(f, "data"):
            n_meas_original = len(f.data)
        else:
            n_meas_original = 0
        print("start:  %d records" % n_meas_original)
        f.put("data", df_initial_meas, format="table", append="true")
        n_meas_final = len(f.data)
        n_meas_added = n_meas_final - n_meas_original
        print("finish: {:d} records ({:d} added)".format(
            n_meas_final,
            n_meas_added
        ))

    if n_meas_added > 0:
        with pd.HDFStore(loc_schlieren, "r") as store:
            schlieren_data = store["data"]

        for date in dates:
            spatial = dict()
            for which in ["near", "far"]:
                loc_spatial = schlieren.get_spatial_loc(date, which)
                # this is !d because .where will change all locations where
                # _mask  is False to have the assigned value, which in this
                # case is our spatial factor
                _mask = schlieren_data["date"] != date
                day_spatial = schlieren.collect_spatial_calibration(loc_spatial)
                spatial[which] = day_spatial
                schlieren_data[f"spatial_{which}"].where(
                    _mask,
                    day_spatial.nominal_value,
                    inplace=True
                )
                # breaks due to maybe_callable if not explicitly cast to float.
                # weird.
                schlieren_data[f"u_spatial_{which}"].where(
                    _mask,
                    float(day_spatial.std_dev),
                    inplace=True
                )
                schlieren_data[f"spatial_{which}_estimated"].where(
                    _mask,
                    False,
                    inplace=True
                )

            spatial = (spatial["near"] + spatial["far"]) / 2
            schlieren_data["spatial_centerline"].where(
                _mask,
                float(spatial.nominal_value),
                inplace=True
            )
            schlieren_data["u_spatial_centerline"].where(
                _mask,
                float(spatial.std_dev),
                inplace=True
            )

        with pd.HDFStore(loc_schlieren, "w") as store:
            store["data"] = schlieren_data


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise ValueError(
            "must include processed_tube_data_h5, schlieren_data_h5, "
            "and *dates_to_process"
        )
    else:
        processed_tube_data_h5 = sys.argv[1]
        schlieren_data_h5 = sys.argv[2]
        dates_to_process = sys.argv[3:]
    process_all_schlieren(
        processed_tube_data_h5,
        schlieren_data_h5,
        dates_to_process
    )
