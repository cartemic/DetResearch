import pandas as pd

columns = [
    "date",
    "shot",
    "fuel",
    "oxidizer",
    "diluent",
    "t_0",
    "u_t_0",
    "p_0_nom",
    "p_0",
    "u_p_0",
    "phi_nom",
    "phi",
    "u_phi",
    "dil_mf_nom",
    "dil_mf",
    "u_dil_mf",
    "wave_speed",
    "u_wave_speed",
]

propane_data_files = ("data_fffff", "data_ggggg", "data_hhhhh", "processed_data")
propane_data = pd.DataFrame()
for data_file in propane_data_files:
    with pd.HDFStore(f"/d/Data/Processed/Data/{data_file}.h5", "r") as store:
        propane_data = pd.concat(
            (
                propane_data,
                store.data[columns],
            )
        )

with pd.HDFStore("/home/mick/DetResearch/scripts/simulation_measurement_comparison/measurements.h5", "r") as store:
    methane_data = store.data[columns]

all_data = pd.concat((methane_data, propane_data))

# filter out unsuccessful detonations
min_successful_velocity = 1000  # m/s
all_data = all_data[all_data["wave_speed"] > min_successful_velocity].reset_index(drop=True)

with pd.HDFStore("/d/Data/Processed/Data/all_tube_data.h5", "w") as store:
    store["data"] = all_data
