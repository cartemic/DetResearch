# stdlib imports
import os

# third party imports
import pandas as pd
from nptdms import TdmsFile

# local imports
from .diodes import find_diode_data, calculate_velocity


class PostProcessDate:
    def __init__(
        self,
        date,
        data_base_dir=os.path.join("D:\\", "Data"),
        dir_raw="Raw",
        dir_processed=os.path.join("Processed", "Data"),
        tdms_state="tube state.tdms",
        tdms_sensor="sensor log.tdms",
        output_file="tube data.h5",
    ):
        self.dir_raw = os.path.join(data_base_dir, dir_raw, date)
        self.dir_processed = os.path.join(data_base_dir, dir_processed, date)
        self.loc_state = os.path.join(self.dir_raw, tdms_state)
        self.loc_sensor = os.path.join(self.dir_raw, tdms_sensor)
        self.output_file = os.path.join(self.dir_processed, output_file)
        self.df_test_info = pd.DataFrame()
        self.tests = []

        if not os.path.exists(self.dir_processed):
            os.makedirs(self.dir_processed)

        if not os.path.exists(self.loc_state):
            raise FileNotFoundError(self.loc_state)
        elif not os.path.exists(self.loc_sensor):
            raise FileNotFoundError(self.loc_sensor)

    def process_tube_data(self, sample_time=pd.Timedelta(seconds=70)):
        df_sensor = TdmsFile(self.loc_sensor).as_dataframe()
        df_pressure = self._extract_sensor_data(
            df_sensor,
            which="pressure"
        )
        df_temperature = self._extract_sensor_data(
            df_sensor,
            which="temperature"
        )
        del df_sensor

        df_state = TdmsFile(self.loc_state).as_dataframe()
        df_state.columns = ["time", "state", "mode"]

        self.df_test_times = self._find_test_times(df_state)

        for i, row in self.df_test_times.iterrows():
            if len(self.tests) == i:
                self.tests.append(
                    dict(
                        pressure=None,
                        temperature=None,
                        state=None
                    )
                )

            df_p_row = df_pressure[
                self._mask_df_by_row_time(df_pressure, row)
            ].reset_index(drop=True)
            df_t_row = df_temperature[
                self._mask_df_by_row_time(df_temperature, row)
            ].reset_index(drop=True)

            # cutoff time determination:
            #     vacuum: start of fuel fill
            #     fuel: start of diluent fil
            #     diluent: start of oxidizer fill
            #     oxidizer: start of reactant mixing
            state_mask = self._mask_df_by_row_time(df_state, row, False)
            df_state_row = pd.DataFrame(data={
                "state": [
                    "vacuum",
                    "fuel",
                    "diluent",
                    "oxidizer"
                ],
                "end": [
                    df_state[
                        state_mask &
                        (df_state["state"] == "Fuel Fill")
                        ]["time"].min(),
                    df_state[
                        state_mask &
                        (df_state["state"] == "Diluent Fill")
                        ]["time"].min(),
                    df_state[
                        state_mask &
                        (df_state["state"] == "Oxidizer Fill")
                        ]["time"].min(),
                    df_state[
                        state_mask &
                        (df_state["state"] == "Mixing")
                        ]["time"].min(),
                ]
            })
            df_state_row["start"] = df_state_row["end"] - sample_time

            fill_pressures = dict()
            for _, state_row in df_state_row.iterrows():
                fill_pressures[state_row["state"]] = df_p_row[
                    self._mask_df_by_row_time(df_p_row, state_row)
                ].mean()[0]

            partial_pressures = dict(
                fuel=fill_pressures["fuel"] - fill_pressures["vacuum"],
                diluent=fill_pressures["diluent"] - fill_pressures["fuel"],
            )
            partial_pressures["oxidizer"] = fill_pressures["oxidizer"] - \
                partial_pressures["fuel"] - partial_pressures["diluent"]

            # TODO: extract nominal test conditions
            self.tests[i]["pressure"] = df_p_row
            self.tests[i]["temperature"] = df_t_row
            self.tests[i]["state"] = df_state_row
            self.tests[i]["fill_pressures"] = fill_pressures
            self.tests[i]["partial_pressures"] = partial_pressures

    @staticmethod
    def _extract_sensor_data(df_sensor, which="pressure", dropna=True):
        if not {"temperature", "pressure"}.intersection({which}):
            raise ValueError("which must be temperature or pressure")

        df_sens_out = df_sensor[[
            "/'%s'/'time'" % which,
            "/'%s'/'manifold'" % which
        ]].dropna()
        df_sens_out.columns = ["time", which]
        if dropna:
            df_sens_out.dropna(inplace=True)
        return df_sens_out

    @staticmethod
    def _find_test_times(df_state):
        # start times
        # The tube will only automatically go `Closed -> Vent` at the end of
        # a completed test cycle
        df_test_times = pd.DataFrame(columns=["start", "end"])
        df_test_times["end"] = df_state[
            (df_state["state"].shift(1) == "Tube Closed") &
            (df_state["state"] == "Tube Vent") &
            (df_state["mode"] == "auto")
            ]["time"]
        df_test_times.reset_index(drop=True, inplace=True)

        # end times
        # A test will be considered to have started at the last automatic mix
        # section purge preceding its end time.
        for i, t in enumerate(df_test_times["end"].values):
            df_test_times.at[i, "start"] = df_state[
                (df_state["time"].values < t) &
                (df_state["state"] == "Mix Section Purge") &
                (df_state["mode"] == "auto")
                ].iloc[-1].time

        return df_test_times

    @staticmethod
    def _mask_df_by_row_time(df_in, test_row, include_ends=True):
        if include_ends:
            return (
                    (df_in["time"] >= test_row["start"]) &
                    (df_in["time"] <= test_row["end"])
            )
        else:
            return (
                    (df_in["time"] > test_row["start"]) &
                    (df_in["time"] < test_row["end"])
            )

    def process_velocity_data(self, diode_spacing=1.0668):
        # TODO: add multiprocess support
        diode_files = find_diode_data(self.dir_raw)
        for i, tdms in enumerate(diode_files):
            if len(self.tests) == i:
                self.tests.append(dict(wave_speed=None))
            self.tests[i]["wave_speed"] = calculate_velocity(
                tdms,
                diode_spacing=diode_spacing
            )[0]

    def save_processed_data(self):
        if len(self.tests) == 0:
            raise AttributeError("No tests have been processed")

        with pd.HDFStore(self.output_file, "w") as store:
            for i, t in enumerate(self.tests):
                test_name = "test_%d" % i
                for k in t.keys():
                    out_key = "/".join(["", test_name, k])
                    store.put(out_key, t[k])
