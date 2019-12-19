# stdlib imports
import os

# third party imports
import pandas as pd
from nptdms import TdmsFile
from numpy import NaN

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
            csv_conditions="test_conditions.csv",
            output_file="tube data.h5",
            diode_spacing=1.0668,
            sample_time=pd.Timedelta(seconds=70),
    ):
        self._dir_raw = os.path.join(data_base_dir, dir_raw, date)
        self._dir_processed = os.path.join(data_base_dir, dir_processed, date)
        self._loc_state = os.path.join(self._dir_raw, tdms_state)
        self._loc_sensor = os.path.join(self._dir_raw, tdms_sensor)
        self._loc_conditions = os.path.join(self._dir_raw, csv_conditions)
        self._output_file = os.path.join(self._dir_processed, output_file)
        self.df_test_info = self._load_nominal_conditions()
        self._diode_spacing = diode_spacing
        self._sample_time = sample_time
        self.tests = []

        if not os.path.exists(self._loc_state):
            raise FileNotFoundError(self._loc_state)
        elif not os.path.exists(self._loc_sensor):
            raise FileNotFoundError(self._loc_sensor)
        elif not os.path.exists(self._loc_conditions):
            raise FileNotFoundError(self._loc_conditions)

    def _load_nominal_conditions(self):
        df_conditions = pd.read_csv(self._loc_conditions)
        df_conditions["datetime"] = pd.to_datetime(
            df_conditions["datetime"],
            utc=True
        )
        old_cols = ["diluent_mol_frac", "p_dil", "p_ox", "p_f"]
        new_cols = ["diluent_mol_frac_nominal", "cutoff_diluent",
                    "cutoff_oxidizer", "cutoff_fuel"]
        df_conditions.rename(
            columns={o: n for o, n in zip(old_cols, new_cols)},
            inplace=True
        )
        df_conditions["start"] = pd.NaT
        df_conditions["end"] = pd.NaT
        return df_conditions

    def process_data(self):
        # TODO: add uncertainty calculation
        # collect necessary data before analyzing each test
        df_sensor = TdmsFile(self._loc_sensor).as_dataframe()
        df_pressure = self._extract_sensor_data(
            df_sensor,
            which="pressure"
        )
        df_temperature = self._extract_sensor_data(
            df_sensor,
            which="temperature"
        )
        del df_sensor
        df_state = TdmsFile(self._loc_state).as_dataframe()
        df_state.columns = ["time", "state", "mode"]
        df_test_times = self._find_test_times(df_state)
        diode_files = find_diode_data(self._dir_raw)

        for i, row in df_test_times.iterrows():
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

            # Determine cutoff times for each state as follows:
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
            df_state_row["start"] = df_state_row["end"] - self._sample_time

            # Get actual fill and partial pressures
            fill_pressures = dict()
            for _, state_row in df_state_row.iterrows():
                fill_pressures[state_row["state"]] = df_p_row[
                    self._mask_df_by_row_time(df_p_row, state_row)
                ].mean()[0]
            partial_pressures = dict(
                fuel=fill_pressures["fuel"] - fill_pressures["vacuum"],
                diluent=fill_pressures["diluent"] - fill_pressures["fuel"],
            )
            partial_pressures["oxidizer"] = fill_pressures["oxidizer"] \
                - partial_pressures["fuel"] - partial_pressures["diluent"]

            df_conditions_actual = pd.concat([
                pd.DataFrame(
                    {"partial_" + k: v for k, v in partial_pressures.items()},
                    index=[i]),
                pd.DataFrame(
                    {"actual_" + k: v for k, v in fill_pressures.items()},
                    index=[i])
            ], axis=1)
            if not set(self.df_test_info.columns).intersection(
                    set(df_conditions_actual.columns)):
                for col in df_conditions_actual.columns:
                    self.df_test_info[col] = NaN

            # There may be more test state changes than actual tests.
            # Determine which row of the dataframe best matches the current
            # test and insert data accordingly
            best_row = pd.Series(
                self.df_test_info["datetime"] < row["start"]
            ).cumsum().idxmax()
            self.df_test_info.loc[
                best_row,
                df_conditions_actual.columns
            ] = df_conditions_actual.values[0]
            # I am removing this type inspection for this line because PyCharm
            # seems to think that diode_files is a base iterable rather than
            # a list, even though it is actually a list (per diodes.py).
            # noinspection PyUnresolvedReferences
            self.df_test_info.at[
                best_row,
                "wave_speed"
            ] = calculate_velocity(
                diode_files[i],
                diode_spacing=self._diode_spacing
            )[0]
            self.df_test_info.loc[
                best_row,
                ["start", "end"]
            ] = row[["start", "end"]]

            # in the case of undiluted mixtures there wil be a very slight
            # difference in fuel and diluent pressure calculations caused by
            # the state detection method; fix it. Fuel will be the correct
            # value.
            if self.df_test_info.loc[
                best_row,
                "diluent_mol_frac_nominal"
            ] == 0:
                self.df_test_info.loc[
                    best_row,
                    "actual_diluent"
                ] = self.df_test_info.loc[best_row, "actual_fuel"]
                self.df_test_info.loc[best_row, "partial_diluent"] = 0.

            # store auxiliary test data
            self.tests[i]["pressure"] = df_p_row
            self.tests[i]["temperature"] = df_t_row
            self.tests[i]["state"] = df_state_row

        # remove unwanted information from test info dataframe
        self.df_test_info.dropna(subset=["partial_fuel"], inplace=True)
        self.df_test_info.reset_index(drop=True, inplace=True)
        self.df_test_info.drop(columns="datetime", inplace=True)

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

    def save_processed_data(self):
        # TODO: output uncertainty calculation
        if len(self.tests) == 0:
            raise AttributeError("No tests have been processed")

        if not os.path.exists(self._dir_processed):
            os.makedirs(self._dir_processed)

        with pd.HDFStore(self._output_file, "w") as store:
            if self.df_test_info is not None:
                store.put("test_info", self.df_test_info)
            for i, t in enumerate(self.tests):
                test_name = "test_%d" % i
                for k in t.keys():
                    out_key = "/".join(["", test_name, k])
                    store.put(out_key, t[k])
