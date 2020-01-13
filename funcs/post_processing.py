# stdlib imports
import os

# third party imports
import cantera as ct
import pandas as pd
import uncertainties as un
from nptdms import TdmsFile
from numpy import NaN

# local imports
from .diodes import find_diode_data, calculate_velocity
from . import diodes, schlieren, uncertainty


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
            diode_spacing=1.0668,  # meters
            sample_time=pd.Timedelta(seconds=70),
            u_pressure_reading=62.345029384733806,  # Pascals
            u_temperature_reading=1.1,  # Kelvin
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

            # TODO: calculate and store equivalence ratio
            pass

            # TODO: calculate and store CJ speed
            pass

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


def _collect_schlieren_dirs(
        base_dir,
        test_date
):
    """
    When reading in camera data from these tests, we will ignore the spatial
    directory since it contains no schlieren information. It will still be
    used, but not in this step. Directories containing a `.old` file have a
    different structure than newer directories, which must be accounted for.

    Parameters
    ----------
    base_dir : str
        Base data directory, (e.g. `/d/Data/Raw/`)
    test_date : str
        ISO 8601 formatted date of test data

    Returns
    -------
    list
        ordered list of directories containing diode output
    """
    raw_dir = os.path.join(
        base_dir,
        test_date
    )
    if not os.path.isdir(raw_dir):
        return []

    contents = os.listdir(raw_dir)

    if ".old" in contents:
        raw_dir = os.path.join(
            base_dir,
            test_date,
            "Camera"
        )
        contents = os.listdir(raw_dir)

    return sorted([
        os.path.join(raw_dir, item)
        for item in contents
        if os.path.isdir(os.path.join(raw_dir, item))
        and "shot" in item.lower()
    ])


def _get_equivalence_ratio(
        p_fuel,
        p_oxidizer,
        f_a_st
):
    """
    Simple equivalence ratio function

    Parameters
    ----------
    p_fuel : float
        Partial pressure of fuel
    p_oxidizer : float
        Partial pressure of oxidizer
    f_a_st : float
        Stoichiometric fuel/air ratio

    Returns
    -------
    float
        Mixture equivalence ratio
    """
    return p_fuel / p_oxidizer / f_a_st


class _ProcessOldData:
    @classmethod
    def _collect_test_dirs(
            cls,
            base_dir,
            test_date
    ):
        """
        The first step of reading in an old test directory is to determine
        which directories contain valid tests. Under the old DAQ system, the
        .vi would generate a new folder each time it was run. Only tests which
        successfully generate a `diodes.tdms` file can be considered completed
        tests. Some of these may still be failed detonations; this issue will
        be dealt with on joining with schlieren data, which contains
        information about whether or not a detonation attempt succeeded.

        Parameters
        ----------
        base_dir : str
            Base data directory, (e.g. `/d/Data/Raw/`)
        test_date : str
            ISO 8601 formatted date of test data

        Returns
        -------
        list
            ordered list of directories containing diode output
        """
        raw_dir = os.path.join(
            base_dir,
            test_date,
            "Sensors"
        )

        return sorted([
            root
            for root, _, files in os.walk(raw_dir, topdown=True)
            if "diodes.tdms" in files
        ])

    @classmethod
    def _get_cutoff_pressure(
            cls,
            df_tdms_pressure,
            kind="fuel",
    ):
        """
        This function accepts a dataframe imported from a `pressure.tdms` file.
        Old test data was output in amps; this was good and I should probably
        have kept it that way. Old tests also logged each fill event
        separately. Extract the desired data, build a confidence interval,
        apply the calibration, and output the resulting value including
        uncertainty.

        Parameters
        ----------
        df_tdms_pressure : pd.DataFrame
            Dataframe containing test-specific pressure trace
        kind : str
            Kind of cutoff pressure to get, e.g. fuel, oxidizer

        Returns
        -------
        un.ufloat
            Float with applied uncertainty
        """
        kind = kind.title()
        if kind not in {"Fuel", "Oxidizer", "Vacuum", "Diluent"}:
            raise ValueError("bad kind")

        # in these tests there is no vacuum logging. These are undiluted tests,
        # which means the diluent pressure is identical to the vacuum pressure.
        if kind == "Vacuum":
            kind = "Diluent"

        pressure = df_tdms_pressure[
                       "/'%s Fill'/'Manifold'" % kind
                   ].dropna() * uncertainty.PRESSURE_CAL["slope"] + \
            uncertainty.PRESSURE_CAL["intercept"]

        return un.ufloat(
            pressure.mean(),
            uncertainty.u_pressure(pressure, daq_err=False, collapse=True)
        )

    @classmethod
    def _get_partial_pressure(
            cls,
            df_tdms_pressure,
            kind="fuel"
    ):
        """
        Fill order: vacuum -> (diluent) -> oxidizer -> fuel

        Parameters
        ----------
        df_tdms_pressure : pd.DataFrame
            Dataframe containing test-specific pressure trace
        kind : str
            Kind of cutoff pressure to get, e.g. fuel, oxidizer

        Returns
        -------
        un.ufloat
            Float with applied uncertainty
        """
        p_ox = cls._get_cutoff_pressure(df_tdms_pressure, "oxidizer")
        if kind.lower() == "fuel":
            return cls._get_cutoff_pressure(df_tdms_pressure, "fuel") - p_ox
        elif kind.lower() == "oxidizer":
            return p_ox
        else:
            raise ValueError("only fuels and oxidizers in this analysis")

    @classmethod
    def _get_initial_pressure(
            cls,
            df_tdms_pressure
    ):
        """
        In old data, the initial mixture pressure is the fuel cutoff pressure

        Parameters
        ----------
        df_tdms_pressure : pd.DataFrame
            Dataframe containing test-specific pressure trace

        Returns
        -------
        un.ufloat
            Float with applied uncertainty
        """
        return cls._get_cutoff_pressure(df_tdms_pressure, kind="fuel")

    @classmethod
    def _get_initial_temperature(cls, df_tdms_temperature):
        return un.ufloat(
            df_tdms_temperature["/'Test Readings'/'Tube'"].mean(),
            uncertainty.u_temperature(
                df_tdms_temperature["/'Test Readings'/'Tube'"],
                tc_type="K",
                collapse=True
            )
        )

    @classmethod
    def __call__(
            cls,
            base_dir,
            test_date,
            f_a_st=0.04201680672268907
    ):
        """

        Parameters
        ----------
        base_dir : str
            Base data directory, (e.g. `/d/Data/Raw/`)
        test_date : str
            ISO 8601 formatted date of test data
        f_a_st : float
            Stoichiometric fuel/air ratio for the test mixture

        Returns
        -------
        List[pd.DataFrame, dict]
            A list in which the first item is a dataframe of the processed
            tube data and the second is a dictionary containing
            background-subtracted schlieren images
        """
        sensor_dirs = cls._collect_test_dirs(base_dir, test_date)
        schlieren_dirs = _collect_schlieren_dirs(base_dir, test_date)
        df = pd.DataFrame(
            columns=["date", "shot", "sensors", "schlieren"],
        )
        df["sensors"] = sensor_dirs
        df["schlieren"] = schlieren_dirs
        df = df[df["schlieren"].apply(lambda x: "failed" not in x)]
        df["date"] = test_date
        df["shot"] = [
            int(os.path.split(d)[1].lower().replace("shot", "").strip())
            for d in schlieren_dirs
        ]

        images = dict()

        for idx, row in df.iterrows():
            # background subtraction
            images[
                "{:s}_shot{:02d}".format(
                    row["date"],
                    row["shot"]
                )
            ] = schlieren.bg_subtract_all_frames(row["schlieren"])

            # gather pressure data
            df_tdms_pressure = TdmsFile(
                os.path.join(
                    row["sensors"],
                    "pressure.tdms"
                )
            ).as_dataframe()
            p_init = cls._get_initial_pressure(df_tdms_pressure)
            p_fuel = cls._get_partial_pressure(df_tdms_pressure, kind="fuel")
            p_oxidizer = cls._get_partial_pressure(df_tdms_pressure, kind="oxidizer")
            phi = _get_equivalence_ratio(p_fuel, p_oxidizer, f_a_st)

            # gather temperature data
            df_tdms_temperature = TdmsFile(
                os.path.join(
                    row["sensors"],
                    "temperature.tdms"
                )
            ).as_dataframe()
            t_init = cls._get_initial_temperature(df_tdms_temperature)

            # wave speed measurement
            diode_loc = os.path.join(row["sensors"], "diodes.tdms")
            wave_speed = diodes.calculate_velocity(diode_loc)[0]

            # output results
            df.at[idx, "phi"] = phi.nominal_value
            df.at[idx, "u_phi"] = phi.std_dev
            df.at[idx, "p_0"] = p_init.nominal_value
            df.at[idx, "u_p_0"] = p_init.std_dev
            df.at[idx, "t_0"] = t_init.nominal_value
            df.at[idx, "u_t_0"] = t_init.std_dev
            df.at[idx, "p_fuel"] = p_fuel.nominal_value
            df.at[idx, "u_p_fuel"] = p_fuel.std_dev
            df.at[idx, "p_oxidizer"] = p_oxidizer.nominal_value
            df.at[idx, "u_p_oxidizer"] = p_oxidizer.std_dev
            df.at[idx, "wave_speed"] = wave_speed.nominal_value
            df.at[idx, "u_wave_speed"] = wave_speed.std_dev

            # TODO: rework for multiprocessing

        return df, images


process_old_data = _ProcessOldData()
