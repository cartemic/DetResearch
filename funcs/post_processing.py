# stdlib imports
import os

# third party imports
import cantera as ct
import multiprocessing as mp
import numpy as np
import pandas as pd
import uncertainties as un
from nptdms import TdmsFile
from numpy import NaN
from uncertainties import unumpy as unp

# local imports
from .diodes import find_diode_data, calculate_velocity
from . import diodes, schlieren, uncertainty


def _get_f_a_st(
        fuel="C3H8",
        oxidizer="O2:1 N2:3.76",
        mech="gri30.cti"
):
    """
    Calculate the stoichiometric fuel/air ratio using Cantera. Calculates using
    only x_fuel to allow for compound oxidizer (e.g. air)

    Parameters
    ----------
    fuel : str
    oxidizer : str
    mech : str
        mechanism file to use

    Returns
    -------
    float
        stoichiometric fuel/air ratio
    """
    gas = ct.Solution(mech)
    gas.set_equivalence_ratio(
        1,
        fuel,
        oxidizer
    )
    x_fuel = gas.mole_fraction_dict()[fuel]
    return x_fuel / (1 - x_fuel)


class _ProcessNewData:
    @staticmethod
    def _load_nominal_conditions(loc_conditions):
        """
        Loads a dataframe of nominal test conditions

        Parameters
        ----------
        loc_conditions : str
            Location of nominal test condition csv

        Returns
        -------
        pd.DataFrame
            Dataframe of nominal test conditions with NaT values for start and
            end times. These will be populated later, and then used to filter
            out incomplete tests.
        """
        df_conditions = pd.read_csv(loc_conditions)
        df_conditions["datetime"] = pd.to_datetime(
            df_conditions["datetime"],
            utc=False
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

    @classmethod
    def _get_fill_pressures(
            cls,
            df_p_row,
            df_state_row
    ):
        """
        Gather fill pressures from a single-test portion of the pressure
        section of a test sensor output dataframe.

        Parameters
        ----------
        df_p_row : pd.DataFrame
            Pressure dataframe for a single test
        df_state_row : pd.DataFrame
            Dataframe containing tube states and corresponding start/end times,
            which are used to slice the relevant portions of df_p_row in order
            to determine fill cutoff pressures.

        Returns
        -------
        dict
            Dictionary containing fuel, oxidizer, diluent, and vacuum cutoff
            pressures
        """
        fill_pressures = dict()
        for _, state_row in df_state_row.iterrows():
            nominal = df_p_row[
                cls._mask_df_by_row_time(df_p_row, state_row)
            ]["pressure"].values
            fill_pressures[state_row["state"]] = unp.uarray(
                nominal,
                uncertainty.u_pressure(nominal, daq_err=False)
            ).mean()
        return fill_pressures

    @classmethod
    def __call__(
            cls,
            base_dir,
            test_date,
            diode_spacing=1.0668,
            sample_time=pd.Timedelta(seconds=70),
            mech="gri30.cti"
    ):
        """
        Analyze test data from a new data set.
        TODO: add schlieren stuffs
        TODO: parallelize

        Parameters
        ----------
        base_dir : str
            Base data directory, (e.g. `/d/Data/Raw/`)
        test_date : str
            ISO 8601 formatted date of test data
        diode_spacing : float
            Spacing of photo diodes, in meters. Defaults to latest (42 in)
        sample_time : pd.Timedelta
            Wait period between tube fill steps during which pressure should
            be relatively constant unless there is a bad leak
        mech : str
            Mechanism file to use for calculating stoichiometric f/a ratio

        Returns
        -------
        tuple
            df_test_info, tests
        """
        dir_data = os.path.join(base_dir, test_date)
        loc_state = os.path.join(dir_data, "tube state.tdms")
        loc_sensor = os.path.join(dir_data, "sensor log.tdms")
        loc_conditions = os.path.join(dir_data, "test_conditions.csv")

        # collect necessary data before analyzing each test
        df_sensor = TdmsFile(loc_sensor).as_dataframe()
        df_pressure = cls._extract_sensor_data(
            df_sensor,
            which="pressure"
        )
        df_temperature = cls._extract_sensor_data(
            df_sensor,
            which="temperature"
        )
        del df_sensor
        df_state = TdmsFile(loc_state).as_dataframe()
        df_state.columns = ["time", "state", "mode"]
        df_test_times = cls._find_test_times(df_state)
        df_test_info = cls._load_nominal_conditions(loc_conditions)
        diode_files = find_diode_data(dir_data)
        tests = [dict() for _ in range(len(df_test_info))]

        for idx, row in df_test_times.iterrows():
            # collect pressure and temperature data from the current test
            df_p_row = df_pressure[
                cls._mask_df_by_row_time(df_pressure, row)
            ].reset_index(drop=True)
            df_t_row = df_temperature[
                cls._mask_df_by_row_time(df_temperature, row)
            ].reset_index(drop=True)

            # Determine cutoff times for each state as follows:
            #     vacuum: start of fuel fill
            #     fuel: start of diluent fil
            #     diluent: start of oxidizer fill
            #     oxidizer: start of reactant mixing
            state_mask = cls._mask_df_by_row_time(df_state, row, False).values
            df_state_row = pd.DataFrame(
                data={
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
                }
            )
            df_state_row["start"] = df_state_row["end"] - sample_time

            # Get actual fill and partial pressures
            fill_pressures = cls._get_fill_pressures(df_p_row, df_state_row)
            partial_pressures = dict(
                fuel=fill_pressures["fuel"] - fill_pressures["vacuum"],
                diluent=fill_pressures["diluent"] - fill_pressures["fuel"],
            )
            partial_pressures["oxidizer"] = fill_pressures["oxidizer"] \
                - partial_pressures["fuel"] - partial_pressures["diluent"]

            p_init = fill_pressures["oxidizer"].nominal_value
            u_p_init = fill_pressures["oxidizer"].std_dev

            # There may be more test state changes than actual tests.
            # Determine which row of the dataframe best matches the current
            # test and insert data accordingly
            best_row = pd.Series(
                df_test_info["datetime"] < row["start"]
            ).cumsum().idxmax()

            # I am removing this type inspection for this line because PyCharm
            # seems to think that diode_files is a base iterable rather than
            # a list, even though it is actually a list (per diodes.py).
            # noinspection PyUnresolvedReferences
            wave_speed = calculate_velocity(
                diode_files[idx],
                diode_spacing=diode_spacing
            )[0]
            df_test_info.loc[
                best_row,
                ["start", "end"]
            ] = row[["start", "end"]]

            # in the case of undiluted mixtures there wil be a very slight
            # difference in fuel and diluent pressure calculations caused by
            # the state detection method; fix it. Fuel will be the correct
            # value.
            if df_test_info.loc[
                best_row,
                "diluent_mol_frac_nominal"
            ] == 0:
                partial_pressures["diluent"] = un.ufloat(0., 0.)

            # collect test-averaged temperature
            test_temp = df_t_row["temperature"].values
            t_init = unp.uarray(
                test_temp,
                uncertainty.u_temperature(test_temp)
            ).mean()

            # calculate equivalence ratio
            fuel = df_test_info.at[best_row, "fuel"]
            oxidizer = df_test_info.at[best_row, "oxidizer"]
            if oxidizer == "air":
                oxidizer = "O2:1 N2:3.76"
            phi = _get_equivalence_ratio(
                partial_pressures["fuel"],
                partial_pressures["oxidizer"],
                _get_f_a_st(
                    fuel,
                    oxidizer,
                    mech
                )
            )

            df_test_info.at[
                best_row, "phi"
            ] = phi.nominal_value

            df_test_info.at[
                best_row, "u_phi"
            ] = phi.std_dev

            df_test_info.at[
                best_row, "p_0"
            ] = p_init

            df_test_info.at[
                best_row, "u_p_0"
            ] = u_p_init

            df_test_info.at[
                best_row, "t_0"
            ] = t_init.nominal_value

            df_test_info.at[
                best_row, "u_t_0"
            ] = t_init.std_dev

            df_test_info.at[
                best_row, "p_fuel"
            ] = partial_pressures["fuel"].nominal_value

            df_test_info.at[
                best_row, "u_p_fuel"
            ] = partial_pressures["fuel"].std_dev

            df_test_info.at[
                best_row, "p_oxidizer"
            ] = partial_pressures["oxidizer"].nominal_value

            df_test_info.at[
                best_row, "u_p_oxidizer"
            ] = partial_pressures["oxidizer"].std_dev

            df_test_info.at[
                best_row, "p_diluent"
            ] = partial_pressures["diluent"].nominal_value

            df_test_info.at[
                best_row, "u_p_diluent"
            ] = partial_pressures["diluent"].std_dev

            df_test_info.at[
                best_row, "wave_speed"
            ] = wave_speed.nominal_value

            df_test_info.at[
                best_row, "u_wave_speed"
            ] = wave_speed.std_dev

            # store auxiliary test data
            tests[best_row] = dict()
            tests[best_row]["pressure"] = df_p_row
            tests[best_row]["temperature"] = df_t_row
            tests[best_row]["state"] = df_state_row

        # remove unwanted information from test info dataframe
        df_test_info.dropna(subset=["p_fuel"], inplace=True)
        df_test_info.reset_index(drop=True, inplace=True)
        df_test_info.drop(columns="datetime", inplace=True)
        tests = [t for t in tests if len(t) > 0]
        return df_test_info, tests

    @staticmethod
    def _extract_sensor_data(
            df_sensor,
            which="pressure",
            dropna=True
    ):
        """
        Extracts temperature or pressure data from a sensor data dataframe

        Parameters
        ----------
        df_sensor : pd.DataFrame
            Dataframe containing full TDMS output from a day of testing
        which : str
            Which trace to extract: `pressure` or `temperature`
        dropna : bool
            Whether or not to drop nan values. This is necessary because
            pressure and temperature traces are different lengths, and nptdms/
            pandas fill the gaps with nans.

        Returns
        -------
        pd.DataFrame
            sub-dataframe containing desired trace
        """
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
        """
        Searches a dataframe to find portions containing completed
        detonation tests

        Parameters
        ----------
        df_state : pd.DataFrame
            Dataframe containing tube state transitions

        Returns
        -------
        pd.DataFrame
            Dataframe containing the start and end times of completed tests
        """
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
    def _mask_df_by_row_time(
            df_in,
            test_row,
            include_ends=True
    ):
        """
        Mask a dataframe by the start and end times for a particular tube state

        Parameters
        ----------
        df_in : pd.DataFrame
            Dataframe of test data to be sliced
        test_row : pd.Series
            Series object containing the start and end times for the desired
            tube state to be sliced
        include_ends : bool
            Decides whether to use >/< or >=/<=

        Returns
        -------
        pd.Series
        """
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

    # def save_processed_data(self):
    #     # TODO: externalize this guy
    #     if len(self.tests) == 0:
    #         raise AttributeError("No tests have been processed")
    #
    #     if not os.path.exists(self._dir_processed):
    #         os.makedirs(self._dir_processed)
    #
    #     with pd.HDFStore(self._output_file, "w") as store:
    #         if self.df_test_info is not None:
    #             store.put("test_info", self.df_test_info)
    #         for i, t in enumerate(self.tests):
    #             test_name = "test_%d" % i
    #             for k in t.keys():
    #                 out_key = "/".join(["", test_name, k])
    #                 store.put(out_key, t[k])


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
    p_fuel : float or un.ufloat
        Partial pressure of fuel
    p_oxidizer : float or un.ufloat
        Partial pressure of oxidizer
    f_a_st : float or un.ufloat
        Stoichiometric fuel/air ratio

    Returns
    -------
    float or un.ufloat
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

        return unp.uarray(
            pressure,
            uncertainty.u_pressure(pressure, daq_err=False)
        ).mean()

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
    def _get_initial_temperature(
            cls,
            df_tdms_temperature
    ):
        """
        Old temperatures need to come from the tube thermocouple, which is
        type K, because the manifold thermocouple was jacked up at the time.

        Parameters
        ----------
        df_tdms_temperature : pd.DataFrame
            Dataframe containing test-specific temperature trace

        Returns
        -------
        un.ufloat
            Test-averaged initial temperature with applied uncertainty
        """
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
            f_a_st=0.04201680672268907,
            multiprocess=False
    ):
        """
        Process data from an old-style data set.

        Parameters
        ----------
        base_dir : str
            Base data directory, (e.g. `/d/Data/Raw/`)
        test_date : str
            ISO 8601 formatted date of test data
        f_a_st : float
            Stoichiometric fuel/air ratio for the test mixture. Default value
            is for propane/air.
        multiprocess : bool
            Set to True to parallelize processing of a single day's tests

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
        if multiprocess:
            pool = mp.Pool()
            results = pool.starmap(
                cls._process_single_test,
                [[idx, row, f_a_st] for idx, row in df.iterrows()]
            )
            for idx, row_results in results:
                df.at[idx, "phi"] = row_results["phi"]
                df.at[idx, "u_phi"] = row_results["u_phi"]
                df.at[idx, "p_0"] = row_results["p_0"]
                df.at[idx, "u_p_0"] = row_results["u_p_0"]
                df.at[idx, "t_0"] = row_results["t_0"]
                df.at[idx, "u_t_0"] = row_results["u_t_0"]
                df.at[idx, "p_fuel"] = row_results["p_fuel"]
                df.at[idx, "u_p_fuel"] = row_results["u_p_fuel"]
                df.at[idx, "p_oxidizer"] = row_results["p_oxidizer"]
                df.at[idx, "u_p_oxidizer"] = row_results["u_p_oxidizer"]
                df.at[idx, "wave_speed"] = row_results["wave_speed"]
                df.at[idx, "u_wave_speed"] = row_results["u_wave_speed"]
                images.update(row_results["schlieren"])

        else:
            for idx, row in df.iterrows():
                _, row_results = cls._process_single_test(idx, row, f_a_st)

                # output results
                df.at[idx, "phi"] = row_results["phi"]
                df.at[idx, "u_phi"] = row_results["u_phi"]
                df.at[idx, "p_0"] = row_results["p_0"]
                df.at[idx, "u_p_0"] = row_results["u_p_0"]
                df.at[idx, "t_0"] = row_results["t_0"]
                df.at[idx, "u_t_0"] = row_results["u_t_0"]
                df.at[idx, "p_fuel"] = row_results["p_fuel"]
                df.at[idx, "u_p_fuel"] = row_results["u_p_fuel"]
                df.at[idx, "p_oxidizer"] = row_results["p_oxidizer"]
                df.at[idx, "u_p_oxidizer"] = row_results["u_p_oxidizer"]
                df.at[idx, "wave_speed"] = row_results["wave_speed"]
                df.at[idx, "u_wave_speed"] = row_results["u_wave_speed"]
                images.update(row_results["schlieren"])

        return df, images

    @classmethod
    def _process_single_test(
            cls,
            idx,
            row,
            f_a_st
    ):
        """
        Process a single row of test data. This has been separated into its
        own function to facilitate the use of multiprocessing.

        Parameters
        ----------
        idx : int
            Index of current dataframe row
        row : pd.Series
            Current row of test data
        f_a_st : float
            Stoichiometric fuel/air ratio for the test mixture.

        Returns
        -------
        Tuple(Int, Dict)
            Calculated test data and associated uncertainty values for the
            current row
        """
        # background subtraction
        image = {
            "{:s}_shot{:02d}".format(
                row["date"],
                row["shot"]
            ): schlieren.bg_subtract_all_frames(row["schlieren"])
        }

        # gather pressure data
        df_tdms_pressure = TdmsFile(
            os.path.join(
                row["sensors"],
                "pressure.tdms"
            )
        ).as_dataframe()
        p_init = cls._get_initial_pressure(df_tdms_pressure)
        p_fuel = cls._get_partial_pressure(
            df_tdms_pressure,
            kind="fuel"
        )
        p_oxidizer = cls._get_partial_pressure(
            df_tdms_pressure,
            kind="oxidizer"
        )
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
        out = dict()
        out["schlieren"] = image
        out["phi"] = phi.nominal_value
        out["u_phi"] = phi.std_dev
        out["p_0"] = p_init.nominal_value
        out["u_p_0"] = p_init.std_dev
        out["t_0"] = t_init.nominal_value
        out["u_t_0"] = t_init.std_dev
        out["p_fuel"] = p_fuel.nominal_value
        out["u_p_fuel"] = p_fuel.std_dev
        out["p_oxidizer"] = p_oxidizer.nominal_value
        out["u_p_oxidizer"] = p_oxidizer.std_dev
        out["wave_speed"] = wave_speed.nominal_value
        out["u_wave_speed"] = wave_speed.std_dev

        return idx, out


def process_old_data(
            base_dir,
            test_date,
            f_a_st=0.04201680672268907,
            multiprocess=False
):
    """
    Process data from an old-style data set.

    Parameters
    ----------
    base_dir : str
        Base data directory, (e.g. `/d/Data/Raw/`)
    test_date : str
        ISO 8601 formatted date of test data
    f_a_st : float
        Stoichiometric fuel/air ratio for the test mixture. Default value
        is for propane/air.
    multiprocess : bool
        Set to True to parallelize processing of a single day's tests

    Returns
    -------
    List[pd.DataFrame, dict]
        A list in which the first item is a dataframe of the processed
        tube data and the second is a dictionary containing
        background-subtracted schlieren images
    """
    proc = _ProcessOldData()
    return proc(
        base_dir,
        test_date,
        f_a_st,
        multiprocess
    )


def process_new_data(
    base_dir,
    test_date,
    diode_spacing=1.0668,
    sample_time=pd.Timedelta(seconds=70),
    mech="gri30.cti"
):
    proc = _ProcessNewData()
    return proc(
        base_dir,
        test_date,
        diode_spacing,
        sample_time,
        mech
    )
