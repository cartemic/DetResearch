# stdlib imports
import os

# third party imports
import cantera as ct
import multiprocessing as mp
import pandas as pd
import uncertainties as un
from nptdms import TdmsFile
from numpy import NaN, sqrt
from scipy.stats import t
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


def _get_dil_mol_frac(
        p_fuel,
        p_oxidizer,
        p_diluent
):
    """

    Parameters
    ----------
    p_fuel : float or un.ufloat
        Fuel partial pressure
    p_oxidizer : float or un.ufloat
        Oxidizer partial pressure
    p_diluent : float or un.ufloat
        Diluent partial pressure

    Returns
    -------
    float or un.ufloat
        Diluent mole fraction
    """
    return p_diluent / (p_fuel + p_oxidizer + p_diluent)


class _ProcessNewData:
    @classmethod
    def __call__(
            cls,
            base_dir,
            test_date,
            sample_time=pd.Timedelta(seconds=70),
            mech="gri30.cti",
            diode_spacing=1.0668,
            multiprocess=False
    ):
        """
        Process data from a day of testing using the newer directory structure

        Parameters
        ----------
        base_dir : str
            Base data directory, (e.g. `/d/Data/Raw/`)
        test_date : str
            ISO 8601 formatted date of test data
        sample_time : None or pd.Timedelta
            Length of hold period at the end of a fill state. If None is passed,
            value will be read from nominal test conditions.
        mech : str
            Mechanism for cantera calculations
        diode_spacing : float
            Diode spacing, in meters
        multiprocess : bool
            Set true to parallelize data analysis

        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Tuple containing a dataframe of test results and a dictionary of
            background subtracted schlieren images
        """
        dir_data = os.path.join(base_dir, test_date)
        df_nominal = cls._load_nominal_conditions(dir_data)
        df_sensor = TdmsFile(os.path.join(
            dir_data, "sensor log.tdms"
        )).as_dataframe()
        df_pressure = cls._extract_sensor_data(df_sensor, "pressure")
        df_temperature = cls._extract_sensor_data(df_sensor, "temperature")
        del df_sensor
        df_tests = cls._find_test_times(base_dir, test_date)
        # todo: raise error if number of tests doesn't match number of fires
        df_schlieren = pd.DataFrame(columns=["shot", "schlieren"])
        df_schlieren["schlieren"] = _collect_schlieren_dirs(
            base_dir, test_date
        )
        df_schlieren["shot"] = df_schlieren["schlieren"]
        df_schlieren["shot"] = [
            int(os.path.split(d)[1].lower().replace("shot", "").strip())
            for d in df_schlieren["schlieren"]
            if "failed" not in d
        ]
        df_tests = df_tests.merge(df_schlieren, on="shot", how="left")
        df_diode_locs = pd.DataFrame(columns=["shot", "diodes"])
        df_diode_locs["diodes"] = find_diode_data(dir_data)
        df_diode_locs["shot"] = [
            int(
                os.path.split(
                    os.path.dirname(d))[1].lower().replace(
                    "shot", ""
                ).strip()
            )
            for d in df_diode_locs["diodes"]
        ]
        df_tests = df_tests.merge(df_diode_locs, on="shot", how="left")
        df_state = TdmsFile(os.path.join(
                base_dir, test_date, "tube state.tdms"
        )).as_dataframe()
        df_state.columns = ["time", "state", "mode"]

        images = dict()
        if multiprocess:
            pool = mp.Pool()
            results = pool.starmap(
                cls._process_single_test,
                [[
                    idx,
                    df_nominal,
                    df_pressure,
                    df_temperature,
                    df_state,
                    sample_time,
                    test_time_row,
                    mech,
                    diode_spacing
                ] for idx, test_time_row in df_tests.iterrows()]
            )
            for idx, row_results in results:
                if row_results["schlieren"] is not None:
                    images.update(row_results["schlieren"])
                df_tests.at[idx, "t_0"] = row_results["t_0"]
                df_tests.at[idx, "u_t_0"] = row_results["u_t_0"]
                df_tests.at[idx, "p_0_nom"] = row_results["p_0_nom"]
                df_tests.at[idx, "p_0"] = row_results["p_0"]
                df_tests.at[idx, "u_p_0"] = row_results["u_p_0"]
                df_tests.at[idx, "phi_nom"] = row_results["phi_nom"]
                df_tests.at[idx, "phi"] = row_results["phi"]
                df_tests.at[idx, "u_phi"] = row_results["u_phi"]
                df_tests.at[idx, "fuel"] = row_results["fuel"]
                df_tests.at[idx, "p_fuel"] = row_results["p_fuel"]
                df_tests.at[idx, "u_p_fuel"] = row_results["u_p_fuel"]
                df_tests.at[idx, "oxidizer"] = row_results["oxidizer"]
                df_tests.at[idx, "p_oxidizer"] = row_results["p_oxidizer"]
                df_tests.at[idx, "u_p_oxidizer"] = row_results["u_p_oxidizer"]
                df_tests.at[idx, "diluent"] = row_results["diluent"]
                df_tests.at[idx, "p_diluent"] = row_results["p_diluent"]
                df_tests.at[idx, "u_p_diluent"] = row_results["u_p_diluent"]
                df_tests.at[idx, "dil_mf_nom"] = row_results["dil_mf_nom"]
                df_tests.at[idx, "dil_mf"] = row_results["dil_mf"]
                df_tests.at[idx, "u_dil_mf"] = row_results["u_dil_mf"]
                df_tests.at[idx, "wave_speed"] = row_results["wave_speed"]
                df_tests.at[idx, "u_wave_speed"] = row_results["u_wave_speed"]

        else:
            for idx, test_time_row in df_tests.iterrows():
                _, row_results = cls._process_single_test(
                    idx,
                    df_nominal,
                    df_pressure,
                    df_temperature,
                    df_state,
                    sample_time,
                    test_time_row,
                    mech,
                    diode_spacing
                )

                # output results
                if row_results["schlieren"] is not None:
                    images.update(row_results["schlieren"])
                df_tests.at[idx, "t_0"] = row_results["t_0"]
                df_tests.at[idx, "u_t_0"] = row_results["u_t_0"]
                df_tests.at[idx, "p_0_nom"] = row_results["p_0_nom"]
                df_tests.at[idx, "p_0"] = row_results["p_0"]
                df_tests.at[idx, "u_p_0"] = row_results["u_p_0"]
                df_tests.at[idx, "phi_nom"] = row_results["phi_nom"]
                df_tests.at[idx, "phi"] = row_results["phi"]
                df_tests.at[idx, "u_phi"] = row_results["u_phi"]
                df_tests.at[idx, "fuel"] = row_results["fuel"]
                df_tests.at[idx, "p_fuel"] = row_results["p_fuel"]
                df_tests.at[idx, "u_p_fuel"] = row_results["u_p_fuel"]
                df_tests.at[idx, "oxidizer"] = row_results["oxidizer"]
                df_tests.at[idx, "p_oxidizer"] = row_results["p_oxidizer"]
                df_tests.at[idx, "u_p_oxidizer"] = row_results["u_p_oxidizer"]
                df_tests.at[idx, "diluent"] = row_results["diluent"]
                df_tests.at[idx, "p_diluent"] = row_results["p_diluent"]
                df_tests.at[idx, "u_p_diluent"] = row_results["u_p_diluent"]
                df_tests.at[idx, "dil_mf_nom"] = row_results["dil_mf_nom"]
                df_tests.at[idx, "dil_mf"] = row_results["dil_mf"]
                df_tests.at[idx, "u_dil_mf"] = row_results["u_dil_mf"]
                df_tests.at[idx, "wave_speed"] = row_results["wave_speed"]
                df_tests.at[idx, "u_wave_speed"] = row_results["u_wave_speed"]

        df_tests["date"] = test_date
        return df_tests, images

    @classmethod
    def _process_single_test(
            cls,
            idx,
            df_nominal,
            df_pressure,
            df_temperature,
            df_state,
            sample_time,
            test_time_row,
            mech="gri30.cti",
            diode_spacing=1.0668
    ):
        """

        Parameters
        ----------
        idx : int
            Index of the test to be analyzed
        df_nominal : pd.DataFrame
            Dataframe of nominal test conditions, untrimmed
        df_pressure : pd.DataFrame
            Dataframe of full-day test pressures
        df_temperature : pd.DataFrame
            Dataframe of full-day test temperatures
        df_state : pd.DataFrame
            Dataframe of tube state changes, untrimmed
        sample_time : None or pd.Timedelta
            Length of hold period at the end of a fill state. If None is passed,
            value will be read from nominal test conditions.
        test_time_row : pd.Series
            Row of current test in the main test dataframe
        mech : str
            Mechanism for cantera calculations
        diode_spacing : float
            Diode spacing, in meters

        Returns
        -------
        Tuple[Int, Dict]
            A tuple containing the index of the analyzed test and a dictionary
            of the test results
        """
        out = dict()

        # collect nominal test conditions
        df_test_nominal = cls._get_test_nominal(df_nominal, test_time_row)
        fuel = df_test_nominal["fuel"]
        oxidizer = df_test_nominal["oxidizer"]
        if oxidizer.lower() == "air":
            oxidizer = "O2:1 N2:3.76"
        diluent = df_test_nominal["diluent"]
        dil_mf_nom = df_test_nominal["diluent_mol_frac_nominal"]
        phi_nom = df_test_nominal["phi_nominal"]
        p_0_nom = df_test_nominal["p_0_nominal"]
        if sample_time is None:
            if hasattr(df_test_nominal, "sample_time"):
                sample_time = pd.Timedelta(
                    seconds=df_test_nominal["sample_time"]
                )
            else:
                sample_time = pd.Timedelta(seconds=70)

        # collect current test temperature with uncertainty
        # TODO: move to separate function and update calculation to be like
        #  new pressure calc
        temps = cls._collect_current_test_df(
            df_temperature,
            test_time_row
        )["temperature"].values
        temps = unp.uarray(
            temps,
            uncertainty.u_temperature(temps)
        )
        t_0 = temps.mean()

        # collect current test pressures
        df_current_test_pressure = cls._collect_current_test_df(
            df_pressure,
            test_time_row
        )
        df_state_cutoff_times = cls._get_pressure_cutoff_times(
            df_state,
            test_time_row,
            sample_time
        )

        # extract cutoff pressures
        p_cutoff_vac = cls._get_cutoff_pressure(
            "vacuum",
            df_current_test_pressure,
            df_state_cutoff_times
        )
        p_cutoff_fuel = cls._get_cutoff_pressure(
            "fuel",
            df_current_test_pressure,
            df_state_cutoff_times
        )
        p_cutoff_oxidizer = cls._get_cutoff_pressure(
            "oxidizer",
            df_current_test_pressure,
            df_state_cutoff_times
        )
        p_cutoff_diluent = cls._get_cutoff_pressure(
            "diluent",
            df_current_test_pressure,
            df_state_cutoff_times
        )

        # calculate partial pressures
        p_fuel = p_cutoff_fuel - p_cutoff_vac
        p_diluent = p_cutoff_diluent - p_cutoff_fuel

        # TODO: since current detonations use air as an oxidizer, add vacuum
        #  cutoff pressure to oxidizer partial pressure. Change this if non-air
        #  oxidizers are  used
        p_oxidizer = p_cutoff_oxidizer - p_cutoff_diluent + p_cutoff_vac

        # oxidizer is the last fill state, which means that p_0 == p_oxidizer
        p_0 = cls._get_cutoff_pressure(
            "oxidizer",
            df_current_test_pressure,
            df_state_cutoff_times
        )

        # calculate equivalence ratio and diluent mole fraction
        phi = _get_equivalence_ratio(
            p_fuel,
            p_oxidizer,
            _get_f_a_st(
                fuel,
                oxidizer,
                mech
            )
        )
        dil_mf = _get_dil_mol_frac(p_fuel, p_oxidizer, p_diluent)

        # get wave speed
        wave_speed = calculate_velocity(
            test_time_row["diodes"],
            diode_spacing=diode_spacing
        )[0]

        # background subtract schlieren
        if not pd.isnull(test_time_row["schlieren"]):
            # do bg subtraction
            date = os.path.split(
                os.path.dirname(
                    os.path.dirname(
                        test_time_row["diodes"]
                    )
                )
            )[1]
            out["schlieren"] = {
                "{:s}_shot{:02d}".format(
                    date,
                    int(test_time_row["shot"])
                ): schlieren.bg_subtract_all_frames(
                    test_time_row["schlieren"])
            }
        else:
            out["schlieren"] = None

        out["t_0"] = t_0.nominal_value
        out["u_t_0"] = t_0.std_dev
        out["p_0_nom"] = p_0_nom
        out["p_0"] = p_0.nominal_value
        out["u_p_0"] = p_0.std_dev
        out["phi_nom"] = phi_nom
        out["phi"] = phi.nominal_value
        out["u_phi"] = phi.std_dev
        out["fuel"] = fuel
        out["p_fuel"] = p_fuel.nominal_value
        out["u_p_fuel"] = p_fuel.std_dev
        out["oxidizer"] = oxidizer
        out["p_oxidizer"] = p_oxidizer.nominal_value
        out["u_p_oxidizer"] = p_oxidizer.std_dev
        out["diluent"] = diluent
        out["p_diluent"] = p_diluent.nominal_value
        out["u_p_diluent"] = p_diluent.std_dev
        out["dil_mf_nom"] = dil_mf_nom
        out["dil_mf"] = dil_mf.nominal_value
        out["u_dil_mf"] = dil_mf.std_dev
        out["wave_speed"] = wave_speed.nominal_value
        out["u_wave_speed"] = wave_speed.std_dev

        return idx, out

    @staticmethod
    def _load_nominal_conditions(dir_data):
        """
        Loads nominal test conditions from disk

        Parameters
        ----------
        dir_data : str
            Directory containing a test_conditions.csv file

        Returns
        -------
         pd.DataFrame
            Dataframe of nominal test conditions
        """
        df_conditions = pd.read_csv(
            os.path.join(
                dir_data,
                "test_conditions.csv"
            )
        )
        df_conditions["datetime"] = pd.to_datetime(
            df_conditions["datetime"],
            utc=False
        )

        # drop unnecessary information
        df_conditions = df_conditions[
            [k for k in df_conditions.keys()
             if k not in {"p_dil", "p_ox", "p_f"}]
        ]
        old_cols = [
            "datetime",
            "diluent_mol_frac",
            "equivalence",
            "init_pressure"
        ]
        new_cols = [
            "time",
            "diluent_mol_frac_nominal",
            "phi_nominal",
            "p_0_nominal"
        ]
        df_conditions.rename(
            columns={o: n for o, n in zip(old_cols, new_cols)},
            inplace=True
        )
        df_conditions["p_0_nominal"] *= 101325  # p_0 recorded in atm cus im dum

        return df_conditions

    @staticmethod
    def _get_test_nominal(
            df_nominal,
            test_time_row
    ):
        """
        Collects nominal test conditions for a given test from a dataframe of
        nominal test conditions

        Parameters
        ----------
        df_nominal : pd.DataFrame
            Nominal test condition dataframe -- see _load_nominal_conditions
        test_time_row : pd.Series
            Row of current test in the main test dataframe

        Returns
        -------
        pd.Series
            Nominal test conditions for a particular test
        """
        # subtract one because n_true is on (1, len) while idx is on (0, len-1)
        # noinspection PyUnresolvedReferences
        # cumsum is on pd.Series you fool
        best_idx = (df_nominal["time"] < test_time_row[
            "end"]).cumsum().max() - 1
        return df_nominal.iloc[best_idx]

    @staticmethod
    def _extract_sensor_data(
            df_sensor,
            which="pressure",
            dropna=True
    ):
        """
        Extracts pressure or temperature data from full sensor dataframe.
        Dropna option is included due to NaNs populated by pandas/nptdms,
        which are caused by temperature and pressure traces having different
        lengths.

        Parameters
        ----------
        df_sensor : pd.DataFrame
            Dataframe of full tube tdms output
        which : str
            `pressure` or `temperature`
        dropna : bool
            Whether to drop NaN values from the output dataframe

        Returns
        -------
        pd.DataFrame
            Desired trace chosen by `which`
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
    def _find_test_times(
            base_dir,
            test_date
    ):
        """
        Locates start and end times of tests in a larger dataframe containing
        all tube data

        Parameters
        ----------
        base_dir : str
            Base data directory, (e.g. `/d/Data/Raw/`)
        test_date : str
            ISO 8601 formatted date of test data

        Returns
        -------
        pd.DataFrame
            Dataframe containing start and end times of each test
        """
        # end times
        # The tube will only automatically go `Closed -> Vent` at the end of
        # a completed test cycle
        loc_state = os.path.join(base_dir, test_date, "tube state.tdms")
        df_state = TdmsFile(loc_state).as_dataframe()
        df_state.columns = ["time", "state", "mode"]

        df_test_times = pd.DataFrame(columns=["shot", "start", "end"])
        df_test_times["end"] = df_state[
            (df_state["state"].shift(1) == "Tube Closed") &
            (df_state["state"] == "Tube Vent") &
            (df_state["mode"] == "auto")
            ]["time"]
        df_test_times.reset_index(drop=True, inplace=True)
        df_test_times["shot"] = df_test_times.index.values

        # start times
        # A test will be considered to have started at the last automatic mix
        # section purge preceding its end time.
        for i, time in enumerate(df_test_times["end"].values):
            df_test_times.at[i, "start"] = df_state[
                (df_state["time"].values < time) &
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
        Creates a mask of a dataframe with a `time` column from a series
        object containing `start` and `end` time stamps.

        Parameters
        ----------
        df_in : pd.DataFrame
            Dataframe containing a `time` column
        test_row : pd.Series
            Series containing `start` and `end` time stamp entries
        include_ends : bool
            True to include end points, false to exclude them

        Returns
        -------
        pd.Series
            Boolean series; True where time is within the desired range and
            False where it isn't.
        """
        start_time = test_row["start"]
        end_time = test_row["end"]

        if isinstance(start_time, pd.Series):
            start_time = start_time.values[0]
            end_time = end_time.values[0]

        if include_ends:
            return (
                (df_in["time"] >= start_time) &
                (df_in["time"] <= end_time)
            )
        else:
            return (
                (df_in["time"] > start_time) &
                (df_in["time"] < end_time)
            )

    @classmethod
    def _get_pressure_cutoff_times(
            cls,
            df_state,
            test_time_row,
            sample_time
    ):
        """
        Locates start and end times of tube fill events

        Parameters
        ----------
        df_state : pd.DataFrame
            Dataframe containing tube state changes
        test_time_row : pd.Series
            Row of current test in the main test dataframe
        sample_time : pd.Timedelta
            Length of hold period at the end of a fill state

        Returns
        -------
        pd.DataFrame
            Dataframe containing start and end times of each portion of the
            tube fill sequence
        """
        # noinspection PyUnresolvedReferences
        # lol it's a pd.Series of booleans you fool
        state_mask = cls._mask_df_by_row_time(
            df_state,
            test_time_row,
            include_ends=False
        ).values
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
        return df_state_row

    @classmethod
    def _get_cutoff_pressure(
            cls,
            state,
            df_current_test_pressure,
            df_state_cutoff_times
    ):
        """
        Gets the cutoff pressure for a particular tube fill state

        Parameters
        ----------
        state : str
            One of the main tube fill states: vacuum, fuel, diluent, oxidizer
        df_current_test_pressure : pd.DataFrame
            Dataframe containing current test pressure trace
        df_state_cutoff_times : pd.DataFrame
            Dataframe of state cutoff times -- see _get_pressure_cutoff_times

        Returns
        -------
        un.ufloat
            Mean pressure value with uncertainty estimate
        """
        press = cls._collect_current_test_df(
            df_current_test_pressure,
            df_state_cutoff_times[df_state_cutoff_times["state"] == state]
        )["pressure"].values

        # calculate sample uncertainty
        num_samples = len(press)
        sem = press.std() / sqrt(num_samples)
        u_sample = un.ufloat(
            0,
            sem * t.ppf(0.975, num_samples - 1),
            tag="sample"
        )

        press = unp.uarray(
            press,
            uncertainty.u_pressure(press, daq_err=False)
        )

        press = press.mean() + u_sample

        return press

    @classmethod
    def _collect_current_test_df(
            cls,
            df_to_slice,
            test_time_row
    ):
        """
        Slices a temperature or pressure dataframe using _mask_by_row_time

        Parameters
        ----------
        df_to_slice : pd.DataFrame
            Dataframe with a `time` column
        test_time_row : pd.Series
            Series containing `start` and `end` timestamps

        Returns
        -------
        pd.DataFrame
            Sliced portion of input dataframe
        """
        return df_to_slice[cls._mask_df_by_row_time(df_to_slice, test_time_row)]


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
        and os.path.exists(os.path.join(raw_dir, item, "frames"))
        and os.path.exists(os.path.join(raw_dir, item, "bg"))
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

        # TODO: update calculation to be like new pressure calc
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
        # TODO: update calculation to be like new pressure calc
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
        df = pd.DataFrame(
            columns=["date", "shot", "sensors", "schlieren"],
        )
        df["sensors"] = cls._collect_test_dirs(base_dir, test_date)
        df["schlieren"] = _collect_schlieren_dirs(base_dir, test_date)
        df = df[df["schlieren"].apply(lambda x: "failed" not in x)]
        df["date"] = test_date
        df["shot"] = [
            int(os.path.split(d)[1].lower().replace("shot", "").strip())
            for d in df["schlieren"].values
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
        loc_temp_tdms = os.path.join(
            row["sensors"],
            "temperature.tdms"
        )
        if os.path.exists(loc_temp_tdms):
            df_tdms_temperature = TdmsFile(
                os.path.join(
                    row["sensors"],
                    "temperature.tdms"
                )
            ).as_dataframe()
            t_init = cls._get_initial_temperature(df_tdms_temperature)
        else:
            t_init = un.ufloat(NaN, NaN)

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
        sample_time=None,
        mech="gri30.cti",
        diode_spacing=1.0668,
        multiprocess=False
):
    """
    Process data from a day of testing using the newer directory structure

    Parameters
    ----------
    base_dir : str
        Base data directory, (e.g. `/d/Data/Raw/`)
    test_date : str
        ISO 8601 formatted date of test data
    sample_time : None or pd.Timedelta
        Length of hold period at the end of a fill state. If None is passed,
        value will be read from nominal test conditions.
    mech : str
        Mechanism for cantera calculations
    diode_spacing : float
        Diode spacing, in meters
    multiprocess : bool
        Set true to parallelize data analysis

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Tuple containing a dataframe of test results and a dictionary of
        background subtracted schlieren images
    """
    proc = _ProcessNewData()
    return proc(
        base_dir,
        test_date,
        sample_time,
        mech,
        diode_spacing,
        multiprocess
    )
