# -*- coding: utf-8 -*-
"""
PURPOSE:
    Database management for data storage

CREATED BY:
    Mick Carter
    Oregon State University
    CIRE and Propulsion Lab
    cartemic@oregonstate.edu
"""
import sqlite3
import inspect
import warnings
import os
import uuid


def _formatwarnmsg_impl(msg):  # pragma: no cover
    # happier warning format :)
    s = ("%s: %s\n" % (msg.category.__name__, msg.message))
    return s


warnings._formatwarnmsg_impl = _formatwarnmsg_impl
warnings.simplefilter('always')


class DataBase:
    """
    A class for database-level operations
    """
    @staticmethod
    def list_all_tables(database):
        """
        Finds every table in a given database.

        Parameters
        ----------
        database : str
            Database to search, e.g. `test.sqlite`

        Returns
        -------
        table_list : list
            All tables within the requested database
        """
        with sqlite3.connect(database) as con:
            cur = con.cursor()
            cur.execute(
                """SELECT name FROM sqlite_master WHERE type='table';"""
            )

        tables = cur.fetchall()
        con.close()
        table_list = [item[0] for item in tables]
        return table_list


class Table:
    """
    A class to help manage tables of computational detonation data.
    """
    _test_table_args = (
        'date_stored',
        'mechanism',
        'initial_temp',
        'initial_press',
        'fuel',
        'oxidizer',
        'equivalence',
        'diluent',
        'diluent_mol_frac',
        'cj_speed',
        'ind_len_west',
        'ind_len_gav',
        'ind_len_ng',
        'cell_size_west',
        'cell_size_gav',
        'cell_size_ng',
        'rxn_table_id'
    )

    def __init__(
            self,
            database,
            table_name,
            testing=False
    ):
        self.table_name = self._clean(table_name)
        self.database = database
        self.con = sqlite3.connect(self.database)
        self._testing = testing
        # todo: rxn table id doesn't make sense here, make a Row class
        self._rxn_table_id = str(uuid.uuid4()).replace('-', '')
        if self.table_name not in DataBase.list_all_tables(database):
            self._create_test_table()
            self._create_rxn_table_base()
            self._create_rxn_table_pert()
        else:
            # todo: look up rxn table id
            pass

    def __del__(self):
        self.con.commit()
        self.con.close()
        if self._testing:
            os.remove(self.database)

    @staticmethod
    def _clean(table_name):
        """
        Cleans a table name string to keep me from doing anything too stupid.
        Alphanumeric values and underscores are allowed; anything else will
        throw a NameError.

        Parameters
        ----------
        table_name : str

        Returns
        -------
        str
        """
        if any([not (char.isalnum() or char == '_') for char in table_name]):
            raise NameError(
                'Table name must be entirely alphanumeric. '
                'Underscores are allowed.'
            )
        else:
            return table_name.lower()

    def test_columns(self):
        """
        Returns
        -------
        table_info : list
            A list of all column names in the current table.
        """
        with sqlite3.connect(self.database) as con:
            cur = con.cursor()
            cur.execute("""PRAGMA table_info({:s});""".format(
                self.table_name)
            )

        table_info = [item[1] for item in cur.fetchall()]

        return table_info

    def base_columns(self):
        """
        Returns
        -------
        table_info : list
            A list of all column names in the current table.
        """
        with sqlite3.connect(self.database) as con:
            cur = con.cursor()
            cur.execute("""PRAGMA table_info({:s});""".format(
                'BASE_' + self._rxn_table_id)
            )

        table_info = [item[1] for item in cur.fetchall()]

        return table_info

    def pert_columns(self):
        """
        Returns
        -------
        table_info : list
            A list of all column names in the current table.
        """
        with sqlite3.connect(self.database) as con:
            cur = con.cursor()
            cur.execute("""PRAGMA table_info({:s});""".format(
                'PERT_' + self._rxn_table_id)
            )

        table_info = [item[1] for item in cur.fetchall()]

        return table_info

    # noinspection PyUnusedLocal
    def check_existing_test(
            self,
            mechanism=None,
            initial_temp=None,
            initial_press=None,
            fuel=None,
            oxidizer=None,
            equivalence=None,
            diluent=None,
            diluent_mol_frac=None
    ):
        """
        Checks the current table for a specific row of data

        Parameters
        ----------
        mechanism : str
            Mechanism used for the desired row's computation
        initial_temp : float
            Initial temperature for the desired row, in Kelvin
        initial_press : float
            Initial pressure for the desired row, in Pascals
        equivalence : float
            Equivalence ratio for the desired row
        fuel : str
            Fuel used in the desired row
        oxidizer : str
            Oxidizer used in the desired row
        diluent : str
            Diluent used in the desired row
        diluent_mol_frac : float
            Mole fraction of diluent used in the desired row

        Returns
        -------
        row_found : bool
            True if a row with the given information was found in the current
            table, False if not
        """
        inputs = {
            key: value for key, value
            in inspect.getargvalues(inspect.currentframe())[3].items()
            if key in self._test_table_args
        }
        with self.con as con:
            cur = con.cursor()
            query_str = self._build_query_str(inputs).format(self.table_name)
            cur.execute(
                query_str,
                {key: value for key, value in inputs.items()
                 if value is not None}
            )
            if len(cur.fetchall()) > 0:
                row_found = True
            else:
                row_found = False
        return row_found

    def _create_test_table(self):
        """
        Creates a table of test conditions and results in the current database
        """
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE {:s} (
                    date_stored TEXT,
                    mechanism TEXT,
                    initial_temp REAL,
                    initial_press REAL,
                    fuel TEXT,
                    oxidizer TEXT,
                    equivalence REAL,
                    diluent TEXT,
                    diluent_mol_frac REAL,
                    cj_speed REAL,
                    ind_len_west REAL,
                    ind_len_gav REAL,
                    ind_len_ng REAL,
                    cell_size_west REAL,
                    cell_size_gav REAL,
                    cell_size_ng REAL,
                    rxn_table_id TEXT
                );
                """.format(self.table_name)
            )

    def _create_rxn_table_base(self):
        """
        Creates a table of base (unperturbed) reactions and their rate constants
        in the current database
        """
        # TODO: implement this!
        base_table_name = 'BASE_' + self._rxn_table_id
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE {:s} (
                    rxn_no INTEGER,
                    rxn TEXT,
                    k_i REAL
                );
                """.format(base_table_name)
            )
        return base_table_name

    def _create_rxn_table_pert(self):
        """
        Creates a table of perturbed reaction results in the current database
        """
        # TODO: implement this!
        pert_table_name = 'PERT_' + self._rxn_table_id
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE {:s} (
                    rxn_no INTEGER,
                    rxn TEXT,
                    k_i REAL,
                    cj_speed REAL,
                    ind_len_west REAL,
                    ind_len_gav REAL,
                    ind_len_ng REAL,
                    cell_size_west REAL,
                    cell_size_gav REAL,
                    cell_size_ng REAL,
                    sens_cj_speed REAL,
                    sens_ind_len_west REAL,
                    sens_ind_len_gav REAL,
                    sens_ind_len_ng REAL,
                    sens_cell_size_west REAL,
                    sens_cell_size_gav REAL,
                    sens_cell_size_ng REAL
                );
                """.format(pert_table_name)
            )
        return pert_table_name

    def _update_test_row(
            self,
            mechanism,
            initial_temp,
            initial_press,
            fuel,
            oxidizer,
            equivalence,
            diluent,
            diluent_mol_frac,
            cj_speed,
            ind_len_west,
            ind_len_gav,
            ind_len_ng,
            cell_size_west,
            cell_size_gav,
            cell_size_ng,
    ):
        """
        Updates the CJ velocity and forward reaction rate (k_i) for a set of
        conditions.

        Parameters
        ----------
        mechanism : str
            Mechanism used for the desired row's computation
        initial_temp : float
            Initial temperature for the desired row, in Kelvin
        initial_press : float
            Initial pressure for the desired row, in Pascals
        fuel : str
            Fuel used in the desired row
        oxidizer : str
            Oxidizer used in the desired row
        equivalence : float
            Equivalence ratio for the desired row
        diluent : str
            Diluent used in the desired row
        diluent_mol_frac : float
            Mole fraction of diluent used in the desired row
        cj_speed : float
            CJ speed to update
        ind_len_west : float
            Induction length (Westbrook)
        ind_len_gav : float
            Induction length (Gavrikov)
        ind_len_ng : float
            Induction length (Ng)
        cell_size_west : float
            Cell size (Westbrook)
        cell_size_gav : float
            Cell size (Gavrikov)
        cell_size_ng : float
            Cell size (Ng)
        """
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                UPDATE {:s} SET 
                    date_stored = datetime('now', 'localtime'),
                    cj_speed = :cj_speed, 
                    ind_len_west = :ind_len_west,
                    ind_len_gav = :ind_len_gav,
                    ind_len_ng = :ind_len_ng,
                    cell_size_west = :cell_size_west,
                    cell_size_gav = :cell_size_gav,
                    cell_size_ng = :cell_size_ng
                WHERE
                    mechanism = :mechanism AND
                    initial_temp = :initial_temp AND
                    initial_press = :initial_press AND
                    equivalence = :equivalence AND
                    fuel = :fuel AND
                    oxidizer = :oxidizer AND
                    diluent = :diluent AND
                    diluent_mol_frac = :diluent_mol_frac
                """.format(self.table_name),
                {
                    'mechanism': mechanism,
                    'initial_temp': initial_temp,
                    'initial_press': initial_press,
                    'fuel': fuel,
                    'oxidizer': oxidizer,
                    'equivalence': equivalence,
                    'diluent': diluent,
                    'diluent_mol_frac': diluent_mol_frac,
                    'cj_speed': cj_speed,
                    'ind_len_west': ind_len_west,
                    'ind_len_gav': ind_len_gav,
                    'ind_len_ng': ind_len_ng,
                    'cell_size_west': cell_size_west,
                    'cell_size_gav': cell_size_gav,
                    'cell_size_ng': cell_size_ng,
                }
            )

    def store_test_row(
            self,
            mechanism,
            initial_temp,
            initial_press,
            fuel,
            oxidizer,
            equivalence,
            diluent,
            diluent_mol_frac,
            cj_speed,
            ind_len_west,
            ind_len_gav,
            ind_len_ng,
            cell_size_west,
            cell_size_gav,
            cell_size_ng,
            overwrite_existing=False
    ):
        """
        Stores a row of data in the current table.

        If a row with this information already exists in the current table,
        overwrite_existing decides whether to overwrite the existing data or
        disregard the current data.

        Parameters
        ----------
        mechanism : str
            Mechanism used for the current row's computation
        initial_temp : float
            Initial temperature for the current row, in Kelvin
        initial_press : float
            Initial pressure for the current row, in Pascals
        equivalence : float
            Equivalence ratio for the current row
        fuel : str
            Fuel used in the current row
        oxidizer : str
            Oxidizer used in the current row
        cj_speed : float
            Current CJ speed
        diluent : str
            Diluent used in the current row
        diluent_mol_frac : float
            Mole fraction of diluent used in the current row
        overwrite_existing : bool
            True to overwrite an existing entry if it exists, False to
            protect existing entries
        ind_len_west : float
            Induction length (Westbrook)
        ind_len_gav : float
            Induction length (Gavrikov)
        ind_len_ng : float
            Induction length (Ng)
        cell_size_west : float
            Cell size (Westbrook)
        cell_size_gav : float
            Cell size (Gavrikov)
        cell_size_ng : float
            Cell size (Ng)
        """
        if self.check_existing_test(
                mechanism=mechanism,
                initial_temp=initial_temp,
                initial_press=initial_press,
                equivalence=equivalence,
                fuel=fuel,
                oxidizer=oxidizer,
                diluent=diluent,
                diluent_mol_frac=diluent_mol_frac
        ):
            # a rew with the current information was found
            if overwrite_existing:
                self._update_test_row(
                    mechanism=mechanism,
                    initial_temp=initial_temp,
                    initial_press=initial_press,
                    fuel=fuel,
                    oxidizer=oxidizer,
                    equivalence=equivalence,
                    diluent=diluent,
                    diluent_mol_frac=diluent_mol_frac,
                    cj_speed=cj_speed,
                    ind_len_west=ind_len_west,
                    ind_len_gav=ind_len_gav,
                    ind_len_ng=ind_len_ng,
                    cell_size_west=cell_size_west,
                    cell_size_gav=cell_size_gav,
                    cell_size_ng=cell_size_ng,
                )
                start_color = '\033[92m'
                end_color = '\033[0m'
                print(start_color+'data row stored successfully'+end_color)
            else:
                # warn the user that the current input was ignored
                warnings.warn(
                    'Cannot overwrite row unless overwrite_existing=True',
                    Warning
                )

        else:
            # no rows with the current information were found
            with self.con as con:
                cur = con.cursor()
                cur.execute(
                    """
                    INSERT INTO {:s} VALUES (
                        datetime('now', 'localtime'),
                        :mechanism,
                        :initial_temp,
                        :initial_press,
                        :fuel,
                        :oxidizer,
                        :equivalence,
                        :diluent,
                        :diluent_mol_frac,
                        :cj_speed,
                        :ind_len_west,
                        :ind_len_gav,
                        :ind_len_ng,
                        :cell_size_west,
                        :cell_size_gav,
                        :cell_size_ng,
                        :rxn_table_id
                    );
                    """.format(self.table_name),
                    {
                        'mechanism': mechanism,
                        'initial_temp': initial_temp,
                        'initial_press': initial_press,
                        'fuel': fuel,
                        'oxidizer': oxidizer,
                        'equivalence': equivalence,
                        'diluent': diluent,
                        'diluent_mol_frac': diluent_mol_frac,
                        'cj_speed': cj_speed,
                        'ind_len_west': ind_len_west,
                        'ind_len_gav': ind_len_gav,
                        'ind_len_ng': ind_len_ng,
                        'cell_size_west': cell_size_west,
                        'cell_size_gav': cell_size_gav,
                        'cell_size_ng': cell_size_ng,
                        'rxn_table_id': self._rxn_table_id,
                    }
                )

    # noinspection PyUnusedLocal
    @staticmethod
    def _build_query_str(kwargs):
        """
        Builds a SQL query string for all of the inputs. Any inputs which are
        None will be left wild.

        todo: this
        Parameters
        ----------

        Returns
        -------
        cmd_str : str
            SQL command to search for the desired inputs
        """
        if hasattr(kwargs, 'self'):
            kwargs.pop('self')

        inputs = {
            key: value for key, value
            in kwargs.items()
            if value is not None
        }
        [where] = [' WHERE ' if len(inputs) > 0 else '']
        sql_varnames = [
            '{:s} = :{:s}'.format(*[item]*2) for item in inputs.keys()
        ]
        [cmd_str] = ['SELECT * FROM {:s}' + where +
                     ' AND '.join(sql_varnames) + ';']
        return cmd_str

    def fetch_test_rows(
            self,
            mechanism=None,
            initial_temp=None,
            initial_press=None,
            equivalence=None,
            fuel=None,
            oxidizer=None,
            diluent=None,
            diluent_mol_frac=None
    ):
        """
        Fetches all rows from the current database with the desired inputs.
        Any inputs which are None will be left wild.

        Parameters
        ----------
        mechanism : str
            Mechanism to search for
        initial_temp : float
            Initial temperature to search for, in Kelvin
        initial_press : float
            Initial pressure to search for, in Pascals
        equivalence : float
            Equivalence ratio to search for
        fuel : str
            Fuel to search for
        oxidizer : str
            Oxidizer to search for
        diluent : str
            Diluent to search for
        diluent_mol_frac : float
            Mole fraction of diluent to search for

        Returns
        -------
        data : dict
            Dictionary containing the rows of the current table which match
            the input criteria. Keys are column names, and values are lists.
        """
        with self.con as con:
            cur = con.cursor()
            cmd_str = self._build_query_str({
                'mechanism': mechanism,
                'initial_temp': initial_temp,
                'initial_press': initial_press,
                'equivalence': equivalence,
                'fuel': fuel,
                'oxidizer': oxidizer,
                'diluent': diluent,
                'diluent_mol_frac': diluent_mol_frac
            })
            cur.execute(
                cmd_str.format(self.table_name),
                {
                    'mechanism': mechanism,
                    'initial_temp': initial_temp,
                    'initial_press': initial_press,
                    'equivalence': equivalence,
                    'fuel': fuel,
                    'oxidizer': oxidizer,
                    'diluent': diluent,
                    'diluent_mol_frac': diluent_mol_frac
                }
            )
            info = cur.fetchall()
            labels = self.test_columns()
            data = {l: [] for l in labels}
            for row in info:
                for l, d in zip(labels, row):
                    data[l].append(d)

            return data


if __name__ == '__main__':  # pragma: no cover
    import subprocess
    subprocess.check_call('pytest -vv tests/test_database.py')
