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
import inspect
import os
import sqlite3
import uuid
import warnings


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
        'inert',
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
        if self.table_name not in DataBase.list_all_tables(database):
            self._create_test_table()

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

    def base_columns(
            self,
            rxn_table_id
    ):
        """
        Returns
        -------
        table_info : list
            A list of all column names in the current table.
        """
        with sqlite3.connect(self.database) as con:
            cur = con.cursor()
            cur.execute("""PRAGMA table_info({:s});""".format(
                'BASE_' + rxn_table_id)
            )

        table_info = [item[1] for item in cur.fetchall()]

        return table_info

    def pert_columns(
            self,
            rxn_table_id
    ):
        """
        Returns
        -------
        table_info : list
            A list of all column names in the current table.
        """
        with sqlite3.connect(self.database) as con:
            cur = con.cursor()
            cur.execute("""PRAGMA table_info({:s});""".format(
                'PERT_' + rxn_table_id)
            )

        table_info = [item[1] for item in cur.fetchall()]

        return table_info

    # noinspection PyUnusedLocal
    def _check_existing_test(
            self,
            mechanism=None,
            initial_temp=None,
            initial_press=None,
            fuel=None,
            oxidizer=None,
            equivalence=None,
            diluent=None,
            diluent_mol_frac=None,
            inert=None
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

    def _check_existing_pert(
            self,
            table_name,
            rxn_no
    ):
        """
        Checks the current table for a specific row of data

        Parameters

        Returns
        -------
        row_found : bool
            True if a row with the given information was found in the current
            table, False if not
        """
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT * from {:s} WHERE rxn_no = :rxn_no
                """.format(table_name),
                {
                    'rxn_no': rxn_no
                }
            )
            if len(cur.fetchall()) > 0:
                row_found = True
            else:
                row_found = False
        return row_found

    def check_for_stored_base_data(
            self,
            rxn_table_id
    ):
        """
        Allows a user to check whether or not a base reaction table has data
        stored in it

        Parameters
        ----------
        rxn_table_id : str
            Reaction table ID corresponding to the BASE_rxn_table_id and
            PERT_rxn_table_id tables. BASE table holds all reactions and
            reaction rate coefficients, while PERT holds all reactions and
            perturbed reaction rate coefficients along with the associated CJ
            speed, induction length, and cell size results.

        Returns
        -------
        data_found : bool
            True if base table has data in it, False if not
        """
        table_name = 'BASE_' + rxn_table_id
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT * from {:s}
                """.format(table_name)
            )
            if len(cur.fetchall()) > 0:
                data_found = True
            else:
                data_found = False
        return data_found

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
                    inert TEXT,
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

    def _create_rxn_table_base(
            self,
            rxn_table_id
    ):
        """
        Creates a table of base (unperturbed) reactions and their rate constants
        in the current database
        """
        base_table_name = 'BASE_' + rxn_table_id
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

    def _create_rxn_table_pert(
            self,
            rxn_table_id
    ):
        """
        Creates a table of perturbed reaction results in the current database
        """
        pert_table_name = 'PERT_' + rxn_table_id
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE {:s} (
                    stored_date TEXT,
                    rxn_no INTEGER,
                    rxn TEXT,
                    k_i REAL,
                    ind_len_west REAL,
                    ind_len_gav REAL,
                    ind_len_ng REAL,
                    cell_size_west REAL,
                    cell_size_gav REAL,
                    cell_size_ng REAL,
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
            inert,
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
                    diluent_mol_frac = :diluent_mol_frac AND
                    inert = :inert
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
                    'inert': inert,
                    'cj_speed': cj_speed,
                    'ind_len_west': ind_len_west,
                    'ind_len_gav': ind_len_gav,
                    'ind_len_ng': ind_len_ng,
                    'cell_size_west': cell_size_west,
                    'cell_size_gav': cell_size_gav,
                    'cell_size_ng': cell_size_ng,
                }
            )

    def _update_pert_row(
            self,
            rxn_table_id,
            rxn_no,
            rxn,
            k_i,
            ind_len_west,
            ind_len_gav,
            ind_len_ng,
            cell_size_west,
            cell_size_gav,
            cell_size_ng,
            sens_ind_len_west,
            sens_ind_len_gav,
            sens_ind_len_ng,
            sens_cell_size_west,
            sens_cell_size_gav,
            sens_cell_size_ng,
    ):
        """
        Updates the CJ velocity and forward reaction rate (k_i) for a set of
        conditions.

        Parameters
        ----------
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
                    k_i = :k_i,
                    ind_len_west = :ind_len_west,
                    ind_len_gav = :ind_len_gav,
                    ind_len_ng = :ind_len_ng,
                    cell_size_west = :cell_size_west,
                    cell_size_gav = :cell_size_gav,
                    cell_size_ng = :cell_size_ng,
                    sens_ind_len_west = :sens_ind_len_west,
                    sens_ind_len_gav = :sens_ind_len_gav,
                    sens_ind_len_ng = :sens_ind_len_ng,
                    sens_cell_size_west = :sens_cell_size_west,
                    sens_cell_size_gav = :sens_cell_size_gav,
                    sens_cell_size_ng = :sens_cell_size_ng
                WHERE
                    rxn_no = :rxn_no AND
                    rxn = :rxn
                """.format(rxn_table_id),
                {
                    'rxn_no': rxn_no,
                    'rxn': rxn,
                    'k_i': k_i,
                    'ind_len_west': ind_len_west,
                    'ind_len_gav': ind_len_gav,
                    'ind_len_ng': ind_len_ng,
                    'cell_size_west': cell_size_west,
                    'cell_size_gav': cell_size_gav,
                    'cell_size_ng': cell_size_ng,
                    'sens_ind_len_west': sens_ind_len_west,
                    'sens_ind_len_gav': sens_ind_len_gav,
                    'sens_ind_len_ng': sens_ind_len_ng,
                    'sens_cell_size_west': sens_cell_size_west,
                    'sens_cell_size_gav': sens_cell_size_gav,
                    'sens_cell_size_ng': sens_cell_size_ng,
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
            inert,
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
        inert : str
            Specie to make inert by removing every reaction where it is a
            reactant or product
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

        Returns
        -------
        rxn_table_id : str
            Reaction table ID corresponding to the BASE_rxn_table_id and
            PERT_rxn_table_id tables. BASE table holds all reactions and
            reaction rate coefficients, while PERT holds all reactions and
            perturbed reaction rate coefficients along with the associated CJ
            speed, induction length, and cell size results.
        """
        if self._check_existing_test(
                mechanism=mechanism,
                initial_temp=initial_temp,
                initial_press=initial_press,
                equivalence=equivalence,
                fuel=fuel,
                oxidizer=oxidizer,
                diluent=diluent,
                diluent_mol_frac=diluent_mol_frac,
                inert=inert
        ):
            # a row with the current information was found
            if overwrite_existing:
                [rxn_table_id] = self.fetch_test_rows(
                    mechanism=mechanism,
                    initial_temp=initial_temp,
                    initial_press=initial_press,
                    equivalence=equivalence,
                    fuel=fuel,
                    oxidizer=oxidizer,
                    diluent=diluent,
                    diluent_mol_frac=diluent_mol_frac,
                    inert=inert
                )['rxn_table_id']
                self._update_test_row(
                    mechanism=mechanism,
                    initial_temp=initial_temp,
                    initial_press=initial_press,
                    fuel=fuel,
                    oxidizer=oxidizer,
                    equivalence=equivalence,
                    diluent=diluent,
                    diluent_mol_frac=diluent_mol_frac,
                    inert=inert,
                    cj_speed=cj_speed,
                    ind_len_west=ind_len_west,
                    ind_len_gav=ind_len_gav,
                    ind_len_ng=ind_len_ng,
                    cell_size_west=cell_size_west,
                    cell_size_gav=cell_size_gav,
                    cell_size_ng=cell_size_ng,
                )
                return rxn_table_id
            else:
                # warn the user that the current input was ignored
                warnings.warn(
                    'Cannot overwrite row unless overwrite_existing=True',
                    Warning
                )

        else:
            # no rows with the current information were found
            with self.con as con:
                rxn_table_id = str(uuid.uuid4()).replace('-', '')
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
                        :inert,
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
                        'inert': inert,
                        'cj_speed': cj_speed,
                        'ind_len_west': ind_len_west,
                        'ind_len_gav': ind_len_gav,
                        'ind_len_ng': ind_len_ng,
                        'cell_size_west': cell_size_west,
                        'cell_size_gav': cell_size_gav,
                        'cell_size_ng': cell_size_ng,
                        'rxn_table_id': rxn_table_id,
                    }
                )
                self._create_rxn_table_base(rxn_table_id)
                self._create_rxn_table_pert(rxn_table_id)
            return rxn_table_id

    def store_perturbed_row(
            self,
            rxn_table_id,
            rxn_no,
            rxn,
            k_i,
            ind_len_west,
            ind_len_gav,
            ind_len_ng,
            cell_size_west,
            cell_size_gav,
            cell_size_ng,
            sens_ind_len_west,
            sens_ind_len_gav,
            sens_ind_len_ng,
            sens_cell_size_west,
            sens_cell_size_gav,
            sens_cell_size_ng,
            overwrite_existing=False
    ):
        """
        Stores a row of data in the current table.

        If a row with this information already exists in the current table,
        overwrite_existing decides whether to overwrite the existing data or
        disregard the current data.

        Parameters
        ----------
        rxn_table_id : str
            Reaction table ID corresponding to the BASE_rxn_table_id and
            PERT_rxn_table_id tables. BASE table holds all reactions and
            reaction rate coefficients, while PERT holds all reactions and
            perturbed reaction rate coefficients along with the associated CJ
            speed, induction length, and cell size results.
        rxn_no : int
            Reaction number of the perturbed reaction in the mechanism's
            reaction list
        rxn : str
            Equation for the perturbed reaction
        k_i : float
            Forward reaction rate constant for the perturbed reaction
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
        sens_ind_len_west : float
            Induction length sensitivity (Westbrook)
        sens_ind_len_gav : float
            Induction length sensitivity (Gavrikov)
        sens_ind_len_ng : float
            Induction length sensitivity (Ng)
        sens_cell_size_west : float
            Cell size (Westbrook)
        sens_cell_size_gav : float
            Cell size sensitivity (Gavrikov)
        sens_cell_size_ng : float
            Cell size sensitivity (Ng)
        overwrite_existing : bool
            True to overwrite an existing entry if it exists, False to
            protect existing entries

        Returns
        -------
        rxn_table_id : str
            Reaction table ID corresponding to the BASE_rxn_table_id and
            PERT_rxn_table_id tables. BASE table holds all reactions and
            reaction rate coefficients, while PERT holds all reactions and
            perturbed reaction rate coefficients along with the associated CJ
            speed, induction length, and cell size results.
        """
        table_name = 'PERT_' + rxn_table_id
        if self._check_existing_pert(table_name, rxn_no):
            # a row with the current information was found
            if overwrite_existing:
                self._update_pert_row(
                    rxn_table_id=table_name,
                    rxn_no=rxn_no,
                    rxn=rxn,
                    k_i=k_i,
                    ind_len_west=ind_len_west,
                    ind_len_gav=ind_len_gav,
                    ind_len_ng=ind_len_ng,
                    cell_size_west=cell_size_west,
                    cell_size_gav=cell_size_gav,
                    cell_size_ng=cell_size_ng,
                    sens_ind_len_west=sens_ind_len_west,
                    sens_ind_len_gav=sens_ind_len_gav,
                    sens_ind_len_ng=sens_ind_len_ng,
                    sens_cell_size_west=sens_cell_size_west,
                    sens_cell_size_gav=sens_cell_size_gav,
                    sens_cell_size_ng=sens_cell_size_ng,
                )
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
                        :rxn_no,
                        :rxn,
                        :k_i,
                        :ind_len_west,
                        :ind_len_gav,
                        :ind_len_ng,
                        :cell_size_west,
                        :cell_size_gav,
                        :cell_size_ng,
                        :sens_ind_len_west,
                        :sens_ind_len_gav,
                        :sens_ind_len_ng,
                        :sens_cell_size_west,
                        :sens_cell_size_gav,
                        :sens_cell_size_ng
                    );
                    """.format(table_name),
                    {
                        'rxn_no': rxn_no,
                        'rxn': rxn,
                        'k_i': k_i,
                        'ind_len_west': ind_len_west,
                        'ind_len_gav': ind_len_gav,
                        'ind_len_ng': ind_len_ng,
                        'cell_size_west': cell_size_west,
                        'cell_size_gav': cell_size_gav,
                        'cell_size_ng': cell_size_ng,
                        'sens_ind_len_west': sens_ind_len_west,
                        'sens_ind_len_gav': sens_ind_len_gav,
                        'sens_ind_len_ng': sens_ind_len_ng,
                        'sens_cell_size_west': sens_cell_size_west,
                        'sens_cell_size_gav': sens_cell_size_gav,
                        'sens_cell_size_ng': sens_cell_size_ng,
                    }
                )

    # noinspection PyUnusedLocal
    @staticmethod
    def _build_query_str(kwargs):
        """
        Builds a SQL query string for all of the inputs. Any inputs which are
        None will be left wild.

        Parameters
        ----------
        kwargs : dict
            Dictionary of keyword arguments to build a query string around.
            This has been left as flexible as possible so that this method can
            build query strings for any of the table types.

        Returns
        -------
        cmd_str : str
            SQL command to search for the desired inputs
        """
        inputs = {
            key: value for key, value
            in kwargs.items()
            if value is not None and 'self' not in key
        }
        [where] = [' WHERE ' if len(inputs) > 0 else '']
        sql_varnames = [
            '{:s} = :{:s}'.format(*[item]*2) for item in inputs.keys()
        ]
        [cmd_str] = ['SELECT * FROM {:s}' + where +
                     ' AND '.join(sql_varnames) + ';']
        return cmd_str

    def fetch_pert_table(
            self,
            rxn_table_id,
            rxn_no=None,
            rxn=None,
            k_i=None,
            ind_len_west=None,
            ind_len_gav=None,
            ind_len_ng=None,
            cell_size_west=None,
            cell_size_gav=None,
            cell_size_ng=None,
            sens_ind_len_west=None,
            sens_ind_len_gav=None,
            sens_ind_len_ng=None,
            sens_cell_size_west=None,
            sens_cell_size_gav=None,
            sens_cell_size_ng=None,
    ):
        """
        Fetches all rows from the current database with the desired inputs.
        Any inputs which are None will be left wild.

        Parameters
        ----------

        Returns
        -------
        data : dict
            Dictionary containing the rows of the current table which match
            the input criteria. Keys are column names, and values are lists.
        """
        rxn_table = 'PERT_' + rxn_table_id
        with self.con as con:
            cur = con.cursor()
            cmd_str = self._build_query_str({
                'rxn_no': rxn_no,
                'rxn': rxn,
                'k_i': k_i,
                'ind_len_west': ind_len_west,
                'ind_len_gav': ind_len_gav,
                'ind_len_ng': ind_len_ng,
                'cell_size_west': cell_size_west,
                'cell_size_gav': cell_size_gav,
                'cell_size_ng': cell_size_ng,
                'sens_ind_len_west': sens_ind_len_west,
                'sens_ind_len_gav': sens_ind_len_gav,
                'sens_ind_len_ng': sens_ind_len_ng,
                'sens_cell_size_west': sens_cell_size_west,
                'sens_cell_size_gav': sens_cell_size_gav,
                'sens_cell_size_ng': sens_cell_size_ng,
            })
            cur.execute(
                cmd_str.format(rxn_table),
                {
                    'rxn_no': rxn_no,
                    'rxn': rxn,
                    'k_i': k_i,
                    'ind_len_west': ind_len_west,
                    'ind_len_gav': ind_len_gav,
                    'ind_len_ng': ind_len_ng,
                    'cell_size_west': cell_size_west,
                    'cell_size_gav': cell_size_gav,
                    'cell_size_ng': cell_size_ng,
                    'sens_ind_len_west': sens_ind_len_west,
                    'sens_ind_len_gav': sens_ind_len_gav,
                    'sens_ind_len_ng': sens_ind_len_ng,
                    'sens_cell_size_west': sens_cell_size_west,
                    'sens_cell_size_gav': sens_cell_size_gav,
                    'sens_cell_size_ng': sens_cell_size_ng,
                }
            )
            info = cur.fetchall()
            labels = self.pert_columns(rxn_table_id)
            data = {lbl: [] for lbl in labels}
            for row in info:
                for lbl, d in zip(labels, row):
                    data[lbl].append(d)

            return data

    def fetch_test_rows(
            self,
            mechanism=None,
            initial_temp=None,
            initial_press=None,
            fuel=None,
            oxidizer=None,
            equivalence=None,
            diluent=None,
            diluent_mol_frac=None,
            inert=None
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
        fuel : str
            Fuel to search for
        oxidizer : str
            Oxidizer to search for
        equivalence : float
            Equivalence ratio to search for
        diluent : str
            Diluent to search for
        diluent_mol_frac : float
            Mole fraction of diluent to search for
        inert : str
            Specie to make inert by removing every reaction where it is a
            reactant or product

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
                'diluent_mol_frac': diluent_mol_frac,
                'inert': inert
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
                    'diluent_mol_frac': diluent_mol_frac,
                    'inert': inert
                }
            )
            info = cur.fetchall()
            labels = self.test_columns()
            data = {lbl: [] for lbl in labels}
            for row in info:
                for lbl, d in zip(labels, row):
                    data[lbl].append(d)

            return data

    def store_base_rxn_table(
            self,
            rxn_table_id,
            gas
    ):
        rxn_table = 'BASE_' + rxn_table_id
        with self.con as con:
            con.execute("""DELETE FROM {:s}""".format(rxn_table))
            for rxn_no, rxn, k_i in zip(
                    range(gas.n_reactions),
                    gas.reaction_equations(),
                    gas.forward_rate_constants
            ):
                con.execute(
                    """
                    INSERT INTO {:s} VALUES (
                        :rxn_no,
                        :rxn,
                        :k_i
                    );
                    """.format(rxn_table),
                    {
                        'rxn_no': rxn_no,
                        'rxn': rxn,
                        'k_i': k_i,
                    }
                )

    def fetch_base_rxn_table(
            self,
            rxn_table_id
    ):
        rxn_table = 'BASE_' + rxn_table_id
        with self.con as con:
            cur = con.cursor()
            cur.execute("""SELECT * FROM {:s}""".format(rxn_table))
            base_rxns = cur.fetchall()
        return base_rxns

    def fetch_single_base_rxn(
            self,
            rxn_table_id,
            rxn_no
    ):
        rxn_table = 'BASE_' + rxn_table_id
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT * FROM {:s}
                WHERE rxn_no = :rxn_no
                """.format(rxn_table),
                {
                    'rxn_no': rxn_no
                }
            )
            base_rxns = cur.fetchone()
        return base_rxns

    def delete_test(
            self,
            rxn_table_id
    ):
        with self.con as con:
            cur = con.cursor()
            for table in ['BASE', 'PERT']:
                cur.execute(
                    """
                    DROP TABLE IF EXISTS {:s}
                    """.format(
                        self._clean(
                            table + '_' + rxn_table_id
                        )
                    )
                )
            cur.execute(
                """
                DELETE FROM {:s} WHERE rxn_table_id = "{:s}"
                """.format(
                    self._clean(
                        self.table_name
                    ),
                    self._clean(
                        rxn_table_id
                    )
                )
            )


if __name__ == '__main__':  # pragma: no cover
    import subprocess
    subprocess.check_call('pytest -vv tests/test_database.py')
