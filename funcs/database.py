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


def _formatwarnmsg_impl(msg):
    # happier warning format :)
    s = ("%s: %s\n" % (msg.category.__name__, msg.message))
    return s


warnings._formatwarnmsg_impl = _formatwarnmsg_impl
warnings.simplefilter('always')


class DataBase:
    @staticmethod
    def list_all_tables(database):
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
    def __init__(
            self,
            database,
            table_name
    ):
        self.table_name = self._clean(table_name)
        self.database = database
        self.con = sqlite3.connect(self.database)
        if self.table_name not in DataBase.list_all_tables(database):
            self._create()

    def __del__(self):
        self.con.commit()
        self.con.close()

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

    def list_all_headers(self):
        with sqlite3.connect(self.database) as con:
            cur = con.cursor()
            cur.execute("""PRAGMA table_info({:s});""".format(
                self.table_name)
            )

        table_info = cur.fetchall()

        con.close()
        return table_info

    def check_existing_row(
            self,
            mechanism,
            initial_temp,
            initial_press,
            equivalence,
            fuel,
            oxidizer,
            reaction_number,
            diluent,
            diluent_mol_frac
    ):
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                SELECT * FROM {:s} WHERE
                    mechanism = :mechanism AND
                    initial_temp = :initial_temp AND
                    initial_press = :initial_press AND
                    equivalence = :equivalence AND
                    fuel = :fuel AND
                    oxidizer = :oxidizer AND
                    diluent = :diluent AND
                    diluent_mol_frac = :diluent_mol_frac AND
                    reaction_number = :reaction_number;
                """.format(self.table_name),
                {
                    'mechanism': mechanism,
                    'initial_temp': initial_temp,
                    'initial_press': initial_press,
                    'equivalence': equivalence,
                    'fuel': fuel,
                    'oxidizer': oxidizer,
                    'diluent': diluent,
                    'diluent_mol_frac': diluent_mol_frac,
                    'reaction_number': reaction_number
                }
            )
            if len(cur.fetchall()) > 0:
                row_found = True
            else:
                row_found = False
        return row_found

    def _create(self):
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE {:s} (
                    date_stored TEXT,
                    mechanism TEXT,
                    initial_temp REAL,
                    initial_press REAL,
                    equivalence REAL,
                    fuel TEXT,
                    oxidizer TEXT,
                    diluent TEXT,
                    diluent_mol_frac REAL,
                    reaction_number INTEGER,
                    k_i REAL,
                    cj_speed REAL
                );
                """.format(self.table_name)
            )

    def _update_row(
            self,
            mechanism,
            initial_temp,
            initial_press,
            equivalence,
            fuel,
            oxidizer,
            reaction_number,
            cj_speed,
            k_i,
            diluent,
            diluent_mol_frac
    ):
        with self.con as con:
            cur = con.cursor()
            cur.execute(
                """
                UPDATE {:s} SET 
                    date_stored = datetime('now', 'localtime'),
                    k_i = :k_i,
                    cj_speed = :cj_speed 
                WHERE
                    mechanism = :mechanism AND
                    initial_temp = :initial_temp AND
                    initial_press = :initial_press AND
                    equivalence = :equivalence AND
                    fuel = :fuel AND
                    oxidizer = :oxidizer AND
                    diluent = :diluent AND
                    diluent_mol_frac = :diluent_mol_frac AND
                    reaction_number = :reaction_number;
                """.format(self.table_name),
                {
                    'mechanism': mechanism,
                    'initial_temp': initial_temp,
                    'initial_press': initial_press,
                    'equivalence': equivalence,
                    'fuel': fuel,
                    'oxidizer': oxidizer,
                    'diluent': str(diluent),
                    'diluent_mol_frac': diluent_mol_frac,
                    'reaction_number': reaction_number,
                    'k_i': k_i,
                    'cj_speed': cj_speed
                }
            )

    def store_row(
            self,
            mechanism,
            initial_temp,
            initial_press,
            equivalence,
            fuel,
            oxidizer,
            k_i,
            cj_speed,
            reaction_number=-1,
            diluent='None',
            diluent_mol_frac=0,
            overwrite_existing=False
    ):
        if self.check_existing_row(
            mechanism=mechanism,
            initial_temp=initial_temp,
            initial_press=initial_press,
            equivalence=equivalence,
            fuel=fuel,
            oxidizer=oxidizer,
            reaction_number=reaction_number,
            diluent=diluent,
            diluent_mol_frac=diluent_mol_frac
        ):
            # a rew with the current information was found
            if overwrite_existing:
                self._update_row(
                    mechanism=mechanism,
                    initial_temp=initial_temp,
                    initial_press=initial_press,
                    equivalence=equivalence,
                    fuel=fuel,
                    oxidizer=oxidizer,
                    reaction_number=reaction_number,
                    diluent=diluent,
                    diluent_mol_frac=diluent_mol_frac,
                    cj_speed=cj_speed,
                    k_i=k_i
                )
                start_color = '\033[92m'
                end_color = '\033[0m'
                print(start_color+'data row stored successfully'+end_color)
            else:
                # warn the user that the current input was ignored
                warnings.warn(
                    'Cannot overwrite row unless overwrite_existing=True'
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
                        :equivalence,
                        :fuel,
                        :oxidizer,
                        :diluent,
                        :diluent_mol_frac,
                        :reaction_number,
                        :k_i,
                        :cj_speed
                    );
                    """.format(self.table_name),
                    {
                        'mechanism': mechanism,
                        'initial_temp': initial_temp,
                        'initial_press': initial_press,
                        'equivalence': equivalence,
                        'fuel': fuel,
                        'oxidizer': oxidizer,
                        'diluent': diluent,
                        'diluent_mol_frac': diluent_mol_frac,
                        'reaction_number': reaction_number,
                        'k_i': k_i,
                        'cj_speed': cj_speed
                    }
                )

    # noinspection PyUnusedLocal
    @staticmethod
    def _build_query_str(
            mechanism,
            initial_temp,
            initial_press,
            equivalence,
            fuel,
            oxidizer,
            reaction_number,
            diluent,
            diluent_mol_frac
    ):
        inputs = {
            key: value for key, value
            in inspect.getargvalues(inspect.currentframe())[3].items()
            if value is not None
        }
        if len(inputs) > 0:
            where = ' WHERE '
        else:
            where = ''
        sql_varnames = [
            '{:s} = :{:s}'.format(*[item]*2) for item in inputs.keys()
        ]
        cmd_str = 'SELECT * FROM {:s} ' + where +\
                  ' AND '.join(sql_varnames) + ';'
        return cmd_str

    def fetch_rows(
            self,
            mechanism=None,
            initial_temp=None,
            initial_press=None,
            equivalence=None,
            fuel=None,
            oxidizer=None,
            reaction_number=None,
            diluent=None,
            diluent_mol_frac=None
    ):
        with self.con as con:
            cur = con.cursor()
            cmd_str = self._build_query_str(
                mechanism=mechanism,
                initial_temp=initial_temp,
                initial_press=initial_press,
                equivalence=equivalence,
                fuel=fuel,
                oxidizer=oxidizer,
                reaction_number=reaction_number,
                diluent=diluent,
                diluent_mol_frac=diluent_mol_frac
            )
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
                    'reaction_number': reaction_number
                }
            )
            info = cur.fetchall()
            labels = [item[1] for item in self.list_all_headers()]
            data = {l: [] for l in labels}
            for row in info:
                for l, d in zip(labels, row):
                    data[l].append(d)

            # row = {key: value for key, value in zip(labels, info)}
            return data


if __name__ == '__main__':
    from pprint import pprint

    db_str = 'test.sqlite'
    table_str = 'test_table'

    test = Table(db_str, table_str)
    test.store_row('test.cti', 300, 101325, 1.15, 'CH4', 'N2O', 1.2e12,
                   2043.87987987987987)
    test.store_row('test.cti', 300, 101325, 1.15, 'CH4', 'N2O', 1.2e12,
                   2043.87987987987987, 0)
    test.store_row('test.cti', 300, 101325, 1.15, 'CH4', 'N2O', 1.2e12,
                   2043.87987987987987, 1, overwrite_existing=True)
    pprint(test.fetch_rows())
    pprint(test.fetch_rows(reaction_number=-1))
