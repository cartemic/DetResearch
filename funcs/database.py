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


class _AllDataBases:

    @staticmethod
    def list_all_tables(database):
        with sqlite3.connect(database) as con:
            cur = con.cursor()
            cur.execute(
                """SELECT name FROM sqlite_master WHERE type='table';"""
            )

        table_list = cur.fetchall()
        con.close()
        return table_list


class _AllTables:
    def __init__(
            self,
            database,
            table_name
    ):
        self.table_name = self._clean(table_name)
        self.database = database
        self.con = sqlite3.connect(self.database)
        self._create(self.table_name)

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
            table_name,
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
        table_name = self._clean(table_name)
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
                """.format(table_name),
                {
                    'mechanism': mechanism,
                    'initial_temp': initial_temp,
                    'initial_press': initial_press,
                    'equivalence': equivalence,
                    'fuel': fuel,
                    'oxidizer': oxidizer,
                    'diluent': str(diluent),
                    'diluent_mol_frac': diluent_mol_frac,
                    'reaction_number': reaction_number
                }
            )
            if len(cur.fetchall()) > 0:
                row_found = True
            else:
                row_found = False
        return row_found


class BaseTable(_AllTables):
    def _create(
            self,
            table_name
    ):
        table_name = self._clean(table_name)
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
                    cj_speed REAL   
                );
                """.format(table_name)
            )


class PerturbedTable(_AllTables):
    def _create(
            self,
            table_name
    ):
        table_name = self._clean(table_name)
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
                """.format(table_name)
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
            k_i,
            cj_speed,
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
            table_name,
            mechanism,
            initial_temp,
            initial_press,
            equivalence,
            fuel,
            oxidizer,
            reaction_number,
            k_i,
            cj_speed,
            diluent=None,
            diluent_mol_frac=0,
            overwrite_existing=False
    ):
        table_name = self._clean(table_name)

        if self.check_existing_row(
            table_name=table_name,
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
                print('ayy')
            else:
                # warn the user that the current input was ignored
                print('aw crap man')

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
                    """.format(table_name),
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


if __name__ == '__main__':
    db_str = 'test.sqlite'
    table_str = 'test_table'
