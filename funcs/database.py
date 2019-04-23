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


def clean(table_name):
    """
    Cleans a table name string to keep me from doing anything too stupid.
    Alphanumeric values and underscores are allowed; anything else will throw
    a NameError.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    str
    """
    if any([not (char.isalnum() or char == '_') for char in table_name]):
        raise NameError(
            'Table name must be entirely alphanumeric. Underscores are allowed.'
        )
    else:
        return table_name.lower()


def list_all_tables(database):
    with sqlite3.connect(database) as con:
        cur = con.cursor()
        cur.execute("""SELECT name FROM sqlite_master WHERE type='table';""")

    table_list = cur.fetchall()

    con.close()
    return table_list


def list_table_headers(database, table_name):
    table_name = clean(table_name)
    with sqlite3.connect(database) as con:
        cur = con.cursor()
        cur.execute("""PRAGMA table_info({:s});""".format(table_name))

    table_info = cur.fetchall()

    con.close()
    return table_info


def create_base_table(
        database,
        table_name
):
    table_name = clean(table_name)
    with sqlite3.connect(database) as con:
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
    con.close()


def create_perturbed_table(
        database,
        table_name
):
    table_name = clean(table_name)
    with sqlite3.connect(database) as con:
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
    con.close()


def check_existing_perturbed_row(
        database,
        table_name,
        mechanism,
        initial_temp,
        initial_press,
        equivalence,
        fuel,
        oxidizer,
        reaction_number,
        diluent,
        diluent_mol_frac):
    table_name = clean(table_name)
    with sqlite3.connect(database) as con:
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
    con.close()
    return row_found


def update_perturbed_row(
        database,
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
        diluent,
        diluent_mol_frac
):
    table_name = clean(table_name)
    with sqlite3.connect(database) as con:
        cur = con.cursor()
        cur.execute(
            """
            UPDATE {:s} SET 
                date_stored = datetime('now', 'localtime') AND
                k_i = :k_i AND
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
    con.close()


def store_perturbed_row(
        database,
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
    table_name = clean(table_name)

    if check_existing_perturbed_row(
        database=database,
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
            update_perturbed_row(
                database=database,
                table_name=table_name,
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
        with sqlite3.connect(database) as con:
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

        con.close()


if __name__=='__main__':
    import os

    db_str = 'test.db'
    table_str = 'test_table'

    if not os.path.exists(db_str):
        create_perturbed_table(db_str, table_str)

    print(list_all_tables(db_str))
    print(list_table_headers(db_str, table_str))
    store_perturbed_row(
        database=db_str,
        table_name=table_str,
        mechanism='test.cti',
        initial_temp=300,
        initial_press=101325,
        equivalence=1,
        fuel='CH4',
        oxidizer='N2O',
        reaction_number=0,
        k_i=1.2,
        cj_speed=2000,
        overwrite_existing=True
    )
    store_perturbed_row(
        database=db_str,
        table_name=table_str,
        mechanism='test.cti',
        initial_temp=300,
        initial_press=101325,
        equivalence=1,
        fuel='CH4',
        oxidizer='N2O',
        reaction_number=0,
        k_i=1.2e12,
        cj_speed=1829.235423456346346345
    )


    # os.remove(db_str)
