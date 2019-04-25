import funcs.database as db
from uuid import uuid4
import string
import random
import os
import sqlite3
import pytest
import inspect


def remove_stragglers():
    stragglers = set(file for file in os.listdir('.') if '.sqlite' in file)
    for file in stragglers:
        os.remove(file)


class TestDataBase:
    @staticmethod
    def test_list_all_tables():
        num_tables = 16
        num_chars = 16
        test_db = str(uuid4()) + '.sqlite'
        table_names = [
            ''.join([random.choice(string.ascii_letters)
                     for _ in range(num_chars)])
            for _ in range(num_tables)
        ]
        with sqlite3.connect(test_db) as con:
            cur = con.cursor()
            for name in table_names:
                cur.execute(
                    """
                    CREATE TABLE {:s} (
                        test TEXT
                    );
                    """.format(name)
                )
        con.close()

        test_names = db.DataBase.list_all_tables(test_db)

        os.remove(test_db)

        assert all([
            test == good for test, good in zip(test_names, table_names)
        ])


# noinspection PyProtectedMember
class TestTable:
    @staticmethod
    def test__clean_bad_input():
        num_chars = 16
        dirty_string = ''.join([
            random.choice(string.ascii_uppercase + string.punctuation)
            for _ in range(num_chars)
        ])
        with pytest.raises(
            NameError,
            match='Table name must be entirely alphanumeric. '
                  'Underscores are allowed.'
        ):
            db.Table._clean(dirty_string)

    @staticmethod
    def test__clean_good_input():
        num_chars = 16
        good_stuff = set(string.ascii_lowercase)
        dirty_string = ''.join([
            random.choice(string.ascii_uppercase)
            for _ in range(num_chars)
        ])
        test_out = set(db.Table._clean(dirty_string))
        assert not test_out.intersection(good_stuff).difference(test_out)

    @staticmethod
    def test__build_query_str():
        # test empty, one input, and two inputs
        good_results = [
            'SELECT * FROM {:s};',
            'SELECT * FROM {:s} WHERE initial_temp = :initial_temp;',
            'SELECT * FROM {:s} WHERE initial_temp = :initial_temp '
            'AND fuel = :fuel;',
        ]
        sig = inspect.signature(db.Table._build_query_str)
        base_inputs = {item: None for item in sig.parameters.keys()}
        inputs = [
            base_inputs,
            {**base_inputs, 'initial_temp': 300},
            {**base_inputs, 'initial_temp': 300, 'fuel': 'CH4'}
        ]
        test_results = [db.Table._build_query_str(**kw) for kw in inputs]
        # dict order is unreliable; use sorted() to compare strings
        assert all([
            ''.join(sorted(test)) == ''.join(sorted(good))
            for test, good in zip(test_results, good_results)
        ])

    @staticmethod
    def test_columns():
        num_cols = 16
        num_chars = 16
        test_db = str(uuid4()) + '.sqlite'
        column_names = [
            ''.join([random.choice(string.ascii_letters)
                     for _ in range(num_chars)])
            for _ in range(num_cols)
        ]
        test_table_name = 'test_table'
        header = 'CREATE TABLE {:s} ('.format(test_table_name)
        footer = ' TEXT);'
        sep = ' TEXT, '

        with sqlite3.connect(test_db) as con:
            cur = con.cursor()
            cur.execute(
                header + sep.join(column_names) + footer
            )
        con.close()

        class TestSelf:
            database = test_db
            table_name = test_table_name
            columns = db.Table.columns

        test_table = TestSelf()
        test_columns = test_table.columns()
        os.remove(test_db)
        assert all([
            test == good for test, good in zip(test_columns, column_names)
        ])


if __name__ == '__main__':
    import subprocess
    try:
        subprocess.check_call('pytest test_database.py -vv --noconftest --cov')
    except subprocess.CalledProcessError as e:
        remove_stragglers()
        raise e
