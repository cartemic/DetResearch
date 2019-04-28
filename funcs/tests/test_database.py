import funcs.database as db
from uuid import uuid4
import string
import random
import os
import sqlite3
import pytest
import inspect
from numpy import isclose, allclose


def remove_stragglers():
    stragglers = set(file for file in os.listdir('.') if '.sqlite' in file)
    for file in stragglers:
        os.remove(file)


def generate_db_name():
    return str(uuid4()) + '.sqlite'


def bind(instance, func, as_name=None):
    """
    https://stackoverflow.com/questions/1015307/python-bind-an-unbound-method

    Bind the function *func* to *instance*, with either provided name *as_name*
    or the existing name of *func*. The provided *func* should accept the
    instance as the first argument, i.e. "self".
    """
    if as_name is None:
        as_name = func.__name__
    bound_method = func.__get__(instance, instance.__class__)
    setattr(instance, as_name, bound_method)
    return bound_method


class TestDataBase:
    def test_list_all_tables(self):
        num_tables = 16
        num_chars = 16
        test_db = generate_db_name()
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


# noinspection PyProtectedMember,PyUnresolvedReferences
class TestTable:
    keys = {
        'date_stored',
        'mechanism',
        'initial_temp',
        'initial_press',
        'equivalence',
        'fuel',
        'oxidizer',
        'diluent',
        'diluent_mol_frac',
        'reaction_number',
        'k_i',
        'cj_speed'
    }

    # noinspection PyProtectedMember,PyUnresolvedReferences
    class FakeTable:
        """
        Fake table object for testing methods independently
        """
        def __init__(
                self,
                test_db,
                test_table_name,
                allow_create=True
        ):
            self.database = test_db
            self.table_name = test_table_name
            self.con = sqlite3.connect(self.database)
            if allow_create and self.table_name not in\
                    db.DataBase.list_all_tables(test_db):
                bind(self, db.Table._create_test_table)
                self._create()

        def __del__(self):
            self.con.commit()
            self.con.close()
            os.remove(self.database)

    def test__clean_bad_input(self):
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

    def test__clean_good_input(self):
        num_chars = 16
        good_stuff = set(string.ascii_lowercase)
        dirty_string = ''.join([
            random.choice(string.ascii_uppercase)
            for _ in range(num_chars)
        ])
        test_out = set(db.Table._clean(dirty_string))
        assert not test_out.intersection(good_stuff).difference(test_out)

    def test__build_query_str(self):
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

    def test_columns(self):
        num_cols = 16
        num_chars = 16
        test_db = generate_db_name()
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

        test_table = self.FakeTable(
            test_db,
            test_table_name,
            allow_create=False
            )
        bind(test_table, db.Table.columns)
        test_columns = test_table.columns()
        assert all([
            test == good for test, good in zip(test_columns, column_names)
        ])

    def test__create(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        FakeTable = self.FakeTable
        FakeTable.columns = db.Table.columns
        test_table = FakeTable(
            test_db,
            test_table_name,
            allow_create=True
            )
        actual_columns = set(test_table.columns())
        assert not self.keys.difference(actual_columns)

    def test_store_and_check_existing(self):
        kwargs = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1
        }
        test_db = generate_db_name()
        test_table_name = 'test_table'
        FakeTable = self.FakeTable
        test_table = FakeTable(
            test_db,
            test_table_name,
            allow_create=True
        )
        bind(test_table, db.Table.store_row)
        bind(test_table, db.Table.check_existing_test)
        test_table.store_row(**{**kwargs, 'k_i': 123, 'cj_speed': 456})
        assert test_table.check_existing_row(**kwargs)

    def test_fetch_rows_blank_table(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        test_rows = test_table.fetch_rows()
        assert all([
            not self.keys.difference(set(test_rows.keys())),
            *[item == [] for item in test_rows.values()]
        ])

    def test_fetch_rows_single(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'k_i': 1.2e12,
            'cj_speed': 1986.12354679687543
        }
        test_table.store_row(**kwargs)
        test_rows = test_table.fetch_rows()
        checks = []
        for key, value in test_rows.items():
            if 'date' not in key:
                if isinstance(kwargs[key], str) or isinstance(kwargs[key], int):
                    checks.append(kwargs[key] == value[0])
                else:
                    checks.append(isclose(kwargs[key], value[0]))
        assert all(checks)

    def test_fetch_rows_multiple(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs = [{
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'k_i': 1.2e12,
            'cj_speed': 1986.12354679687543
        }, {
            'mechanism': 'gri30.cti',
            'initial_temp': 900,
            'initial_press': 202156,
            'equivalence': 2,
            'fuel': 'C3H8',
            'oxidizer': 'O2:1 N2:3.76',
            'reaction_number': 12,
            'diluent': 'None',
            'diluent_mol_frac': 0,
            'k_i': 1.34e12,
            'cj_speed': 1467.546546539687543
        }]
        for kw in kwargs:
            test_table.store_row(**kw)
        test_rows = test_table.fetch_rows()
        checks = []
        good_answer = {
            key: [value, kwargs[1][key]] for key, value in kwargs[0].items()
        }
        for key, value in test_rows.items():
            if 'date' not in key:
                if (isinstance(kwargs[0][key], str)
                        or isinstance(kwargs[0][key], int)):
                    for good, test in zip(good_answer[key], test_rows[key]):
                        checks.append(test == good)
                else:
                    checks.append(allclose(good_answer[key], value))
        assert all(checks)

    def test__update_row(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs_init = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'k_i': 1.2e12,
            'cj_speed': 1986.12354679687543
        }
        kwargs_repl = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'k_i': 1.34e12,
            'cj_speed': 1467.546546539687543
        }
        test_table.store_row(**kwargs_init)
        test_table._update_row(**kwargs_repl)
        test_rows = test_table.fetch_rows()
        checks = []
        for key, value in test_rows.items():
            if 'date' not in key:
                if (isinstance(kwargs_repl[key], str)
                        or isinstance(kwargs_repl[key], int)):
                    checks.append(kwargs_repl[key] == value[0])
                else:
                    checks.append(isclose(kwargs_repl[key], value[0]))
        assert all(checks)

    def test_store_row_update(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs_init = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'k_i': 1.2e12,
            'cj_speed': 1986.12354679687543
        }
        kwargs_repl = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'k_i': 1.34e12,
            'cj_speed': 1467.546546539687543
        }
        test_table.store_row(**kwargs_init)
        test_table.store_row(**kwargs_repl, overwrite_existing=True)
        test_rows = test_table.fetch_rows()
        checks = []
        for key, value in test_rows.items():
            if 'date' not in key:
                if (isinstance(kwargs_repl[key], str)
                        or isinstance(kwargs_repl[key], int)):
                    checks.append(kwargs_repl[key] == value[0])
                else:
                    checks.append(isclose(kwargs_repl[key], value[0]))
        assert all(checks)

    @pytest.mark.filterwarnings('ignore')
    def test_store_row_update_no_overwrite(self):
        test_db = generate_db_name()
        test_table_name = 'test_table'
        test_table = db.Table(
            test_db,
            test_table_name,
            testing=True
            )
        kwargs_init = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'k_i': 1.2e12,
            'cj_speed': 1986.12354679687543
        }
        kwargs_repl = {
            'mechanism': 'gri30.cti',
            'initial_temp': 300,
            'initial_press': 101325,
            'equivalence': 1,
            'fuel': 'CH4',
            'oxidizer': 'N2O',
            'reaction_number': 0,
            'diluent': 'N2',
            'diluent_mol_frac': 0.1,
            'k_i': 1.34e12,
            'cj_speed': 1467.546546539687543
        }
        test_table.store_row(**kwargs_init)
        test_table.store_row(**kwargs_repl, overwrite_existing=False)
        test_rows = test_table.fetch_rows()

        checks = []
        for key, value in test_rows.items():
            if 'date' not in key:
                if (isinstance(kwargs_init[key], str)
                        or isinstance(kwargs_init[key], int)):
                    checks.append(kwargs_init[key] == value[0])
                else:
                    checks.append(isclose(kwargs_init[key], value[0]))
        assert all(checks)


if __name__ == '__main__':  # pragma: no cover
    import subprocess
    try:
        subprocess.check_call('pytest test_database.py -vv --noconftest --cov')
    except subprocess.CalledProcessError as e:
        # clean up in case of an unexpected error cropping up
        remove_stragglers()
        raise e
