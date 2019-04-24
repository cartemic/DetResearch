import funcs.database as db
from uuid import uuid4
import string
import random
import os
import sqlite3


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


if __name__ == '__main__':
    import subprocess
    subprocess.check_call('pytest test_database.py -vv --noconftest --cov')
