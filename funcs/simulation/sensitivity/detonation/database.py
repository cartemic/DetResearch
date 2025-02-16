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

import dataclasses
import sqlite3
from functools import lru_cache
from typing import List, Optional

import cantera as ct


class DatabaseError(Exception):
    pass


@dataclasses.dataclass
class TestConditions:
    mechanism: str
    initial_temp: float
    initial_press: float
    fuel: str
    oxidizer: str
    equivalence: float
    diluent: str
    diluent_mol_frac: float
    cj_speed: Optional[float] = None
    ind_len_west: Optional[float] = None
    ind_len_gav: Optional[float] = None
    ind_len_ng: Optional[float] = None
    cell_size_west: Optional[float] = None
    cell_size_gav: Optional[float] = None
    cell_size_ng: Optional[float] = None
    _test_id: Optional[int] = None

    @property
    def test_id(self):
        return self._test_id

    def to_dict(self):
        d = dataclasses.asdict(self)
        return {k.lstrip("_"): v for (k, v) in d.items()}

    def needs_cj_calc(self) -> bool:
        return self.cj_speed is None

    def needs_cell_size_calc(self) -> bool:
        return None in (
            self.ind_len_ng,
            self.ind_len_gav,
            self.ind_len_west,
            self.cell_size_ng,
            self.cell_size_gav,
            self.cell_size_west,
        )


@dataclasses.dataclass
class PerturbedResults:
    test_id: int
    rxn_no: int
    perturbation_fraction: float
    rxn: str
    k_i: float
    ind_len_west: float
    ind_len_gav: float
    ind_len_ng: float
    cell_size_west: float
    cell_size_gav: float
    cell_size_ng: float
    sens_ind_len_west: float
    sens_ind_len_gav: float
    sens_ind_len_ng: float
    sens_cell_size_west: float
    sens_cell_size_gav: float
    sens_cell_size_ng: float

    def to_dict(self):
        d = dataclasses.asdict(self)
        return dict(d.items())


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("select name from sqlite_master where type='table' and name=:name", {"name": name})

    return cur.fetchone() is not None


class DataBase:  # todo: implement these changes throughout the code base
    """
    A class for database-level operations
    """

    def __init__(self, path: str):
        self._path = path
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.test_conditions_table = TestConditionsTable(con=self.con)
        self.base_rxn_table = BaseReactionTable(con=self.con)
        self.perturbed_results_table = PerturbedResultsTable(
            con=self.con,
            base_rxn_table=self.base_rxn_table,
            test_conditions_table=self.test_conditions_table,
        )

    def __del__(self):
        self.con.commit()
        self.con.close()

    def new_test(self, test_conditions: TestConditions) -> TestConditions:
        if test_conditions.test_id is not None:
            raise DatabaseError("Cannot create a new test unless test_conditions has test_id = None")

        if not self.base_rxn_table.has_mechanism(mechanism=test_conditions.mechanism):
            self.base_rxn_table.store_all_reactions(
                gas=ct.Solution(test_conditions.mechanism),
                mechanism=test_conditions.mechanism,
            )

        return self.test_conditions_table.insert_new_row(test_conditions=test_conditions)


class BaseReactionTable:
    name = "rxn_base"

    def __init__(self, con: sqlite3.Connection):
        self.cur = con.cursor()
        if not _table_exists(self.cur, self.name):
            self._create()

    def _create(self):
        """
        Creates a table of base (unperturbed) reactions and their rate constants
        in the current database
        """
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                mechanism TEXT,
                rxn_no INTEGER,
                rxn TEXT,
                k_i REAL,
                PRIMARY KEY (mechanism, rxn_no)
            );
            """
        )

    @property
    @lru_cache(maxsize=None)
    def columns(self) -> List[str]:
        """
        A list of all column names in the current table.
        """
        self.cur.execute(f"PRAGMA table_info({self.name});")

        return [item[1] for item in self.cur.fetchall()]

    def store_all_reactions(self, gas: ct.Solution, mechanism: str):
        self.cur.execute(
            f"DELETE FROM {self.name} WHERE mechanism = :mechanism",
            {"mechanism": mechanism},
        )
        for rxn_no, rxn, k_i in zip(range(gas.n_reactions), gas.reaction_equations(), gas.forward_rate_constants):
            self.cur.execute(
                f"""
                INSERT INTO {self.name} VALUES (
                    :mechanism,
                    :rxn_no,
                    :rxn,
                    :k_i
                );
                """,
                {
                    "mechanism": mechanism,
                    "rxn_no": rxn_no,
                    "rxn": rxn,
                    "k_i": k_i,
                },
            )
            self.cur.connection.commit()

    def has_mechanism(self, mechanism: str) -> bool:
        self.cur.execute(
            f"SELECT COUNT(*) > 0 has_some FROM {self.name} WHERE mechanism = :mechanism",
            {"mechanism": mechanism},
        )

        return bool(self.cur.fetchone()["has_some"])


# Allow manipulation of _test_id
# ruff: noqa: SLF001
class TestConditionsTable:
    name = "test_conditions"

    def __init__(self, con: sqlite3.Connection):
        self.cur = con.cursor()
        if not _table_exists(self.cur, self.name):
            self._create()

    def _create(self):
        """
        Creates a table of test conditions and results in the current database
        """
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                test_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
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
                cell_size_ng REAL
            );
            """
        )

    @property
    @lru_cache(maxsize=None)
    def columns(self) -> List[str]:
        """
        A list of all column names in the current table.
        """
        self.cur.execute(f"PRAGMA table_info({self.name});")

        return [item[1] for item in self.cur.fetchall()]

    def test_exists(self, test_id: Optional[int]) -> bool:
        """
        Checks the current table for a specific row of data

        Parameters
        ----------
        test_id
            Test ID corresponding to updated row
        """
        self.cur.execute(f"SELECT * from {self.name} WHERE test_id = :test_id", {"test_id": test_id})

        return len(self.cur.fetchall()) > 0

    @staticmethod
    def _row_to_test_conditions(row: sqlite3.Row) -> TestConditions:
        return TestConditions(
            _test_id=row["test_id"],
            mechanism=row["mechanism"],
            initial_temp=row["initial_temp"],
            initial_press=row["initial_press"],
            fuel=row["fuel"],
            oxidizer=row["oxidizer"],
            equivalence=row["equivalence"],
            diluent=row["diluent"],
            diluent_mol_frac=row["diluent_mol_frac"],
            cj_speed=row["cj_speed"],
            ind_len_west=row["ind_len_west"],
            ind_len_gav=row["ind_len_gav"],
            ind_len_ng=row["ind_len_ng"],
            cell_size_west=row["cell_size_west"],
            cell_size_gav=row["cell_size_gav"],
            cell_size_ng=row["cell_size_ng"],
        )

    def fetch_rows(self, test_ids: List[int]) -> List[TestConditions]:
        cleaned_ids = []
        for test_id in test_ids:
            if not isinstance(test_id, int):
                raise DatabaseError("Test IDs must be integers.")
            cleaned_ids.append(str(test_id))
        self.cur.execute(f"SELECT * FROM {self.name} where test_id in ({','.join(cleaned_ids)})")

        return [self._row_to_test_conditions(row) for row in self.cur.fetchall()]

    def fetch_row(self, test_id: int) -> TestConditions:
        if not isinstance(test_id, int):
            raise DatabaseError("Test IDs must be integers.")
        self.cur.execute(f"SELECT * FROM {self.name} where test_id = {test_id}")

        return self._row_to_test_conditions(self.cur.fetchone())

    def insert_new_row(self, test_conditions: TestConditions) -> TestConditions:
        """
        Stores a row of test data in the current table.
        """
        if test_conditions.test_id is not None:
            raise DatabaseError("Cannot create a new test unless test_conditions has test_id = None")

        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                Null,
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
                :cell_size_ng
            );
            """,
            test_conditions.to_dict(),
        )
        self.cur.connection.commit()
        test_conditions._test_id = self.cur.lastrowid

        return test_conditions

    def update_row(self, test_conditions: TestConditions):
        """
        Updates the following items in the database:

        * cj_speed
        * ind_len_west
        * ind_len_gav
        * ind_len_ng
        * cell_size_west
        * cell_size_gav
        * cell_size_ng
        """
        self.cur.execute(
            f"""
            UPDATE {self.name} SET
                date_stored = datetime('now', 'localtime'),
                cj_speed = :cj_speed,
                ind_len_west = :ind_len_west,
                ind_len_gav = :ind_len_gav,
                ind_len_ng = :ind_len_ng,
                cell_size_west = :cell_size_west,
                cell_size_gav = :cell_size_gav,
                cell_size_ng = :cell_size_ng
            WHERE
                test_id = :test_id
            """,
            test_conditions.to_dict(),
        )
        self.cur.connection.commit()


class PerturbedResultsTable:
    name = "perturbed_results"

    def __init__(
        self,
        con: sqlite3.Connection,
        base_rxn_table: BaseReactionTable,
        test_conditions_table: TestConditionsTable,
        testing=False,
    ):
        self.cur = con.cursor()
        self._testing = testing
        self._base_rxn_table = base_rxn_table
        self._test_conditions_table = test_conditions_table
        if not _table_exists(self.cur, self.name):
            self._create()

    def _create(self):
        """
        Creates a table of perturbed reaction results in the current database
        """
        self.cur.execute(
            f"""
            CREATE TABLE {self.name} (
                test_id INTEGER,
                rxn_no INTEGER,
                stored_date TEXT,
                perturbation_fraction REAL,
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
                sens_cell_size_ng REAL,
                PRIMARY KEY (test_id, rxn_no)
            );
            """
        )

    @property
    @lru_cache(maxsize=None)
    def columns(self):
        """
        A list of all column names in the current table.
        """
        self.cur.execute(f"PRAGMA table_info({self.name});")

        return [item[1] for item in self.cur.fetchall()]

    def row_exists(self, test_id: int, rxn_no: int):
        """
        Checks the current table for a specific row of data
        """
        self.cur.execute(
            f"SELECT * from {self.name} WHERE (test_id, rxn_no) = (:test_id, :rxn_no)",
            {"test_id": test_id, "rxn_no": rxn_no},
        )

        return len(self.cur.fetchall()) > 0

    def insert_new_row(self, perturbed_results: PerturbedResults):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :test_id,
                :rxn_no,
                datetime('now', 'localtime'),
                :perturbation_fraction,
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
            """,
            perturbed_results.to_dict(),
        )
        self.cur.connection.commit()

    def update_row(self, perturbed_results: PerturbedResults):
        self.cur.execute(
            f"""
            UPDATE {self.name} SET
                perturbation_fraction = :perturbation_fraction,
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
                (test_id, rxn_no) = (:test_id, :rxn_no)
            """,
            perturbed_results.to_dict(),
        )
        self.cur.connection.commit()

    @staticmethod
    def _row_to_perturbed_results(row: sqlite3.Row) -> PerturbedResults:
        return PerturbedResults(
            test_id=row["test_id"],
            rxn_no=row["rxn_no"],
            perturbation_fraction=row["perturbation_fraction"],
            rxn=row["rxn"],
            k_i=row["k_i"],
            ind_len_west=row["ind_len_west"],
            ind_len_gav=row["ind_len_gav"],
            ind_len_ng=row["ind_len_ng"],
            cell_size_west=row["cell_size_west"],
            cell_size_gav=row["cell_size_gav"],
            cell_size_ng=row["cell_size_ng"],
            sens_ind_len_west=row["sens_ind_len_west"],
            sens_ind_len_gav=row["sens_ind_len_gav"],
            sens_ind_len_ng=row["sens_ind_len_ng"],
            sens_cell_size_west=row["sens_cell_size_west"],
            sens_cell_size_gav=row["sens_cell_size_gav"],
            sens_cell_size_ng=row["sens_cell_size_ng"],
        )

    def fetch_rows(self, test_id: int) -> List[PerturbedResults]:
        """
        Fetches all rows from the current test.
        """
        self.cur.execute(f"SELECT * FROM {self.name} WHERE test_id = :test_id", {"test_id": test_id})

        return [self._row_to_perturbed_results(row) for row in self.cur.fetchall()]

    def fetch_row(self, test_id: int, rxn_no: int) -> PerturbedResults:
        """
        Fetches a single reaction row from the current test.
        """
        self.cur.execute(
            f"SELECT * FROM {self.name} WHERE (test_id, rxn_no) = (:test_id, :rxn_no)",
            {"test_id": test_id, "rxn_no": rxn_no},
        )

        return self._row_to_perturbed_results(self.cur.fetchone())
