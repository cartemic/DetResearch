import dataclasses
import sqlite3
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

from cantera import Species
from retry import retry


class DatabaseError(Exception):
    pass


class TableName(Enum):
    Conditions: str = "conditions"
    Reactions: str = "reactions"
    Species: str = "species"
    BulkProperties: str = "bulk_properties"


class SimulationType(Enum):
    Znd: str = "znd"
    Cv: str = "cv"


class SqliteDataBase:
    def __init__(self, path: Union[str, Path], timeout: float = 600):
        self.path = path
        self.timeout = timeout
        self.con = self.connect()
        self.con.row_factory = sqlite3.Row

    def __del__(self):
        try:
            self.con.commit()
        except sqlite3.ProgrammingError:
            self.con = self.connect()
        self.con.close()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=self.timeout)


class SqliteTable:
    def __init__(self, db: SqliteDataBase, table_name: str, clear_existing_data: bool = False):
        self.db = db
        self.cur = self.db.con.cursor()
        self.name = table_name
        if self.table_exists() and clear_existing_data:
            self.__clear()

    def table_exists(self) -> bool:
        self.cur.execute("select name from sqlite_master where type='table' and name=:name", {"name": self.name})
        return self.cur.fetchone() is not None

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __clear(self):
        self.cur.execute(
            f"""
            DROP TABLE IF EXISTS {self.name};
            """
        )


@dataclasses.dataclass
class Conditions:
    sim_type: str
    mech: str
    match: Optional[str]
    dil_condition: str
    initial_temp: float
    initial_press: float
    fuel: str
    oxidizer: str
    equivalence: float
    phi_nom: float
    diluent: Optional[str]
    dil_mf: float


class ConditionTable(SqliteTable):
    def __init__(self, db: SqliteDataBase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Conditions.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                start DATETIME NOT NULL,
                end DATETIME,
                sim_type TEXT NOT NULL,
                mech TEXT NOT NULL,
                match TEXT,
                dil_condition TXT NOT NULL,
                initial_temp REAL NOT NULL,
                initial_press REAL NOT NULL,
                fuel TEXT NOT NULL,
                oxidizer TEXT NOT NULL,
                equivalence REAL NOT NULL,
                phi_nom REAL NOT NULL,
                diluent TEXT,
                dil_mf REAL NOT NULL,
                temp_vn REAL,
                t_ind REAL,
                u_znd REAL,
                u_cj REAL,
                cell_size REAL,
                cell_size_2 REAL
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def insert(self, test_conditions: Conditions) -> int:
        """
        Stores a row of test data in the current table.
        """

        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                Null,
                CURRENT_TIMESTAMP,
                Null,
                :sim_type,
                :mech,
                :match,
                :dil_condition,
                :initial_temp,
                :initial_press,
                :fuel,
                :oxidizer,
                :equivalence,
                :phi_nom,
                :diluent,
                :dil_mf,
                Null,
                Null,
                Null,
                Null,
                Null,
                Null
            );
            """,
            dataclasses.asdict(test_conditions),
        )
        self.cur.connection.commit()
        return self.cur.lastrowid


@dataclasses.dataclass
class BulkPropertiesData:
    condition_id: int
    run_no: int
    time: float
    temperature: float
    pressure: float
    cp: float
    cv: float
    temperature_gradient: Optional[float] = None
    velocity: Optional[float] = None

    @cached_property
    def gamma(self) -> float:
        return self.cp / self.cv


class BulkPropertiesTable(SqliteTable):
    def __init__(self, db: SqliteDataBase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.BulkProperties.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                condition_id INTEGER NOT NULL,
                run_no INTEGER NOT NULL,
                time REAL NOT NULL,
                temperature REAL NOT NULL,
                temperature_gradient REAL,
                pressure REAL NOT NULL,
                cp REAL NOT NULL,
                cv REAL NOT NULL,
                gamma REAL NOT NULL,
                velocity REAL,
                PRIMARY KEY (condition_id, run_no, time),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def insert_or_update(self, data: BulkPropertiesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :condition_id,
                :run_no,
                :time,
                :temperature,
                :temperature_gradient,
                :pressure,
                :cp,
                :cv,
                :gamma,
                :velocity
            )
            ON CONFLICT(condition_id, run_no, time) DO UPDATE SET
                temperature=excluded.temperature,
                pressure=excluded.pressure,
                velocity=excluded.velocity;
            """,
            {
                "condition_id": data.condition_id,
                "run_no": data.run_no,
                "time": data.time,
                "temperature": data.temperature,
                "temperature_gradient": data.temperature_gradient,
                "pressure": data.pressure,
                "cp": data.cp,
                "cv": data.cv,
                "gamma": data.gamma,
                "velocity": data.velocity,
            },
        )
        if commit:
            self.cur.connection.commit()


@dataclasses.dataclass
class ReactionData:
    condition_id: int
    run_no: int
    time: float
    reaction: str
    fwd_rate_constant: float
    fwd_rate_of_progress: float
    rev_rate_constant: float
    rev_rate_of_progress: float
    net_rate_of_progress: float


class ReactionTable(SqliteTable):
    def __init__(self, db: SqliteDataBase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Reactions.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                condition_id INTEGER NOT NULL,
                run_no INTEGER NOT NULL,
                time REAL NOT NULL,
                reaction TEXT NOT NULL,
                fwd_rate_constant REAL NOT NULL,
                fwd_rate_of_progress REAL NOT NULL,
                rev_rate_constant REAL NOT NULL,
                rev_rate_of_progress REAL NOT NULL,
                net_rate_of_progress REAL NOT NULL,
                PRIMARY KEY (condition_id, run_no, time, reaction),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def insert_or_update(self, data: ReactionData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :condition_id,
                :run_no,
                :time,
                :reaction,
                :fwd_rate_constant,
                :fwd_rate_of_progress,
                :rev_rate_constant,
                :rev_rate_of_progress,
                :net_rate_of_progress
            )
            ON CONFLICT(condition_id, run_no, time, reaction) DO UPDATE SET
                fwd_rate_constant=excluded.fwd_rate_constant,
                fwd_rate_of_progress=excluded.fwd_rate_of_progress,
                rev_rate_constant=excluded.rev_rate_constant,
                rev_rate_of_progress=excluded.rev_rate_of_progress,
                net_rate_of_progress=excluded.net_rate_of_progress;
            """,
            {
                "condition_id": data.condition_id,
                "run_no": data.run_no,
                "time": data.time,
                "reaction": data.reaction,
                "fwd_rate_constant": data.fwd_rate_constant,
                "fwd_rate_of_progress": data.fwd_rate_of_progress,
                "rev_rate_constant": data.rev_rate_constant,
                "rev_rate_of_progress": data.rev_rate_of_progress,
                "net_rate_of_progress": data.net_rate_of_progress,
            },
        )
        if commit:
            self.cur.connection.commit()


@dataclasses.dataclass
class SpeciesData:
    condition_id: int
    run_no: int
    time: float
    species: Species
    mole_frac: float
    concentration: float
    creation_rate: float
    destruction_rate: float
    net_production_rate: float
    a: Optional[float] = None
    b: Optional[float] = None
    dy_dt: Optional[float] = None


class SpeciesTable(SqliteTable):
    def __init__(self, db: SqliteDataBase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Species.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                condition_id INTEGER NOT NULL,
                run_no INTEGER NOT NULL,
                time REAL NOT NULL,
                species TEXT NOT NULL,
                mole_frac REAL NOT NULL,
                concentration REAL NOT NULL,
                creation_rate REAL NOT NULL,
                destruction_rate REAL NOT NULL,
                net_production_rate REAL NOT NULL,
                a REAL,
                b REAL,
                dy_dt REAL,
                PRIMARY KEY (condition_id, run_no, time, species),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """
        )

    @retry(sqlite3.OperationalError, tries=10, backoff=2, max_delay=2)
    def insert_or_update(self, data: SpeciesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                :condition_id,
                :run_no,
                :time,
                :species,
                :mole_frac,
                :concentration,
                :creation_rate,
                :destruction_rate,
                :net_production_rate,
                :a,
                :b,
                :dy_dt
            )
            ON CONFLICT(condition_id, run_no, time, species) DO UPDATE SET
                mole_frac=excluded.mole_frac,
                concentration=excluded.concentration,
                creation_rate=excluded.creation_rate,
                destruction_rate=excluded.destruction_rate,
                net_production_rate=excluded.net_production_rate;
            """,
            {
                "condition_id": data.condition_id,
                "run_no": data.run_no,
                "time": data.time,
                "species": data.species.name,
                "mole_frac": data.mole_frac,
                "concentration": data.concentration,
                "creation_rate": data.creation_rate,
                "destruction_rate": data.destruction_rate,
                "net_production_rate": data.net_production_rate,
                "a": data.a,
                "b": data.b,
                "dy_dt": data.dy_dt,
            }
        )
        if commit:
            self.cur.connection.commit()


class SimulationDatabase:
    db: SqliteDataBase
    conditions: ConditionTable
    conditions_id: int
    reactions: ReactionTable
    species: SpeciesTable

    def __init__(self, db: SqliteDataBase, conditions: Conditions):
        self.db = db
        self.conditions = ConditionTable(db)
        self.conditions_id = self.conditions.insert(conditions)
        self.reactions = ReactionTable(db)
        self.species = SpeciesTable(db)
        self.bulk_properties = BulkPropertiesTable(self.db)

    def reconnect(self):
        self.db = SqliteDataBase(path=self.db.path)
        self.conditions = ConditionTable(self.db)
        self.reactions = ReactionTable(self.db)
        self.species = SpeciesTable(self.db)
        self.bulk_properties = BulkPropertiesTable(self.db)


def clear_simulation_database(path: Union[str, Path]):
    """
    Convenience function to make resets easier between simulation re-runs
    """
    db = SqliteDataBase(path=path)
    _ = ConditionTable(db, clear_existing_data=True)
    _ = ReactionTable(db, clear_existing_data=True)
    _ = SpeciesTable(db, clear_existing_data=True)
    _ = BulkPropertiesTable(db, clear_existing_data=True)
