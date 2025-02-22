import dataclasses
import sqlite3
from enum import Enum
from functools import cached_property
from typing import Optional

import backoff
import psycopg
# Cantera's API has gotten kinda unreliable w.r.t. types in more recent versions, but it's there... unfortunately this
# is still not helpful when it comes to introspection, but at least we know what to look for in the docs
# noinspection PyUnresolvedReferences
from cantera import Species


BACKOFF_MAX_RETRIES = 1
BACKOFF_JITTER = backoff.random_jitter
BACKOFF_STRATEGY = backoff.fibo


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


class PostgresDatabase:
    def __init__(self, conninfo: str):
        self.conninfo = conninfo
        self.con = self.connect()

    def __del__(self):
        try:
            self.con.commit()
        except psycopg.OperationalError:
            self.con = self.connect()
        self.con.close()

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.conninfo)


class SqliteTable:
    def __init__(
            self,
            db: PostgresDatabase,
            table_name: str,
            clear_existing_data: bool = False,
    ):
        self.db = db
        self.cur = self.db.con.cursor()
        self.name = table_name
        if self.table_exists() and clear_existing_data:
            self.__clear()

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def table_exists(self) -> bool:
        self.cur.execute(
            "select true from pg_tables where schemaname='public' and tablename=%(name)s",
            {"name": self.name},
        )
        return self.cur.fetchone() is not None

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def __clear(self):
        self.cur.execute(f"DROP TABLE IF EXISTS {self.name} CASCADE;".encode("utf-8"))


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
    perturbed_rxn: int
    perturbation_fraction: float


class ConditionTable(SqliteTable):
    def __init__(self, db: PostgresDatabase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Conditions.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def __create(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                id SERIAL PRIMARY KEY NOT NULL,
                sim_start TIMESTAMPTZ NOT NULL,
                sim_end TIMESTAMPTZ,
                sim_type TEXT NOT NULL,
                mech TEXT NOT NULL,
                match TEXT,
                dil_condition TEXT NOT NULL,
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
                cell_size_2 REAL,
                perturbed_rxn INTEGER,
                perturbation_fraction REAL
            )
            """.encode("utf-8")
        )

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def insert(self, test_conditions: Conditions) -> int:
        """
        Stores a row of test data in the current table.
        """

        # noinspection PyTypeChecker
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                DEFAULT,
                NOW(),
                Null,
                %(sim_type)s,
                %(mech)s,
                %(match)s,
                %(dil_condition)s,
                %(initial_temp)s,
                %(initial_press)s,
                %(fuel)s,
                %(oxidizer)s,
                %(equivalence)s,
                %(phi_nom)s,
                %(diluent)s,
                %(dil_mf)s,
                Null,
                Null,
                Null,
                Null,
                Null,
                Null,
                %(perturbed_rxn)s,
                %(perturbation_fraction)s
            )
            RETURNING id;
            """.encode("utf-8"),
            dataclasses.asdict(test_conditions),
        )
        self.cur.connection.commit()
        return self.cur.fetchone()[0]

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def finalize_run(
        self,
        temp_vn: float | None,
        t_ind: float | None,
        u_znd: float | None,
        u_cj: float | None,
        cell_size: float | None,
        cell_size_2: float | None,
        condition_ids: list[int]
    ) -> None:
        with self.db.connect() as con:
            con.execute(
                """
                UPDATE
                    conditions
                SET
                    temp_vn = %(temp_vn)s,
                    t_ind = %(t_ind)s,
                    u_znd = %(u_znd)s,
                    u_cj = %(u_cj)s,
                    cell_size = %(cell_size)s,
                    cell_size_2 = %(cell_size_2)s,
                    sim_end = NOW()
                WHERE
                    id = ANY(%(condition_ids)s);
                """,
                {
                    "temp_vn": temp_vn,
                    "t_ind": t_ind,
                    "u_znd": u_znd,
                    "u_cj": u_cj,
                    "cell_size": cell_size,
                    "cell_size_2": cell_size_2,
                    "condition_ids": condition_ids,
                },
            )
            con.commit()


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
    def __init__(self, db: PostgresDatabase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.BulkProperties.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
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
            """.encode("utf-8")
        )

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def insert_or_update(self, data: BulkPropertiesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                %(condition_id)s,
                %(run_no)s,
                %(time)s,
                %(temperature)s,
                %(temperature_gradient)s,
                %(pressure)s,
                %(cp)s,
                %(cv)s,
                %(gamma)s,
                %(velocity)s
            )
            ON CONFLICT(condition_id, run_no, time) DO UPDATE SET
                temperature=excluded.temperature,
                pressure=excluded.pressure,
                velocity=excluded.velocity;
            """.encode("utf-8"),
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
    relative_chemical_contribution: float


class ReactionTable(SqliteTable):
    def __init__(self, db: PostgresDatabase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Reactions.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
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
                relative_chemical_contribution REAL NOT NULL,
                PRIMARY KEY (condition_id, run_no, time, reaction),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """.encode("utf-8")
        )

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def insert_or_update(self, data: ReactionData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                %(condition_id)s,
                %(run_no)s,
                %(time)s,
                %(reaction)s,
                %(fwd_rate_constant)s,
                %(fwd_rate_of_progress)s,
                %(rev_rate_constant)s,
                %(rev_rate_of_progress)s,
                %(net_rate_of_progress)s,
                %(relative_chemical_contribution)s
            )
            ON CONFLICT(condition_id, run_no, time, reaction) DO UPDATE SET
                fwd_rate_constant=excluded.fwd_rate_constant,
                fwd_rate_of_progress=excluded.fwd_rate_of_progress,
                rev_rate_constant=excluded.rev_rate_constant,
                rev_rate_of_progress=excluded.rev_rate_of_progress,
                net_rate_of_progress=excluded.net_rate_of_progress,
                relative_chemical_contribution=excluded.relative_chemical_contribution;
            """.encode("utf-8"),
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
                "relative_chemical_contribution": data.relative_chemical_contribution,
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
    def __init__(self, db: PostgresDatabase, clear_existing_data: bool = False):
        super().__init__(db=db, table_name=TableName.Species.value, clear_existing_data=clear_existing_data)
        if not self.table_exists():
            self.__create()

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
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
            """.encode("utf-8")
        )

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def insert_or_update(self, data: SpeciesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                %(condition_id)s,
                %(run_no)s,
                %(time)s,
                %(species)s,
                %(mole_frac)s,
                %(concentration)s,
                %(creation_rate)s,
                %(destruction_rate)s,
                %(net_production_rate)s,
                %(a)s,
                %(b)s,
                %(dy_dt)s
            )
            ON CONFLICT(condition_id, run_no, time, species) DO UPDATE SET
                mole_frac=excluded.mole_frac,
                concentration=excluded.concentration,
                creation_rate=excluded.creation_rate,
                destruction_rate=excluded.destruction_rate,
                net_production_rate=excluded.net_production_rate;
            """.encode("utf-8"),
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
    db: PostgresDatabase
    conditions: ConditionTable
    conditions_id: int
    reactions: ReactionTable
    species: SpeciesTable

    def __init__(self, db: PostgresDatabase, conditions: Conditions):
        self.db = db
        self.conditions = ConditionTable(db)
        self.conditions_id = self.conditions.insert(conditions)
        self.reactions = ReactionTable(db)
        self.species = SpeciesTable(db)
        self.bulk_properties = BulkPropertiesTable(db)

    def reconnect(self):
        self.db = PostgresDatabase(conninfo=self.db.conninfo)
        self.conditions = ConditionTable(self.db)
        self.reactions = ReactionTable(self.db)
        self.species = SpeciesTable(self.db)
        self.bulk_properties = BulkPropertiesTable(self.db)

    def commit_all(self):
        self.bulk_properties.cur.connection.commit()
        self.species.cur.connection.commit()
        self.reactions.cur.connection.commit()


def clear_simulation_database(conninfo: str):
    """
    Convenience function to make resets easier between simulation re-runs
    """
    db = PostgresDatabase(conninfo=conninfo)
    _ = ConditionTable(db, clear_existing_data=True)
    _ = ReactionTable(db, clear_existing_data=True)
    _ = SpeciesTable(db, clear_existing_data=True)
    _ = BulkPropertiesTable(db, clear_existing_data=True)
