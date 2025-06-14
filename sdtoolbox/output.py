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


class PostgresTable:
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
        self.cur.execute(f"DROP TABLE IF EXISTS {self.name} CASCADE;".encode())


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


class ConditionTable(PostgresTable):
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
                initial_temp DOUBLE PRECISION NOT NULL,
                initial_press DOUBLE PRECISION NOT NULL,
                fuel TEXT NOT NULL,
                oxidizer TEXT NOT NULL,
                equivalence DOUBLE PRECISION NOT NULL,
                phi_nom DOUBLE PRECISION NOT NULL,
                diluent TEXT,
                dil_mf DOUBLE PRECISION NOT NULL,
                temp_vn DOUBLE PRECISION,
                t_ind DOUBLE PRECISION,
                u_znd DOUBLE PRECISION,
                u_cj DOUBLE PRECISION,
                cell_size DOUBLE PRECISION,
                cell_size_2 DOUBLE PRECISION,
                perturbed_rxn INTEGER,
                perturbation_fraction DOUBLE PRECISION
            )
            """.encode()
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
            """.encode(),
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


class BulkPropertiesTable(PostgresTable):
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
                time DOUBLE PRECISION NOT NULL,
                temperature DOUBLE PRECISION NOT NULL,
                temperature_gradient DOUBLE PRECISION,
                pressure DOUBLE PRECISION NOT NULL,
                cp DOUBLE PRECISION NOT NULL,
                cv DOUBLE PRECISION NOT NULL,
                gamma DOUBLE PRECISION NOT NULL,
                velocity DOUBLE PRECISION,
                PRIMARY KEY (condition_id, time),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """.encode()
        )

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def clear(self, condition_id: int) -> None:
        self.cur.connection.rollback()
        self.cur.execute(
            f"""
            DELETE
            FROM {self.name}
            WHERE condition_id = %(condition_id)s;
            """.encode(),
            {"condition_id": condition_id}
        )
        self.cur.connection.commit()

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def insert(self, data: BulkPropertiesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                %(condition_id)s,
                %(time)s,
                %(temperature)s,
                %(temperature_gradient)s,
                %(pressure)s,
                %(cp)s,
                %(cv)s,
                %(gamma)s,
                %(velocity)s
            );
            """.encode(),
            {
                "condition_id": data.condition_id,
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
    time: float
    reaction_no: int
    reaction: str
    fwd_rate_constant: float
    fwd_rate_of_progress: float
    rev_rate_constant: float
    rev_rate_of_progress: float
    net_rate_of_progress: float
    abs_rate_of_progress_rxn: float
    abs_rate_of_progress_total: float


class ReactionTable(PostgresTable):
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
                time DOUBLE PRECISION NOT NULL,
                reaction_no INT NOT NULL,
                reaction TEXT NOT NULL,
                fwd_rate_constant DOUBLE PRECISION NOT NULL,
                fwd_rate_of_progress DOUBLE PRECISION NOT NULL,
                rev_rate_constant DOUBLE PRECISION NOT NULL,
                rev_rate_of_progress DOUBLE PRECISION NOT NULL,
                net_rate_of_progress DOUBLE PRECISION NOT NULL,
                abs_rate_of_progress_rxn DOUBLE PRECISION NOT NULL,
                abs_rate_of_progress_total DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (condition_id, reaction_no, time),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """.encode()
        )

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def clear(self, condition_id: int, reaction_no: int) -> None:
        self.cur.connection.rollback()
        self.cur.execute(
            f"""
            DELETE
            FROM {self.name}
            WHERE condition_id = %(condition_id)s AND reaction_no = %(reaction_no)s;
            """.encode(),
            {"condition_id": condition_id, "reaction_no": reaction_no}
        )
        self.cur.connection.commit()

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def insert(self, data: ReactionData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                %(condition_id)s,
                %(time)s,
                %(reaction_no)s,
                %(reaction)s,
                %(fwd_rate_constant)s,
                %(fwd_rate_of_progress)s,
                %(rev_rate_constant)s,
                %(rev_rate_of_progress)s,
                %(net_rate_of_progress)s,
                %(abs_rate_of_progress_rxn)s,
                %(abs_rate_of_progress_total)s
            );
            """.encode(),
            {
                "condition_id": data.condition_id,
                "time": data.time,
                "reaction_no": data.reaction_no,
                "reaction": data.reaction,
                "fwd_rate_constant": data.fwd_rate_constant,
                "fwd_rate_of_progress": data.fwd_rate_of_progress,
                "rev_rate_constant": data.rev_rate_constant,
                "rev_rate_of_progress": data.rev_rate_of_progress,
                "net_rate_of_progress": data.net_rate_of_progress,
                "abs_rate_of_progress_rxn": data.abs_rate_of_progress_rxn,
                "abs_rate_of_progress_total": data.abs_rate_of_progress_total,
            },
        )
        if commit:
            self.cur.connection.commit()


@dataclasses.dataclass
class SpeciesData:
    condition_id: int
    time: float
    species_no: int
    species: Species
    mole_frac: float
    concentration: float
    creation_rate: float
    destruction_rate: float
    net_production_rate: float
    a: Optional[float] = None
    b: Optional[float] = None
    dy_dt: Optional[float] = None


class SpeciesTable(PostgresTable):
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
                time DOUBLE PRECISION NOT NULL,
                species_no INT NOT NULL,
                species TEXT NOT NULL,
                mole_frac DOUBLE PRECISION NOT NULL,
                concentration DOUBLE PRECISION NOT NULL,
                creation_rate DOUBLE PRECISION NOT NULL,
                destruction_rate DOUBLE PRECISION NOT NULL,
                net_production_rate DOUBLE PRECISION NOT NULL,
                a DOUBLE PRECISION,
                b DOUBLE PRECISION,
                dy_dt DOUBLE PRECISION,
                PRIMARY KEY (condition_id, species_no, time),
                FOREIGN KEY(condition_id) REFERENCES {TableName.Conditions.value}(id)
                ON UPDATE CASCADE ON DELETE CASCADE
            );
            """.encode()
        )

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def clear(self, condition_id: int, species_no: int) -> None:
        self.cur.connection.rollback()
        self.cur.execute(
            f"""
            DELETE
            FROM {self.name}
            WHERE condition_id = %(condition_id)s AND species_no = %(species_no)s;
            """.encode(),
            {"condition_id": condition_id, "species_no": species_no}
        )
        self.cur.connection.commit()

    @backoff.on_exception(
        BACKOFF_STRATEGY,
        sqlite3.OperationalError,
        max_tries=BACKOFF_MAX_RETRIES,
        jitter=BACKOFF_JITTER,
    )
    def insert(self, data: SpeciesData, commit: bool = True):
        self.cur.execute(
            f"""
            INSERT INTO {self.name} VALUES (
                %(condition_id)s,
                %(time)s,
                %(species_no)s,
                %(species)s,
                %(mole_frac)s,
                %(concentration)s,
                %(creation_rate)s,
                %(destruction_rate)s,
                %(net_production_rate)s,
                %(a)s,
                %(b)s,
                %(dy_dt)s
            );
            """.encode(),
            {
                "condition_id": data.condition_id,
                "time": data.time,
                "species_no": data.species_no,
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
        self.this_run = conditions
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

    def clear_results(self, reaction_no: int, species_no: int) -> None:
        self.reactions.clear(self.conditions_id, reaction_no)
        self.species.clear(self.conditions_id, species_no)
        self.bulk_properties.clear(self.conditions_id)


def clear_simulation_database(conninfo: str):
    """
    Convenience function to make resets easier between simulation re-runs
    """
    db = PostgresDatabase(conninfo=conninfo)
    _ = ConditionTable(db, clear_existing_data=True)
    _ = ReactionTable(db, clear_existing_data=True)
    _ = SpeciesTable(db, clear_existing_data=True)
    _ = BulkPropertiesTable(db, clear_existing_data=True)
