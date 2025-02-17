import warnings
from dataclasses import dataclass

import pandas as pd
import psycopg

CONN_INFO = "postgresql://postgres@localhost:5432/perturbation_study"


@dataclass
class SimulationResults:
    diluted: pd.DataFrame
    undiluted: pd.DataFrame


def load_reactions() -> SimulationResults:
    with psycopg.connect(CONN_INFO) as con, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        reactions = pd.read_sql(
            """
                with combined as (
                select
                    bp.condition_id condition_id_,
                    *
                from
                    reactions bp
                join (
                    select
                        *
                    from
                        conditions c
                    join (
                        select
                            condition_id,
                            max(run_no) mrn
                        from
                            bulk_properties
                        group by
                            condition_id
                    ) maxes on
                        c.id = maxes.condition_id
                ) jcon on
                    bp.condition_id = jcon.condition_id
                where
                    sim_type = 'cv'
                    and match = 'tad'
                    and bp.run_no = jcon.mrn  -- only use last run
                    and dil_condition != 'medium'
                    and not phi_nom = 0.4
                ),
                results as (
                select
                    diluent,
                    dil_condition,
                    condition_id_ condition_id,
                    "time",
                    ("time" / t_ind) as progress,
                    phi_nom,
                    reaction,
                    relative_chemical_contribution
                from
                    combined
                )
                select
                    *
                from
                    results
                where
                    progress <= 1
                order by
                    condition_id,
                    reaction,
                    progress
                ;
            """,
            con,
        )
    diluted_mask = reactions.dil_condition.eq("undiluted")
    undiluted = (
        reactions[diluted_mask]
        .set_index(["condition_id", "phi_nom", "reaction"])
        .drop(["dil_condition", "diluent"], axis=1)
    )
    diluted = reactions[~diluted_mask].set_index(["condition_id", "dil_condition", "phi_nom", "reaction"])
    # Interpolation should happen prior to plotting, not here

    return SimulationResults(diluted=diluted, undiluted=undiluted)


def load_conditions() -> SimulationResults:
    with psycopg.connect(CONN_INFO) as con, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        conditions = pd.read_sql(
            """
            select
                dil_condition,
                phi_nom,
                diluent,
                id,
                equivalence,
                dil_mf,
                cell_size,
                perturbed_rxn
            from
                conditions
            where
                sim_type = 'cv'
                and match = 'tad'
                and dil_condition != 'medium'
                and not phi_nom = 0.4
            order by
                id
            ;
            """,
            con,
            dtype={"perturbed_rxn": pd.Int64Dtype()},
        )
        diluted_mask = conditions.dil_condition.eq("undiluted")
        undiluted = (
            conditions[diluted_mask]
            .set_index(["id", "phi_nom"])
            .drop(["dil_condition", "diluent"], axis=1)
        )
        diluted = conditions[~diluted_mask].set_index(["id", "dil_condition", "phi_nom"])

    return SimulationResults(diluted=diluted, undiluted=undiluted)
