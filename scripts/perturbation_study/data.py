import re
import warnings
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import numpy as np
import pandas as pd
import psycopg
from scipy.integrate import simpson

CONN_INFO = "postgresql://postgres@localhost:5432/perturbation_study"

CHEMICAL_REGEX = re.compile(r"([A-Z])+(\d+)")

T = TypeVar("T")


@dataclass
class SimulationResults(Generic[T]):
    perturbed: T
    unperturbed: T


def load_reactions(condition_ids: tuple[int]) -> SimulationResults[pd.DataFrame]:
    index = ["condition_id", "reaction_no"]
    with psycopg.connect(CONN_INFO) as con, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # direct substitution is bad sql practice, but it's fine here
        reactions = pd.read_sql(
            f"""
                with combined as (
                select
                    bp.condition_id condition_id_,
                    *
                from
                    reactions bp
                join
                    conditions c
                on
                    bp.condition_id = c.id
                where
                    c.id in {condition_ids}
                ),
                results as (
                select
                    diluent,
                    dil_condition,
                    perturbed_rxn,
                    condition_id_ condition_id,
                    "time",
                    ("time" / t_ind) as progress,
                    phi_nom,
                    reaction_no,
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
                    reaction_no,
                    progress
                ;
            """,
            con,
            dtype={"perturbed_rxn": pd.Int64Dtype(), "diluent": pd.StringDtype()},
        )
    reactions["reaction"] = reactions.apply(to_chemical_string, axis=1)
    perturbed_mask = pd.notna(reactions.perturbed_rxn)
    unperturbed = reactions[~perturbed_mask].set_index(index)
    perturbed = reactions[perturbed_mask].set_index(index)

    return SimulationResults(perturbed=perturbed, unperturbed=unperturbed)


def to_chemical_string(row: pd.Series) -> str:
    r_no = row["reaction_no"]
    rxn = re.sub(CHEMICAL_REGEX, r"\1$_{\2}$", row["reaction"]).replace("<=>", "⟺")
    return f"{rxn} - [{r_no:03}]"


def load_conditions() -> SimulationResults[pd.DataFrame]:
    index = ["condition_id"]
    with psycopg.connect(CONN_INFO) as con, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        conditions = pd.read_sql(
            """
            select
                dil_condition,
                phi_nom,
                diluent,
                id condition_id,
                equivalence,
                dil_mf,
                cell_size,
                perturbed_rxn,
                perturbation_fraction
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
            dtype={"perturbed_rxn": pd.Int64Dtype(), "diluent": pd.StringDtype(), "cell_size": pd.Float64Dtype()},
        )
        perturbed = pd.notna(conditions.perturbed_rxn)
        unperturbed = conditions[~perturbed].set_index(index)
        perturbed = conditions[perturbed].set_index(index)

    return SimulationResults(perturbed=perturbed, unperturbed=unperturbed)


def analyze_cell_sizes(conditions: SimulationResults) -> pd.DataFrame:
    index = ["diluent", "dil_condition"]
    main_cols = ["cell_size"]
    unperturbed = conditions.unperturbed.set_index(index)[main_cols]
    perturbed = conditions.perturbed.set_index(index)[["perturbed_rxn", "perturbation_fraction"] + main_cols]
    result = unperturbed.join(perturbed, rsuffix="_p")
    for column in main_cols:
        pert_col = f"{column}_p"
        result[f"delta_{column}"] = result[column] - result[pert_col]
        result = result.drop(pert_col, axis=1)
    return (
        result
        .reset_index()
        .set_index(index + ["perturbed_rxn"])
    )


def calculate_rcc(
    conditions: SimulationResults[pd.DataFrame],
    reactions: SimulationResults[pd.DataFrame],
) -> pd.DataFrame:
    r"""
    Assumes reaction data is ordered by ``progress``, which should happen on database read. ``values`` are unperturbed.

    .. math::
        \Delta RCC_{i} = RCC_{i, unperturbed} - RCC_{i, perturbed}
    """
    unperturbed = _calc_rcc_by_condition(conditions.unperturbed, reactions.unperturbed)
    perturbed = _calc_rcc_by_condition(conditions.perturbed, reactions.perturbed)
    result = unperturbed.copy().to_frame()
    result["delta_RCC"] = unperturbed - perturbed
    return result


def _calc_rcc_by_condition(conditions: pd.DataFrame, reactions: pd.DataFrame) -> pd.Series:
    return (
        cast(
            pd.Series,
            reactions.groupby(["condition_id", "reaction_no", "reaction"]).apply(_calc_single_rcc, include_groups=False),
        )
        .rename("RCC")
        .to_frame()
        .join(conditions[["diluent", "dil_condition"]])
        .reset_index(drop=False)
        .set_index(["reaction_no", "diluent", "dil_condition", "reaction"])["RCC"]
    )


def _calc_single_rcc(rxn_data: pd.DataFrame) -> float:
    r"""
    Assumes reaction data is ordered by ``progress``, which should happen on database read.

    .. math::
        RCC_{i} =
            \frac{1}{t_{ind}}
            \int_{0}^{t_{ind}}
                \frac{\left | R_{net, i} \right |}
                {\sum_{0}^{N} \left | R_{net, j} \right |}
            dt

    - ``progress`` is :math:`t/t_ind`, which gives us :math:`\frac{1}{t_{ind}}` as well as :math:`dt`
    - ``relative_chemical_contribution`` is
        :math:`\frac{\left | R_{net, i} \right |}{\sum_{0}^{N} \left | R_{net, j} \right |}`
    - Integration is performed using Simpson's rule
    """
    result = simpson(rxn_data["relative_chemical_contribution"], rxn_data["progress"])
    return result


def calculate_normalized_sensitivity_coefficients(
    cell_sizes: pd.DataFrame,
    rcc: pd.DataFrame,
) -> pd.DataFrame:
    r"""
    .. math::
        c_{i} = \frac{\lambda_{i,u} - \lambda_{_i,p}}{k_{i,u} - k_{i,p}} \frac{k_{i,u}}{\lambda_{i, u}}
        = \frac{1 - \frac{\lambda_{i,p}}{\lambda_{i,u}}}{1 - \frac{k_{i,p}}{k_{i,u}}}

    Where Δk is ΔRCC or perturbation fraction

    Parameters
    ----------
    cell_sizes
    rcc

    Returns
    -------

    """
    output = cell_sizes.copy()
    output.index = output.index.set_names(["diluent", "dil_condition", "reaction_no"])
    output = output.join(rcc)

    # rates
    output["dl/dmul"] = output["delta_cell_size"] / output["perturbation_fraction"]
    output["dl/dmul_magnitude"] = output["dl/dmul"].abs()  # for sorting
    output["dl/dRCC"] = output["delta_cell_size"] / output["delta_RCC"]

    # normalized coefficients
    # todo: double check that the math for dpf is correct, and write it down. We don't necessarily want pf - pf, we want
    #  multiplier - multiplier and also /multiplier, which is 1 for unperturbed and 1 + pf for perturbed
    output["c_mul"] = output["dl/dmul"] / output["cell_size"]
    output["c_mul_magnitude"] = output["c_mul"].abs()  # for sorting
    output["c_RCC"] = output["dl/dRCC"] * output["RCC"] / output["cell_size"]

    return output


def group_coefficients_by_diluent(coefficients: pd.DataFrame) -> pd.DataFrame:
    return pd.concat((
        _group_coefficients_by_diluent(coefficients, "CO2"),
        _group_coefficients_by_diluent(coefficients, "N2"),
    ))


def _group_coefficients_by_diluent(coefficients: pd.DataFrame, diluent: str) -> pd.DataFrame:
    grouped = coefficients[coefficients.index.get_level_values("diluent").str.startswith(diluent)].copy()
    index = grouped.index.names
    grouped = grouped.reset_index()
    grouped["method"] = grouped["diluent"].str.endswith("i").map({True: "inert", False: "active"})
    grouped["diluent"] = diluent
    return grouped.set_index(index)


def less_broken_nlargest(data: pd.DataFrame, target: str, n: int) -> pd.DataFrame:
    # Pandas NaN checks don't seem to be working for some reason so we have to manually fix things
    result = data.copy()
    result = result[~np.isnan(data[target].to_numpy())]
    return result.sort_values("target_abs", ascending=False).head(n)


def top_n_by_method(grouped: pd.DataFrame, target: str, n: int) -> pd.DataFrame:
    grouped = grouped.reset_index()
    result = pd.DataFrame()
    for dil_condition, group in grouped.groupby("dil_condition"):
        group["target_abs"] = group[target].abs()
        this_top_n = less_broken_nlargest(group[group["method"] == "active"], "target_abs", n)
        result = pd.concat((result, group[group["reaction_no"].isin(this_top_n["reaction_no"])]))

    return result.sort_values("target_abs")


def top_n_inert_diffs(grouped: pd.DataFrame, target: str, n: int) -> pd.DataFrame:
    grouped = (
        grouped.loc[grouped["method"] == "active", target]
        - grouped.loc[grouped["method"] == "inert", target]
    ).to_frame().reset_index()
    result = pd.DataFrame()
    for dil_condition, group in grouped.groupby("dil_condition"):
        group["target_abs"] = group[target].abs()
        this_top_n = less_broken_nlargest(group, "target_abs", n)
        result = pd.concat((result, group[group["reaction_no"].isin(this_top_n["reaction_no"])]))

    return result.sort_values("target_abs")
