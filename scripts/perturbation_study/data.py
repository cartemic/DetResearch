import re
import warnings
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, cast

import numpy as np
import pandas as pd
import pandera.pandas as pa
import psycopg
from pandera.typing.pandas import DataFrame, Index
from scipy.integrate import simpson

CONN_INFO = "postgresql://postgres@localhost:5432/perturbation_study"

CHEMICAL_REGEX = re.compile(r"([A-Z])+(\d+)")

T = TypeVar("T")


@dataclass
class SimulationResults(Generic[T]):
    perturbed: T
    unperturbed: T


_DiluentType = str
_DiluentField = pa.Field(isin=["CO2", "CO2i", "N2", "N2i"], nullable=True)

_DilConditionType = str
_DilConditionField = pa.Field(isin=["low", "high", "undiluted"])

_PerturbedRxnType = int
_PerturbedRxnField = pa.Field(nullable=True)

_CellSizeType = float
_CellSizeField = pa.Field(gt=0)

_PerturbationFractionType = float
_PerturbationFractionField = pa.Field(coerce=True)

_ReactionNoType = int
_ReactionNoField = pa.Field(ge=0)

_ReactionType = str
_ReactionField = pa.Field()

_RccType = float
_RccField = pa.Field(alias="RCC")

_DeltaRccType = float
_DeltaRccField =  pa.Field(alias="delta_RCC")

_DlType = float
_DlDmulField = pa.Field(alias="dl/dmul")
_DlDrccField = pa.Field(alias="dl/dRCC")

_CType = float
_CMulField = pa.Field(alias="c_mul")
_CRccField = pa.Field(alias="c_RCC")


class _BaseDataframeModel(pa.DataFrameModel):
    class Config:
        strict = True


class ReactionsDataframeModel(_BaseDataframeModel):
    condition_id: Index[int] = pa.Field(ge=0)
    reaction_no: Index[_ReactionNoType] = _ReactionNoField
    diluent: _DiluentType = _DiluentField
    dil_condition: _DilConditionType = _DilConditionField
    perturbed_rxn: _PerturbedRxnType = _PerturbedRxnField
    time: float = pa.Field(ge=0)
    progress: float = pa.Field(ge=0, le=1)
    phi_nom: float = pa.Field(gt=0)
    reaction: _ReactionType = _ReactionField
    abs_rate_of_progress_total: float = pa.Field(ge=0)
    relative_chemical_contribution: float = pa.Field(ge=0, le=1)


type ReactionsDataframe = DataFrame[ReactionsDataframeModel]


def load_reactions(condition_ids: tuple[int, ...]) -> SimulationResults[ReactionsDataframe]:
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
                    abs_rate_of_progress_total,
                    (abs_rate_of_progress_rxn / abs_rate_of_progress_total) as relative_chemical_contribution
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
    unperturbed = ReactionsDataframeModel.validate(reactions[~perturbed_mask].set_index(index))
    perturbed = ReactionsDataframeModel.validate(reactions[perturbed_mask].set_index(index))

    return SimulationResults(perturbed=perturbed, unperturbed=unperturbed)


def to_chemical_string(row: pd.Series) -> str:
    r_no = row["reaction_no"]
    rxn = re.sub(CHEMICAL_REGEX, r"\1$_{\2}$", row["reaction"]).replace("<=>", "⟺")
    return f"{rxn} - [{r_no:03}]"


class ConditionsDataframeModel(_BaseDataframeModel):
    condition_id: Index[int] = pa.Field(ge=0)
    dil_condition: str = pa.Field(isin=["low", "high", "undiluted"])
    phi_nom: float = pa.Field(gt=0)
    diluent: str | None = pa.Field(isin=["CO2", "CO2i", "N2", "N2i"], nullable=True)
    equivalence: float = pa.Field(gt=0)
    dil_mf: float = pa.Field(ge=0, coerce=True)
    cell_size: _CellSizeType = _CellSizeField
    perturbed_rxn: _PerturbedRxnType = _PerturbedRxnField
    perturbation_fraction: _PerturbationFractionType = _PerturbationFractionField


type ConditionsDataframe = DataFrame[ConditionsDataframeModel]


def load_conditions() -> SimulationResults[ConditionsDataframe]:
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
        perturbed_mask = pd.notna(conditions.perturbed_rxn)
        unperturbed = ConditionsDataframeModel.validate(conditions[~perturbed_mask].set_index(index))
        perturbed = ConditionsDataframeModel.validate(conditions[perturbed_mask].set_index(index))

    return SimulationResults(perturbed=perturbed, unperturbed=unperturbed)


class CellSizeDataframeModel(_BaseDataframeModel):
    diluent: Index[_DiluentType] = _DiluentField
    dil_condition: Index[_DilConditionType] = _DilConditionField
    perturbed_rxn: Index[_PerturbedRxnType] = _PerturbedRxnField
    cell_size: _CellSizeType = _CellSizeField
    perturbation_fraction: _PerturbationFractionType = _PerturbationFractionField
    delta_cell_size: float = pa.Field()


type CellSizeDataframe = DataFrame[CellSizeDataframeModel]


def analyze_cell_sizes(conditions: SimulationResults[ConditionsDataframe]) -> CellSizeDataframe:
    index = ["diluent", "dil_condition"]
    main_cols = ["cell_size"]
    unperturbed = conditions.unperturbed.set_index(index)[main_cols]
    perturbed = conditions.perturbed.set_index(index)[["perturbed_rxn", "perturbation_fraction", *main_cols]]
    result = unperturbed.join(perturbed, rsuffix="_p")
    for column in main_cols:
        pert_col = f"{column}_p"
        result[f"delta_{column}"] = result[column] - result[pert_col]
        result = result.drop(pert_col, axis=1)
    result = result.reset_index().set_index([*index, "perturbed_rxn"])
    return CellSizeDataframeModel.validate(result)


class RccDataframeModel(_BaseDataframeModel):
    reaction_no: Index[_ReactionNoType] = _ReactionNoField
    diluent: Index[_DiluentType] = _DiluentField
    dil_condition: Index[_DilConditionType] = _DilConditionField
    reaction: Index[_ReactionType] = _ReactionField
    rcc: _RccType = _RccField
    delta_rcc: _DeltaRccType = _DeltaRccField


type RccDataframe = DataFrame[RccDataframeModel]


def calculate_rcc(
    conditions: SimulationResults[ConditionsDataframe],
    reactions: SimulationResults[ReactionsDataframe],
) -> RccDataframe:
    r"""
    Assumes reaction data is ordered by ``progress``, which should happen on database read. ``values`` are unperturbed.

    .. math::
        \Delta RCC_{i} = RCC_{i, unperturbed} - RCC_{i, perturbed}
    """
    unperturbed = _calc_rcc_by_condition(conditions.unperturbed, reactions.unperturbed)
    perturbed = _calc_rcc_by_condition(conditions.perturbed, reactions.perturbed)
    result = unperturbed.copy().to_frame()
    result["delta_RCC"] = unperturbed - perturbed
    return RccDataframeModel.validate(result)


def _calc_rcc_by_condition(conditions: ConditionsDataframe, reactions: ReactionsDataframe) -> pd.Series:
    return (
        cast(
            pd.Series,
            reactions.groupby(["condition_id", "reaction_no", "reaction"]).apply(
                 _calc_single_rcc, include_groups=False
            ),
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


class NormalizedSensitivityCoefficientModel(_BaseDataframeModel):
    diluent: Index[_DiluentType] = _DiluentField
    dil_condition: Index[_DilConditionType] = _DilConditionField
    reaction_no: Index[_ReactionNoType] = _ReactionNoField
    cell_size: _CellSizeType = _CellSizeField
    perturbation_fraction: _PerturbationFractionType = _PerturbationFractionField
    delta_cell_size: _CellSizeType = pa.Field()
    rcc: _RccType = _RccField
    delta_rcc: _DeltaRccType = _DeltaRccField
    dl_dmul: _DlType = _DlDmulField
    """
    Corresponds to C_s (eq. 6)
    """

    dl_drcc: _DlType = _DlDrccField
    """
    Corresponds to C_s (eq. 6)
    """

    c_mul: _CType = _CMulField
    """
    Corresponds to c_s (eq. 8)
    """

    c_rcc: _CType = _CRccField
    """
    Corresponds to c_s (eq. 8)
    """


type NormalizedSensitivityCoefficientDataframe = DataFrame[NormalizedSensitivityCoefficientModel]


def calculate_normalized_sensitivity_coefficients(
    cell_sizes: CellSizeDataframe,
    rcc: RccDataframe,
) -> NormalizedSensitivityCoefficientDataframe:
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
    output["dl/dRCC"] = output["delta_cell_size"] / output["delta_RCC"]

    # normalized coefficients
    # cell_size is unperturbed
    rcc_numerator = 1
    output["c_mul"] = output["dl/dmul"] / output["cell_size"]
    output["c_RCC"] = output["dl/dRCC"] * rcc_numerator / output["cell_size"]

    return NormalizedSensitivityCoefficientModel.validate(output)


class NSCByDiluentDataframeModel(_BaseDataframeModel):
    diluent: Index[str] = pa.Field(isin=("CO2", "N2"))  # diluted only, inert subscript removed
    dil_condition: Index[_DilConditionType] = _DilConditionField
    reaction_no: Index[_ReactionNoType] = _ReactionNoField
    reaction: Index[_ReactionType] = _ReactionField
    cell_size: _CellSizeType = _CellSizeField
    perturbation_fraction: _PerturbationFractionType = _PerturbationFractionField
    delta_cell_size: _CellSizeType = pa.Field()
    rcc: _RccType = _RccField
    delta_rcc: _DeltaRccType = _DeltaRccField
    dl_dmul: _DlType = _DlDmulField
    dl_drcc: _DlType = _DlDrccField
    c_mul: _CType = _CMulField
    c_rcc: _CType = _CRccField
    method: str = pa.Field(isin=("active", "inert"))


type NSCByDiluentDataframe = DataFrame[NSCByDiluentDataframeModel]


def group_coefficients_by_diluent(coefficients: NormalizedSensitivityCoefficientDataframe) -> NSCByDiluentDataframe:
    result = pd.concat(
        (
            _group_coefficients_by_diluent(coefficients, "CO2"),
            _group_coefficients_by_diluent(coefficients, "N2"),
        )
    )
    return NSCByDiluentDataframeModel.validate(result)


def _group_coefficients_by_diluent(
    coefficients: NormalizedSensitivityCoefficientDataframe,
    diluent: str,
) -> pd.DataFrame:
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
        (grouped.loc[grouped["method"] == "active", target] - grouped.loc[grouped["method"] == "inert", target])
        .to_frame()
        .reset_index()
    )
    result = pd.DataFrame()
    for dil_condition, group in grouped.groupby("dil_condition"):
        group["target_abs"] = group[target].abs()
        this_top_n = less_broken_nlargest(group, "target_abs", n)
        result = pd.concat((result, group[group["reaction_no"].isin(this_top_n["reaction_no"])]))

    return result.sort_values("target_abs")


SpeciesDataColumn = Literal[
    "mole_frac",
    "concentration",
    "creation_rate",
    "destruction_rate",
    "net_production_rate",
    "a",
    "b",
    "dy_dt",
]
SpeciesTimeseriesSchema = pa.DataFrameSchema(
    {
        "diluent": pa.Column(str),
        "dil_condition": pa.Column(str),
        "method": pa.Column(str, checks=pa.Check.isin(("active", "inert"))),
        "species": pa.Column(str),
        "time": pa.Column(float, checks=pa.Check.ge(0)),
        "progress": pa.Column(float, checks=pa.Check.ge(0).le(1)),
        "|".join(SpeciesDataColumn.__dict__["__args__"]): pa.Column(float, coerce=True, regex=True)
    },
    strict=True,
)
type SpeciesTimeseriesDataframe = DataFrame[SpeciesTimeseriesSchema]


def load_species_timeseries(
    data_column: SpeciesDataColumn,
    species: tuple[str, ...],
    dil_conditions: tuple[str, ...],
) -> SpeciesTimeseriesDataframe:
    with psycopg.connect(CONN_INFO) as con, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # direct substitution is bad sql practice, but it's fine here
        ts_data = pd.read_sql(
            f"""
                with combined as (
                    select
                        s.condition_id condition_id_,
                        *
                    from
                        species s
                    join
                        conditions c
                    on
                        s.condition_id = c.id
                    where
                        c.perturbation_fraction = 0
                        and c.sim_type = 'cv'
                        and c.dil_condition in {dil_conditions}
                        and c.phi_nom = 1
                        and (c.diluent like 'CO2%' or c.diluent like 'N2%')
                ),
                results as (
                    select
                        case when diluent like '%i' then 'inert' else 'active' end as "method",
                        replace(diluent, 'i', '') as diluent,
                        dil_condition,
                        "time",
                        ("time" / t_ind) as progress,
                        species,
                        {data_column}
                    from
                        combined
                    where species in {species}
                )
                select
                    *
                from
                    results
                where
                    progress <= 1
                order by
                    diluent,
                    method,
                    species,
                    dil_condition,
                    progress
                ;
            """,
            con,
        )
    return (
        SpeciesTimeseriesSchema
        .update_columns(
            {
                "species": {"checks": pa.Check.isin(species)},
                "dil_condition": {"checks": pa.Check.isin(dil_conditions)}
            }
        )
        .validate(ts_data))
