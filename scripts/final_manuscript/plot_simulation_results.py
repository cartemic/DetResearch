import enum
import sqlite3
from dataclasses import dataclass
from functools import cached_property
from sqlite3 import Connection
from typing import Literal, Optional, Tuple, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import SubplotSpec
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d

from scripts.final_manuscript.plot_settings import set_palette, set_style


def load_conditions_data(con: Connection) -> pd.DataFrame:
    return pd.read_sql_query("select * from conditions", con)


def load_reactions_data(con: Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        select
            rc.condition_id,
            rc.run_no,
            rc.sim_type,
            rc.reaction,
            rc.diluent,
            rc.dil_mf,
            rc.time,
            rc.fwd_rate_constant,
            rc.fwd_rate_of_progress,
            rc.rev_rate_constant,
            rc.rev_rate_of_progress,
            rc.net_rate_of_progress,
            b.temperature,
            b.pressure,
            b.velocity
        from
            (
                (
            select
                *
            from
                reactions r
            inner join conditions c
                on
                c.id = r.condition_id
                ) rc
        inner join bulk_properties b
                on
            b.condition_id = rc.condition_id
            AND b.time = rc.time
            AND b.run_no = rc.run_no
           )
        """,
        con,
    )


def load_species_data(con: Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        select
            rc.condition_id,
            rc.run_no,
            rc.sim_type,
            rc.species,
            rc.diluent,
            rc.dil_mf,
            rc.time,
            rc.mole_frac,
            rc.concentration,
            rc.creation_rate,
            rc.destruction_rate,
            rc.net_production_rate,
            b.temperature,
            b.pressure,
            b.velocity
        from
            (
                (
            select
                *
            from
                species r
            inner join conditions c
                on
                c.id = r.condition_id
                ) rc
        inner join bulk_properties b
                on
            b.condition_id = rc.condition_id
            AND b.time = rc.time
            AND b.run_no = rc.run_no
           )
        """,
        con,
    )


@dataclass(frozen=True)
class DataColumn:
    column_name: str
    plot_name: str
    data_type: Union[Literal["reaction"], Literal["species"]]
    units: Optional[str]
    offset: Optional[float] = None


class Species:
    # Will have to generalize this a bit if we start using more complex species
    def __init__(self, element: str, number: int):
        self.element = element
        self.number = number

    def as_string(self) -> str:
        return f"{self.element}{self.number}"

    def as_tex_string(self) -> str:
        return f"{self.element}_{{{self.number}}}"


class DataColumnSpecies:
    mole_frac = DataColumn(
        column_name="mole_frac",
        plot_name="Mole Fraction",
        units="-",
        data_type="species",
        offset=1e-1,
    )
    concentration = DataColumn(
        column_name="concentration",
        plot_name="Concentration",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} }",
        data_type="species",
        offset=1e-2,
    )
    creation_rate = DataColumn(
        column_name="creation_rate",
        plot_name="Creation Rate",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="species",
        offset=1e8,
    )
    destruction_rate = DataColumn(
        column_name="destruction_rate",
        plot_name="Destruction Rate",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="species",
        offset=1e8,
    )
    net_production_rate = DataColumn(
        column_name="net_production_rate",
        plot_name="Net Production Rate",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="species",
        offset=1e8,
    )


class DataColumnReaction:
    fwd_rate_constant = DataColumn(
        column_name="fwd_rate_constant",
        plot_name="Forward Rate Constant",
        units=None,
        data_type="reaction",
        offset=1e9,
    )
    fwd_rate_of_progress = DataColumn(
        column_name="fwd_rate_of_progress",
        plot_name="Forward Rate of Progress",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="reaction",
        offset=1e6,
    )
    rev_rate_constant = DataColumn(
        column_name="rev_rate_constant",
        plot_name="Reverse Rate Constant",
        units=None,
        data_type="reaction",
        offset=1e9,
    )
    rev_rate_of_progress = DataColumn(
        column_name="rev_rate_of_progress",
        plot_name="Reverse Rate of Progress",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="reaction",
        offset=1e6,
    )
    net_rate_of_progress = DataColumn(
        column_name="net_rate_of_progress",
        plot_name="Net Rate of Progress",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="reaction",
        offset=1e6,
    )


class SimulationPlot:
    def __init__(
        self,
        velocity: Optional[plt.Axes],
        conditions: plt.Axes,
        results: plt.Axes,
        relative_dilution: Union[Literal["High"], Literal["Low"]],
        diluent: Optional[Species] = None,
        title_fontsize: int = 9,
        axis_fontsize: int = 6,
        legend_fontsize: int = 5,
    ):
        self.velocity = velocity
        self.temperature = conditions
        self.pressure = conditions.twinx()
        self.results = results
        self.relative_dilution = relative_dilution
        self.diluent = diluent

        self._dil_mf = None

        self._title_fontsize = title_fontsize
        self._axis_fontsize = axis_fontsize
        self._legend_fontsize = legend_fontsize

        all_axes = [self.temperature, self.pressure, self.results]
        if self.velocity is not None:
            all_axes.append(self.velocity)
        for axes in all_axes:
            axes.yaxis.get_offset_text().set_fontsize(self._axis_fontsize)
            axes.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:1.2f}"))
            for ax in (axes.xaxis, axes.yaxis):
                ax.set_tick_params(labelsize=self._axis_fontsize)
        self.set_title()

    def align_all_axes(self):
        self.pressure.yaxis.set_label_coords(1.125, 0.5)
        self.temperature.yaxis.set_label_coords(-0.125, 0.5)
        self.results.yaxis.set_label_coords(-0.125, 0.5)

    def set_dil_mf(self, dil_mf: float):
        self._dil_mf = dil_mf
        self.set_title()

    def set_title(self):
        title_axes = self.velocity or self.temperature
        title = f"{self.relative_dilution} dilution"
        if self._dil_mf is not None:
            title += r" $\left(\chi_" + f"{{{self.diluent.as_tex_string()}}} = {self._dil_mf:3.2f}" + r"\right)$"
        # we should be able to set this on temperature or pressure since they share a subplot
        title_axes.set_title(title, fontsize=self._title_fontsize)

    def plot_data(
        self,
        data: pd.DataFrame,
        data_column: DataColumn,
        result_designator_column: str,
        conditions: pd.Series,
        induction_window: Optional[float] = None,
    ):
        """
        Parameters
        ==========
        data: simulation data to plot
        data_column: column designating data to plot in the lower half of the combined plot
        result_designator_column : column designating reaction or species values, each of which will have a data plot
            entry
        conditions : series of current plot's test conditions and results
        induction_window : (Optional) time in seconds - plot x scale will be set to t_ind +/- `induction_window`
        """
        plot_data = data.sort_values("time")
        time_scale = 1_000_000  # s -> us
        temp_scale = 1e-3
        press_scale = 1e-7

        if induction_window:
            # shift reference time from start to induction time
            plot_data["time"] -= conditions.t_ind

        plot_data["time"] *= time_scale
        plot_data["pressure"] *= press_scale
        plot_data["temperature"] *= temp_scale

        dil_mfs = plot_data["dil_mf"].unique()
        n_dil_mfs = len(dil_mfs)
        if n_dil_mfs > 1:
            raise RuntimeError(f"Data does not have a unique `dil_mf`: {dil_mfs}")
        if n_dil_mfs < 1:
            raise RuntimeError("Data does not have any `dil_mf`")
        self.set_dil_mf(dil_mfs[0])

        line_pressure = self.pressure.plot(plot_data["time"], plot_data["pressure"], ls="--", color="C1", label="P")
        line_temperature = self.temperature.plot(
            plot_data["time"], plot_data["temperature"], ls="-", color="C0", label="T"
        )

        t_max = plot_data["time"].max()
        if induction_window:
            # we want windowed plots to go below zero
            # induction_time = 0
            t_min = -induction_window * time_scale
            t_max = 0  # not centered
        else:
            t_min = 0
            # induction_time = conditions.t_ind * time_scale

        # Prevent post-induction data from throwing off my y-axis scaling.
        # I'm not worried about t_min since that's a flat tail anyway.
        plot_data = plot_data[plot_data["time"] <= t_max]

        self.temperature.set_xlim(t_min, t_max)
        self.temperature.set_xlim(t_min, t_max)
        self.results.set_xlim(t_min, t_max)

        if self.velocity is not None:
            self.velocity.plot(plot_data["time"], plot_data["velocity"], "C5")
            self.velocity.set_xlim(t_min, t_max)
            self.velocity.set_ylabel("Velocity (m/s)", fontsize=self._axis_fontsize, verticalalignment="baseline")

        # Induction time is determined by a temperature threshold in the CV simulations, so that is where we plot it
        # if conditions.sim_type == "cv":
        #     for ax in (self.temperature, self.results):
        #         ax.axvspan(t_min, induction_time, zorder=-1, color="#eee")

        self.pressure.set_ylabel(
            f"Pressure (Pa)\nx{press_scale}", fontsize=self._axis_fontsize, verticalalignment="top"
        )
        self.temperature.set_ylabel(
            f"Temperature (K)\nx{temp_scale:1.1e}", fontsize=self._axis_fontsize, verticalalignment="bottom"
        )

        lines = line_pressure + line_temperature
        labels = [line.get_label() for line in lines]
        # we should be able to set this on temperature or pressure since they share a subplot
        self.temperature.legend(lines, labels, fontsize=self._legend_fontsize)

        # reaction/species plots
        color_indices = [2, 3, 4, 5, 6, 7]
        n_colors = len(color_indices)
        line_styles = ["-", "--", "-.", ":"]
        n_styles = len(line_styles)
        lines = []
        for i, (label, data_group) in enumerate(plot_data.groupby(result_designator_column)):
            color_idx = color_indices[i % n_colors]
            # if we expand the number of reactions/species and need more differentiation between lines, we can always
            # cycle through styles at a slower rate
            ls = line_styles[(i // n_colors) % n_styles]
            lines += self.results.plot(
                data_group["time"],
                data_group[data_column.column_name] / (data_column.offset or 1),
                color=f"C{color_idx}",
                label=label,
                ls=ls,
            )
        labels = [line.get_label() for line in lines]

        ylabel = data_column.plot_name
        if data_column.units:
            ylabel += (
                r" $\left(" f"{data_column.units}" r"\right)$" f"\nx{1/data_column.offset:1.1e}"
                if data_column.offset
                else ""
            )
        self.results.legend(lines, labels, fontsize=self._legend_fontsize)
        self.results.set_ylabel(ylabel, fontsize=self._axis_fontsize)
        time_label = "Time" + ("" if induction_window is None else " from induction")
        self.results.set_xlabel(time_label + r" $\left( \mu \mathrm{s} \right)$", fontsize=self._axis_fontsize)
        self.align_all_axes()


@dataclass
class SimulationPlotRow:
    dil_low: SimulationPlot
    dil_high: SimulationPlot
    row: plt.Axes


@dataclass(frozen=True)
class ResultConditions:
    dil_high: pd.Series
    dil_low: pd.Series


@dataclass(frozen=True)
class PlotConditions:
    sim_type: Union[Literal["cv"], Literal["znd"]]  # this can probably be smarter but whatever
    co2: ResultConditions
    n2: ResultConditions


class NitrogenRow(enum.Enum):
    tad: str = "N$_{2}$ diluted, T$_{ad}$ matched"
    mf: str = "N$_{2}$ diluted, mole fraction matched"


class SimulationPlots:
    def __init__(
        self,
        data: pd.DataFrame,
        data_column: DataColumn,
        condition_ids: PlotConditions,
        n2_row: NitrogenRow,
        show_title: bool = False,
        save_to: Optional[str] = None,
        suptitle_fontsize: int = 12,
        row_fontsize: int = 10,
        induction_window: Optional[float] = None,
        plot_difference: bool = False,
    ):
        if plot_difference and (induction_window is None):
            raise RuntimeError("plot_difference should only be used with induction_window")

        # noinspection PyTypeChecker
        self._figure: plt.Figure = None
        self._suptitle_fontsize = suptitle_fontsize

        # these need to match self_rows in self._create_plots(), and are set using setattr()
        # noinspection PyTypeChecker
        self.co2: SimulationPlotRow = None  # set in _create_plots
        # noinspection PyTypeChecker
        self.n2: SimulationPlotRow = None  # set in _create_plots
        self.diff: Optional[SimulationPlotRow] = None

        self.plot_velocity = condition_ids.sim_type == "znd"

        self._create_plots(plot_difference)

        self.co2.row.set_title("CO$_{2}$ diluted", fontsize=row_fontsize)
        self.n2.row.set_title(n2_row.value, fontsize=row_fontsize)
        if plot_difference:
            self.diff.row.set_title("Difference")

        if show_title:
            sim_kind = condition_ids.sim_type.upper()
            plot_kind = data_column.plot_name.replace("\n", " ")
            self._figure.suptitle(
                f"{sim_kind} Simulations ({plot_kind})",
                fontsize=self._suptitle_fontsize,
                y=0.925,
            )

        self._plot_data(data, data_column, condition_ids, induction_window)

        self._share_ylim()

        self._figure.align_ylabels()

        if isinstance(save_to, str):
            self._figure.savefig(save_to)

    def _share_ylim(self):
        axes = []
        temp_bounds = [1e16, 0]
        press_bounds = [1e16, 0]
        result_bounds = [1e16, 0]
        vel_bounds = [1e16, 0]
        for row in (self.co2, self.n2):
            if row is not None:
                for ax in (row.dil_low, row.dil_high):
                    axes.append(ax)

                    y_lim = ax.temperature.get_ylim()
                    temp_bounds[0] = min(temp_bounds[0], y_lim[0])
                    temp_bounds[1] = max(temp_bounds[1], y_lim[1])

                    y_lim = ax.pressure.get_ylim()
                    press_bounds[0] = min(press_bounds[0], y_lim[0])
                    press_bounds[1] = max(press_bounds[1], y_lim[1])

                    y_lim = ax.results.get_ylim()
                    result_bounds[0] = min(result_bounds[0], y_lim[0])
                    result_bounds[1] = max(result_bounds[1], y_lim[1])

                    if ax.velocity is not None:
                        y_lim = ax.velocity.get_ylim()
                        vel_bounds[0] = min(vel_bounds[0], y_lim[0])
                        vel_bounds[1] = max(vel_bounds[1], y_lim[1])

        for ax in axes:
            ax.temperature.set_ylim(*temp_bounds)
            ax.pressure.set_ylim(*press_bounds)
            ax.results.set_ylim(*result_bounds)

            if ax.velocity is not None:
                ax.velocity.set_ylim(*vel_bounds)

        if self.diff is not None:
            y_lim_temp_low = self.diff.dil_low.temperature.get_ylim()
            y_lim_temp_high = self.diff.dil_high.temperature.get_ylim()
            y_lim_press_low = self.diff.dil_low.pressure.get_ylim()
            y_lim_press_high = self.diff.dil_high.pressure.get_ylim()
            y_lim_results_low = self.diff.dil_low.results.get_ylim()
            y_lim_results_high = self.diff.dil_high.results.get_ylim()
            if self.plot_velocity:
                y_lim_vel_low = self.diff.dil_low.velocity.get_ylim()
                y_lim_vel_high = self.diff.dil_high.velocity.get_ylim()
            for plot in (self.diff.dil_low, self.diff.dil_high):
                plot.temperature.set_ylim(
                    min(y_lim_temp_low[0], y_lim_temp_high[0]),
                    max(y_lim_temp_low[1], y_lim_temp_high[1]),
                )
                plot.pressure.set_ylim(
                    min(y_lim_press_low[0], y_lim_press_high[0]),
                    max(y_lim_press_low[1], y_lim_press_high[1]),
                )
                plot.results.set_ylim(
                    min(y_lim_results_low[0], y_lim_results_high[0]),
                    max(y_lim_results_low[1], y_lim_results_high[1]),
                )
                if self.plot_velocity:
                    # guarded above
                    # noinspection PyUnboundLocalVariable
                    plot.velocity.set_ylim(
                        min(y_lim_vel_low[0], y_lim_vel_high[0]),
                        max(y_lim_vel_low[1], y_lim_vel_high[1]),
                    )

    def _create_plots(self, plot_difference: bool):
        self._figure = plt.figure(figsize=(8.5, 11))

        n2 = Species("N", 2)
        co2 = Species("CO", 2)

        n_rows = 2 + int(plot_difference)
        rows_gridspec = self._figure.add_gridspec(nrows=n_rows, ncols=1, hspace=0.3)

        self._create_single_plot("co2", co2, rows_gridspec[0])
        self._create_single_plot("n2", n2, rows_gridspec[1])
        if plot_difference:
            self._create_single_plot("diff", None, rows_gridspec[2])

    def _create_single_plot(
        self,
        self_row: str,
        diluent: Optional[Species],
        row_gs: SubplotSpec,
    ):
        # noinspection PyTypeChecker
        row = self._figure.add_subplot(row_gs)
        row.axis("off")

        # This little guy keeps the high/low dilution titles from overlapping with the row titles
        row_plots_and_title_gs = row_gs.subgridspec(2, 1, height_ratios=[1, 100])

        n_windows_per_plot = 3 if self.plot_velocity else 2

        row_plots = row_plots_and_title_gs[1].subgridspec(n_windows_per_plot, 2, hspace=0, wspace=0.4)
        row_axes = []
        relative_dilutions = ["Low", "High"]
        for col, relative_dilution in enumerate(relative_dilutions):
            relative_dilution: Union[Literal["Low"], Literal["High"]]

            plot_idx = 0
            if self.plot_velocity:
                ax_velocity = self._figure.add_subplot(row_plots[0, col])
                ax_velocity.get_xaxis().set_visible(False)
                plot_idx += 1
            else:
                ax_velocity = None
            ax_conditions = self._figure.add_subplot(row_plots[plot_idx, col])
            ax_conditions.get_xaxis().set_visible(False)
            plot_idx += 1

            ax_results = self._figure.add_subplot(row_plots[plot_idx, col])
            row_axes.append(
                SimulationPlot(
                    velocity=ax_velocity,
                    conditions=ax_conditions,
                    results=ax_results,
                    relative_dilution=relative_dilution,
                    diluent=diluent,
                )
            )
        setattr(
            self,
            self_row,
            SimulationPlotRow(row=row, dil_low=row_axes[0], dil_high=row_axes[1]),
        )

    def _plot_data(
        self,
        data: pd.DataFrame,
        data_column: DataColumn,
        conditions: PlotConditions,
        induction_window: Optional[float],
    ):
        is_reaction_data = "reaction" in data.columns
        is_species_data = "species" in data.columns
        if is_reaction_data and is_species_data:
            raise RuntimeError("Data contains both reaction and species columns!")
        if not (is_reaction_data or is_species_data):
            raise RuntimeError("Data doesn't contain reaction or species columns!")
        result_designator_column = "reaction" if is_reaction_data else "species"
        if data_column.data_type == "reaction" and is_species_data:
            raise RuntimeError("Reaction plots were requested but species data was provided!")
        if data_column.data_type == "species" and is_reaction_data:
            raise RuntimeError("Species plots were requested but reaction data was provided!")

        if self.diff is not None:
            diff_data = co2_minus_tad_matched_n2(
                data=data,
                conditions=conditions,
                group_column=result_designator_column,
                data_column=data_column.column_name,
            )

        # these need to match the keys for DataColumn and PlotConditionIds
        for relative_amount in ("dil_low", "dil_high"):
            co2_conditions = conditions.co2.__dict__[relative_amount]
            n2_conditions = conditions.n2.__dict__[relative_amount]
            for plot_row, these_conditions in (
                (self.co2, co2_conditions),
                (self.n2, n2_conditions),
            ):
                plot_row.__dict__[relative_amount].plot_data(
                    data=data[data["condition_id"] == these_conditions.id],
                    data_column=data_column,
                    result_designator_column=result_designator_column,
                    conditions=these_conditions,
                    induction_window=induction_window,
                )
            if self.diff is not None:
                # this is checked above
                # noinspection PyUnboundLocalVariable
                self.diff.__dict__[relative_amount].plot_data(
                    data=diff_data[
                        (diff_data["condition_id"] == co2_conditions.id)
                        | (diff_data["condition_id"] == n2_conditions.id)
                    ],
                    data_column=data_column,
                    result_designator_column=result_designator_column,
                    conditions=co2_conditions,
                    induction_window=induction_window,
                )


def validate_sim_type(maybe: str) -> Union[Literal["cv"], Literal["znd"]]:
    if maybe == "cv":
        return "cv"
    elif maybe == "znd":
        return "znd"
    else:
        raise RuntimeError(f"Invalid simulation type: {maybe}")


def get_condition_ids(conditions: pd.DataFrame) -> tuple[PlotConditions, ...]:
    all_condition_ids = []
    for sim_type, sim_type_conditions in conditions.groupby("sim_type"):
        all_condition_ids.append(
            PlotConditions(
                sim_type=validate_sim_type(str(sim_type)),
                co2=ResultConditions(
                    dil_high=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "CO2")
                            & (sim_type_conditions["match"].isna())
                            & (sim_type_conditions["dil_condition"] == "high")
                        )
                    ].iloc[0],
                    dil_low=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "CO2")
                            & (sim_type_conditions["match"].isna())
                            & (sim_type_conditions["dil_condition"] == "low")
                        )
                    ].iloc[0],
                ),
                # n2=ResultConditions(
                #     dil_high=sim_type_conditions[
                #         (
                #             (sim_type_conditions["diluent"] == "N2")
                #             & (sim_type_conditions["match"] == "mf")
                #             & (sim_type_conditions["dil_condition"] == "high")
                #         )
                #     ].iloc[0],
                #     dil_low=sim_type_conditions[
                #         (
                #             (sim_type_conditions["diluent"] == "N2")
                #             & (sim_type_conditions["match"] == "mf")
                #             & (sim_type_conditions["dil_condition"] == "low")
                #         )
                #     ].iloc[0],
                # ),
                n2=ResultConditions(
                    dil_high=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "N2")
                            & (sim_type_conditions["match"] == "tad")
                            & (sim_type_conditions["dil_condition"] == "high")
                        )
                    ].iloc[0],
                    dil_low=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "N2")
                            & (sim_type_conditions["match"] == "tad")
                            & (sim_type_conditions["dil_condition"] == "low")
                        )
                    ].iloc[0],
                ),
            )
        )
    return tuple(all_condition_ids)


@dataclass
class PlotArgs:
    save_plots: bool
    show_title: bool
    n2_row: NitrogenRow
    con: Connection

    data_columns: Tuple[Tuple[DataColumnReaction, ...], Tuple[DataColumnSpecies, ...]] = (
        (
            DataColumnReaction.fwd_rate_of_progress,
            DataColumnReaction.fwd_rate_constant,
            DataColumnReaction.rev_rate_constant,
            DataColumnReaction.rev_rate_of_progress,
            DataColumnReaction.net_rate_of_progress,
        ),
        (
            DataColumnSpecies.concentration,
            DataColumnSpecies.creation_rate,
            DataColumnSpecies.mole_frac,
            DataColumnSpecies.destruction_rate,
            DataColumnSpecies.net_production_rate,
        ),
    )

    @cached_property
    def conditions(self) -> pd.DataFrame:
        return load_conditions_data(self.con)

    @cached_property
    def reactions(self) -> pd.DataFrame:
        return load_reactions_data(self.con)

    @cached_property
    def species(self) -> pd.DataFrame:
        return load_species_data(self.con)

    @cached_property
    def data_sources(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.reactions, self.species

    @cached_property
    def condition_ids(self) -> Tuple[PlotConditions, ...]:
        return get_condition_ids(self.conditions)

    @cached_property
    def cv_condition_ids(self) -> Optional[PlotConditions]:
        for c_id in self.condition_ids:
            # there can be only one (of each type)
            if c_id.sim_type == "cv":
                return c_id
        return None


def full_window_plots(plot_args: PlotArgs):
    save_to = None
    for data_source, data_columns in zip(plot_args.data_sources, plot_args.data_columns):
        for condition_ids in plot_args.condition_ids:
            for data_column in data_columns:
                if plot_args.save_plots:
                    save_to = (
                        f"plots/{condition_ids.sim_type} - {data_column.data_type} - {data_column.column_name} "
                        "(full window).pdf"
                    )
                SimulationPlots(
                    data=data_source,
                    condition_ids=condition_ids,
                    data_column=data_column,
                    n2_row=plot_args.n2_row,
                    show_title=plot_args.show_title,
                    save_to=save_to,
                )

    # Zoomed in on reactions with lower forward progress rates, CV simulations only, full window
    data_column = DataColumnReaction.fwd_rate_of_progress
    cv_condition_ids = plot_args.cv_condition_ids
    if cv_condition_ids is None:
        raise RuntimeError("Plot data does not contain CV simulation results")
    if plot_args.save_plots:
        save_to = f"plots/cv - {data_column.data_type} - {data_column.column_name} (zoomed, induction window).pdf"
    SimulationPlots(
        data=plot_args.reactions[~plot_args.reactions.reaction.str.startswith("CO + OH")],
        condition_ids=cv_condition_ids,
        data_column=data_column,
        n2_row=plot_args.n2_row,
        show_title=plot_args.show_title,
        save_to=save_to,
    )


def induction_time_centered_plots(plot_args: PlotArgs, induction_window: float):
    save_to = None
    for data_source, data_columns in zip(plot_args.data_sources, plot_args.data_columns):
        for condition_ids in plot_args.condition_ids:
            for data_column in data_columns:
                if plot_args.save_plots:
                    save_to = (
                        f"plots/{condition_ids.sim_type} - {data_column.data_type} - {data_column.column_name} "
                        "(induction window).pdf"
                    )
                SimulationPlots(
                    data=data_source,
                    condition_ids=condition_ids,
                    data_column=data_column,
                    n2_row=plot_args.n2_row,
                    show_title=plot_args.show_title,
                    save_to=save_to,
                    induction_window=induction_window,
                )

    # Zoomed in on reactions with lower forward progress rates, CV simulations only, full window
    data_column = DataColumnReaction.fwd_rate_of_progress
    cv_condition_ids = plot_args.cv_condition_ids
    if cv_condition_ids is None:
        raise RuntimeError("Plot data does not contain CV simulation results")
    if plot_args.save_plots:
        save_to = f"plots/cv - {data_column.data_type} - {data_column.column_name} (zoomed, induction window).pdf"
    SimulationPlots(
        data=plot_args.reactions[~plot_args.reactions.reaction.str.startswith("CO + OH")],
        condition_ids=cv_condition_ids,
        data_column=data_column,
        n2_row=plot_args.n2_row,
        show_title=plot_args.show_title,
        save_to=save_to,
        induction_window=induction_window,
    )


def induction_time_centered_diff_plots(plot_args: PlotArgs, induction_window: float):
    save_to = None
    for data_source, data_columns in zip(plot_args.data_sources, plot_args.data_columns):
        for condition_ids in plot_args.condition_ids:
            for data_column in data_columns:
                if plot_args.save_plots:
                    save_to = (
                        f"plots/{condition_ids.sim_type} - {data_column.data_type} - {data_column.column_name} "
                        "(induction window, differences).pdf"
                    )
                SimulationPlots(
                    data=data_source,
                    condition_ids=condition_ids,
                    data_column=data_column,
                    n2_row=plot_args.n2_row,
                    show_title=plot_args.show_title,
                    save_to=save_to,
                    induction_window=induction_window,
                    plot_difference=True,
                )

    # Zoomed in on reactions with lower forward progress rates, CV simulations only, full window
    data_column = DataColumnReaction.fwd_rate_of_progress
    cv_condition_ids = plot_args.cv_condition_ids
    if cv_condition_ids is None:
        raise RuntimeError("Plot data does not contain CV simulation results")
    if plot_args.save_plots:
        save_to = (
            f"plots/cv - {data_column.data_type} - {data_column.column_name} "
            "(zoomed, induction window, differences).pdf"
        )
    SimulationPlots(
        data=plot_args.reactions[~plot_args.reactions.reaction.str.startswith("CO + OH")],
        condition_ids=cv_condition_ids,
        data_column=data_column,
        n2_row=plot_args.n2_row,
        show_title=plot_args.show_title,
        save_to=save_to,
        induction_window=induction_window,
        plot_difference=True,
    )


def co2_minus_tad_matched_n2(
    data: pd.DataFrame,
    conditions: PlotConditions,
    group_column: str,
    data_column: str,
) -> pd.DataFrame:
    subtracted = []
    for group_label, grouped in data.groupby(group_column):
        for co2_conditions, n2_conditions in (
            (conditions.co2.dil_low, conditions.n2.dil_low),
            (conditions.co2.dil_high, conditions.n2.dil_high),
        ):
            co2 = grouped[grouped.condition_id == co2_conditions.id].copy()
            n2 = grouped[grouped.condition_id == n2_conditions.id]
            for column in (data_column, "pressure", "temperature"):
                fit = interp1d(
                    # Extend interp range to accommodate induction time shift
                    [-np.inf, *n2.time.sub(n2_conditions.t_ind).to_numpy(), np.inf],
                    [n2[column].iloc[0], *n2[column].to_numpy(), n2[column].iloc[-1]],
                )
                co2[column] = co2[column] - fit(co2.time - co2_conditions.t_ind)
            # columns are no longer relevant, but used in later grouping
            co2["diluent"] = None
            co2["dil_mf"] = None
            subtracted.append(co2)

    return pd.concat(subtracted).sort_values("time")


def main():
    set_palette()
    set_style()
    mpl.rcParams["lines.linewidth"] = 1

    db_path = "/home/mick/DetResearch/scripts/final_manuscript/co2_reaction_study.sqlite"
    show_plots = False
    induction_window = 1e-7
    plot_args = PlotArgs(save_plots=True, show_title=True, n2_row=NitrogenRow.tad, con=sqlite3.connect(db_path))

    # full_window_plots(plot_args)
    # induction_time_centered_plots(plot_args, induction_window)
    induction_time_centered_diff_plots(plot_args, induction_window)

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
