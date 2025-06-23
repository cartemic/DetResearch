from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

from scripts.perturbation_study import data


@dataclass
class CoefficientData:
    subscript: str
    title: str

    @cached_property
    def normalized_column(self) -> str:
        return f"c_{self.subscript}"

    @cached_property
    def non_normalized_column(self) -> str:
        return f"dl/d{self.subscript}"


class Coefficients(Enum):
    mul = CoefficientData(subscript="mul", title="Perturbation Fraction")
    rcc = CoefficientData(subscript="RCC", title="Relative Chemical Contribution")


def load_coefficient_data() -> data.NSCByDiluentDataframe:
    conditions = data.load_conditions()
    condition_ids = tuple(conditions.perturbed.index.union(conditions.unperturbed.index))
    reactions = data.load_reactions(condition_ids)
    cell_sizes = data.analyze_cell_sizes(conditions)
    rcc = data.calculate_rcc(conditions, reactions)
    normalized_coefficients = data.calculate_normalized_sensitivity_coefficients(cell_sizes, rcc)
    coefficients_by_diluent = data.group_coefficients_by_diluent(normalized_coefficients)

    return coefficients_by_diluent


def load_species_timeseries_data(data_column: data.SpeciesDataColumn) -> tuple[str, data.SpeciesTimeseriesDataframe]:
    return data_column, data.load_species_timeseries(
        data_column=data_column,
        species=(
            "C2H",
            "C2H2",
            "C2H5",
            "C2H6",
            "CH2",
            "CH2(S)",
            "CH3",
            "CH3O",
            "CH4",
            "CO2",
            "H",
            "H2",
            "H2O",
            "H2O2",
            "HCN",
            "HCNN",
            "HO2",
            "N2",
            "N2O",
            "NH",
            "NO",
            "O",
            "O2",
            "OH",
        ),
        dil_conditions=("low", "high"),
    )


def plot_normalized_sensitivity_coefficients(
    plot_data: data.NSCByDiluentDataframe,
    with_title: bool,
) -> dict[Path, plt.Figure]:
    diluent: str
    figures = {}
    for diluent, grouped_data in plot_data.groupby("diluent"):
        for coefficient in Coefficients:
            column = coefficient.value.normalized_column
            diluent_fmt = pretty_species(diluent)
            grid = sns.catplot(
                x=column,
                y="reaction",
                hue="method",
                hue_order=["active", "inert"],
                col="dil_condition",
                col_order=["low", "high"],
                data=data.top_n_by_method(grouped_data, column, 10),
                kind="bar",
                orient="h",
                sharey=False,
            )
            grid.despine()
            grid.set_xlabels("Normalized Sensitivity Coefficient")
            grid.set_ylabels("Reaction")
            grid.legend.set_title(diluent_fmt)
            for ax in grid.axes.flatten():
                condition = ax.get_title().replace("dil_condition = ", "").capitalize()
                ax.set_title(f"{condition} Dilution")

            if with_title:
                grid.fig.suptitle(f"{coefficient.value.title} Sensitivity ({diluent_fmt})", weight="bold")
                grid.fig.subplots_adjust(top=0.875)

            figures[Path(coefficient.value.subscript) / diluent / "normalized.png"] = grid.figure

    return figures


def plot_inert_diffs(
    plot_data: data.NSCByDiluentDataframe,
    with_title: bool,
) -> dict[Path, plt.Figure]:
    diluent: str
    figures = {}
    for diluent, grouped_data in plot_data.groupby("diluent"):
        for coefficient in Coefficients:
            diluent_fmt = pretty_species(diluent)
            column = coefficient.value.normalized_column
            subscript = coefficient.value.subscript
            grid = sns.catplot(
                x=column,
                y="reaction",
                col="dil_condition",
                col_order=["low", "high"],
                data=data.top_n_inert_diffs(grouped_data, column, 10),
                kind="bar",
                orient="h",
                sharey=False,
            )
            grid.despine()
            grid.set_xlabels(f"$c_{{active,{subscript}}} - c_{{inert,{subscript}}}$")
            grid.set_ylabels("Reaction")
            for ax in grid.axes.flatten():
                condition = ax.get_title().replace("dil_condition = ", "").capitalize()
                ax.set_title(f"{condition} Dilution")
                ax.grid(alpha=0.5)
                ax.set_axisbelow(True)

            if with_title:
                grid.fig.suptitle(f"Change in {coefficient.value.title} Sensitivity ({diluent_fmt})", weight="bold")
                grid.fig.subplots_adjust(top=0.875)

            figures[Path(coefficient.value.subscript) / diluent / "inert_diffs.png"] = grid.figure

    return figures


def pretty_species(diluent: str) -> str:
    return f"${diluent.replace('2', '_{2}')}$"


def plot_non_normalized_coefficients(
    plot_data: data.NSCByDiluentDataframe,
    with_title: bool,
) -> dict[Path, plt.Figure]:
    diluent: str
    figures = {}
    for diluent, grouped_data in plot_data.groupby("diluent"):
        for coefficient in Coefficients:
            diluent_fmt = pretty_species(diluent)
            column = coefficient.value.non_normalized_column
            grid = sns.catplot(
                x=column,
                y="reaction",
                hue="method",
                hue_order=["active", "inert"],
                col="dil_condition",
                col_order=["low", "high"],
                data=data.top_n_by_method(grouped_data, column, 10),
                kind="bar",
                orient="h",
                sharey=False,
            )
            grid.despine()
            if with_title:
                grid.fig.suptitle(f"{coefficient.value.title} Sensitivity ({diluent_fmt})", weight="bold")
                grid.fig.subplots_adjust(top=0.875)

            grid.set_xlabels("Sensitivity Coefficient")
            grid.set_ylabels("Reaction")
            grid.legend.set_title(diluent_fmt)
            for ax in grid.axes.flatten():
                condition = ax.get_title().replace("dil_condition = ", "").capitalize()
                ax.set_title(f"{condition} Dilution")

            figures[Path(coefficient.value.subscript) / diluent / "non_normalized.png"] = grid.figure

    return figures


def plot_diluent_species_timeseries_grid(
    diluent: str,
    time_basis_column: str,
    time_basis_display: str,
    data_column: str,
    data_column_display: str,
    diluent_data: data.SpeciesTimeseriesDataframe,
) -> plt.Figure:
    # method: str
    fig = plt.figure(figsize=(6, 6), layout="constrained")
    plot_row = fig.subfigures(2, 1, wspace=0.1)
    for main_grid_row, (dil_condition, df_dil_condition) in enumerate(diluent_data.groupby("dil_condition")):
        # noinspection PyUnresolvedReferences
        plot_row[main_grid_row].suptitle(f"{dil_condition} dilution".title())
        # noinspection PyUnresolvedReferences
        # plot_col = plot_row[main_grid_row].subfigures(1, 2, hspace=0.1)
        # for dil_grid_col, (method, df_method) in enumerate(df_dil_condition.groupby("method")):
        dil_grid_col = 0
        show_x_label = main_grid_row == 1
        show_y_label = dil_grid_col == 0
        show_legend = (main_grid_row == 0 and dil_grid_col == 0)
        grid_fig = plot_row[main_grid_row]
        plot_diluent_species_timeseries(
            fig=grid_fig,
            diluent=diluent,
            diluent_activity_type=dil_condition,
            diluent_data=df_dil_condition,
            time_basis_column=time_basis_column,
            time_basis_display=time_basis_display if show_x_label else None,
            data_column=data_column,
            data_column_display=data_column_display if show_y_label else None,
            show_legend=show_legend,
        )
    return fig


def plot_diluent_species_timeseries(
    fig: plt.Figure,
    diluent: str,
    diluent_activity_type: str,
    diluent_data: data.SpeciesTimeseriesDataframe,
    time_basis_column: str,
    time_basis_display: str | None,
    data_column: str,
    data_column_display: str | None,
    show_legend: bool = True,
) -> None:
    fig.suptitle(f"{diluent_activity_type.title()} {pretty_species(diluent)} Dilution")
    ax = fig.subplots(1, 1)
    sns.lineplot(
        diluent_data,
        x=time_basis_column,
        y=data_column,
        ax=ax,
        hue="species",
        style="method",
    )
    ax.set(yscale="log")
    ax.set(xlabel=time_basis_display, ylabel=data_column_display)
    sns.despine(ax=ax)

    if show_legend:
        leg = ax.get_legend()
        leg.set_title(leg.get_title().get_text().title())
    else:
        ax.get_legend().remove()


def plot_all_species_timeseries(
    data_column: str,
    data_column_display: str,
    species_data: data.SpeciesTimeseriesDataframe,
) -> dict[Path, plt.Figure]:
    diluent: str
    figures = {}
    time_basis = {
        "time": "Time (sec)",
        "progress": "Induction Progress",
    }
    for time_basis_column, time_basis_display in time_basis.items():
        for diluent, diluent_data in species_data.groupby("diluent"):
            fig = plot_diluent_species_timeseries_grid(
                diluent=diluent,
                time_basis_column=time_basis_column,
                time_basis_display=time_basis_display,
                data_column=data_column,
                data_column_display=data_column_display,
                diluent_data=diluent_data,
            )
            figures[Path(time_basis_column) / f"{diluent}.png"] = fig

    return figures


def main(show: bool, save: bool):
    with_title = not save

    plot_data = load_coefficient_data()
    data_column, species_data = load_species_timeseries_data("mole_frac")
    data_column_display = "Mole Fraction"
    minimum_progress = 0.9

    non_normalized = plot_non_normalized_coefficients(plot_data, with_title)
    normalized = plot_normalized_sensitivity_coefficients(plot_data, with_title)
    species = plot_all_species_timeseries(
        data_column=data_column,
        data_column_display=data_column_display,
        species_data=species_data[species_data["progress"] >= minimum_progress],
    )
    if save:
        output_dir = Path(__file__).parent / "writeup" / "images"
        coefficients_dir = output_dir / "coefficients"
        timeseries_dir = output_dir / "timeseries"
        if output_dir.exists():
            rm_rf(output_dir)

        for coefficient_plots in (normalized, non_normalized):
            for pth, fig in coefficient_plots.items():
                fig_path = coefficients_dir / pth
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path)

        for pth, fig in species.items():
            fig_path = timeseries_dir / pth
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path)


    if show:
        plt.show()


def rm_rf(path: Path) -> None:
    for thing in path.iterdir():
        if thing.is_file():
            thing.unlink()
        else:
            rm_rf(thing)
            thing.rmdir()


if __name__ == "__main__":
    main(
        show=False,
        save=True,
    )
