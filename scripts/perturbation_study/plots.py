import seaborn as sns
from matplotlib import pyplot as plt

from scripts.perturbation_study import data

COEFFICIENT_NAMES = {
    # "c_mul": "Perturbation Fraction",
    "c_RCC": "Relative Chemical Contribution",
}

NON_NORMALIZED_COEFFICIENT_EQUATIONS = {
    "dl/dmul": 5,
    "dl/dRCC": 10,
}

NON_NORMALIZED_COEFFICIENT_NAMES = {
    # "dl/dmul": "Perturbation Fraction",
    "dl/dRCC": "Relative Chemical Contribution",
}


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
        species=("H", "OH"),
        dil_conditions=("low", "high"),
    )


def plot_normalized_sensitivity_coefficients(plot_data: data.NSCByDiluentDataframe) -> None:
    """
    c (eq. 1, using eq. 5 or 10 as the basis)
    """
    diluent: str
    for diluent, grouped_data in plot_data.groupby("diluent"):
        for target, target_title in COEFFICIENT_NAMES.items():
            diluent_fmt = pretty_species(diluent)
            grid = sns.catplot(
                x=target,
                y="reaction",
                hue="method",
                hue_order=["active", "inert"],
                col="dil_condition",
                col_order=["low", "high"],
                data=data.top_n_by_method(grouped_data, target, 10),
                kind="bar",
                orient="h",
                sharey=False,
            )
            grid.despine()
            grid.fig.suptitle(f"{target_title} Sensitivity ({diluent_fmt}) (eq. 1)", weight="bold")
            grid.fig.subplots_adjust(top=0.875)
            grid.set_xlabels("Normalized Sensitivity Coefficient")
            grid.set_ylabels("Reaction")
            grid.legend.set_title(diluent_fmt)
            for ax in grid.axes.flatten():
                condition = ax.get_title().replace("dil_condition = ", "").capitalize()
                ax.set_title(f"{condition} Dilution")


def plot_inert_diffs(plot_data: data.NSCByDiluentDataframe) -> None:
    """
    delta c (eq. 2, using eq. 5 or 10 as the basis)
    """
    diluent: str
    for diluent, grouped_data in plot_data.groupby("diluent"):
        for target, target_title in COEFFICIENT_NAMES.items():
            coefficient_subscript = target.replace("c_", "")
            diluent_fmt = pretty_species(diluent)
            grid = sns.catplot(
                x=target,
                y="reaction",
                col="dil_condition",
                col_order=["low", "high"],
                data=data.top_n_inert_diffs(grouped_data, target, 10),
                kind="bar",
                orient="h",
                sharey=False,
            )
            grid.despine()
            grid.fig.suptitle(f"Change in {target_title} Sensitivity ({diluent_fmt}) (eq. 2)", weight="bold")
            grid.fig.subplots_adjust(top=0.875)
            grid.set_xlabels(f"$c_{{s,active,{coefficient_subscript}}} - c_{{s,inert,{coefficient_subscript}}}$")
            grid.set_ylabels("Reaction")
            for ax in grid.axes.flatten():
                condition = ax.get_title().replace("dil_condition = ", "").capitalize()
                ax.set_title(f"{condition} Dilution")
                ax.grid(alpha=0.5)
                ax.set_axisbelow(True)


def pretty_species(diluent: str) -> str:
    return f"${diluent.replace('2', '_{2}')}$"


def plot_non_normalized_coefficients(plot_data: data.NSCByDiluentDataframe) -> None:
    """
    C (eq 5, 10)
    """
    diluent: str
    for diluent, grouped_data in plot_data.groupby("diluent"):
        for target, target_title in NON_NORMALIZED_COEFFICIENT_NAMES.items():
            eq_no = NON_NORMALIZED_COEFFICIENT_EQUATIONS[target]
            diluent_fmt = pretty_species(diluent)
            grid = sns.catplot(
                x=target,
                y="reaction",
                hue="method",
                hue_order=["active", "inert"],
                col="dil_condition",
                col_order=["low", "high"],
                data=data.top_n_by_method(grouped_data, target, 10),
                kind="bar",
                orient="h",
                sharey=False,
            )
            grid.despine()
            grid.fig.suptitle(f"{target_title} Sensitivity ({diluent_fmt}) (eq. {eq_no})", weight="bold")
            grid.fig.subplots_adjust(top=0.875)
            grid.set_xlabels("Sensitivity Coefficient")
            grid.set_ylabels("Reaction")
            grid.legend.set_title(diluent_fmt)
            for ax in grid.axes.flatten():
                condition = ax.get_title().replace("dil_condition = ", "").capitalize()
                ax.set_title(f"{condition} Dilution")


def plot_diluent_species_timeseries_grid(
    diluent: str,
    time_basis_column: str,
    time_basis_display: str,
    data_column: str,
    data_column_display: str,
    diluent_data: data.SpeciesTimeseriesDataframe,
) -> None:
    method: str
    fig = plt.figure(figsize=(6, 6), layout="constrained")
    plot_row = fig.subfigures(2, 1, wspace=0.1)
    for main_grid_row, (dil_condition, df_dil_condition) in enumerate(diluent_data.groupby("dil_condition")):
        # noinspection PyUnresolvedReferences
        plot_row[main_grid_row].suptitle(f"{dil_condition} dilution".title())
        # noinspection PyUnresolvedReferences
        plot_col = plot_row[main_grid_row].subfigures(1, 2, hspace=0.1)
        for dil_grid_col, (method, df_method) in enumerate(df_dil_condition.groupby("method")):
            show_x_label = main_grid_row == 1
            show_y_label = dil_grid_col == 0
            show_legend = (main_grid_row == 0 and dil_grid_col == 0)
            grid_fig = plot_col[dil_grid_col]
            plot_diluent_species_timeseries(
                fig=grid_fig,
                diluent=diluent,
                diluent_activity_type=method,
                diluent_data=df_method,
                time_basis_column=time_basis_column,
                time_basis_display=time_basis_display if show_x_label else None,
                data_column=data_column,
                data_column_display=data_column_display if show_y_label else None,
                show_legend=show_legend,
            )


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
    fig.suptitle(f"{diluent_activity_type.title()} {pretty_species(diluent)}")
    ax = fig.subplots(1, 1)
    sns.lineplot(
        diluent_data,
        x=time_basis_column,
        y=data_column,
        ax=ax,
        hue="species",
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
    time_basis_column: str,
    time_basis_display: str,
    data_column: str,
    data_column_display: str,
    species_data: data.SpeciesTimeseriesDataframe,
) -> None:
    diluent: str
    for diluent, diluent_data in species_data.groupby("diluent"):
        plot_diluent_species_timeseries_grid(
            diluent=diluent,
            time_basis_column=time_basis_column,
            time_basis_display=time_basis_display,
            data_column=data_column,
            data_column_display=data_column_display,
            diluent_data=diluent_data,
        )


def main():
    plot_data = load_coefficient_data()
    data_column, species_data = load_species_timeseries_data("mole_frac")
    data_column_display = "Mole Fraction"
    minimum_progress = 0.9

    plot_non_normalized_coefficients(plot_data)
    plot_normalized_sensitivity_coefficients(plot_data)
    # plot_inert_diffs(plot_data)
    time_basis = {
        "time": "Time (sec)",
        "progress": "Induction Progress",
    }
    for time_basis_column, time_basis_display in time_basis.items():
        plot_all_species_timeseries(
            time_basis_column=time_basis_column,
            time_basis_display=time_basis_display,
            data_column=data_column,
            data_column_display=data_column_display,
            species_data=species_data[species_data["progress"] >= minimum_progress],
        )

    plt.show()


if __name__ == "__main__":
    main()
