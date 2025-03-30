from typing import TYPE_CHECKING

import seaborn as sns
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    from pandas import DataFrame

from scripts.perturbation_study import data

COEFFICIENT_NAMES = {
    "c_mul": "Perturbation Fraction",
    "c_RCC": "Relative Chemical Contribution",
}

def load_data() -> "DataFrame":
    conditions = data.load_conditions()
    # noinspection PyTypeChecker
    condition_ids: tuple[int] = tuple(
        set(conditions.perturbed.index.get_level_values("condition_id")).union(
            conditions.unperturbed.index.get_level_values("condition_id")
        )
    )
    reactions = data.load_reactions(condition_ids)
    cell_sizes = data.analyze_cell_sizes(conditions)
    rcc = data.calculate_rcc(conditions, reactions)
    coefficients = data.calculate_normalized_sensitivity_coefficients(cell_sizes, rcc)
    coefficients_by_diluent = data.group_coefficients_by_diluent(coefficients)

    return coefficients_by_diluent



def plot_all(plot_data: "DataFrame") -> None:
    diluent: str
    for diluent, grouped_data in plot_data.groupby("diluent"):
        for target, target_title in COEFFICIENT_NAMES.items():
            diluent_fmt = f"${diluent.replace('2', '_{2}')}$"
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
                # sharey=False,
            )
            grid.despine()
            grid.fig.suptitle(f"{target_title} Sensitivity ({diluent_fmt})", weight="bold")
            grid.fig.subplots_adjust(top=0.875)
            grid.set_xlabels("Normalized Sensitivity Coefficient")
            grid.set_ylabels("Reaction")
            grid.legend.set_title(diluent_fmt)
            for ax in grid.axes.flatten():
                condition = ax.get_title().replace("dil_condition = ", "").capitalize()
                ax.set_title(f"{condition} Dilution")


def plot_inert_diffs(plot_data: "DataFrame") -> None:
    diluent: str
    for diluent, grouped_data in plot_data.groupby("diluent"):
        for target, target_title in COEFFICIENT_NAMES.items():
            coefficient_subscript = target.replace("c_", "")
            diluent_fmt = f"${diluent.replace('2', '_{2}')}$"
            grid = sns.catplot(
                x=target,
                y="reaction",
                col="dil_condition",
                col_order=["low", "high"],
                data=data.top_n_inert_diffs(grouped_data, target, 10),
                kind="bar",
                orient="h",
                # sharey=False,
            )
            grid.despine()
            grid.fig.suptitle(f"Change in {target_title} Sensitivity ({diluent_fmt})", weight="bold")
            grid.fig.subplots_adjust(top=0.875)
            grid.set_xlabels(f"$k_{{active,{coefficient_subscript}}} - k_{{inert,{coefficient_subscript}}}$")
            grid.set_ylabels("Reaction")
            for ax in grid.axes.flatten():
                condition = ax.get_title().replace("dil_condition = ", "").capitalize()
                ax.set_title(f"{condition} Dilution")


def main():
    plot_data = load_data()

    plot_all(plot_data)
    plot_inert_diffs(plot_data)

    plt.show()


if __name__ == "__main__":
    main()
