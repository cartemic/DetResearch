import concurrent
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import cantera as ct
import numpy as np
import pandas as pd
import pint
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from pypbomb import DDT, Bolt, Tube, Window
from tqdm import tqdm

from funcs.simulation.thermo import diluted_species_dict

for_slides = False
if for_slides:
    bg_color = "#eff0ea"
    axes_color = "#565d5d"
else:
    bg_color = "#ffffff"
    axes_color = "#000000"

good = "#0CDA17"
bad = "#D81B60"
colors = [
    "#fe6100",
    "#648fff",
    "#dc267f",
]

warnings.simplefilter("ignore", pint.UnitStrippedWarning)

ureg = pint.UnitRegistry()
quant = ureg.Quantity

sns.set(
    font="serif",
    style="white",
    context="paper",
    rc={
        "axes.edgecolor": axes_color,
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color,
        "axes.labelcolor": axes_color,
        "xtick.color": axes_color,
        "ytick.color": axes_color,
        "axes.grid": False,
        "hatch.linewidth": 2,
    },
)
sns.set_palette(colors)


def plot_max_initial_pressure(
    plot_data: pd.DataFrame,
    temperature_key: str,
    pressure_key: str,
    save_plot: bool,
):
    grid = sns.relplot(
        x=temperature_key,
        y=pressure_key,
        col="NPS",
        style="Schedule",
        hue="Schedule",
        kind="line",
        data=plot_data,
        aspect=0.5,
    )
    sns.despine()
    grid.fig.set_size_inches((6.5, 2))
    grid.tight_layout()
    if save_plot:
        grid.savefig("plots/max_initial_pressure.pdf")


def plot_dlf(plot_data: pd.DataFrame):
    sns.catplot(x="Schedule", y="DLF", col="NPS", kind="bar", data=plot_data, aspect=0.5)


def plot_window_length(
    window_lengths: np.array,
    window_thicknesses: np.array,
    max_desired_thickness: pint.Quantity,
    show_title: bool,
    save_plot: bool,
):
    fig, ax = plt.subplots(figsize=(6.5, 2))
    ax.plot(window_lengths, window_thicknesses, "k")
    ax.fill_between(
        window_lengths[window_thicknesses <= max_desired_thickness],
        window_thicknesses[window_thicknesses <= max_desired_thickness],
        0,
        zorder=-1,
        color=bad,
        alpha=0.25,
        hatch="xxx",
        facecolor="none",
        lw=2,
    )
    ax.set_xlim([window_lengths.min().magnitude, window_lengths.max().magnitude])
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlabel("Length (mm)")
    ax.set_ylabel("Thickness\n(mm)")
    if show_title:
        ax.set_title(
            "Window minimum thickness vs. length\n" "Max length {:3.2f} mm at {:3.2f} mm thick".format(
                window_lengths[window_thicknesses <= max_desired_thickness].max().magnitude,
                max_desired_thickness,
            )
        )
    sns.despine()
    plt.tight_layout()
    if save_plot:
        plt.savefig("plots/window_length.pdf")


def plot_bolt_safety_factors(
    num_bolts: np.array,
    bolt_safety_factors: dict,
    show_title: bool,
    save_plot: bool,
):
    fig, ax = plt.subplots(figsize=(6.5, 2))
    ax.plot(num_bolts, bolt_safety_factors["bolt"], "--", c=axes_color, label="Bolt")
    ax.plot(num_bolts, bolt_safety_factors["plate"], "-.", c=axes_color, label="Viewing Section")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlim([num_bolts.min() - 1, num_bolts.max() + 1])
    ax.set_xlabel("Number of bolts")
    ax.set_ylabel("FS")
    if show_title:
        ax.set_title("Bolt and plate safety factors vs. number of bolts")
    ax.fill_between(
        ax.get_xlim(),
        1,
        zorder=-1,
        color=bad,
        alpha=0.25,
        hatch="xxx",
        facecolor="none",
        lw=2,
    )
    sns.despine()
    plt.tight_layout()
    if save_plot:
        plt.savefig("plots/bolt_safety_factors.pdf")


def calculate_single_tube_result(
    size: str,
    schedule: str,
    initial_temperature: pint.Quantity,
    material: str,
    spec: dict,
    mechanism: str,
    temperature_key: str,
    pressure_key: str,
) -> pd.Series:
    dims = Tube.get_dimensions(size, schedule, unit_registry=ureg)
    max_stress = Tube.calculate_max_stress(initial_temperature, material, welded=False, unit_registry=ureg)
    max_pressure = Tube.calculate_max_pressure(
        dims["inner_diameter"],
        dims["outer_diameter"],
        max_stress,
        # safety_factor=0.25,
    ).to("kPa")
    elastic_modulus = Tube.get_elastic_modulus(material, ureg)
    density = Tube.get_density(material, ureg)
    poisson = Tube.get_poisson(material)
    initial_pressure, dlf = Tube.calculate_max_initial_pressure(
        dims["inner_diameter"],
        dims["outer_diameter"],
        initial_temperature,
        spec,
        mechanism,
        max_pressure.to("Pa"),  # required conversion due to bug in pypbomb!
        elastic_modulus,
        density,
        poisson,
        use_multiprocessing=True,
        return_dlf=True,
    )

    current_results = pd.Series(dtype="object")
    current_results["Schedule"] = schedule
    current_results["NPS"] = size
    current_results[pressure_key] = initial_pressure.to("kPa").magnitude
    current_results[temperature_key] = initial_temperature.to("K").magnitude
    current_results["tube_temp"] = initial_temperature
    current_results["max_pressure"] = max_pressure
    current_results["inner_diameter"] = dims["inner_diameter"]
    current_results["DLF"] = dlf

    return current_results


def plot_mixture_comparison(
    results: pd.DataFrame,
    results_2: pd.DataFrame,
    temperature_key: str,
    pressure_key: str,
    show_title: bool,
    save_plot: bool,
):
    results["Mixture"] = "2"
    results_2["Mixture"] = "1"
    grid = sns.relplot(
        x=temperature_key,
        y=pressure_key,
        col="NPS",
        row="Schedule",
        style="Mixture",
        style_order=["1", "2"],
        hue="Mixture",
        hue_order=["1", "2"],
        kind="line",
        data=pd.concat((results, results_2)),
    )
    if show_title:
        grid.fig.suptitle("Mixture comparison\nMax safe initial pressure vs. Temperature")
    for ax in grid.axes.flatten():
        title = ax.get_title()
        ax.set_title(title.replace(" = ", " ").replace(" | ", "\n"))
        if ax.get_xlabel():
            ax.set_xlabel("$T_{0}$ (K)")
        if ax.get_ylabel():
            ax.set_ylabel("$P_{0, max}$ (kPa)")
    grid.fig.set_size_inches((6.5, 6.5))
    grid.tight_layout()
    sns.despine()
    if save_plot:
        grid.savefig("plots/max_initial_pressure_mixture_comparison.pdf")


def main(
    calculate_first_results: bool,
    calculate_second_results: bool,
    show_plots: bool,
    show_titles: bool,
    save_plots: bool,
):
    fuel = "CH4"
    oxidizer = "N2O"
    mechanism = "gri30_highT.yaml"
    gas = ct.Solution(mechanism)
    gas.set_equivalence_ratio(1, fuel, oxidizer)
    spec = diluted_species_dict(
        gas.mole_fraction_dict(),
        "CO2",
        0.2,
    )
    material = "316L"
    nps_main = "6"
    schedule_main = "80"
    nps_mix = "1"
    schedule_mix = "40"
    window_height = quant(5.75, "in")
    potential_sizes = ["1", "3", "4", "6"]  # strings because sizes aren't necessarily numeric

    common_sizes = set(Tube.get_available_pipe_schedules(potential_sizes[0]))
    for size in potential_sizes[1:]:
        common_sizes.intersection_update(set(Tube.get_available_pipe_schedules(size)))
    potential_schedules = ["40", "80", "XXH"]

    results_file = "tube_size_results.h5"
    pressure_key = "Max initial pressure (kPa)"
    temperature_key = "Initial temperature (K)"
    initial_temperatures = quant(np.linspace(0, 100, 11), "degF").to("K")
    combinations = list(product(potential_schedules, potential_sizes, initial_temperatures))
    n_runs = len(combinations)
    if calculate_first_results:
        results = pd.DataFrame(
            columns=[
                "Schedule",
                "NPS",
                pressure_key,
                temperature_key,
                "tube_temp",
                "max_pressure",
                "DLF",
            ]
        )
        with (
            ProcessPoolExecutor() as executor,
            tqdm(total=n_runs, unit="calc", file=sys.stdout, colour="green", desc="Running") as counter,
        ):
            futures = {
                executor.submit(
                    calculate_single_tube_result,
                    size,
                    schedule,
                    initial_temperature,
                    material,
                    spec,
                    mechanism,
                    temperature_key,
                    pressure_key,
                )
                for (schedule, size, initial_temperature) in combinations
            }

            finished = []

            for done in concurrent.futures.as_completed(futures):
                finished.append(done.result())
                counter.update()

            counter.set_description_str("Done")

        for result in finished:
            results = pd.concat((results, result.to_frame().T), ignore_index=True)

        float_keys = [pressure_key, temperature_key, "DLF"]
        results[float_keys] = results[float_keys].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.HDFStore(results_file, "w") as store:
                store.put("data", results)

    else:
        with pd.HDFStore(results_file, "r") as store:
            results = store.data
    results.sort_values("NPS", inplace=True)

    plot_max_initial_pressure(results, temperature_key=temperature_key, pressure_key=pressure_key, save_plot=save_plots)
    plot_dlf(results)

    results_main = results[(results["NPS"] == nps_main) & (results["Schedule"] == schedule_main)].copy()
    results_main.drop([pressure_key, temperature_key], axis=1, inplace=True)

    results_mix = results[(results["NPS"] == nps_mix) & (results["Schedule"] == schedule_mix)].drop(
        [pressure_key, temperature_key, "inner_diameter"], axis=1
    )
    results_mix["max_main_pressure"] = results_main["max_pressure"].to_numpy()
    results_mix["safe"] = results_mix["max_pressure"] > results_mix["max_main_pressure"]
    if all(results_mix["safe"]):
        print("Mix tube is safe :)")
    else:
        print("Mix tube is unsafe :(")

    # results_main["flange class"] = results_main.apply(
    #     lambda x: Flange.get_class(x["max_pressure"], x["tube_temp"], material, ureg), axis=1
    # )

    window_lengths = quant(np.linspace(1, 3, 10), "in").to("mm")
    window_thicknesses = (
        Window.minimum_thickness(
            length=window_height,
            width=window_lengths,
            safety_factor=4,
            pressure=results_main["max_pressure"].max(),
            rupture_modulus=(197.9, "MPa"),
            unit_registry=ureg,
        )
        .to("mm")
        .magnitude
    )

    max_desired_thickness = quant(1, "in").to("mm").magnitude  # inch
    plot_window_length(window_lengths, window_thicknesses, max_desired_thickness, show_titles, save_plots)

    window_length = quant(2.25, "in")
    Window.safety_factor(
        window_length,
        window_height,
        quant(1, "in"),
        pressure=results_main["max_pressure"].max(),
        rupture_modulus=quant(197.9, "MPa"),
    )

    num_bolts = np.array(range(1, 25))
    bolt_safety_factors = Bolt.calculate_safety_factors(
        max_pressure=results_main["max_pressure"].max(),
        window_area=window_length * window_height,
        num_bolts=num_bolts,
        thread_size="1/4-28",
        thread_class="2",
        bolt_max_tensile=(150, "kpsi"),  # grade 8
        plate_max_tensile=(485, "MPa"),  # 316L,
        engagement_length=(0.5, "in"),
        unit_registry=ureg,
    )
    plot_bolt_safety_factors(num_bolts, bolt_safety_factors, show_titles, save_plots)

    main_id = results_main["inner_diameter"].iloc[0]
    target_blockage_diameter = DDT.calculate_blockage_diameter(main_id, 0.45, unit_registry=ureg)
    print("Target blockage diameter: {:3.2f}".format(target_blockage_diameter.to("mm")))

    blockage_actual = DDT.calculate_blockage_ratio(main_id, quant(0.75, "in").to("mm"), unit_registry=ureg)
    print("Actual blockage ratio: {:4.1f}%".format(blockage_actual * 100))

    runup = DDT.calculate_run_up(
        blockage_actual, main_id, (70, "degF"), (1, "atm"), gas.mole_fraction_dict(), mechanism, unit_registry=ureg
    )
    print("Runup distance: {:1.2f}".format(runup.to("m")))

    results_file_2 = "second_mixture_results.h5"
    fuel_2 = "C3H8"
    oxidizer_2 = "O2:1 N2:3.76"
    gas_2 = ct.Solution(mechanism)
    gas_2.set_equivalence_ratio(1, fuel_2, oxidizer_2)

    if calculate_second_results:
        results_2 = pd.DataFrame(columns=[pressure_key, temperature_key])
        with (
            ProcessPoolExecutor() as executor,
            tqdm(total=n_runs, unit="calc", file=sys.stdout, colour="green", desc="Running") as counter,
        ):
            futures = {
                executor.submit(
                    calculate_single_tube_result,
                    size,
                    schedule,
                    initial_temperature,
                    material,
                    gas_2.mole_fraction_dict(),
                    mechanism,
                    temperature_key,
                    pressure_key,
                )
                for (schedule, size, initial_temperature) in combinations
            }

            finished = []

            for done in concurrent.futures.as_completed(futures):
                finished.append(done.result())
                counter.update()

            counter.set_description_str("Done")

        for result in finished:
            results_2 = pd.concat((results_2, result.to_frame().T), ignore_index=True)

        float_keys = [pressure_key, temperature_key]
        results_2[float_keys] = results_2[float_keys].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.HDFStore(results_file_2, "w") as store:
                store.put("data", results_2)

    else:
        with pd.HDFStore(results_file_2, "r") as store:
            results_2 = store.data

    plot_mixture_comparison(results, results_2, temperature_key, pressure_key, show_titles, save_plots)

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main(
        calculate_first_results=False,
        calculate_second_results=False,
        show_plots=False,
        show_titles=False,
        save_plots=True,
    )
