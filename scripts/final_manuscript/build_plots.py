from collections import defaultdict
from pathlib import Path
from typing import Annotated

from file_utils import rm_rf
from matplotlib import pyplot as plt
from typer import Typer, Option

from plotting import data
from plotting.plots import (
    cell_size,
    formatting,
    gamma,
    cj_speed_vs_equivalence,
    t_ind_vs_equivalence,
)
from plotting.plots.velocity import cj_ratio_vs_equivalence, sound_speed_vs_equivalence, \
    mach_vs_equivalence, cj_mach_vs_equivalence, measured_speed_vs_equivalence


app = Typer()


@app.command()
def main(
    show: bool = False,
    save: bool = True,
    out_dir: Annotated[
        Path,
        Option(help="Directory where plots will be saved, WILL BE EMPTIED BEFORE SAVE"),
    ] = Path(__file__).parent / "plots",
    out_filetype: str = "png",
    clean: bool = True,
):
    cell_size_data_all = data.load_cell_size_data()
    cell_size_data_measured = cell_size_data_all[cell_size_data_all["method"] == "measured"]
    cell_size_data_simulated = cell_size_data_all[cell_size_data_all["method"] == "simulated"]
    cell_size_data_ratio = data.calculate_simulated_measured_ratio(
        simulated=cell_size_data_simulated,
        measured=cell_size_data_measured,
    )
    sim_bulk_properties = data.load_cp_cv_sim_bulk_properties()
    sim_bulk_property_ratios = data.calculate_cp_cv_gamma_ratios(sim_bulk_properties)
    sim_conditions = data.load_sim_conditions()
    sim_inert = data.load_sim_inert()
    sim_inert_and_active = data.combine_inert_and_active(inert=sim_inert, active=sim_conditions)
    cell_size_data_ratio_inert = data.calculate_simulated_measured_ratio(
        simulated=sim_inert_and_active[sim_inert_and_active["diluent_kind"] == "inert"],
        measured=cell_size_data_measured,
    )

    formatting.set_style()

    plots = defaultdict(dict)
    plots["01_measured"]["01_measured_vs_raw_mole_concentration"] = cell_size.measured_vs_raw_mole_concentration.plot(cell_size_data_measured)
    plots["01_measured"]["02_measured_vs_co2_tad_normalized"] = cell_size.measured_vs_co2_tad_normalized.plot(cell_size_data_measured)

    plots["02_simulated"]["01_simulated_vs_raw_mole_concentration"] = cell_size.simulated_vs_raw_mole_concentration.plot(cell_size_data_simulated)
    plots["02_simulated"]["02_simulated_vs_co2_tad_normalized"] = cell_size.simulated_vs_co2_tad_normalized.plot(cell_size_data_simulated)

    plots["03_measured_vs_simulated"]["equivalence"] = cell_size.ratio_vs_equivalence.plot(cell_size_data_ratio)

    plots["04_gamma"]["01_actual"] = gamma.plot(sim_bulk_properties)
    plots["04_gamma"]["02_ratio"] = gamma.plot_ratios(sim_bulk_property_ratios)

    plots["05_speed"]["01_measured_vs_equivalence"] = measured_speed_vs_equivalence.plot(sim_conditions)
    plots["05_speed"]["02_cj_vs_equivalence"] = cj_speed_vs_equivalence.plot(sim_conditions)
    plots["05_speed"]["03_cj_ratio_vs_equivalence"] = cj_ratio_vs_equivalence.plot(sim_conditions)

    plots["06_speed"]["04_sound_speed_vs_equivalence"] = sound_speed_vs_equivalence.plot(sim_conditions)
    plots["06_speed"]["05_mach_vs_equivalence"] = mach_vs_equivalence.plot(sim_conditions)
    plots["06_speed"]["06_cj_mach_vs_equivalence"] = cj_mach_vs_equivalence.plot(sim_conditions)

    # # In these plots we see only what we'd expect
    # t_ind_vs_equivalence.plot(sim_conditions)  # no physical measurement for comparison other than cell size

    plots["07_simulated_inert"]["01_inert_vs_active"] = cell_size.inert.vs_active(sim_inert_and_active)
    # plots["07_simulated_inert"]["01_inert_vs_active"] = cell_size.inert.vs_active_2(sim_inert_and_active)
    plots["07_simulated_inert"]["02_inert_vs_measured"] = cell_size.inert.vs_measured(active=cell_size_data_ratio, inert=cell_size_data_ratio_inert)

    if show:
        plt.show()

    if save:
        if out_dir.exists() and clean:
            rm_rf(out_dir)

        for dir_base, grouped_plots in plots.items():
            this_dir = out_dir / dir_base
            this_dir.mkdir(parents=True, exist_ok=True)
            for plot_name, plot_facet in  grouped_plots.items():
                plot_facet.figure.savefig(this_dir / f"{plot_name}.{out_filetype}")


if __name__ == "__main__":
    app()
