from matplotlib import pyplot as plt

from scripts.final_manuscript.plotting import data
from scripts.final_manuscript.plotting.plots import (
    cell_size,
    formatting,
    gamma,
    cj_speed_vs_equivalence,
    t_ind_vs_equivalence,
)
from scripts.final_manuscript.plotting.plots.velocity import cj_ratio_vs_equivalence, sound_speed_vs_equivalence, \
    mach_vs_equivalence, cj_mach_vs_equivalence, measured_speed_vs_equivalence


def main():
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

    # cell_size.measured_vs_raw_mole_concentration.plot(cell_size_data_measured)
    # cell_size.measured_vs_co2_tad_normalized.plot(cell_size_data_measured)
    #
    # cell_size.simulated_vs_raw_mole_concentration.plot(cell_size_data_simulated)
    # cell_size.simulated_vs_co2_tad_normalized.plot(cell_size_data_simulated)
    #
    # cell_size.ratio_vs_equivalence.plot(cell_size_data_ratio)
    #
    # # todo: inert plots
    #
    # gamma.plot(sim_bulk_properties)
    # gamma.plot_ratios(sim_bulk_property_ratios)
    #
    # measured_speed_vs_equivalence.plot(sim_conditions)
    # cj_speed_vs_equivalence.plot(sim_conditions)
    # cj_ratio_vs_equivalence.plot(sim_conditions)
    #
    # sound_speed_vs_equivalence.plot(sim_conditions)
    # mach_vs_equivalence.plot(sim_conditions)
    # cj_mach_vs_equivalence.plot(sim_conditions)
    #
    # # In these plots we see only what we'd expect
    # t_ind_vs_equivalence.plot(sim_conditions)  # no physical measurement for comparison other than cell size

    # cell_size.inert.vs_active(sim_inert_and_active)
    # cell_size.inert.vs_active_2(sim_inert_and_active)
    cell_size.inert.vs_measured(active=cell_size_data_ratio, inert=cell_size_data_ratio_inert)

    # cell_size.ratio_vs_equivalence.plot(cell_size_data_ratio)

    plt.show()


if __name__ == "__main__":
    main()
