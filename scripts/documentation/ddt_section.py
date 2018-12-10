# calculate the diameter of a Shchelkin spiral (part of tube documentation)
import beaverdet as bd
import pint
import numpy as np
import cantera as ct
from matplotlib import pyplot as plt        # externalize plotting later!
import palettable                           # remove when externalized


# defines a figure function, which should be externalized later to ensure
# uniformity across all plots
def figure(
        xlabel,
        ylabel,
        plot_title,
        axis_color
):
    # formatting
    axis_font_size = 10
    line_thk = 1.5

    # set colors
    bmap = palettable.tableau.get_map('ColorBlind_10')

    # format text
    plt.rc('text', color=axis_color)
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)

    # define figure and axes
    fig = plt.figure(figsize=(7.5, 2.5))
    axes = fig.add_subplot(111)

    # format axes
    axes.tick_params(labelsize=axis_font_size)
    axes.set_color_cycle(bmap.mpl_colors)

    # set labels
    plt.xlabel(xlabel,
               fontsize=axis_font_size * 1.125,
               fontweight='bold',
               color=axis_color)
    plt.ylabel(ylabel,
               fontsize=axis_font_size * 1.125,
               fontweight='bold',
               color=axis_color)
    plt.title(plot_title,
              fontsize=axis_font_size * 1.5,
              fontweight='bold',
              color='k')

    # format axis lines
    for ctr, spine in enumerate(axes.spines.values()):
        spine.set_color(axis_color)
        if ctr % 2:
            spine.set_visible(False)
        else:
            spine.set_linewidth(line_thk)

    plt.tight_layout()

    return [fig, axes]


# initialize unit registry
ureg = pint.UnitRegistry()
quant = ureg.Quantity

# define initial state
init_temp = quant(20, 'degC')
init_press = quant(1, 'atm')

# define tube and mixture quantities
tube_diam = quant(5.76, 'in').to('cm')      # inner diameter
desired_br = 0.45                           # blockage ratio
mech = 'gri30.cti'                          # mechanism for Cantera calculations
gas = ct.Solution(mech)
equivs = np.linspace(
    0.4,
    1.0,
    20
)                     # desired equivalence ratios
fuels = ['CH4', 'C3H8']                     # fuels to test with
fuel_names = {
    'CH4': 'Methane',
    'C3H8': 'Propane'
}
oxidizer = 'N2O'

# find spiral diameter resulting in desired blockage ratio
spiral_diam = bd.tube.DDT.calculate_spiral_diameter(
    tube_diam,
    desired_br
)
print(
    'Spiral diameter: {0:.2f} ({1:.2f})'.format(
        spiral_diam.to('cm'),
        spiral_diam.to('in')
    )
)

# calculate actual blockage ratio, since a 3/4 inch spiral will be hard to make
num_struts = 4                              # number of blockage struts
washer_diam = tube_diam * np.sqrt(desired_br / num_struts)
print(
    'Calculated diameter: {0:.2f} ({1:.2f})'.format(
        washer_diam.to('cm'),
        washer_diam.to('in')
    )
)

# round washer diameter to the nearest 1/4 inch
washer_diam = np.round(washer_diam.to('in') * 4) / 4.
print(
    'Washer diameter: {0:.2f} ({1:.2f})'.format(
        washer_diam.to('cm'),
        washer_diam.to('in')
    )
)

# calculate new blockage ratio
actual_br = num_struts * (washer_diam.to('m') / tube_diam.to('m')).magnitude**2
print(
    'Actual blockage ratio: {:.2f}%'.format(
        actual_br * 100
                                            )
      )

# calculate run-up distance as a function of equivalence for each mixture
axis_color = '#333333'
fig, axes = figure(
    'Equivalence Ratio',
    'Run-up Distance (m)',
    'DDT Section Minimum Length',
    axis_color
)
for fuel in fuels:
    print(fuel)
    runup_lengths = []

    for phi in equivs:
        gas.set_equivalence_ratio(
            phi,
            fuel,
            oxidizer
        )
        species = gas.mole_fraction_dict()

        runup = bd.tube.DDT.calculate_run_up(
            actual_br,
            tube_diam,
            init_temp,
            init_press,
            species,
            mech,
            ureg
        )
        runup_lengths.append(runup)
        print(
            '    {0}: {1:.2f} ({2:.2f})'.format(
                phi,
                runup,
                runup.to('ft')
            )
        )

    axes.plot(
        equivs,
        [length.to('m').magnitude for length in runup_lengths],
        label=fuel_names[fuel]
    )

axes.legend(
    edgecolor='none'
)

fig.savefig('runup_calcs.pdf ', format='pdf')
plt.show()
