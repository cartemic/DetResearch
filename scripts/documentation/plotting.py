from matplotlib import pyplot as plt
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
    axes.set_prop_cycle('color', bmap.mpl_colors)

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
